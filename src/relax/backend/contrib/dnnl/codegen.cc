/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/backend/contrib/dnnl/codegen.cc
 * \brief Implementation of the DNNL JSON serializer.
 *
 * The DNNL serializer converts Relax composite functions into JSON graph nodes
 * that can be consumed by the DNNL runtime.
 *
 * A composite function may contain one or more primitive Relax operator calls,
 * representing either a single operation or a fused sequence of operations.
 * The serializer walks the composite body to:
 *   - collect the leaf tensor inputs of the composite;
 *   - extract attributes from each primitive operator call;
 *   - serialize constant scalar and shape arguments as node attributes;
 *   - record the primitive operation sequence in the "fused_ops" attribute.
 *
 * All primitive calls in a composite are represented by a single JSON kernel
 * node. The serializer does not depend on specific composite names or
 * hardcoded fused patterns, allowing new composite patterns to be handled
 * without changes to the codegen logic.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using JSONSerializer = backend::contrib::JSONSerializer;
using backend::contrib::NodeEntries;

namespace {

// Serializes an op's non-tensor arguments (scalars/shapes) as "arg_<name>" attributes on `node`.
// This is where clip's min/max (relax.clip(x, min, max) -- plain call args, not op attrs) get
// picked up, along with any other op's scalar/shape args, without needing to know the op ahead
// of time. The "arg_" prefix avoids colliding with JSONGraphNode's reserved "shape"/"dtype" keys.
void SetArgumentAttributes(const JSONGraphObjectPtr& node, const CallNode* call_node) {
  const auto* op_node = call_node->op.as<OpNode>();
  if (op_node == nullptr) return;
  const ffi::Array<ArgumentInfo>& arg_infos = op_node->arguments;
  for (size_t i = 0; i < call_node->args.size() && i < arg_infos.size(); ++i) {
    const Expr& arg = call_node->args[i];
    const std::string key = "arg_" + std::string(arg_infos[i]->name);
    if (auto prim_value = arg.as<PrimExpr>()) {
      PrimExpr value = prim_value.value();
      if (const auto* imm = value.as<IntImmNode>()) {
        node->SetAttr(key, static_cast<int64_t>(imm->value));
      } else if (const auto* fimm = value.as<FloatImmNode>()) {
        node->SetAttr(key, static_cast<double>(fimm->value));
      }
    } else if (const auto* shape_expr = arg.as<ShapeExprNode>()) {
      ffi::Array<int64_t> values;
      bool all_const = true;
      for (const PrimExpr& e : shape_expr->values) {
        const auto* eimm = e.as<IntImmNode>();
        if (eimm == nullptr) {
          all_const = false;
          break;
        }
        values.push_back(eimm->value);
      }
      if (all_const) node->SetAttr(key, std::move(values));
    }
  }
}

}  // namespace

/*!
 * \brief Serializes a DNNL composite function (a fused chain of one or more primitive Relax ops,
 * e.g. conv2d -> add -> relu -> clip) into a single JSON kernel node.
 *
 * Deliberately contains no conditional dispatch on composite_name or op name: every primitive
 * call inside the composite body contributes its attrs/args to the same node, and the sequence
 * of op names is recorded verbatim in "fused_ops" for the runtime to interpret. Adding a new
 * fused pattern (a new Relax-side pattern registration + a runtime-side post-op handler) requires
 * no codegen.cc change, mirroring how TensorRTJSONSerializer needs no change per new pattern.
 */
class DNNLJSONSerializer : public JSONSerializer {
 public:
  DNNLJSONSerializer(ffi::Map<Constant, ffi::String> constant_names, ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    TVM_FFI_ICHECK(fn_var);
    const auto fn = bindings_[ffi::GetRef<Var>(fn_var)].as_or_throw<Function>();
    TVM_FFI_ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<ffi::String>(attr::kComposite);
    TVM_FFI_ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";
    std::string composite_name = composite_opt.value();

    // Map each composite-function parameter to the entries already produced for the
    // corresponding argument at the outer call site.
    std::unordered_map<const VarNode*, NodeEntries> param_entries;
    for (size_t i = 0; i < fn->params.size(); ++i) {
      param_entries[fn->params[i].get()] = VisitExpr(call_node->args[i]);
    }

    NodeEntries inputs;
    std::unordered_set<const ffi::Object*> seen;
    auto add_leaf_if_new = [&](const Expr& e) {
      const ffi::Object* key = e.get();
      if (seen.count(key)) return;
      if (const auto* var_node = e.as<VarNode>()) {
        auto it = param_entries.find(var_node);
        if (it == param_entries.end()) return;  // internal binding var, not a leaf
        seen.insert(key);
        inputs.insert(inputs.end(), it->second.begin(), it->second.end());
      } else if (e.as<ConstantNode>()) {
        seen.insert(key);
        auto res = VisitExpr(e);
        inputs.insert(inputs.end(), res.begin(), res.end());
      }
    };

    // Single pass over every primitive call in the composite body: gather leaf inputs and record
    // which calls need attr/arg extraction. No call is singled out as "the" root call, and no op
    // name is ever compared against a string. Actual extraction is deferred to after the node is
    // constructed below, since JSONGraphNode needs its final input list at construction time.
    const auto* seq = fn->body.as<SeqExprNode>();
    TVM_FFI_ICHECK(seq) << "Expected composite function body to be a SeqExpr.";
    std::vector<const CallNode*> op_calls;
    for (const auto& block : seq->blocks) {
      for (const auto& binding : block->bindings) {
        const auto* var_binding = binding.as<VarBindingNode>();
        TVM_FFI_ICHECK(var_binding) << "Expected VarBinding inside composite function.";
        const auto* inner_call = var_binding->value.as<CallNode>();
        if (inner_call == nullptr) continue;

        for (const auto& arg : inner_call->args) {
          add_leaf_if_new(arg);
        }

        if (inner_call->op.as<OpNode>() != nullptr) {
          op_calls.push_back(inner_call);
        }
      }
    }
    TVM_FFI_ICHECK(!op_calls.empty()) << "DNNL composite function " << composite_name
                                      << " must contain at least one primitive Relax operator call";

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    ffi::Array<ffi::String> fused_ops;
    for (const CallNode* inner_call : op_calls) {
      const auto* op_node = inner_call->op.as<OpNode>();
      fused_ops.push_back(op_node->name);
      SetCallNodeAttribute(node, inner_call);  // existing generic attr extractor (utils.h)
      SetArgumentAttributes(node, inner_call);
    }
    node->SetAttr("fused_ops", std::move(fused_ops));

    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

ffi::Array<ffi::Module> DNNLCompiler(ffi::Array<Function> functions,
                                     ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                     ffi::Map<Constant, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;

  for (const auto& func : functions) {
    DNNLJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.DNNLJSONRuntimeCreate");
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<ffi::Module>());
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.dnnl", DNNLCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
