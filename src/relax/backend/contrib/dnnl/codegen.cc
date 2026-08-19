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
 * Unlike TensorRT's composite functions (which always wrap exactly one primitive op), a DNNL
 * composite represents a *fused chain* -- e.g. "dnnl.conv2d_bias_relu" contains three primitive
 * calls: conv2d, add (bias), relu. So there is no single "root call" to resolve by name.
 * Baseline for this file is TensorRT's codegen.cc: instead of matching composite_name (or an op
 * name) against a table to decide which call to extract attrs from, we walk every binding in the
 * composite body once and, for every primitive call found:
 *   - copy its op attrs via the existing SetCallNodeAttribute() helper (same helper the original
 *     conv2d-only code used for its single root_call)
 *   - serialize its non-tensor scalar/shape arguments as "arg_<name>" attrs -- this is what
 *     replaces the old hardcoded "_clip" check; relax.clip's min/max are call args, not attrs,
 *     and this now applies to any op with such args, not just clip
 *   - append its op name to a "fused_ops" attr, so the runtime knows the fusion sequence without
 *     codegen needing a hardcoded list of which ops DNNL supports as post-ops
 * The leaf-tensor-input-gathering logic below (param_entries / add_leaf_if_new) was already fully
 * generic before this change and is unmodified in spirit -- it's folded into the same single walk
 * of the composite body's bindings so the body is only traversed once.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

/*!
 * \brief Resolves a composite pattern name (e.g. "dnnl.conv2d_bias_relu") to the
 * CallNode for its underlying root op inside the composite function.
 *
 * Exact-match entries are used for patterns that have no fused variants sharing
 * a name prefix (matmul, layer_norm). Prefix-match entries are used for op
 * families with fused variants that all legitimately share a root op (the
 * conv2d family: dnnl.conv2d, dnnl.conv2d_relu, dnnl.conv2d_bias_relu, ...).
 *
 * Prefix matching is deliberately anchored at the start of the string
 * (not a substring search) so that, e.g., a future "dnnl.qnn.conv2d" pattern
 * does not incorrectly match the "dnnl.conv2d" prefix.
 */
const CallNode* ResolveRootCall(const std::string& composite_name, const Function& fn) {
  static const std::unordered_map<std::string, std::string> kExactResolvers = {
      {"dnnl.matmul", "relax.matmul"},
      {"dnnl.layer_norm", "relax.nn.layer_norm"},
  };
  static const std::vector<std::pair<std::string, std::string>> kPrefixResolvers = {
      {"dnnl.conv2d", "relax.nn.conv2d"},
  };

  auto exact_it = kExactResolvers.find(composite_name);
  if (exact_it != kExactResolvers.end()) {
    return backend::GetOpInFunction(fn, exact_it->second);
  }

  for (const auto& prefix_and_op : kPrefixResolvers) {
    const std::string& prefix = prefix_and_op.first;
    if (composite_name.rfind(prefix, 0) == 0) {  // composite_name starts with prefix
      return backend::GetOpInFunction(fn, prefix_and_op.second);
    }
  }

  TVM_FFI_THROW(InternalError) << "Unimplemented pattern: " << composite_name;
  return nullptr;
}

}  // namespace
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

    // Map each composite-function parameter to the entries already produced
    // for the corresponding argument at the outer call site.
    std::unordered_map<const VarNode*, NodeEntries> param_entries;
    for (size_t i = 0; i < fn->params.size(); ++i) {
      param_entries[fn->params[i].get()] = VisitExpr(call_node->args[i]);
    }

    NodeEntries inputs;
    std::unordered_set<const ffi::Object*> seen;
    std::unordered_map<const VarNode*, Expr> local_bindings;

    auto add_leaf_if_new = [&](const Expr& e) {
      const ffi::Object* key = e.get();
      if (seen.count(key)) return;
      if (const auto* var_node = e.as<VarNode>()) {
        if (auto it = param_entries.find(var_node); it != param_entries.end()) {
          seen.insert(key);
          inputs.insert(inputs.end(), it->second.begin(), it->second.end());
          return;
        }
        // Not an external param -- may be an internal var bound to a constant.
        if (auto lb_it = local_bindings.find(var_node);
            lb_it != local_bindings.end() && lb_it->second.as<ConstantNode>()) {
          seen.insert(key);
          auto res = VisitExpr(lb_it->second);
          inputs.insert(inputs.end(), res.begin(), res.end());
        }
        // else: bound to a non-constant (e.g. another call's result) -- genuinely not a leaf
      } else if (e.as<ConstantNode>()) {
        seen.insert(key);
        auto res = VisitExpr(e);
        inputs.insert(inputs.end(), res.begin(), res.end());
      }
    };

    const auto* seq = fn->body.as<SeqExprNode>();
    TVM_FFI_ICHECK(seq) << "Expected composite function body to be a SeqExpr.";
    for (const auto& block : seq->blocks) {
      for (const auto& binding : block->bindings) {
        const auto* var_binding = binding.as<VarBindingNode>();
        TVM_FFI_ICHECK(var_binding) << "Expected VarBinding inside composite function.";
        local_bindings[var_binding->var.get()] = var_binding->value;
        if (const auto* inner_call = var_binding->value.as<CallNode>()) {
          for (const auto& arg : inner_call->args) {
            add_leaf_if_new(arg);
          }
        }
      }
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    const CallNode* root_call = ResolveRootCall(composite_name, fn);

    // clip's bounds are plain TIR FloatImm call args in Relax (relax.clip(x, min, max)),
    // not op attrs, so SetCallNodeAttribute(node, root_call) below -- which only extracts
    // the *root op's* own attrs (e.g. conv2d's strides/padding) -- never sees them.
    // Extract them here and attach as "a_min"/"a_max" JSON attrs, which is what the DNNL
    // runtime's ParseAttrs() fallback (dnnl_json_runtime.cc) reads for "_clip"-suffixed
    // composites. Without this, clip silently runs with bounds (0, 0).
    if (composite_name.find("_clip") != std::string::npos) {
      const CallNode* clip_call = backend::GetOpInFunction(fn, "relax.clip");
      TVM_FFI_ICHECK(clip_call) << "Expected to find relax.clip inside composite "
                                << composite_name;
      TVM_FFI_ICHECK_EQ(clip_call->args.size(), 3U)
          << "Expected relax.clip(x, min, max) to have 3 args";

      const auto* min_imm = clip_call->args[1].as<FloatImmNode>();
      const auto* max_imm = clip_call->args[2].as<FloatImmNode>();
      TVM_FFI_ICHECK(min_imm) << "Expected relax.clip's min arg to be a FloatImm";
      TVM_FFI_ICHECK(max_imm) << "Expected relax.clip's max arg to be a FloatImm";

      node->SetAttr("a_min", min_imm->value);
      node->SetAttr("a_max", max_imm->value);
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
