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
 * The DNNL serializer converts Relax composite functions into JSON graph nodes that can be
 * consumed by the DNNL runtime. A composite function may contain one or more primitive Relax
 * operator calls, representing either a single operation or a fused chain of operations
 * (e.g. conv2d -> add[bias] -> add[residual sum] -> relu -> clip).
 *
 * Every primitive call in the composite contributes to a single JSON kernel node:
 *   - op attrs (SetCallNodeAttribute) and plain scalar/shape call args (SetArgumentAttributes)
 *     are extracted generically from *every* primitive call in the chain, not just the "main"
 *     op -- this is what lets a parametrized post-op (e.g. a fused leaky_relu's alpha) survive
 *     without codegen.cc needing to know about it ahead of time. The op name sequence is
 *     recorded verbatim in "fused_ops".
 *   - Three details can't be derived generically and are extracted structurally instead, keyed
 *     under the exact attribute names the DNNL runtime's ParseAttrs() (dnnl_json_runtime.cc)
 *     expects:
 *       * clip's bounds ("a_min"/"a_max") -- these are plain FloatImm call args in Relax
 *         (relax.clip(x, min, max)), not an Attrs struct, so the generic extractors above never
 *         see them, and the generic "arg_min"/"arg_max" keys SetArgumentAttributes would produce
 *         are not what the runtime reads. Without this explicit step clip silently runs with
 *         bounds (0, 0).
 *       * a fused residual add ("sum_idx") -- identified by locating a `relax.add` binding whose
 *         first operand is downstream of the composite's primary op and whose second operand is
 *         a tracked composite input; the input's index is recorded.
 *       * a fused QNN dequantize ("o_scl_idx", "dst_zp_idx", "dst_axis") -- identified by
 *         locating a `relax.dequantize` binding and recording the composite-input indices of its
 *         scale/zero_point operands plus its axis attr.
 *   - Every composite/body pairing above is guarded by a naming/structure consistency check
 *     (HasNameToken) that fails loudly if e.g. a composite named with "_sum" has no residual-add
 *     in its body, or vice versa, rather than silently mis-serializing.
 *
 * The chain's primary op (needed only to anchor residual-add detection) is not resolved via a
 * per-op-family name table. Composite bodies here are straight-line (no branches): every op
 * after the first consumes, directly or transitively, the output of an earlier one. So the
 * first primitive call encountered in binding order is always the primary op, and is captured
 * as such during the same single pass that builds `op_calls` -- no table to maintain when a new
 * op family (e.g. a new conv variant) is added. See the gaps note at the bottom of this file for
 * the assumption this relies on.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/qdq.h>
#include <tvm/relax/expr.h>

#include <algorithm>
#include <optional>
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
// This picks up any op's plain PrimExpr/ShapeExpr call args generically, without needing to know
// the op ahead of time. The "arg_" prefix avoids colliding with JSONGraphNode's reserved
// "shape"/"dtype" keys, and with the "a_"/"o_"/"dst_"-prefixed keys set explicitly below for
// clip/sum/QNN (which use runtime-mandated names that don't follow this generic convention).
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

/*!
 * \brief Returns true if composite_name contains `token` as a full underscore/dot-delimited
 * segment (e.g. HasNameToken("dnnl.conv2d_bias_sum_relu", "sum") is true, but
 * HasNameToken("dnnl.conv2d_summary", "sum") is false). Used only for name/body consistency
 * assertions -- never to drive control flow.
 */
bool HasNameToken(const std::string& composite_name, const std::string& token) {
  size_t start = 0;
  while (start <= composite_name.size()) {
    size_t end_underscore = composite_name.find('_', start);
    size_t end_dot = composite_name.find('.', start);
    size_t end = std::min(end_underscore, end_dot);
    if (end == std::string::npos) end = composite_name.size();
    if (composite_name.compare(start, end - start, token) == 0) return true;
    start = end + 1;
  }
  return false;
}

/*!
 * \brief Structurally searches the composite function body for a call to `op_name`. Returns
 * nullptr if none is found (never throws), so callers can branch on presence rather than
 * relying on the composite's name.
 */
const CallNode* FindOpCall(const SeqExprNode* seq, const std::string& op_name) {
  for (const auto& block : seq->blocks) {
    for (const auto& binding : block->bindings) {
      const auto* vb = binding.as<VarBindingNode>();
      if (!vb) continue;
      const auto* call = vb->value.as<CallNode>();
      if (!call) continue;
      const auto* op_node = call->op.as<OpNode>();
      if (op_node && op_node->name == op_name) return call;
    }
  }
  return nullptr;
}

/*!
 * \brief Structurally searches for a residual-add binding: add(chain, leaf) where chain is an
 * internal var downstream of (but not equal to) root_var, and leaf is a tracked composite
 * input. Returns std::nullopt if no such binding exists.
 */
std::optional<const VarNode*> FindResidualLeaf(
    const SeqExprNode* seq, const VarNode* root_var,
    const std::unordered_map<const VarNode*, NodeEntries>& param_entries) {
  for (const auto& block : seq->blocks) {
    for (const auto& binding : block->bindings) {
      const auto* vb = binding.as<VarBindingNode>();
      if (!vb) continue;
      const auto* add_call = vb->value.as<CallNode>();
      if (!add_call || add_call->args.size() != 2) continue;
      const auto* op_node = add_call->op.as<OpNode>();
      if (!op_node || op_node->name != "relax.add") continue;

      for (int i = 0; i < 2; ++i) {
        const auto* chain_var = add_call->args[i].as<VarNode>();
        const auto* leaf_var = add_call->args[1 - i].as<VarNode>();
        bool chain_is_downstream_internal =
            chain_var && chain_var != root_var && !param_entries.count(chain_var);
        if (chain_is_downstream_internal && leaf_var && param_entries.count(leaf_var)) {
          return leaf_var;
        }
      }
    }
  }
  return std::nullopt;
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

    // Map each composite-function parameter to the entries already produced for the
    // corresponding argument at the outer call site.
    std::unordered_map<const VarNode*, NodeEntries> param_entries;
    for (size_t i = 0; i < fn->params.size(); ++i) {
      param_entries[fn->params[i].get()] = VisitExpr(call_node->args[i]);
    }

    NodeEntries inputs;
    std::unordered_set<const ffi::Object*> seen;
    std::unordered_map<const ffi::Object*, size_t> leaf_start_index;
    auto add_leaf_if_new = [&](const Expr& e) {
      const ffi::Object* key = e.get();
      if (seen.count(key)) return;
      if (const auto* var_node = e.as<VarNode>()) {
        auto it = param_entries.find(var_node);
        if (it == param_entries.end()) return;  // internal binding var, not a leaf
        seen.insert(key);
        leaf_start_index[key] = inputs.size();
        inputs.insert(inputs.end(), it->second.begin(), it->second.end());
      } else if (e.as<ConstantNode>()) {
        seen.insert(key);
        leaf_start_index[key] = inputs.size();
        auto res = VisitExpr(e);
        inputs.insert(inputs.end(), res.begin(), res.end());
      }
    };

    // Single pass over every primitive call in the composite body: gather leaf inputs (composite
    // params + constants) and the ordered list of primitive op calls. No call is singled out
    // here as "the" root call -- that resolution happens below, after the node exists, since
    // JSONGraphNode needs its final input list at construction time.
    const auto* seq = fn->body.as<SeqExprNode>();
    TVM_FFI_ICHECK(seq) << "Expected composite function body to be a SeqExpr.";
    std::vector<const CallNode*> op_calls;
    const CallNode* root_call = nullptr;
    const VarNode* root_var = nullptr;
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
          if (root_call == nullptr) {
            root_call = inner_call;
            root_var = var_binding->var.get();
          }
        }
      }
    }
    TVM_FFI_ICHECK(!op_calls.empty()) << "DNNL composite function " << composite_name
                                      << " must contain at least one primitive Relax operator call";

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    // Generic per-call attribute extraction: every primitive call in the chain contributes its
    // op attrs and any plain scalar/shape call args -- not just the primary op. This is what
    // lets a parametrized post-op (e.g. a fused leaky_relu's alpha) survive without codegen.cc
    // needing a special case for it. The op name sequence is recorded verbatim in "fused_ops".
    ffi::Array<ffi::String> fused_ops;
    for (const CallNode* inner_call : op_calls) {
      const auto* op_node = inner_call->op.as<OpNode>();
      fused_ops.push_back(op_node->name);
      SetCallNodeAttribute(node, inner_call);
      SetArgumentAttributes(node, inner_call);
    }
    node->SetAttr("fused_ops", std::move(fused_ops));

    // clip's bounds are plain TIR FloatImm call args (relax.clip(x, min, max)), not op attrs, so
    // the generic extraction above never captures them under the runtime-expected keys. Extract
    // them here as "a_min"/"a_max", which is what the DNNL runtime's ParseAttrs() fallback
    // (dnnl_json_runtime.cc) reads for "_clip"-suffixed composites. Without this, clip silently
    // runs with bounds (0, 0).
    const CallNode* clip_call = FindOpCall(seq, "relax.clip");
    TVM_FFI_ICHECK_EQ(clip_call != nullptr, HasNameToken(composite_name, "clip"))
        << "Composite " << composite_name << " naming/body mismatch for clip pattern "
        << "(name implies clip=" << HasNameToken(composite_name, "clip")
        << ", body has relax.clip=" << (clip_call != nullptr) << ")";

    if (clip_call != nullptr) {
      TVM_FFI_ICHECK_EQ(clip_call->args.size(), 3U)
          << "Expected relax.clip(x, min, max) to have 3 args";
      const auto* min_imm = clip_call->args[1].as<FloatImmNode>();
      const auto* max_imm = clip_call->args[2].as<FloatImmNode>();
      TVM_FFI_ICHECK(min_imm) << "Expected relax.clip's min arg to be a FloatImm";
      TVM_FFI_ICHECK(max_imm) << "Expected relax.clip's max arg to be a FloatImm";
      node->SetAttr("a_min", min_imm->value);
      node->SetAttr("a_max", max_imm->value);
    }

    auto residual = FindResidualLeaf(seq, root_var, param_entries);
    TVM_FFI_ICHECK_EQ(residual.has_value(), HasNameToken(composite_name, "sum"))
        << "Composite " << composite_name << " naming/body mismatch for residual-add pattern "
        << "(name implies sum=" << HasNameToken(composite_name, "sum")
        << ", body has residual=" << residual.has_value() << ")";

    if (residual.has_value()) {
      auto idx_it = leaf_start_index.find(static_cast<const ffi::Object*>(*residual));
      TVM_FFI_ICHECK(idx_it != leaf_start_index.end())
          << "Residual leaf for " << composite_name << " was not tracked as a composite input";
      node->SetAttr("sum_idx", static_cast<int64_t>(idx_it->second));
    }

    bool is_qnn_composite = composite_name.rfind("dnnl.qnn.", 0) == 0;
    if (is_qnn_composite) {
      const CallNode* dequantize_call = FindOpCall(seq, "relax.dequantize");
      TVM_FFI_ICHECK(dequantize_call != nullptr)
          << "Composite " << composite_name << " naming implies a fused dequantize "
          << "but none was found in the body.";
      TVM_FFI_ICHECK_EQ(dequantize_call->args.size(), 3U)
          << "Expected relax.dequantize(data, scale, zero_point) to have 3 args";

      const Expr& scale_expr = dequantize_call->args[1];
      const Expr& zp_expr = dequantize_call->args[2];
      auto scale_idx_it = leaf_start_index.find(scale_expr.get());
      auto zp_idx_it = leaf_start_index.find(zp_expr.get());
      TVM_FFI_ICHECK(scale_idx_it != leaf_start_index.end())
          << "QNN scale for " << composite_name << " was not tracked as a composite input "
          << "(expected a constant leaf, per the is_const() pattern constraint)";
      TVM_FFI_ICHECK(zp_idx_it != leaf_start_index.end())
          << "QNN zero_point for " << composite_name << " was not tracked as a composite input";

      node->SetAttr("o_scl_idx", static_cast<int64_t>(scale_idx_it->second));
      node->SetAttr("dst_zp_idx", static_cast<int64_t>(zp_idx_it->second));

      const auto* qattrs = dequantize_call->attrs.as<QuantizeAttrs>();
      TVM_FFI_ICHECK(qattrs != nullptr) << "Expected relax.dequantize to carry QuantizeAttrs";
      node->SetAttr("dst_axis", static_cast<int64_t>(qattrs->axis));
    } else {
      const CallNode* stray_dq = FindOpCall(seq, "relax.dequantize");
      TVM_FFI_ICHECK(stray_dq == nullptr)
          << "Composite " << composite_name << " body contains relax.dequantize but its "
          << "name doesn't start with 'dnnl.qnn.' -- naming/body mismatch.";
    }

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
