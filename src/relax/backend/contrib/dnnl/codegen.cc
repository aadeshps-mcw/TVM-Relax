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
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/attrs/qdq.h>

#include <algorithm>
#include <optional>
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
      {"dnnl.layer_norm", "relax.nn.layer_norm"},
      {"dnnl.batch_norm", "relax.nn.batch_norm"},
      {"dnnl.softmax", "relax.nn.softmax"},
      {"dnnl.add", "relax.add"},
      {"dnnl.multiply", "relax.multiply"},
      {"dnnl.abs", "relax.abs"},
      {"dnnl.exp", "relax.exp"},
      {"dnnl.log", "relax.log"},
      {"dnnl.sqrt", "relax.sqrt"},
      {"dnnl.round", "relax.round"},
      {"dnnl.relu", "relax.nn.relu"},
      {"dnnl.leaky_relu", "relax.nn.leakyrelu"},
      {"dnnl.tanh", "relax.tanh"},
      {"dnnl.sigmoid", "relax.sigmoid"},
      {"dnnl.clip", "relax.clip"},
      {"dnnl.max_pool1d", "relax.nn.max_pool1d"},
      {"dnnl.max_pool2d", "relax.nn.max_pool2d"},
      {"dnnl.max_pool3d", "relax.nn.max_pool3d"},
      {"dnnl.avg_pool1d", "relax.nn.avg_pool1d"},
      {"dnnl.avg_pool2d", "relax.nn.avg_pool2d"},
      {"dnnl.avg_pool3d", "relax.nn.avg_pool3d"},
      {"dnnl.global_avg_pool2d", "relax.nn.adaptive_avg_pool2d"},
      {"dnnl.conv2d", "relax.nn.conv2d"},
      {"dnnl.matmul", "relax.matmul"},
      {"dnnl.batch_matmul", "relax.nn.batch_matmul"},
      {"dnnl.qnn.conv2d", "relax.nn.conv2d"},
      {"dnnl.qnn.matmul", "relax.matmul"},
  };
  static const std::vector<std::pair<std::string, std::string>> kPrefixResolvers = {
      {"dnnl.conv2d", "relax.nn.conv2d"},
      {"dnnl.matmul", "relax.matmul"},
      {"dnnl.batch_matmul", "relax.nn.batch_matmul"},
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

/*!
 * \brief Returns true if composite_name contains `token` as a full underscore-delimited
 * segment (e.g. HasNameToken("dnnl.conv2d_bias_sum_relu", "sum") is true, but
 * HasNameToken("dnnl.conv2d_summary", "sum") is false). Used only for name/body
 * consistency assertions -- never to drive control flow.
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
 * \brief Structurally searches the composite function body for a relax.clip call.
 * Returns nullptr if none is found (never throws), so callers can branch on presence
 * rather than relying on the composite's name.
 */
const CallNode* FindOpCall(const SeqExprNode* seq, const std::string& op_name) {
  for (const auto& block : seq->blocks) {
    for (const auto& binding : block->bindings) {
      const auto* vb = binding.as<VarBindingNode>();
      const auto* call = vb->value.as<CallNode>();
      if (!call) continue;
      const auto* op_node = call->op.as<OpNode>();
      if (op_node && op_node->name == op_name) return call;
    }
  }
  return nullptr;
}

/*!
 * \brief Structurally searches for a residual-add binding: add(chain, leaf) where
 * chain is an internal var downstream of (but not equal to) root_var, and leaf is a
 * tracked composite input. Returns std::nullopt if no such binding exists.
 */
std::optional<const VarNode*> FindResidualLeaf(
    const SeqExprNode* seq, const VarNode* root_var,
    const std::unordered_map<const VarNode*, NodeEntries>& param_entries) {
  for (const auto& block : seq->blocks) {
    for (const auto& binding : block->bindings) {
      const auto* vb = binding.as<VarBindingNode>();
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

    // Map each composite-function parameter to the entries already produced
    // for the corresponding argument at the outer call site.
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
        leaf_start_index[key] = inputs.size();  // <-- new: record constants too
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

    const VarNode* root_var = nullptr;
    for (const auto& block : seq->blocks) {
      for (const auto& binding : block->bindings) {
        const auto* vb = binding.as<VarBindingNode>();
        if (vb->value.get() == root_call) {
          root_var = vb->var.get();
          break;
        }
      }
      if (root_var) break;
    }
    TVM_FFI_ICHECK(root_var) << "Could not locate binding for root call of " << composite_name;

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
      // Use FindOpCall, which safely returns nullptr instead of crashing
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

      // Reuse the EXISTING attr names the runtime already reads in ParseAttrs --
      node->SetAttr("o_scl_idx", static_cast<int64_t>(scale_idx_it->second));
      node->SetAttr("dst_zp_idx", static_cast<int64_t>(zp_idx_it->second));

      const auto* qattrs = dequantize_call->attrs.as<QuantizeAttrs>();
      TVM_FFI_ICHECK(qattrs != nullptr) << "Expected relax.dequantize to carry QuantizeAttrs";
      node->SetAttr("dst_axis", static_cast<int64_t>(qattrs->axis));

    } else {
      // Safely check for stray dequantize without crashing if it's missing
      const CallNode* stray_dq = FindOpCall(seq, "relax.dequantize");
      TVM_FFI_ICHECK(stray_dq == nullptr)
          << "Composite " << composite_name << " body contains relax.dequantize but its "
          << "name doesn't start with 'dnnl.qnn.' -- naming/body mismatch.";
    }

    SetCallNodeAttribute(node, root_call);
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
