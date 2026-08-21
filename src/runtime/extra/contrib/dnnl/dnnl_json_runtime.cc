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
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

// TODO(@apeskov): Have to mute warning from dnnl headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

#include "dnnl_tensor_requisite.h"
#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

namespace {
inline bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
template <typename... Args>
inline bool contains_any(const std::string& s, const Args&... args) {
  return (contains(s, args) || ...);
}
inline std::string StripDnnlPrefix(const std::string& s) {
  static const std::string kPrefix = "dnnl.";
  return s.rfind(kPrefix, 0) == 0 ? s.substr(kPrefix.size()) : s;
}
}  // namespace

class DNNLJSONRuntime : public JSONRuntimeBase {
 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        next_unique_eid_offset_(data_entry_.size()),
        run_arg_eid_(input_var_eid_) {
    for (const auto e : outputs_) run_arg_eid_.push_back(EntryID(e));
  }

  const char* kind() const override { return "dnnl_json"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
    BuildEngine();
  }

  /* Unused stub implementation */
  void Run() override { TVM_FFI_THROW(InternalError) << "Unreachable code"; }

  /* Thread safe implementation of Run. Keep runtime instance immutable */
  void Run(const ffi::PackedArgs& args) const {
    auto arg_data_provider = makeIODataProvider(args);
    auto mem_solver = tensor_registry_.MakeSolver(arg_data_provider);
    // Execute primitives one by one
    for (const auto& act : net_) {
      auto prim = std::get<0>(act);
      auto arg_reqs = std::get<1>(act);

      // Find proper dnnl::memory buffers
      std::unordered_map<int, dnnl::memory> mem_args;
      for (const auto& kvp : arg_reqs) mem_args[kvp.first] = mem_solver(kvp.second);

      // skip the reorder if src==dst to enable inplace operation
      if (prim.get_kind() == dnnl::primitive::kind::reorder) {
        const auto& mem_src = mem_args.at(DNNL_ARG_SRC);
        const auto& mem_dst = mem_args.at(DNNL_ARG_DST);
        if ((mem_src.get_desc() == mem_dst.get_desc()) &&
            (mem_src.get_data_handle() == mem_dst.get_data_handle())) {
          continue;
        }
      }

      prim.execute(stream_, mem_args);
    }
  }

  /* Override GetFunction to reimplement Run method */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    if (this->symbol_name_ == name) {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK(this->initialized_) << "The module has not been initialized";

        TVM_FFI_ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
            << "Found mismatch in the number of provided data entries and required.";

        Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name);
    }
  }

  /* Same as makeInitDataProvider but in case of InputOutput return real DLTensor */
  TensorRegistry::DLTensorProvider makeIODataProvider(const ffi::PackedArgs& args) const {
    std::map<uint32_t, const DLTensor*> io_map;  // eid to dl tensor map
    for (size_t i = 0; i < run_arg_eid_.size(); i++) {
      io_map[run_arg_eid_[i]] = args[i].cast<DLTensor*>();
    }

    // lambda with captured IO data handlers
    return [io_map](uint32_t eid) -> const DLTensor* { return io_map.at(eid); };
  }

 private:
  const std::map<std::string, dnnl::algorithm> elt_name2algo{
      {"abs", dnnl::algorithm::eltwise_abs},
      {"exp", dnnl::algorithm::eltwise_exp},
      {"log", dnnl::algorithm::eltwise_log},
      {"sqrt", dnnl::algorithm::eltwise_sqrt},
      {"round", dnnl::algorithm::eltwise_round},
      // {"logsumexp", dnnl::algorithm::eltwise_logsigmoid},
      {"nn.relu", dnnl::algorithm::eltwise_relu},
      {"nn.leaky_relu", dnnl::algorithm::eltwise_relu},
      {"relu", dnnl::algorithm::eltwise_relu},
      {"leaky_relu", dnnl::algorithm::eltwise_relu},
      {"tanh", dnnl::algorithm::eltwise_tanh},
      {"sigmoid", dnnl::algorithm::eltwise_logistic},
      {"clip", dnnl::algorithm::eltwise_clip},
      {"gelu_erf", dnnl::algorithm::eltwise_gelu_erf},
      {"gelu", dnnl::algorithm::eltwise_gelu_erf},
      {"silu", dnnl::algorithm::eltwise_swish}};

  dnnl::primitive_attr ParseAttrs(const size_t& nid, TensorRequisite* bias_tr,
                                  TensorRequisite* o_scl_tr_out, int* scale_post_op_idx_out) {
    dnnl::primitive_attr attr;
    *scale_post_op_idx_out = -1;

    auto dst_zp_tr = GetInputByName(nid, "dst_zp_idx");
    auto o_scl_tr = GetInputByName(nid, "o_scl_idx");
    auto sum_scl_tr = GetInputByName(nid, "sum_scl_idx");

    auto activation = GetNodeAttr<std::vector<std::string>>(nodes_[nid], "activation", {"none"});
    if (activation[0] != "none") {
      TVM_FFI_ICHECK(elt_name2algo.count(activation[0]))
          << "Unhandled activation string in ParseAttrs: '" << activation[0] << "'";
      auto a_type = elt_name2algo.at(activation[0]);
      auto a_alfa = GetInput(nid, std::stoi(activation[2])).GetConstScalarData<float>();
      auto a_beta = GetInput(nid, std::stoi(activation[3])).GetConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_eltwise(a_type, a_alfa, a_beta);
      attr.set_post_ops(ops);
    }

    if (sum_scl_tr) {
      auto scl = sum_scl_tr.GetConstScalarData<float>();
      auto ops = attr.get_post_ops();
      ops.append_sum(scl);
      attr.set_post_ops(ops);
    }

    // Resolve dst zero-point value up front (needed standalone or combined
    // with a scalar scale, and possibly consumed early in the per-channel
    // branch below).
    bool has_zp = static_cast<bool>(dst_zp_tr);
    float zp = 0.f;
    if (has_zp) {
      TVM_FFI_ICHECK(dst_zp_tr.IsConstant());
      TVM_FFI_ICHECK(dst_zp_tr.IsScalar())
          << "DNNL's output-rescale zero-point is applied via a scalar eltwise "
          << "post-op (alpha=1, beta=-zp); per-channel zero_point is not "
          << "supported by this path -- pair a per-channel scale with a "
          << "scalar (size-1) zero_point, as is standard for per-channel "
          << "(symmetric) quantization schemes.";
      auto zp_dtype = dst_zp_tr.desc().get_data_type();
      if (zp_dtype == dnnl::memory::data_type::f32) {
        zp = dst_zp_tr.GetConstScalarData<float>();
      } else if (zp_dtype == dnnl::memory::data_type::s32) {
        zp = static_cast<float>(dst_zp_tr.GetConstScalarData<int32_t>());
      } else {
        TVM_FFI_THROW(InternalError) << "Unsupported dst_zp dtype for DNNL output-rescale post-op";
      }
    }

    // Dst scale: oneDNN's runtime DST-scale *attribute* is rejected for
    // pure-f32 primitives ("unsupported attribute", confirmed via
    // ONEDNN_VERBOSE), so both the scalar and per-channel cases are folded
    // into post-ops instead of set_scales_mask.
    if (o_scl_tr) {
      TVM_FFI_ICHECK(o_scl_tr.IsConstant());
      auto data = o_scl_tr.GetConstDataLikeVec<float>();

      if (data.size() == 1) {
        // Scalar: combine directly with zp into one eltwise_linear post-op.
        // (x - zp) * scale == scale*x + (-zp*scale)
        float scale = data[0];
        auto ops = attr.get_post_ops();
        ops.append_eltwise(dnnl::algorithm::eltwise_linear, scale, has_zp ? -zp * scale : 0.f);
        attr.set_post_ops(ops);
        has_zp = false;
        *o_scl_tr_out = TensorRequisite{};  // fully folded; nothing to pass to Submit
      } else {
        // Per-channel: apply zp first (if any) as its own eltwise post-op,
        // to preserve (x - zp) * scale ordering, then the scale as a
        // binary_mul post-op with a broadcastable [1,...,C,...,1] operand.
        if (has_zp) {
          auto zp_ops = attr.get_post_ops();
          zp_ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, -zp);
          attr.set_post_ops(zp_ops);
          has_zp = false;
        }

        int dst_axis = GetNodeAttr<int>(nodes_[nid], "dst_axis", 1);
        auto dst_dims_full = GetOutput(nid, 0).dims();
        if (dst_axis < 0) dst_axis += static_cast<int>(dst_dims_full.size());

        std::vector<int64_t> scale_dims(dst_dims_full.size(), 1);
        scale_dims[dst_axis] = static_cast<int64_t>(data.size());
        auto scale_tr = o_scl_tr.Reshape(scale_dims);

        auto ops = attr.get_post_ops();
        *scale_post_op_idx_out = ops.len();
        ops.append_binary(dnnl::algorithm::binary_mul, scale_tr.desc());
        attr.set_post_ops(ops);

        *o_scl_tr_out = scale_tr;
      }
    }

    // Standalone zp: only reached if o_scl_tr was absent, or present but
    // hadn't already consumed it above.
    if (has_zp) {
      auto ops = attr.get_post_ops();
      ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f, -zp);
      attr.set_post_ops(ops);
    }

    *bias_tr = GetInputByName(nid, "bias_idx");

    if (activation[0] != "none" || sum_scl_tr || dst_zp_tr) return attr;

    auto op_name = nodes_[nid].GetOpName();
    dnnl::post_ops ops;
    if (contains(op_name, "_sum")) ops.append_sum(1.f);
    if (contains(op_name, "_relu")) ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
    if (contains(op_name, "_tanh")) ops.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
    if (contains(op_name, "_clip")) {
      float a_min = GetNodeAttr<float>(nodes_[nid], "a_min");
      float a_max = GetNodeAttr<float>(nodes_[nid], "a_max");
      ops.append_eltwise(dnnl::algorithm::eltwise_clip, a_min, a_max);
    }
    if (contains(op_name, "_sigmoid"))
      ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
    if (contains(op_name, "_swish")) ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.f, 1.f);
    if (contains(op_name, "_gelu")) ops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
    if (contains(op_name, "_mish")) ops.append_eltwise(dnnl::algorithm::eltwise_mish, 1.f, 0.f);
    if (ops.len() != 0) attr.set_post_ops(ops);

    if (!bias_tr->defined()) {
      *bias_tr = contains(op_name, "_bias") ? GetInput(nid, 2) : TensorRequisite{};
    }

    return attr;
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    std::set<uint32_t> io_eid_set(run_arg_eid_.begin(), run_arg_eid_.end());
    tensor_registry_ = TensorRegistry(engine_, io_eid_set);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        TVM_FFI_ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        auto stripped_op_name = StripDnnlPrefix(op_name);
        if (contains_any(op_name, "deconv1d", "deconv2d", "deconv3d", "conv1d_transpose",
                         "conv2d_transpose", "conv3d_transpose")) {
          Deconvolution(nid);
        } else if (contains_any(op_name, "conv1d", "conv2d", "conv3d")) {
          Convolution(nid);
        } else if (contains(op_name, "dense")) {
          Dense(nid);
        } else if ("batch_norm" == stripped_op_name) {
          BatchNorm(nid);
        } else if (contains(op_name, "global_avg_pool2d")) {
          GlobalAvgPooling(nid);
        } else if (contains_any(op_name, "max_pool1d", "max_pool2d", "max_pool3d")) {
          Pooling(nid, dnnl::algorithm::pooling_max);
        } else if (contains_any(op_name, "avg_pool1d", "avg_pool2d", "avg_pool3d")) {
          Pooling(nid, dnnl::algorithm::pooling_avg_exclude_padding);
        } else if (elt_name2algo.count(stripped_op_name)) {
          Eltwise(nid);
        } else if ("softmax" == stripped_op_name) {
          Softmax(nid);
        } else if ("add" == stripped_op_name) {
          Binary(nid, dnnl::algorithm::binary_add);
        } else if ("multiply" == stripped_op_name) {
          Binary(nid, dnnl::algorithm::binary_mul);
        } else if (contains(op_name, "layer_norm")) {
          LayerNorm(nid);
        } else if (contains(op_name, "matmul") && !contains(op_name, "batch_matmul")) {
          MatMul(nid);
        } else if (contains(op_name, "batch_matmul")) {
          BatchMatMul(nid);
        } else {
          TVM_FFI_THROW(InternalError) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  void Convolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = nodes_[nid].GetOpName();

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};
    auto o_scl_tr = TensorRequisite{};
    int scale_post_op_idx = -1;
    auto attr = ParseAttrs(nid, &bias_tr, &o_scl_tr, &scale_post_op_idx);

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    auto groups = GetNodeAttr<int>(node, "groups");
    auto src_layout = GetNodeAttr<std::string>(node, "data_layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    auto wgh_layout = GetNodeAttr<std::string>(node, "kernel_layout");

    // dst_layout == "" means to use data_layout
    if (dst_layout.empty()) dst_layout = src_layout;

    // Minus one for DNNL representation. No dilation for DNNL is 0
    for (auto& d : dilates) d--;

    // Take into account provided layout strings
    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);
    wgh_tr = wgh_tr.TreatAs(wgh_layout);

    // Should support G mixed with O. Like { G*O, I, H, W }
    // Use { G, O, I, H, W } weight format even if groups == 1.
    // Regular conv keeps the undivided output-channel count at logical axis 0 (O).
    if (wgh_layout.find("G") == std::string::npos) {
      wgh_tr = TensorRequisite::ApplyGroupWeightLayout(wgh_tr, groups, /*full_axis=*/0);
    }

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // TODO(@apeskov): This is WA. In case of padded blocked tensor format we do not know original
    //  shapes. Example tensor {1, 10, 224, 224} with layout "NCNH8c" will lead to tensor
    //  {1, 2, 224, 224, 8}. Identically as for shapes {1, 11, 224, 224} or {1, 15, 224, 224}.
    //
    // Let's try to compensate it for weight tensor. Weight IC should match with source IC.
    // Example src: [1, 3, 224, 224] with layout NCHW
    //         wgh: [16, 3, 3, 3] with layout OIHW2i8o -> [2, 2, 3, 3, 2, 8]
    // Similarly, Weight OC should match with destination OC.
    // Example dst: [1, 1000, 7, 7] with layout NCHW
    //         wgh: [1000, 1024, 1, 1] with layout OIHW48o -> [21, 1024, 1, 1, 48]
    if (wgh_tr.dims()[0] != groups || wgh_tr.dims()[1] != dst_tr.dims()[1] / groups ||
        wgh_tr.dims()[2] != src_tr.dims()[1] / groups) {
      auto wgh_croped_dims = wgh_tr.dims();
      wgh_croped_dims[0] = groups;
      wgh_croped_dims[1] = dst_tr.dims()[1] / groups;  // wgh_OC = dst_OC / groups
      wgh_croped_dims[2] = src_tr.dims()[1] / groups;  // wgh_IC = src_IC / groups
      auto zero_offset = dnnl::memory::dims(wgh_tr.dims().size(), 0);
      wgh_tr = wgh_tr.Crop(wgh_croped_dims, zero_offset);
    }

    // Conv description.
    // Interim fix: Force plain layouts for int8/uint8 to bypass a known bug in
    // memory allocation for reordered blocked layouts with zero-point compensation.
    bool is_int8 = (src_tr.desc().get_data_type() == dnnl::memory::data_type::s8 ||
                    src_tr.desc().get_data_type() == dnnl::memory::data_type::u8);

    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        is_int8 ? src_tr.desc() : src_tr.LayoutAny().desc(),
        is_int8 ? wgh_tr.desc() : wgh_tr.LayoutAny().desc(),
        is_int8 ? bias_tr.desc() : bias_tr.LayoutAny().desc(),
        is_int8 ? dst_tr.desc() : dst_tr.LayoutAny().desc(), strides, dilates, padding_l, padding_r,
        attr);

    src_tr = src_tr.RequestLayout(conv_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(conv_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(conv_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(conv_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(conv_prim_desc.scratchpad_desc());

    // TODO(@apeskov): Simulation of inplace primitive. just as PoC.
    auto sum_in_tr = GetInputByName(nid, "sum_idx").TreatAs(dst_layout);
    if (op_name.find("_sum") != std::string::npos) {
      sum_in_tr = GetInput(nid, node.GetInputs().size() - 1);
      sum_in_tr = sum_in_tr.TreatAs(dst_layout);
    }

    std::unordered_map<int, TensorRequisite> conv_args = {{DNNL_ARG_SRC, src_tr},
                                                          {DNNL_ARG_WEIGHTS, wgh_tr},
                                                          {DNNL_ARG_BIAS, bias_tr},
                                                          {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                                          {DNNL_ARG_DST, dst_tr}};
    if (scale_post_op_idx >= 0) {
      conv_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(scale_post_op_idx) | DNNL_ARG_SRC_1] = o_scl_tr;
    }
    Submit(dnnl::convolution_forward(conv_prim_desc), conv_args, {sum_in_tr, DNNL_ARG_DST});
  }

  void Deconvolution(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};
    auto o_scl_tr = TensorRequisite{};
    int scale_post_op_idx = -1;
    auto attr = ParseAttrs(nid, &bias_tr, &o_scl_tr, &scale_post_op_idx);

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");
    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());
    auto groups = GetNodeAttr<int>(node, "groups");
    auto src_layout = GetNodeAttr<std::string>(node, "data_layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    auto wgh_layout = GetNodeAttr<std::string>(node, "kernel_layout");

    if (dst_layout.empty()) dst_layout = src_layout;

    for (auto& d : dilates) d--;
    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);
    wgh_tr = wgh_tr.TreatAs(wgh_layout);

    // Should support G mixed with I. Like { G*I, O, H, W }.
    // Deconv's default weight layout ("IOHW"-derived) keeps the undivided channel count at
    // logical axis 1 (I), not axis 0 (O) as in regular conv -- see
    // TensorRequisite::ApplyGroupWeightLayout for why full_axis differs here.
    if (groups != 1 && wgh_layout.find("G") == std::string::npos) {
      wgh_tr = TensorRequisite::ApplyGroupWeightLayout(wgh_tr, groups, /*full_axis=*/1);
    }

    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    bool is_int8 = (src_tr.desc().get_data_type() == dnnl::memory::data_type::s8 ||
                    src_tr.desc().get_data_type() == dnnl::memory::data_type::u8);

    auto deconv_prim_desc = dnnl::deconvolution_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
        is_int8 ? src_tr.desc() : src_tr.LayoutAny().desc(),
        is_int8 ? wgh_tr.desc() : wgh_tr.LayoutAny().desc(),
        is_int8 ? bias_tr.desc() : bias_tr.LayoutAny().desc(),
        is_int8 ? dst_tr.desc() : dst_tr.LayoutAny().desc(), strides, dilates, padding_l, padding_r,
        attr);

    src_tr = src_tr.RequestLayout(deconv_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(deconv_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(deconv_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(deconv_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(deconv_prim_desc.scratchpad_desc());

    std::unordered_map<int, TensorRequisite> deconv_args = {{DNNL_ARG_SRC, src_tr},
                                                            {DNNL_ARG_WEIGHTS, wgh_tr},
                                                            {DNNL_ARG_BIAS, bias_tr},
                                                            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                                            {DNNL_ARG_DST, dst_tr}};
    if (scale_post_op_idx >= 0) {
      deconv_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(scale_post_op_idx) | DNNL_ARG_SRC_1] = o_scl_tr;
    }
    Submit(dnnl::deconvolution_forward(deconv_prim_desc), deconv_args);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};
    auto o_scl_tr = TensorRequisite{};
    int scale_post_op_idx = -1;
    auto attr = ParseAttrs(nid, &bias_tr, &o_scl_tr, &scale_post_op_idx);

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    bool is_int8 = (src_tr.desc().get_data_type() == dnnl::memory::data_type::s8 ||
                    src_tr.desc().get_data_type() == dnnl::memory::data_type::u8);

    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference,
        is_int8 ? src_tr.desc() : src_tr.LayoutAny().desc(),
        is_int8 ? wgh_tr.desc() : wgh_tr.LayoutAny().desc(),
        is_int8 ? bias_tr.desc() : bias_tr.LayoutAny().desc(),
        is_int8 ? dst_tr.desc() : dst_tr.LayoutAny().desc(), attr);

    src_tr = src_tr.RequestLayout(dense_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(dense_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(dense_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(dense_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(dense_prim_desc.scratchpad_desc());

    auto sum_in_tr = GetInputByName(nid, "sum_idx");
    if (op_name.find("_sum") != std::string::npos) {
      sum_in_tr = GetInput(nid, node.GetInputs().size() - 1);
    }

    std::unordered_map<int, TensorRequisite> dense_args = {{DNNL_ARG_SRC, src_tr},
                                                           {DNNL_ARG_WEIGHTS, wgh_tr},
                                                           {DNNL_ARG_BIAS, bias_tr},
                                                           {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                                           {DNNL_ARG_DST, dst_tr}};
    if (scale_post_op_idx >= 0) {
      dense_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(scale_post_op_idx) | DNNL_ARG_SRC_1] = o_scl_tr;
    }
    Submit(dnnl::inner_product_forward(dense_prim_desc), dense_args, {sum_in_tr, DNNL_ARG_DST});
  }

  void MatMul(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();

    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};
    auto o_scl_tr = TensorRequisite{};
    int scale_post_op_idx = -1;
    auto attr = ParseAttrs(nid, &bias_tr, &o_scl_tr, &scale_post_op_idx);

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (bias_tr.defined()) {
      // oneDNN's matmul primitive (unlike inner_product) requires the bias's
      // dimension count to match dst's rank -- broadcastable size-1 dims
      // are fine, a smaller rank is not. Reshape (n,) -> (1, ..., 1, n).
      auto dst_dims = dst_tr.dims();
      std::vector<int64_t> bias_dims(dst_dims.size(), 1);
      bias_dims.back() = dst_dims.back();
      bias_tr = bias_tr.Reshape(bias_dims);
    }

    bool is_int8 = (src_tr.desc().get_data_type() == dnnl::memory::data_type::s8 ||
                    src_tr.desc().get_data_type() == dnnl::memory::data_type::u8);

    auto matmul_prim_desc =
        dnnl::matmul::primitive_desc(engine_, is_int8 ? src_tr.desc() : src_tr.LayoutAny().desc(),
                                     is_int8 ? wgh_tr.desc() : wgh_tr.LayoutAny().desc(),
                                     is_int8 ? bias_tr.desc() : bias_tr.LayoutAny().desc(),
                                     is_int8 ? dst_tr.desc() : dst_tr.LayoutAny().desc(), attr);

    src_tr = src_tr.RequestLayout(matmul_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(matmul_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(matmul_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(matmul_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(matmul_prim_desc.scratchpad_desc());

    auto sum_in_tr = GetInputByName(nid, "sum_idx");
    if (op_name.find("_sum") != std::string::npos) {
      sum_in_tr = GetInput(nid, node.GetInputs().size() - 1);
    }

    std::unordered_map<int, TensorRequisite> matmul_args = {{DNNL_ARG_SRC, src_tr},
                                                            {DNNL_ARG_WEIGHTS, wgh_tr},
                                                            {DNNL_ARG_BIAS, bias_tr},
                                                            {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                                            {DNNL_ARG_DST, dst_tr}};
    if (scale_post_op_idx >= 0) {
      matmul_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(scale_post_op_idx) | DNNL_ARG_SRC_1] = o_scl_tr;
    }
    Submit(dnnl::matmul(matmul_prim_desc), matmul_args, {sum_in_tr, DNNL_ARG_DST});
  }

  void BatchMatMul(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_tr = GetInput(nid, 0);
    auto wgh_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);
    auto bias_tr = TensorRequisite{};
    auto o_scl_tr = TensorRequisite{};
    int scale_post_op_idx = -1;
    auto attr = ParseAttrs(nid, &bias_tr, &o_scl_tr, &scale_post_op_idx);

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    bool transpose_a = GetNodeAttr<bool>(node, "transpose_a");
    bool transpose_b = GetNodeAttr<bool>(node, "transpose_b");

    if (transpose_a) {
      src_tr = src_tr.Permute({0, 2, 1});
    }
    if (transpose_b) {
      wgh_tr = wgh_tr.Permute({0, 2, 1});
    }

    // Assumption that bias is correct and can be squeezed to 1D
    bias_tr = bias_tr.Reshape({dst_tr.dims()[1]});

    // Matmul description.
    bool is_int8 = (src_tr.desc().get_data_type() == dnnl::memory::data_type::s8 ||
                    src_tr.desc().get_data_type() == dnnl::memory::data_type::u8);

    auto bmm_prim_desc =
        dnnl::matmul::primitive_desc(engine_, is_int8 ? src_tr.desc() : src_tr.LayoutAny().desc(),
                                     is_int8 ? wgh_tr.desc() : wgh_tr.LayoutAny().desc(),
                                     is_int8 ? bias_tr.desc() : bias_tr.LayoutAny().desc(),
                                     is_int8 ? dst_tr.desc() : dst_tr.LayoutAny().desc(), attr);

    src_tr = src_tr.RequestLayout(bmm_prim_desc.src_desc());
    wgh_tr = wgh_tr.RequestLayout(bmm_prim_desc.weights_desc());
    dst_tr = dst_tr.RequestLayout(bmm_prim_desc.dst_desc());
    bias_tr = bias_tr.RequestLayout(bmm_prim_desc.bias_desc());

    auto scratchpad_tr = TensorRequisite::AsIs(bmm_prim_desc.scratchpad_desc());

    Submit(dnnl::matmul(bmm_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                         {DNNL_ARG_WEIGHTS, wgh_tr},
                                         {DNNL_ARG_BIAS, bias_tr},
                                         {DNNL_ARG_SCRATCHPAD, scratchpad_tr},
                                         {DNNL_ARG_DST, dst_tr}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto gamma_tr = GetInput(nid, 1);
    auto beta_tr = GetInput(nid, 2);
    auto mean_tr = GetInput(nid, 3);
    auto var_tr = GetInput(nid, 4);
    auto dst_tr = GetOutput(nid, 0);

    auto axis = GetNodeAttr<int>(node, "axis");
    auto epsilon = GetNodeAttr<float>(node, "epsilon");
    auto center = GetNodeAttr<bool>(node, "center");
    auto scale = GetNodeAttr<bool>(node, "scale");

    TVM_FFI_ICHECK(axis == 1 && center && scale) << "Unimplemented BatchNorm case";

    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, src_tr.desc(), dst_tr.desc(), epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale |
            dnnl::normalization_flags::use_shift);

    // With separate use_scale/use_shift (not the deprecated combined
    // use_scale_shift), oneDNN takes gamma/beta as two independent 1-D
    // tensors passed straight through DNNL_ARG_SCALE/DNNL_ARG_SHIFT --
    // there is no combined [2, C] weights tensor to extract via
    // weights_desc(), so no crop/squeeze/copy is needed at all (mirrors
    // LayerNorm() below).
    Submit(dnnl::batch_normalization_forward(bn_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                                             {DNNL_ARG_DST, dst_tr},
                                                             {DNNL_ARG_SCALE, gamma_tr},
                                                             {DNNL_ARG_SHIFT, beta_tr},
                                                             {DNNL_ARG_MEAN, mean_tr},
                                                             {DNNL_ARG_VARIANCE, var_tr}});
  }

  void LayerNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto gamma_tr = GetInput(nid, 1);
    auto beta_tr = GetInput(nid, 2);
    auto dst_tr = GetOutput(nid, 0);

    auto axes = GetNodeAttr<std::vector<int64_t>>(node, "axes");
    auto epsilon = GetNodeAttr<float>(node, "epsilon");
    auto center = GetNodeAttr<bool>(node, "center");
    auto scale = GetNodeAttr<bool>(node, "scale");

    TVM_FFI_ICHECK_EQ(axes.size(), 1U)
        << "DNNL LayerNorm currently only supports a single normalization axis, got "
        << axes.size();
    int axis = static_cast<int>(axes[0]);
    auto rank = static_cast<int>(src_tr.dims().size());
    bool is_last_axis = (axis == -1) || (axis == rank - 1);
    TVM_FFI_ICHECK(is_last_axis && center && scale)
        << "Unimplemented LayerNorm case: axis=" << axis << " rank=" << rank << " center=" << center
        << " scale=" << scale;

    // LN description.
    auto lnorm_prim_desc = dnnl::layer_normalization_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, src_tr.desc(), dst_tr.desc(), epsilon,
        dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

    // With separate use_scale/use_shift (not the deprecated combined
    // use_scale_shift), oneDNN takes gamma/beta as two independent 1-D
    // tensors passed straight through DNNL_ARG_SCALE/DNNL_ARG_SHIFT --
    // there is no combined [2, C] weights tensor to extract via
    // weights_desc(), so no crop/squeeze/copy is needed at all.
    Submit(dnnl::layer_normalization_forward(lnorm_prim_desc), {{DNNL_ARG_SRC, src_tr},
                                                                {DNNL_ARG_DST, dst_tr},
                                                                {DNNL_ARG_SCALE, gamma_tr},
                                                                {DNNL_ARG_SHIFT, beta_tr}});
  }

  void GlobalAvgPooling(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    auto src_layout = GetNodeAttr<std::string>(node, "layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    if (dst_layout.empty()) dst_layout = src_layout;

    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);
    auto src_dims = src_tr.dims();
    TVM_FFI_ICHECK_GT(src_dims.size(), 2U) << "Expected at least one spatial dimension";
    std::vector<int64_t> kernel(src_dims.begin() + 2, src_dims.end());
    std::vector<int64_t> strides(kernel.size(), 1);
    std::vector<int64_t> dilates(kernel.size(), 0);
    std::vector<int64_t> padding_l(kernel.size(), 0);
    std::vector<int64_t> padding_r(kernel.size(), 0);

    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_avg_exclude_padding,
        src_tr.desc(), dst_tr.LayoutAny().desc(), strides, kernel, dilates, padding_l, padding_r);

    src_tr = src_tr.RequestLayout(pool_prim_desc.src_desc());
    dst_tr = dst_tr.RequestLayout(pool_prim_desc.dst_desc());
    auto scratchpad_tr = TensorRequisite::AsIs(pool_prim_desc.scratchpad_desc());

    Submit(dnnl::pooling_forward(pool_prim_desc),
           {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}, {DNNL_ARG_SCRATCHPAD, scratchpad_tr}});
  }

  void Pooling(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    auto src_layout = GetNodeAttr<std::string>(node, "layout");
    auto dst_layout = GetNodeAttr<std::string>(node, "out_layout");
    if (dst_layout.empty()) dst_layout = src_layout;

    src_tr = src_tr.TreatAs(src_layout);
    dst_tr = dst_tr.TreatAs(dst_layout);

    auto kernel = GetNodeAttr<std::vector<int64_t>>(node, "pool_size");
    auto strides = GetNodeAttr<std::vector<int64_t>>(node, "strides");
    auto dilates = GetNodeAttr<std::vector<int64_t>>(node, "dilation");
    auto padding = GetNodeAttr<std::vector<int64_t>>(node, "padding");

    std::vector<int64_t> padding_l(padding.begin(), padding.begin() + padding.size() / 2);
    std::vector<int64_t> padding_r(padding.begin() + padding.size() / 2, padding.end());

    for (auto& d : dilates)
      d--;  // No dilation for DNNL is 0, Relax's is 1 -- same as Convolution().

    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, algo, src_tr.desc(), dst_tr.LayoutAny().desc(),
        strides, kernel, dilates, padding_l, padding_r);

    src_tr = src_tr.RequestLayout(pool_prim_desc.src_desc());
    dst_tr = dst_tr.RequestLayout(pool_prim_desc.dst_desc());
    auto scratchpad_tr = TensorRequisite::AsIs(pool_prim_desc.scratchpad_desc());

    Submit(dnnl::pooling_forward(pool_prim_desc),
           {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}, {DNNL_ARG_SCRATCHPAD, scratchpad_tr}});
  }

  void Eltwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = StripDnnlPrefix(node.GetOpName());
    auto algo = elt_name2algo.at(op_name);

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    float alpha = 0., beta = 0.;
    if (contains(op_name, "clip")) {
      alpha = GetNodeAttr<float>(node, "a_min");
      beta = GetNodeAttr<float>(node, "a_max");
    } else if (contains(op_name, "leaky_relu")) {
      alpha = GetNodeAttr<float>(node, "alpha");
    }

    auto elt_prim_desc =
        dnnl::eltwise_forward::primitive_desc(engine_, dnnl::prop_kind::forward_inference, algo,
                                              src_tr.desc(), dst_tr.desc(), alpha, beta);

    Submit(dnnl::eltwise_forward(elt_prim_desc), {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    auto src_tr = GetInput(nid, 0);
    auto dst_tr = GetOutput(nid, 0);

    auto axis = GetNodeAttr<int>(node, "axis");
    if (axis < 0) {
      axis = src_tr.dims().size() + axis;
    }

    auto softmax_prim_desc = dnnl::softmax_forward::primitive_desc(
        engine_, dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate,
        src_tr.desc(), dst_tr.desc(), axis);
    TVM_FFI_ICHECK(dst_tr.desc() == softmax_prim_desc.dst_desc());

    Submit(dnnl::softmax_forward(softmax_prim_desc),
           {{DNNL_ARG_SRC, src_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  void Binary(const size_t& nid, dnnl::algorithm algo) {
    auto node = nodes_[nid];
    TVM_FFI_ICHECK_EQ(node.GetInputs().size(), 2U);

    // Memory and compute description.
    auto lhs_tr = GetInput(nid, 0);
    auto rhs_tr = GetInput(nid, 1);
    auto dst_tr = GetOutput(nid, 0);

    lhs_tr = lhs_tr.Broadcast(dst_tr.dims());
    rhs_tr = rhs_tr.Broadcast(dst_tr.dims());

    auto binary_prim_desc =
        dnnl::binary::primitive_desc(engine_, algo, lhs_tr.desc(), rhs_tr.desc(), dst_tr.desc());

    Submit(dnnl::binary(binary_prim_desc),
           {{DNNL_ARG_SRC_0, lhs_tr}, {DNNL_ARG_SRC_1, rhs_tr}, {DNNL_ARG_DST, dst_tr}});
  }

  /*!
   * \brief Helper to extract node attribute with ability to specify default value and result type.
   */
  template <typename T>
  const T GetNodeAttr(const json::JSONGraphNode& node, const std::string& name, T def = {}) {
    if (!node.HasAttr(name)) return def;
    if constexpr (std::is_same_v<T, bool>) {
      return static_cast<bool>(node.GetAttr<int64_t>(name));
    } else if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(node.GetAttr<int64_t>(name));
    } else if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(node.GetAttr<double>(name));
    } else if constexpr (std::is_same_v<T, std::string>) {
      return std::string(node.GetAttr<ffi::String>(name));
    } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
      auto arr = node.GetAttr<ffi::Array<int64_t>>(name);
      std::vector<int64_t> res;
      for (size_t i = 0; i < arr.size(); ++i) res.push_back(arr[i]);
      return res;
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      auto arr = node.GetAttr<ffi::Array<ffi::String>>(name);
      std::vector<std::string> res;
      for (size_t i = 0; i < arr.size(); ++i) res.push_back(std::string(arr[i]));
      return res;
    }
  }

  TensorRequisite GetInput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.

    const JSONGraphNode& node = nodes_[nid];

    TVM_FFI_ICHECK_LT(idx, node.GetInputs().size());
    auto data_entry = node.GetInputs()[idx];

    auto shape_arr = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto dtype = nodes_[data_entry.id_].GetOpDataType()[data_entry.index_];
    auto eid = node_row_ptr_[data_entry.id_] + data_entry.index_;
    auto const_dl_tensor = data_entry_[eid];

    std::vector<int64_t> shape(shape_arr.begin(), shape_arr.end());
    auto desc = MakePlainDesc(shape, dtype);

    TensorRequisite res;
    if (const_dl_tensor) {
      TVM_FFI_ICHECK(const_dl_tensor->data);
      TVM_FFI_ICHECK(ffi::IsContiguous(*const_dl_tensor));
      auto mem = dnnl::memory(desc, engine_, const_dl_tensor->data);
      res = TensorRequisite::AsIs(mem, eid);
    } else {
      res = TensorRequisite::AsIs(desc, eid);
    }
    return res;
  }

  TensorRequisite GetInputByName(const size_t& nid, const std::string& name) {
    auto idx = GetNodeAttr<int>(nodes_[nid], name, -1);
    return GetInput(nid, idx);
  }

  TensorRequisite GetOutput(const size_t& nid, const int idx) {
    if (idx == -1) return {};  // -1 reserved value for empty input.
    const JSONGraphNode& node = nodes_[nid];

    TVM_FFI_ICHECK_LT(idx, node.GetNumOutput());
    auto shape_arr = node.GetOpShape()[idx];
    auto dtype = node.GetOpDataType()[idx];
    auto eid = node_row_ptr_[nid] + static_cast<uint32_t>(idx);

    TVM_FFI_ICHECK(data_entry_[eid] == nullptr);

    std::vector<int64_t> shape(shape_arr.begin(), shape_arr.end());
    auto desc = MakePlainDesc(shape, dtype);

    return TensorRequisite::AsIs(desc, eid).Backward();
  }

  bool IsIntermidate(const TensorRequisite& tr) {
    auto eid = tr.eid();
    bool is_input = std::find(input_nodes_.begin(), input_nodes_.end(), eid) != input_nodes_.end();
    bool is_output = std::any_of(outputs_.begin(), outputs_.end(),
                                 [eid](auto& output) { return output.id_ == eid; });
    if (is_input || is_output) {
      return false;
    } else {
      return true;
    }
  }

  /*! \brief Helper function to register primitive into execution queue */
  void Submit(const dnnl::primitive& prim, const std::unordered_map<int, TensorRequisite>& tr_args,
              const std::pair<TensorRequisite, int>& inplace_conf = {}) {
    // Register all provided TR arguments
    std::unordered_map<int, TensorRegistry::ArgId> prim_arg_id;
    TensorRegistry::ActionQue post_prim_actions;

    // mark inplace tr
    if (auto tr_in = inplace_conf.first) {
      auto tr_out = tr_args.at(inplace_conf.second);
      if (IsIntermidate(tr_in) && IsIntermidate(tr_out)) {
        tensor_registry_.Register(tr_in, &net_);
        tensor_registry_.MarkInplace(tr_out, tr_in);
      }
    }

    for (const auto& kvp : tr_args) {
      const auto& key = kvp.first;
      const auto& tr = kvp.second;

      if (!tr.defined()) continue;  // empty arg is admitted. Just skip it
      auto arg_id = tensor_registry_.Register(tr, tr.IsReversed() ? &post_prim_actions : &net_);
      prim_arg_id[key] = arg_id;
    }

    // Simulate inplace primitive, the reorder with src==dst will be skipped in Run()
    if (auto tr = inplace_conf.first) {
      auto arg_id = tensor_registry_.Register(tr, &net_);
      auto dst_tr = tr_args.at(inplace_conf.second);
      auto dst_arg_id = prim_arg_id.at(inplace_conf.second);

      // Register copy action direct before main primitive
      dnnl::reorder::primitive_desc io_copy_pd(engine_, tr.desc(), engine_, dst_tr.desc());
      net_.push_back(
          {dnnl::reorder(io_copy_pd), {{DNNL_ARG_SRC, arg_id}, {DNNL_ARG_DST, dst_arg_id}}});
    }

    // Register main primitive
    net_.push_back({prim, prim_arg_id});

    // Register post actions
    net_.insert(net_.end(), post_prim_actions.begin(), post_prim_actions.end());
  }

  uint32_t GenUniqueEid() { return next_unique_eid_offset_++; }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  TensorRegistry::ActionQue net_;
  /* Storage for all memory objects */
  TensorRegistry tensor_registry_;
  /* Generator of new unique eid which doesn't match with existing data entry */
  uint32_t next_unique_eid_offset_;
  /* Map of Run arg idx to corresponding eid */
  std::vector<uint32_t> run_arg_eid_;
};

ffi::Module DNNLJSONRuntimeCreate(ffi::String symbol_name, ffi::String graph_json,
                                  const ffi::Array<ffi::String>& const_names) {
  auto n = ffi::make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.DNNLJSONRuntimeCreate", DNNLJSONRuntimeCreate)
      .def("ffi.Module.load_from_bytes.dnnl_json", JSONRuntimeBase::LoadFromBytes<DNNLJSONRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
