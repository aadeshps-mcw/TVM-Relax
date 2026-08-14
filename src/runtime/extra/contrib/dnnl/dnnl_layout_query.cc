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
 * \file src/runtime/contrib/dnnl/dnnl_layout_query.cc
 * \brief FFI entry point that asks oneDNN what physical layout it would choose for a conv2d's
 * activation/weight tensors, given only shape/dtype/attrs. Used by the frontend's
 * partition_for_dnnl() to seed relax.transform.ConvertLayout with a real oneDNN-blocked layout
 * (e.g. "NCHW8c"/"OIHW8i8o") instead of a hardcoded plain "NCHW"/"OIHW".
 *
 * Verified against this codebase's actual TVM build (not assumed): relax.transform.ConvertLayout
 * DOES support blocked layout strings -- it lowers them to real IndexMap-based R.layout_transform
 * ops and threads the blocked layout into data_layout/kernel_layout attrs correctly. See the
 * dnnl_pattern_layout_query_patch.py companion file for the frontend integration and an important
 * note on where this fix's actual performance benefit comes from (chained DNNL conv2d layers
 * avoiding blocked<->plain round-trips at subgraph seams -- NOT per-layer reorder speedup).
 *
 * SCOPE: groups == 1 only. Grouped-conv weights would need a 5D "G..." layout string for the
 * *query* (oneDNN needs the true grouped shape to pick a group-aware blocking), but the runtime's
 * TreatAs() in dnnl_tensor_requisite.h requires the kernel_layout token count to match the
 * constant weight tensor's ORIGINAL (ungrouped, 4D OIHW) physical shape -- see BuildEngine's
 * Convolution(), which inserts the G dimension itself via a post-TreatAs Reshape(), specifically
 * because the incoming weight tensor is always plain 4D before that point. Emitting a "G..."
 * layout string here would break that contract and fail TreatAs's dims-count check. Supporting
 * groups > 1 properly needs a second design pass (query with a 5D grouped shape, but express the
 * result back to the frontend as a 4D-token-count-compatible string, or change the runtime side
 * to accept pre-grouped constants) -- deliberately deferred, not silently done wrong.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <dnnl.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "dnnl_tensor_requisite.h"
#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

namespace {

// Reverse of TensorRequisite::TreatAs's forward direction: given a RESOLVED blocked
// dnnl::memory::desc (e.g. from primitive_desc::weights_desc() after format_tag::any), find
// which entry in TensorRequisite::FormatTagsByCanonicalName() produces an identical descriptor
// for these dims/dtype, then convert that canonical tag name (oneDNN's own abcd-style string,
// e.g. "ABcd16b16a") into a TVM-style layout string (e.g. "OIHW16i16o") using desired_logic_layout
// to map abstract positions back to concrete letters.
//
// Deliberately brute-forces against the SAME table TreatAs() already uses (via
// dnnl_fmt_tag2str), rather than hand-decoding dnnl_blocking_desc_t fields, so this stays
// consistent with TreatAs by construction instead of by two independently-written
// implementations agreeing by luck.
std::string DescToTvmLayoutString(const dnnl::memory::desc& resolved,
                                  const std::string& desired_logic_layout) {
  const auto dims = resolved.get_dims();
  const auto dtype = resolved.get_data_type();

  std::string found_canonical;
  for (const auto& kv : TensorRequisite::FormatTagsByCanonicalName()) {
    const std::string& canonical_name = kv.first;
    const dnnl::memory::format_tag tag = kv.second;
    dnnl::memory::desc candidate;
    try {
      candidate = dnnl::memory::desc(dims, dtype, tag);
    } catch (const dnnl::error&) {
      continue;  // this tag doesn't apply to this rank/dims -- expected, not an error
    }
    if (candidate == resolved) {
      found_canonical = canonical_name;
      break;
    }
  }
  TVM_FFI_ICHECK(!found_canonical.empty())
      << "Could not find a matching format_tag for the resolved oneDNN descriptor -- "
         "TensorRequisite::FormatTagsByCanonicalName() table may be incomplete for this "
         "oneDNN build, or the descriptor isn't a standard blocked format.";

  // Invert CanonicalFormatTagName's letter mapping: 'a'/'A' -> position 0, 'b'/'B' -> position 1,
  // etc. -- desired_logic_layout[position] gives the concrete TVM letter (O/I/H/W/N/C).
  std::string result;
  size_t i = 0;
  while (i < found_canonical.size()) {
    char c = found_canonical[i];
    if (std::isdigit(static_cast<unsigned char>(c))) {
      // Inner (blocked) token: digits followed by exactly one lowercase letter.
      size_t start = i;
      while (i < found_canonical.size() &&
             std::isdigit(static_cast<unsigned char>(found_canonical[i])))
        i++;
      TVM_FFI_ICHECK_LT(i, found_canonical.size());
      char letter = found_canonical[i];
      int pos = std::tolower(letter) - 'a';
      TVM_FFI_ICHECK_LT(static_cast<size_t>(pos), desired_logic_layout.size());
      result += found_canonical.substr(start, i - start);
      result += static_cast<char>(std::tolower(desired_logic_layout[pos]));
      i++;
    } else {
      // Outer token: single letter, case indicates blocked-or-not (irrelevant to TVM's string
      // convention -- outer axes are always written uppercase in TVM layout strings regardless).
      int pos = std::tolower(c) - 'a';
      TVM_FFI_ICHECK_LT(static_cast<size_t>(pos), desired_logic_layout.size());
      result += static_cast<char>(std::toupper(desired_logic_layout[pos]));
      i++;
    }
  }
  return result;
}

}  // namespace

/*!
 * \brief Query oneDNN's preferred src/weight layout for a single ungrouped conv2d shape.
 * \return {src_layout, wgh_layout} as TVM-style layout strings, e.g. {"NCHW16c", "OIHW16i16o"}.
 *         Returns {"NCHW", "OIHW"} (the existing plain default) if oneDNN resolves to a
 *         non-blocked format for this shape/dtype/ISA, or if the shape/dtype combination can't
 *         be built into a primitive at all on this machine, or if groups != 1 -- all legitimate
 *         answers, not failures.
 */
ffi::Array<ffi::String> QueryOptimalConv2DLayout(ffi::Array<int64_t> src_shape,
                                                 ffi::Array<int64_t> wgh_shape,
                                                 ffi::Array<int64_t> strides,
                                                 ffi::Array<int64_t> dilation,
                                                 ffi::Array<int64_t> padding, int64_t groups,
                                                 ffi::String dtype_str) {
  TVM_FFI_ICHECK_EQ(src_shape.size(), 4u);
  TVM_FFI_ICHECK_EQ(wgh_shape.size(), 4u);
  TVM_FFI_ICHECK_EQ(strides.size(), 2u);
  TVM_FFI_ICHECK_EQ(dilation.size(), 2u);
  TVM_FFI_ICHECK_EQ(padding.size(), 4u);

  if (groups != 1) {
    // See file-level comment: grouped-conv weight layout querying needs a separate design pass
    // to stay compatible with the runtime's TreatAs()/Reshape() contract. Fall back rather than
    // emit something that will fail (or worse, silently corrupt) at TreatAs() time.
    return {"NCHW", "OIHW"};
  }

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  const DLDataType dltype = ffi::StringToDLDataType(dtype_str);
  const dnnl::memory::data_type dt = dtype_dl2dnnl(dltype);

  if (dt == dnnl::memory::data_type::undef) {
    TVM_FFI_THROW(ValueError) << "Unsupported dtype for DNNL layout query: " << dtype_str;
  }

  auto to_dims = [](const ffi::Array<int64_t>& v) {
    return dnnl::memory::dims(v.begin(), v.end());
  };

  dnnl::memory::dims src_dims = to_dims(src_shape);
  dnnl::memory::dims wgh_dims = to_dims(wgh_shape);
  dnnl::memory::dims strides_dims = to_dims(strides);
  dnnl::memory::dims dilates = to_dims(dilation);
  for (auto& d : dilates) d -= 1;  // TVM dilation=1 means "no dilation" == oneDNN's 0

  dnnl::memory::dims pad_l = {padding[0], padding[1]};
  dnnl::memory::dims pad_r = {padding[2], padding[3]};

  auto out_dim = [](int64_t in, int64_t k, int64_t dil0, int64_t pl, int64_t pr, int64_t s) {
    const int64_t eff_k = (k - 1) * (dil0 + 1) + 1;
    return (in + pl + pr - eff_k) / s + 1;
  };
  const int64_t out_h =
      out_dim(src_dims[2], wgh_dims[2], dilates[0], pad_l[0], pad_r[0], strides_dims[0]);
  const int64_t out_w =
      out_dim(src_dims[3], wgh_dims[3], dilates[1], pad_l[1], pad_r[1], strides_dims[1]);
  dnnl::memory::dims dst_dims = {src_dims[0], wgh_dims[0], out_h, out_w};

  auto src_any = dnnl::memory::desc(src_dims, dt, dnnl::memory::format_tag::any);
  auto wgh_any = dnnl::memory::desc(wgh_dims, dt, dnnl::memory::format_tag::any);
  auto dst_any = dnnl::memory::desc(dst_dims, dt, dnnl::memory::format_tag::any);

  dnnl::convolution_forward::primitive_desc conv_pd;
  try {
    conv_pd = dnnl::convolution_forward::primitive_desc(
        engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, src_any,
        wgh_any, dst_any, strides_dims, dilates, pad_l, pad_r);
  } catch (const dnnl::error&) {
    return {"NCHW", "OIHW"};
  }

  const std::string src_layout = DescToTvmLayoutString(conv_pd.src_desc(), "NCHW");
  const std::string wgh_layout = DescToTvmLayoutString(conv_pd.weights_desc(), "OIHW");

  return {src_layout, wgh_layout};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.contrib.dnnl.query_optimal_conv2d_layout",
                        QueryOptimalConv2DLayout);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
