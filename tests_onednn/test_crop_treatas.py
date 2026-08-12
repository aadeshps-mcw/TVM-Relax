# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Crop()/TreatAs() are ruled out (groups=1 fails as badly as groups=4;
alter_layout True/False give byte-identical mismatches). This means the
bug is in the int8 convolution computation itself -- on the DNNL side,
the plain-TVM reference side, or both. `_compile_and_compare` only checks
the two against *each other*, so it can't tell us which one (if either)
is actually right. This script adds a from-scratch numpy ground truth,
computed independently of both TVM code paths, as a tiebreaker.

Run with:
    pytest verify_int8_ground_truth.py -v -s
"""

import numpy as np
import pytest
import test_dhi as dhi  # adjust import if needed

import tvm
from tvm import relax


def _numpy_conv2d_s32(data, weight, padding, stride=1):
    """Independent reference: explicit int64 accumulation (to avoid any
    accidental overflow obscuring the comparison), no groups (groups=1
    only -- keep this minimal since groups is already cleared as a
    suspect). Treats `data`/`weight` as literal integer values per their
    numpy dtype (i.e. u8 is genuinely 0..255, s8 is genuinely -128..127;
    no implicit zero-point/offset is applied anywhere)."""
    n, c, h, w = data.shape
    oc, ic, kh, kw = weight.shape
    assert ic == c, "this helper assumes groups=1"

    data_i = data.astype(np.int64)
    weight_i = weight.astype(np.int64)
    data_p = np.pad(data_i, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, oc, oh, ow), dtype=np.int64)

    for o in range(oc):
        for y in range(oh):
            for x in range(ow):
                patch = data_p[:, :, y * stride : y * stride + kh, x * stride : x * stride + kw]
                out[:, o, y, x] = np.sum(patch * weight_i[o], axis=(1, 2, 3))

    return out.astype(np.int32)


def _get_dnnl_and_ref_outputs(mod, data_np, weight_np, alter_layout=False):
    """Same two computations _compile_and_compare does, but returns both
    instead of just asserting they match each other."""
    target = tvm.target.Target("llvm")
    tvm_args = [tvm.runtime.tensor(data_np), tvm.runtime.tensor(weight_np)]

    ref_ex = relax.build(mod, target=target)
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    plain_tvm_out = ref_vm["main"](*tvm_args).numpy()

    partitioned = dhi.partition_for_dnnl(mod, alter_layout=alter_layout, run_codegen=False)
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    dnnl_out = vm["main"](*tvm_args).numpy()

    return dnnl_out, plain_tvm_out


@pytest.mark.parametrize(
    "dtype,weight_dtype",
    [
        ("int8", "int8"),
        ("uint8", "int8"),
    ],
)
def test_int8_conv_against_independent_ground_truth(dtype, weight_dtype):
    np.random.seed(42)
    data_shape = (1, 4, 5, 5)
    out_channels = 4

    mod, weight_shape = dhi._make_conv2d_module(
        data_shape=data_shape,
        out_channels=out_channels,
        groups=1,
        kernel_size=(3, 3),
        padding=(1, 1),
        dtype=dtype,
        weight_dtype=weight_dtype,
        out_dtype="int32",
    )
    data_np = dhi._random_for_dtype(data_shape, dtype)
    weight_np = dhi._random_for_dtype(weight_shape, weight_dtype)

    dnnl_out, plain_tvm_out = _get_dnnl_and_ref_outputs(mod, data_np, weight_np, alter_layout=False)
    ground_truth = _numpy_conv2d_s32(data_np, weight_np, padding=1, stride=1)

    dnnl_matches_gt = np.array_equal(dnnl_out, ground_truth)
    tvm_matches_gt = np.array_equal(plain_tvm_out, ground_truth)

    print(f"\n[{dtype}] DNNL matches independent ground truth:      {dnnl_matches_gt}")
    print(f"[{dtype}] plain-TVM matches independent ground truth:  {tvm_matches_gt}")

    if not dnnl_matches_gt:
        diff = dnnl_out.astype(np.int64) - ground_truth.astype(np.int64)
        print(
            f"[{dtype}] DNNL vs ground truth: mismatched={np.count_nonzero(diff)}/{diff.size}, "
            f"max_abs_diff={np.max(np.abs(diff))}"
        )
    if not tvm_matches_gt:
        diff = plain_tvm_out.astype(np.int64) - ground_truth.astype(np.int64)
        print(
            f"[{dtype}] plain-TVM vs ground truth: "
            f"mismatched={np.count_nonzero(diff)}/{diff.size}, "
            f"max_abs_diff={np.max(np.abs(diff))}"
        )

    # Don't assert yet -- this test is diagnostic. Both prints above tell
    # us which backend (if either) is actually correct. Uncomment once
    # you know which side should be trusted:
    # assert dnnl_matches_gt, "DNNL path diverges from independent ground truth"
    # assert tvm_matches_gt, "plain-TVM path diverges from independent ground truth"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
