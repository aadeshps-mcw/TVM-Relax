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
Tests for the DNNL "Path B" QNN output-rescale fusion:

    conv2d(data, weight)  -> dequantize(out, scale, zp)   =>  dnnl.qnn.conv2d
    matmul(lhs, rhs)      -> dequantize(out, scale, zp)   =>  dnnl.qnn.matmul

conv2d/matmul run in float; the dequantize's (scale, zero_point) are folded
into DNNL's existing output-rescale post-op machinery (o_scl_idx / dst_zp_idx,
read in ParseAttrs in dnnl_json_runtime.cc). These tests do NOT exercise real
int8 execution -- they check:

  1. FuseOpsByPattern recognizes and names the composite correctly
     (dnnl.qnn.conv2d / dnnl.qnn.matmul -- not misrouted via the "dense"
     substring in BuildEngine's dispatch).
  2. End-to-end numerics of the offloaded (DNNL) build match the native
     TVM (llvm, unpartitioned) reference.
  3. Per-channel dequantize (a real, non-size-1 scale on the output-channel
     axis) round-trips correctly -- this exercises the ParseAttrs fix where
     the scale mask previously hardcoded axis=1 and now reads dst_axis.
  4. The pre-existing conv2d+bias+sum(+relu) and conv2d+clip composites
     still fuse and execute correctly after leaf_start_index was
     generalized from `const VarNode*` to `const ffi::Object*` to also
     track constant leaves (scale/zp). This is a shared-code regression
     risk, not QNN-specific behavior, flagged explicitly in the plan.

NOTE on dtype: relax.dequantize's native (non-offloaded) compute may assume
integer `data` input in some TVM versions. Path B intentionally calls
dequantize on a *float* conv2d/matmul output (that's the whole point -- the
op never actually runs in int8). If the un-partitioned reference build in
_compile_and_compare fails to lower/build with a dtype error, that is a
signal to check relax.dequantize's actual compute definition in this tree,
not necessarily a bug in the DNNL-side changes -- see the comment at the
call site below.
"""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

has_dnnl = tvm.get_global_func("relax.ext.dnnl", allow_missing=True) is not None
pytestmark = pytest.mark.skipif(not has_dnnl, reason="DNNL BYOC not enabled in this build")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorType(shape, dtype))


def _compile_and_compare(mod, inputs_np, rtol=1e-5, atol=1e-5, expect_composite=None):
    """
    1. Runs the native TVM CPU reference (unpartitioned).
    2. Partitions for DNNL and verifies at least one function was offloaded,
       and (if given) that `expect_composite` is among the composite names.
    3. Builds the DNNL graph and runs it.
    4. Compares the outputs.
    """
    target = tvm.target.Target("llvm")
    tvm_args = [tvm.runtime.tensor(x) for x in inputs_np]

    # 1. Native TVM reference
    ex_ref = relax.build(mod, target=target)
    vm_ref = relax.VirtualMachine(ex_ref, tvm.cpu())
    out_ref = vm_ref["main"](*tvm_args).numpy()

    # 2. Partition and verify offload happened
    partitioned = partition_for_dnnl(mod, run_codegen=False)

    dnnl_funcs = [
        gv.name_hint
        for gv, func in partitioned.functions.items()
        if func.attrs is not None and func.attrs.get("Codegen") == "dnnl"
    ]
    assert len(dnnl_funcs) >= 1, (
        "Expected ops to be offloaded, but got 0 subgraphs. "
        "Check that _DNNL_COMPUTE_OPS contains the op!"
    )

    composite_names = [
        str(func.attrs.get("Composite"))
        for gv, func in partitioned.functions.items()
        if func.attrs is not None and func.attrs.get("Composite") is not None
    ]
    # `Composite`-tagged functions are frequently nested *inside* the body
    # of the outer `Codegen="dnnl"` function (a local function, not a
    # separate module-level GlobalVar) -- e.g. `local_func` bound inside
    # `fused_..._dnnl_dnnl`'s body with `R.func_attr({"Composite": ...})`.
    # `.functions.items()` only walks top-level globals, so it misses
    # those. Fall back to scanning the printed script text, which is the
    # one place both top-level and nested composite attrs reliably show up.
    composite_names += re.findall(r'"Composite":\s*"([^"]+)"', partitioned.script())
    composite_names = sorted(set(composite_names))
    if expect_composite is not None:
        assert expect_composite in composite_names, (
            f"expected composite '{expect_composite}', got {composite_names}"
        )
        assert not any("dense" in n for n in composite_names), (
            "a composite name contains 'dense' -- this is exactly the "
            "BuildEngine substring-dispatch misrouting the naming choice "
            "(dnnl.qnn.matmul, not dnnl.qnn.dense) was meant to avoid"
        )

    # 3. Lower and build for DNNL
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)

    ex_dnnl = relax.build(codegen_mod, target=target)
    vm_dnnl = relax.VirtualMachine(ex_dnnl, tvm.cpu())
    out_dnnl = vm_dnnl["main"](*tvm_args).numpy()

    # 4. Compare numerics
    np.testing.assert_allclose(out_dnnl, out_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# module builders
# ---------------------------------------------------------------------------


def _conv2d_dequant_mod(data_shape, weight_shape, scale_shape, zp_shape, axis, dtype="float32"):
    data = _var("data", data_shape, dtype)
    weight = _var("weight", weight_shape, dtype)
    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            # NOTE: relax.dequantize requires zero_point to be an integer
            # dtype (int8/uint8/int16/uint16/int32/uint32/float16) -- it
            # rejects float32 outright at type-inference time, regardless
            # of what dtype `data` is. Confirmed via InferTypeDequantize's
            # explicit allow-list. Scale stays float32.
            scale = relax.const(np.random.uniform(0.01, 0.05, scale_shape).astype("float32"))
            zp = relax.const(np.random.randint(-2, 3, zp_shape).astype("int32"))
            out = bb.emit(relax.op.nn.conv2d(data, weight, out_dtype=dtype))
            out = bb.emit(relax.op.dequantize(out, scale, zp, axis=axis, out_dtype=dtype))
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def _matmul_dequant_mod(lhs_shape, rhs_shape, scale_shape, zp_shape, axis, dtype="float32"):
    lhs = _var("lhs", lhs_shape, dtype)
    rhs = _var("rhs", rhs_shape, dtype)
    bb = relax.BlockBuilder()
    with bb.function("main", [lhs, rhs]):
        with bb.dataflow():
            scale = relax.const(np.random.uniform(0.01, 0.05, scale_shape).astype("float32"))
            zp = relax.const(np.random.randint(-2, 3, zp_shape).astype("int32"))
            out = bb.emit(relax.op.matmul(lhs, rhs, out_dtype=dtype))
            out = bb.emit(relax.op.dequantize(out, scale, zp, axis=axis, out_dtype=dtype))
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def _conv2d_bias_sum_relu_mod(data_shape, weight_shape, bias_shape, dtype="float32"):
    # Regression coverage: the residual/sum lookup now goes through the
    # generalized leaf_start_index map (const ffi::Object* keyed instead of
    # const VarNode*), via an explicit upcast at the lookup site.
    data = _var("data", data_shape, dtype)
    weight = _var("weight", weight_shape, dtype)
    residual = _var("residual", data_shape, dtype)
    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight, residual]):
        with bb.dataflow():
            bias = relax.const(np.random.uniform(-0.1, 0.1, bias_shape).astype(dtype))
            # padding=(1, 1) keeps the spatial dims at 8x8 (3x3 kernel,
            # stride 1) so the conv2d output broadcasts against `residual`,
            # which has the same shape as `data`.
            out = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1), out_dtype=dtype))
            out = bb.emit(relax.op.add(out, bias))
            out = bb.emit(relax.op.add(out, residual))
            out = bb.emit(relax.op.nn.relu(out))
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def _conv2d_clip_mod(data_shape, weight_shape, dtype="float32"):
    data = _var("data", data_shape, dtype)
    weight = _var("weight", weight_shape, dtype)
    bb = relax.BlockBuilder()
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = bb.emit(relax.op.nn.conv2d(data, weight, out_dtype=dtype))
            # clip's min/max want raw PrimExpr (plain Python numbers get
            # auto-converted); a relax.const Constant is rejected.
            out = bb.emit(relax.op.clip(out, 0.0, 6.0))
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


# ---------------------------------------------------------------------------
# QNN fusion: structural naming + numerical correctness together
# (mirrors the working reference's _compile_and_compare, which already
# checks offload happened; expect_composite adds the specific name check)
# ---------------------------------------------------------------------------


def test_qnn_conv2d_dequantize_correctness():
    np.random.seed(0)
    data_shape, weight_shape = (1, 4, 8, 8), (6, 4, 3, 3)
    data = np.random.uniform(-1, 1, data_shape).astype("float32")
    weight = np.random.uniform(-1, 1, weight_shape).astype("float32")

    mod = _conv2d_dequant_mod(data_shape, weight_shape, (1,), (1,), axis=1)
    _compile_and_compare(
        mod, [data, weight], rtol=1e-4, atol=1e-4, expect_composite="dnnl.qnn.conv2d"
    )


def test_qnn_conv2d_dequantize_per_channel_axis():
    # Exercises the ParseAttrs fix: a real per-channel scale on the
    # output-channel axis (axis=1 in NCHW), so the mask must be (1 << 1)
    # rather than falling back to the size==1 per-tensor case.
    np.random.seed(1)
    data_shape, weight_shape = (1, 4, 8, 8), (6, 4, 3, 3)
    out_channels = weight_shape[0]
    data = np.random.uniform(-1, 1, data_shape).astype("float32")
    weight = np.random.uniform(-1, 1, weight_shape).astype("float32")

    mod = _conv2d_dequant_mod(data_shape, weight_shape, (out_channels,), (1,), axis=1)
    _compile_and_compare(
        mod, [data, weight], rtol=1e-4, atol=1e-4, expect_composite="dnnl.qnn.conv2d"
    )


def test_qnn_matmul_dequantize_correctness():
    np.random.seed(2)
    lhs_shape, rhs_shape = (1, 16, 32), (1, 32, 24)
    lhs = np.random.uniform(-1, 1, lhs_shape).astype("float32")
    rhs = np.random.uniform(-1, 1, rhs_shape).astype("float32")

    mod = _matmul_dequant_mod(lhs_shape, rhs_shape, (1,), (1,), axis=-1)
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    print(partitioned.script())
    _compile_and_compare(mod, [lhs, rhs], rtol=1e-4, atol=1e-4, expect_composite="dnnl.qnn.matmul")


# ---------------------------------------------------------------------------
# regression: shared codegen paths touched by the leaf_start_index refactor
# ---------------------------------------------------------------------------


def test_conv2d_bias_sum_relu_regression():
    np.random.seed(3)
    data_shape, weight_shape, bias_shape = (1, 4, 8, 8), (4, 4, 3, 3), (1, 4, 1, 1)
    data = np.random.uniform(-1, 1, data_shape).astype("float32")
    weight = np.random.uniform(-1, 1, weight_shape).astype("float32")
    residual = np.random.uniform(-1, 1, data_shape).astype("float32")

    mod = _conv2d_bias_sum_relu_mod(data_shape, weight_shape, bias_shape)
    _compile_and_compare(mod, [data, weight, residual], rtol=1e-4, atol=1e-4)


def test_conv2d_clip_regression():
    np.random.seed(4)
    data_shape, weight_shape = (1, 4, 8, 8), (4, 4, 3, 3)
    data = np.random.uniform(-1, 1, data_shape).astype("float32")
    weight = np.random.uniform(-1, 1, weight_shape).astype("float32")

    mod = _conv2d_clip_mod(data_shape, weight_shape)
    _compile_and_compare(mod, [data, weight], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
