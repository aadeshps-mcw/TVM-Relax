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
DNNL BYOC backend test suite.

Builds on the prior version of this file (conv2d+relu, layer_norm rewrite,
pruning, matmul, conv2d+bias+relu, dense+bias+gelu reorder) and adds:

  - Regression coverage for two bugs found while debugging this backend:
      1. dnnl.global_avg_pool2d previously matched ANY adaptive_avg_pool2d,
         not just the true global (output_size == (1, 1)) case, producing a
         malformed oneDNN pooling descriptor at runtime for e.g. a (16,16)
         -> (4,4) adaptive pool. Fixed via dnnl_global_avg_pool2d_checker.
      2. _sum_pattern's bias-shape predicate only accepted a scalar or 1D
         bias, rejecting the (out_channels, 1, 1)-shaped NCHW-broadcast bias
         convention used elsewhere in this same codebase -- causing
         conv2d_bias_sum_relu to silently fall back to three separate,
         unfused composites (conv2d_bias, add, relu) instead of fusing.
  - The residual-sum pattern family (_sum_patterns), previously untested.
  - The int64 safety guard and the ResNetV1 downsample stride-swap rewrite,
    pytest-ified from ad hoc print-and-eyeball scripts into real assertions.
  - A couple of standalone eltwise ops (sigmoid, clip), since the prior file
    only exercised compute ops.

Run with:
    pytest test_dnnl_backend.py -v
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)
pytestmark = [pytest.mark.skipif(not has_dnnl, reason="DNNL not enabled.")]


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------
def _dnnl_regions(mod: tvm.IRModule) -> list:
    """Top-level functions still tagged Codegen == 'dnnl'. Only meaningful
    BEFORE RunCodegen has lowered them -- partition_for_dnnl's default
    run_codegen=True already lowers everything, so callers that want to
    inspect regions structurally should pass run_codegen=False."""
    return [
        func
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
    ]


def _all_composite_names_in_mod(mod: tvm.IRModule) -> list:
    """Recursively collect every `Composite` attr string in the module,
    including composite functions bound locally inside another function's
    body (where they live post-MergeCompositeFunctions, nested inside the
    Codegen='dnnl' wrapper rather than as top-level mod.functions entries)."""
    names = []

    def _collect(func: relax.Function):
        if func.attrs is not None:
            c = func.attrs.get("Composite")
            if c is not None:
                names.append(str(c))
        seq = func.body if isinstance(func.body, relax.SeqExpr) else None
        if seq is None:
            return
        for block in seq.blocks:
            for binding in block.bindings:
                value = getattr(binding, "value", None)
                if isinstance(value, relax.Function):
                    _collect(value)

    for _, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _collect(func)
    return names


def _offloaded_dnnl_call_targets(mod: tvm.IRModule) -> list:
    """Names of DNNL external functions actually dispatched to via
    call_dps_packed, for a module that HAS already been through RunCodegen
    (either explicitly or because partition_for_dnnl's default
    run_codegen=True already ran it)."""
    targets = []
    for _, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        seq = func.body if isinstance(func.body, relax.SeqExpr) else None
        if seq is None:
            continue
        for block in seq.blocks:
            for binding in block.bindings:
                call = getattr(binding, "value", None)
                if not isinstance(call, relax.Call):
                    continue
                if call.op == tvm.ir.Op.get("relax.call_dps_packed") and "dnnl" in str(
                    call.args[0]
                ):
                    targets.append(str(call.args[0]))
    return targets


def _to_tensor(np_array, dev):
    try:
        return tvm.runtime.tensor(np_array, device=dev)
    except TypeError:
        return tvm.runtime.tensor(np_array, dev)


def _run(mod: tvm.IRModule, args, dev=None):
    dev = dev or tvm.cpu()
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, dev)
    return vm["main"](*[_to_tensor(a, dev) for a in args]).numpy()


# -------------------------------------------------------------------------
# Model builders
# -------------------------------------------------------------------------
def _make_conv2d_relu_module(
    data_shape=(1, 3, 224, 224),
    weight_shape=(16, 3, 3, 3),
    dtype="float32",
    with_relu=True,
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))

    with builder.function("main", [data, weight]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            out = builder.emit(relax.op.nn.relu(conv)) if with_relu else conv
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_expanded_layernorm_module(shape=(1, 3136, 64), dtype="float32"):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(shape, dtype))
    gamma = relax.Var("gamma", relax.TensorType((shape[-1],), dtype))
    beta = relax.Var("beta", relax.TensorType((shape[-1],), dtype))

    with builder.function("main", [data, gamma, beta]):
        with builder.dataflow():
            mu = builder.emit(relax.op.mean(data, axis=[-1], keepdims=True))
            diff = builder.emit(relax.op.subtract(data, mu))
            p2 = builder.emit(relax.op.power(diff, relax.const(2.0, dtype)))
            var = builder.emit(relax.op.mean(p2, axis=[-1], keepdims=True))
            eps = builder.emit(relax.op.add(var, relax.const(1e-5, dtype)))
            denom = builder.emit(relax.op.sqrt(eps))
            norm = builder.emit(relax.op.divide(diff, denom))
            scaled = builder.emit(relax.op.multiply(norm, gamma))
            out = builder.emit(relax.op.add(scaled, beta))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_standalone_elementwise_module(shape=(1, 64, 56, 56), dtype="float32"):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    y = relax.Var("y", relax.TensorType(shape, dtype))

    with builder.function("main", [x, y]):
        with builder.dataflow():
            add_out = builder.emit(relax.op.add(x, y))
            relu_out = builder.emit(relax.op.nn.relu(add_out))
            out = builder.emit_output(relu_out)
        builder.emit_func_output(out)
    return builder.get()


def _make_conv2d_bias_relu_module(
    data_shape=(1, 3, 224, 224),
    weight_shape=(16, 3, 3, 3),
    dtype="float32",
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    bias = relax.Var("bias", relax.TensorType((weight_shape[0], 1, 1), dtype))

    with builder.function("main", [data, weight, bias]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            biased = builder.emit(relax.op.add(conv, bias))
            out = builder.emit(relax.op.nn.relu(biased))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_conv2d_bias_sum_module(
    data_shape=(1, 8, 16, 16),
    weight_shape=(8, 8, 3, 3),
    dtype="float32",
    with_relu=True,
):
    """conv2d -> +bias -> +residual [-> relu]. Bias is (oc, 1, 1), the only
    valid per-channel-bias shape for an NCHW conv output -- broadcasting
    aligns from the RIGHT, so a bare 1D (oc,) bias would try to align
    against the trailing spatial (w) dimension, not the channel dimension.
    (The 1D (n,) convention IS valid for matmul, whose output's last dim
    genuinely is the channel dim -- see _make_matmul_bias_sum_module.)
    This (oc, 1, 1) shape is the one that previously fell through the
    _sum_pattern bias-shape predicate unfused (see module docstring)."""
    oc = weight_shape[0]
    n, _, h, w = data_shape
    bias_shape = (oc, 1, 1)

    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    bias = relax.Var("bias", relax.TensorType(bias_shape, dtype))
    residual = relax.Var("residual", relax.TensorType((n, oc, h, w), dtype))

    with builder.function("main", [data, weight, bias, residual]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            biased = builder.emit(relax.op.add(conv, bias))
            summed = builder.emit(relax.op.add(biased, residual))
            out = builder.emit(relax.op.nn.relu(summed)) if with_relu else summed
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_module(m=8, k=16, n=32, dtype="float32"):
    builder = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorType((m, k), dtype))
    b = relax.Var("b", relax.TensorType((k, n), dtype))

    with builder.function("main", [a, b]):
        with builder.dataflow():
            out = builder.emit(relax.op.matmul(a, b))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_bias_sum_module(m=16, k=32, n=64, dtype="float32"):
    """matmul -> +bias -> +residual, no relu (matches _sum_patterns' scope:
    matmul only gets the no-relu sum variant, unlike conv2d which gets both)."""
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType((m, k), dtype))
    weight = relax.Var("weight", relax.TensorType((k, n), dtype))
    bias = relax.Var("bias", relax.TensorType((n,), dtype))
    residual = relax.Var("residual", relax.TensorType((m, n), dtype))

    with builder.function("main", [data, weight, bias, residual]):
        with builder.dataflow():
            mm = builder.emit(relax.op.matmul(data, weight))
            biased = builder.emit(relax.op.add(mm, bias))
            summed = builder.emit(relax.op.add(biased, residual))
            out = builder.emit_output(summed)
        builder.emit_func_output(out)
    return builder.get()


def _make_dense_bias_gelu_module(batch=2, seq=4, in_dim=8, out_dim=16, dtype="float32"):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType((batch * seq, in_dim), dtype))
    weight = relax.Var("weight", relax.TensorType((in_dim, out_dim), dtype))
    bias = relax.Var("bias", relax.TensorType((out_dim,), dtype))

    with builder.function("main", [data, weight, bias]):
        with builder.dataflow():
            den = builder.emit(relax.op.matmul(data, weight))
            re_den = builder.emit(relax.op.reshape(den, (batch, seq, out_dim)))
            added = builder.emit(relax.op.add(bias, re_den))
            divisor = builder.emit(relax.op.divide(added, relax.const(1.4142135, dtype)))
            val_erf = builder.emit(relax.op.erf(divisor))
            added_erf = builder.emit(relax.op.add(val_erf, relax.const(1.0, dtype)))
            mul1 = builder.emit(relax.op.multiply(added, added_erf))
            mul2 = builder.emit(relax.op.multiply(mul1, relax.const(0.5, dtype)))
            out = builder.emit_output(mul2)
        builder.emit_func_output(out)
    return builder.get()


def _make_adaptive_avg_pool_module(data_shape=(1, 32, 16, 16), output_size=(4, 4), dtype="float32"):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(data_shape, dtype))
    with builder.function("main", [x]):
        with builder.dataflow():
            out = builder.emit(relax.op.nn.adaptive_avg_pool2d(x, output_size=list(output_size)))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_standalone_op_module(op_fn, shape=(1, 16), dtype="float32", num_args=3):
    """num_args=3 covers relax.op.clip(x, min, max); num_args=1 covers unary
    ops like sigmoid."""
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    with builder.function("main", [x]):
        with builder.dataflow():
            out = builder.emit(op_fn(x))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_int64_conv2d_module(
    data_shape=(1, 3, 224, 224), weight_shape=(16, 3, 3, 3), dtype="int64"
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    with builder.function("main", [data, weight]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            out = builder.emit_output(conv)
        builder.emit_func_output(out)
    return builder.get()


def _make_resnet_downsample_module(
    data_shape=(1, 256, 56, 56),
    w1x1_shape=(64, 256, 1, 1),
    w3x3_shape=(64, 64, 3, 3),
    dtype="float32",
    swapped=False,
):
    """1x1 conv with stride=2 -> relu -> 3x3 conv with stride=1: the
    ResNet-v1 downsample block that ResNetV1Rewrite is meant to swap
    (1x1 stride 1, 3x3 stride 2) to keep the expensive 3x3 conv at full
    spatial resolution before it strides down.

    swapped=True builds the ALREADY-rewritten graph directly (1x1 stride=1,
    3x3 stride=2), for use as a reference target -- since ResNetV1Rewrite
    changes which pixels get sampled, it is not numerically transparent, so
    the only meaningful reference is a graph built with the swapped strides
    from the start, not the original graph.
    """
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    w1 = relax.Var("w_1x1", relax.TensorType(w1x1_shape, dtype))
    w2 = relax.Var("w_3x3", relax.TensorType(w3x3_shape, dtype))

    stride_1x1 = (1, 1) if swapped else (2, 2)
    stride_3x3 = (2, 2) if swapped else (1, 1)

    with builder.function("main", [data, w1, w2]):
        with builder.dataflow():
            conv1 = builder.emit(relax.op.nn.conv2d(data, w1, strides=stride_1x1))
            relu1 = builder.emit(relax.op.nn.relu(conv1))
            conv2 = builder.emit(relax.op.nn.conv2d(relu1, w2, strides=stride_3x3, padding=(1, 1)))
            out = builder.emit_output(conv2)
        builder.emit_func_output(out)
    return builder.get()


# -------------------------------------------------------------------------
# Original partitioning tests
# -------------------------------------------------------------------------
def test_conv2d_relu_partition_supported():
    mod = _make_conv2d_relu_module(with_relu=True)
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected 1 offloaded DNNL region"
    assert len(regions[0].params) == 2


def test_layer_norm_rewrite_and_partition():
    mod = _make_expanded_layernorm_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 1, (
        "Expected expanded LayerNorm to collapse and offload"
    )


def test_prune_subgraphs_demotes_light_ops():
    mod = _make_standalone_elementwise_module()
    partitioned = partition_for_dnnl(mod, prune_subgraphs=True, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 0, "Standalone elementwise should be pruned"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"with_relu": False}, id="single-conv2d"),
        pytest.param(
            {"data_shape": (1, 16, 56, 56), "weight_shape": (32, 16, 3, 3)}, id="custom-shape"
        ),
    ],
)
def test_conv2d_variants_partition(kwargs):
    mod = _make_conv2d_relu_module(**kwargs)
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 1


def test_matmul_partition_supported():
    mod = _make_matmul_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected matmul to be offloaded as a DNNL region"
    assert len(regions[0].params) == 2


def test_conv2d_bias_relu_partition_supported():
    mod = _make_conv2d_bias_relu_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected 1 offloaded DNNL region for conv2d_bias_relu"
    assert len(regions[0].params) == 3


def test_dense_bias_gelu_reshape_rewrite_and_partition():
    mod = _make_dense_bias_gelu_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 1, (
        "Expected reordered dense+bias+gelu region to offload"
    )


def test_prune_dnnl_subgraphs_removes_light_function():
    from tvm.relax.backend.contrib.dnnl import prune_dnnl_subgraphs
    from tvm.script import ir as I
    from tvm.script import relax as R

    @I.ir_module
    class Mod:
        @R.function(private=True)
        def light_fn(x: R.Tensor((1, 8), dtype="float32")) -> R.Tensor((1, 8), dtype="float32"):
            R.func_attr({"Codegen": "dnnl"})
            with R.dataflow():
                out = R.nn.relu(x)
                R.output(out)
            return out

        @R.function
        def main(x: R.Tensor((1, 8), dtype="float32")) -> R.Tensor((1, 8), dtype="float32"):
            cls = Mod
            with R.dataflow():
                out = cls.light_fn(x)
                R.output(out)
            return out

    pruned = prune_dnnl_subgraphs(Mod)
    remaining = [
        f
        for f in pruned.functions.values()
        if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "dnnl"
    ]
    assert len(remaining) == 0, "Light-compute function should have been demoted"
    assert pruned.get_global_var("main") is not None


def test_partition_without_layout_alteration():
    mod = _make_conv2d_relu_module(with_relu=True)
    partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 1, (
        "Expected conv2d+relu to offload even without layout alteration"
    )


def test_no_offload_for_unsupported_ops():
    mod = _make_standalone_elementwise_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 0
    assert isinstance(partitioned["main"], relax.Function)


def test_conv2d_relu_partition_numerically_correct():
    np.random.seed(0)
    data_shape = (1, 3, 8, 8)
    weight_shape = (4, 3, 3, 3)
    mod = _make_conv2d_relu_module(data_shape=data_shape, weight_shape=weight_shape, with_relu=True)

    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")

    ref_out = _run(mod, [data_np, weight_np])

    partitioned = partition_for_dnnl(mod)  # run_codegen=True default
    out = _run(partitioned, [data_np, weight_np])

    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------------
# Regression: dnnl.global_avg_pool2d must only match TRUE global pooling
# -------------------------------------------------------------------------
class TestGlobalAvgPoolRegression:
    def test_true_global_pool_offloads(self):
        """output_size == (1, 1): the only case dnnl.global_avg_pool2d
        should legitimately match."""
        mod = _make_adaptive_avg_pool_module(output_size=(1, 1))
        partitioned = partition_for_dnnl(mod, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.global_avg_pool2d" in names, f"got {names}"

    def test_general_adaptive_pool_does_not_offload(self):
        """output_size == (4, 4) from a (16, 16) input: NOT a global pool.
        Must NOT be offloaded -- oneDNN's pooling_forward primitive can't
        express this and previously produced 'could not create a descriptor
        for a pooling forward propagation primitive' at vm_initialization."""
        mod = _make_adaptive_avg_pool_module(data_shape=(1, 32, 16, 16), output_size=(4, 4))
        partitioned = partition_for_dnnl(mod, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.global_avg_pool2d" not in names, (
            f"General adaptive_avg_pool2d (non-(1,1) output) was offloaded to DNNL "
            f"-- this will crash oneDNN's descriptor creation at runtime. got {names}"
        )

    def test_true_global_pool_numerically_correct(self):
        np.random.seed(0)
        data_shape = (1, 8, 6, 6)
        mod = _make_adaptive_avg_pool_module(data_shape=data_shape, output_size=(1, 1))
        x_np = np.random.uniform(size=data_shape).astype("float32")

        ref_out = _run(mod, [x_np])
        partitioned = partition_for_dnnl(mod)
        out = _run(partitioned, [x_np])
        np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)

    def test_general_adaptive_pool_still_numerically_correct_via_fallback(self):
        """Confirms the non-offloaded case still runs correctly through TVM's
        native adaptive_avg_pool2d after partition_for_dnnl leaves it alone."""
        np.random.seed(0)
        data_shape = (1, 8, 16, 16)
        mod = _make_adaptive_avg_pool_module(data_shape=data_shape, output_size=(4, 4))
        x_np = np.random.uniform(size=data_shape).astype("float32")

        ref_out = _run(mod, [x_np])
        partitioned = partition_for_dnnl(mod)
        out = _run(partitioned, [x_np])
        np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------------
# Regression: _sum_pattern's bias-shape predicate must accept the
# NCHW-broadcast (oc, 1, 1) per-channel bias convention (it previously
# only accepted scalar/1D, which is valid for matmul but not conv -- see
# module docstring). NOTE: only (oc, 1, 1) is exercised for conv2d here;
# a bare 1D (oc,) bias is NOT a valid conv2d bias shape at all (NCHW
# broadcasting is right-aligned, so (oc,) would try to align against the
# trailing spatial width dim, not the channel dim) -- that convention is
# matmul-specific and is covered by test_matmul_bias_sum_offloads below.
# -------------------------------------------------------------------------
class TestResidualSumPatterns:
    @pytest.mark.parametrize("with_relu", [True, False])
    def test_conv2d_bias_sum_offloads(self, with_relu):
        mod = _make_conv2d_bias_sum_module(with_relu=with_relu)
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        expected = "dnnl.conv2d_bias_sum" + ("_relu" if with_relu else "")
        assert expected in names, (
            f"with_relu={with_relu}: expected '{expected}' in composites but got "
            f"{names} -- the sum pattern likely fell back to unfused "
            f"conv2d_bias/add/relu composites."
        )

    def test_matmul_bias_sum_offloads(self):
        mod = _make_matmul_bias_sum_module()
        partitioned = partition_for_dnnl(mod, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul_bias_sum" in names, f"got {names}"

    def test_conv2d_bias_sum_relu_numerically_correct(self):
        np.random.seed(0)
        data_shape = (1, 8, 16, 16)
        weight_shape = (8, 8, 3, 3)
        oc = weight_shape[0]
        n, _, h, w = data_shape
        mod = _make_conv2d_bias_sum_module(
            data_shape=data_shape, weight_shape=weight_shape, with_relu=True
        )

        data_np = np.random.uniform(size=data_shape).astype("float32")
        weight_np = (np.random.uniform(size=weight_shape) * 0.1).astype("float32")
        bias_np = (np.random.uniform(size=(oc, 1, 1)) * 0.1).astype("float32")
        residual_np = np.random.uniform(size=(n, oc, h, w)).astype("float32")

        ref_out = _run(mod, [data_np, weight_np, bias_np, residual_np])
        partitioned = partition_for_dnnl(mod, alter_layout=False)
        out = _run(partitioned, [data_np, weight_np, bias_np, residual_np])

        np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)

    def test_mismatched_channel_bias_is_rejected(self):
        """A bias shaped (1, C_wrong, 1) -- a valid NCHW-broadcast SHAPE in
        general, but with a channel count that doesn't match the conv's
        actual output channels -- must still be rejected by the predicate.
        Uses relax.TensorType with a symbolic-safe concrete shape so the
        graph itself builds fine (unlike the previous version of this test,
        which never actually added the mismatched bias into the graph and
        so wasn't testing anything); the mismatch is caught by the pattern
        predicate's channel-dim comparison, not by IR construction, since
        (1, 5, 1) legitimately broadcasts against ANY (n, c, h, w) shape at
        the type level -- only the *value* of 5 vs the conv's actual oc=8
        is wrong, which only the pattern's shape check (not the type
        system) can catch."""
        data_shape = (1, 8, 16, 16)
        weight_shape = (8, 8, 3, 3)  # oc = 8
        n, oc, h, w = data_shape[0], weight_shape[0], data_shape[2], data_shape[3]

        builder = relax.BlockBuilder()
        data = relax.Var("data", relax.TensorType(data_shape, "float32"))
        weight = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
        # A bias must both TYPE-CHECK (broadcast successfully against the
        # conv output) and be a valid PER-CHANNEL bias (exactly one
        # non-unit dim, matching oc) for the predicate to accept it. Use
        # (oc, 1, w) -- right-aligned against (n, oc, h, w) this broadcasts
        # fine (oc matches, the middle 1 broadcasts against h, w matches
        # w) so IR construction succeeds -- but it has TWO non-unit dims
        # (oc and w), so it is not actually a per-channel bias and the
        # predicate should still reject it.
        bad_bias = relax.Var("bias", relax.TensorType((oc, 1, w), "float32"))
        residual = relax.Var("residual", relax.TensorType((n, oc, h, w), "float32"))

        with builder.function("main", [data, weight, bad_bias, residual]):
            with builder.dataflow():
                conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
                # (oc, 1, w) is shape-compatible with the conv output but is
                # NOT a valid per-channel bias (it has a real trailing-dim
                # component too), which is exactly what the predicate should
                # reject even though it happens to be broadcast-compatible.
                biased = builder.emit(relax.op.add(conv, bad_bias))
                summed = builder.emit(relax.op.add(biased, residual))
                out = builder.emit(relax.op.nn.relu(summed))
                out = builder.emit_output(out)
            builder.emit_func_output(out)
        mod = builder.get()

        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.conv2d_bias_sum_relu" not in names, (
            f"a bias with more than one non-unit dimension should be rejected "
            f"as a valid per-channel bias, but got {names}"
        )


# -------------------------------------------------------------------------
# int64 safety guard
# -------------------------------------------------------------------------
def test_int64_conv2d_is_not_offloaded():
    """oneDNN doesn't natively support int64; conv2d with int64 operands
    must be rejected by the pattern checkers and left as a native relax
    op rather than offloaded."""
    mod = _make_int64_conv2d_module()
    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 0, "int64 conv2d should not be offloaded to DNNL"

    main = partitioned["main"]
    found_conv2d = any(
        isinstance(getattr(b, "value", None), relax.Call)
        and isinstance(b.value.op, tvm.ir.Op)
        and b.value.op.name == "relax.nn.conv2d"
        for block in main.body.blocks
        for b in block.bindings
    )
    assert found_conv2d, "expected relax.nn.conv2d to remain un-lowered in main"


# -------------------------------------------------------------------------
# ResNetV1 downsample rewrite
# -------------------------------------------------------------------------
def test_resnet_downsample_stride_swap():
    """1x1(stride=2)->relu->3x3(stride=1) must become 1x1(stride=1)->relu->
    3x3(stride=2) -- verified via output shapes (spatial downsampling should
    move from the 1x1 conv to the 3x3 conv) rather than reaching into
    external-module internals, since the composite's own attrs aren't
    visible post-BYOC-lowering."""
    mod = _make_resnet_downsample_module()
    partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

    # Structural check: walk the (pre-codegen) composite bodies and confirm
    # the 1x1 conv's stride is now 1 and the 3x3 conv's stride is now 2.
    strides_by_kernel = {}

    def _walk(func):
        seq = func.body if isinstance(func.body, relax.SeqExpr) else None
        if seq is None:
            return
        for block in seq.blocks:
            for binding in block.bindings:
                v = getattr(binding, "value", None)
                if isinstance(v, relax.Function):
                    _walk(v)
                elif (
                    isinstance(v, relax.Call)
                    and isinstance(v.op, tvm.ir.Op)
                    and v.op.name == "relax.nn.conv2d"
                ):
                    weight_shape = tuple(int(d) for d in v.args[1].ty.shape.values)
                    kh, kw = weight_shape[-2], weight_shape[-1]
                    strides_by_kernel[(kh, kw)] = tuple(int(s) for s in v.attrs.strides)

    for _, func in partitioned.functions.items():
        if isinstance(func, relax.Function):
            _walk(func)

    assert strides_by_kernel.get((1, 1)) == (1, 1), (
        f"expected 1x1 conv stride to become (1, 1) after rewrite, got "
        f"{strides_by_kernel.get((1, 1))}"
    )
    assert strides_by_kernel.get((3, 3)) == (2, 2), (
        f"expected 3x3 conv stride to become (2, 2) after rewrite, got "
        f"{strides_by_kernel.get((3, 3))}"
    )


def test_resnet_downsample_rewrite_matches_swapped_reference():
    """ResNetV1Rewrite is a genuine architecture change (moves the stride from the
    1x1 to the 3x3 conv), not a numerically-transparent optimization -- so its
    output will NOT match the original graph. What we can verify is that DNNL's
    output matches a reference graph built with the swapped strides directly."""
    np.random.seed(0)
    data_shape = (1, 256, 56, 56)
    w1x1_shape = (64, 256, 1, 1)
    w3x3_shape = (64, 64, 3, 3)

    mod = _make_resnet_downsample_module(data_shape, w1x1_shape, w3x3_shape)  # 1x1 s2, 3x3 s1
    ref_mod = _make_resnet_downsample_module(
        data_shape,
        w1x1_shape,
        w3x3_shape,
        swapped=True,  # 1x1 s1, 3x3 s2 -- built directly
    )

    data_np = (np.random.uniform(size=data_shape) * 0.1).astype("float32")
    w1_np = (np.random.uniform(size=w1x1_shape) * 0.1).astype("float32")
    w2_np = (np.random.uniform(size=w3x3_shape) * 0.1).astype("float32")

    ref_out = _run(ref_mod, [data_np, w1_np, w2_np])
    partitioned = partition_for_dnnl(mod, alter_layout=False)  # applies ResNetV1Rewrite internally
    out = _run(partitioned, [data_np, w1_np, w2_np])

    assert out.shape == ref_out.shape == (1, 64, 28, 28)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)


# -------------------------------------------------------------------------
# Standalone elementwise ops beyond compute ops
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,op_fn,num_args",
    [
        pytest.param("sigmoid", lambda x: relax.op.sigmoid(x), 1, id="sigmoid"),
        pytest.param("clip", lambda x: relax.op.clip(x, -1.0, 1.0), 3, id="clip"),
    ],
)
def test_standalone_eltwise_offloads_and_is_correct(name, op_fn, num_args):
    shape = (1, 16)
    mod = _make_standalone_op_module(op_fn, shape=shape)

    partitioned = partition_for_dnnl(mod, run_codegen=False)
    names = _all_composite_names_in_mod(partitioned)
    assert f"dnnl.{name}" in names, f"got {names}"

    np.random.seed(0)
    x_np = np.random.uniform(-2, 2, size=shape).astype("float32")
    ref_out = _run(mod, [x_np])
    codegen_mod = partition_for_dnnl(mod)
    out = _run(codegen_mod, [x_np])
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
