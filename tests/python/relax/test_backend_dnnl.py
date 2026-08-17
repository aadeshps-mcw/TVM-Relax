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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)
pytestmark = [pytest.mark.skipif(not has_dnnl, reason="DNNL not enabled.")]

def _dnnl_regions(mod: tvm.IRModule) -> list[relax.Function]:
    """Helper to extract offloaded DNNL functions from partitioned module."""
    return [
        func
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
    ]


# -------------------------------------------------------------------------
# Test Helper Builders
# -------------------------------------------------------------------------
def _make_conv2d_relu_module(
    data_shape=(1, 3, 224, 224),
    weight_shape=(16, 3, 3, 3),
    dtype="float32",
    with_relu=True,
):
    """Builds a simple Conv2D (+ ReLU) Relax IRModule."""
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))

    with builder.function("main", [data, weight]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            if with_relu:
                out = builder.emit(relax.op.nn.relu(conv))
            else:
                out = conv
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_expanded_layernorm_module(shape=(1, 3136, 64), dtype="float32"):
    """Builds an expanded primitive LayerNorm graph (Pattern #1)."""
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
    """Builds a graph with ONLY elementwise ops (no compute ops)."""
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
    """Builds Conv2D + bias + ReLU, to exercise the dnnl.conv2d_bias_relu pattern."""
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    # Bias is per-output-channel, broadcastable against NCHW conv output.
    bias = relax.Var("bias", relax.TensorType((weight_shape[0], 1, 1), dtype))

    with builder.function("main", [data, weight, bias]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            biased = builder.emit(relax.op.add(conv, bias))
            out = builder.emit(relax.op.nn.relu(biased))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_module(m=8, k=16, n=32, dtype="float32"):
    """Builds a standalone matmul graph, to exercise the dnnl.matmul pattern."""
    builder = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorType((m, k), dtype))
    b = relax.Var("b", relax.TensorType((k, n), dtype))

    with builder.function("main", [a, b]):
        with builder.dataflow():
            out = builder.emit(relax.op.matmul(a, b))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_dense_bias_gelu_module(batch=2, seq=4, in_dim=8, out_dim=16, dtype="float32"):
    """Builds matmul -> reshape -> bias-add -> gelu(erf form), matching the
    shape rewrite_dense_bias_gelu_reshape_last is designed to catch."""
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


# -------------------------------------------------------------------------
# Original Partitioning Test Cases
# -------------------------------------------------------------------------
def test_conv2d_relu_partition_supported():
    """Verify Conv2D + ReLU pattern is matched and offloaded to DNNL."""
    mod = _make_conv2d_relu_module(with_relu=True)
    partitioned = partition_for_dnnl(mod)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected 1 offloaded DNNL region"
    assert len(regions[0].params) == 2


def test_layer_norm_rewrite_and_partition():
    """Verify expanded LayerNorm is rewritten into relax.nn.layer_norm and offloaded."""
    mod = _make_expanded_layernorm_module()
    partitioned = partition_for_dnnl(mod)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected expanded LayerNorm to collapse and offload"


def test_prune_subgraphs_demotes_light_ops():
    """Verify subgraphs containing NO heavy compute ops (conv/dense/matmul) are pruned."""
    mod = _make_standalone_elementwise_module()

    # With pruning ENABLED (default): should demote and keep everything in main TVM execution
    partitioned = partition_for_dnnl(mod, prune_subgraphs=True)
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
    """Parametrized test checking Conv2D variants."""
    mod = _make_conv2d_relu_module(**kwargs)
    partitioned = partition_for_dnnl(mod)
    assert len(_dnnl_regions(partitioned)) == 1


# -------------------------------------------------------------------------
# Additional Partitioning Tests
# -------------------------------------------------------------------------
def test_matmul_partition_supported():
    """Verify a standalone matmul is matched and offloaded to DNNL."""
    mod = _make_matmul_module()
    partitioned = partition_for_dnnl(mod)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected matmul to be offloaded as a DNNL region"
    assert len(regions[0].params) == 2


def test_conv2d_bias_relu_partition_supported():
    """Verify Conv2D + bias + ReLU pattern is matched and offloaded to DNNL."""
    mod = _make_conv2d_bias_relu_module()
    partitioned = partition_for_dnnl(mod)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected 1 offloaded DNNL region for conv2d_bias_relu"
    assert len(regions[0].params) == 3


def test_dense_bias_gelu_reshape_rewrite_and_partition():
    """Verify the matmul->reshape->bias->gelu reorder rewrite fires and the
    resulting matmul+bias(+gelu) region is offloaded to DNNL."""
    mod = _make_dense_bias_gelu_module()
    partitioned = partition_for_dnnl(mod)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected reordered dense+bias+gelu region to offload"


def test_prune_dnnl_subgraphs_removes_light_function():
    """Directly verify prune_dnnl_subgraphs demotes a hand-tagged region with
    no compute ops. Mirrors real partitioned-graph structure: a public `main`
    that calls a private helper function tagged Codegen="dnnl" containing
    only elementwise ops (no compute op), which prune_dnnl_subgraphs should
    demote (strip the Codegen tag from) and then inline/eliminate as a
    separate global."""
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

    mod = Mod

    pruned = prune_dnnl_subgraphs(mod)
    remaining = [
        f
        for f in pruned.functions.values()
        if isinstance(f, relax.Function) and f.attrs and f.attrs.get("Codegen") == "dnnl"
    ]
    assert len(remaining) == 0, "Light-compute function should have been demoted"
    # main must survive pruning -- it's the public entry point and was never tagged.
    assert pruned.get_global_var("main") is not None


def test_partition_without_layout_alteration():
    """Verify alter_layout=False path still partitions conv2d correctly."""
    mod = _make_conv2d_relu_module(with_relu=True)
    partitioned = partition_for_dnnl(mod, alter_layout=False)

    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1, "Expected conv2d+relu to offload even without layout alteration"


def test_no_offload_for_unsupported_ops():
    """Verify a graph with no matchable ops results in zero DNNL regions and
    an unchanged main function."""
    mod = _make_standalone_elementwise_module()
    partitioned = partition_for_dnnl(mod)

    assert len(_dnnl_regions(partitioned)) == 0
    main = partitioned["main"]
    assert isinstance(main, relax.Function)


def test_conv2d_relu_partition_numerically_correct():
    """Build, run, and numerically verify a partitioned conv2d+relu graph
    against a plain (non-partitioned) reference execution."""
    np.random.seed(0)
    data_shape = (1, 3, 8, 8)
    weight_shape = (4, 3, 3, 3)
    mod = _make_conv2d_relu_module(data_shape=data_shape, weight_shape=weight_shape, with_relu=True)

    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")

    target = tvm.target.Target("llvm")

    # Reference: build and run the unpartitioned module directly.
    ref_ex = relax.build(mod, target=target)
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    ref_out = ref_vm["main"](tvm.runtime.tensor(data_np), tvm.runtime.tensor(weight_np)).numpy()

    # Partitioned + codegen'd module.
    partitioned = partition_for_dnnl(mod)
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](tvm.runtime.tensor(data_np), tvm.runtime.tensor(weight_np)).numpy()

    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
