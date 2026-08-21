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
DNNL BYOC backend/partition tests.

Pure partitioning tests: attribute validation (rejecting unsupported dtypes/attrs) and graph
structure (fusion into a single offloaded region). Everything runs with run_codegen=False, so
no DNNL runtime build or execution happens -- host CPU, no GPU/runtime needed.
"""

import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl


def _dnnl_regions(mod):
    return [
        func
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
    ]


def _composite_names(mod):
    names = []
    seen_funcs = set()

    def _collect(func):
        if id(func) in seen_funcs:
            return
        seen_funcs.add(id(func))
        if func.attrs is not None and func.attrs.get("Composite") is not None:
            names.append(str(func.attrs.get("Composite")))

        def _visit(expr):
            if isinstance(expr, relax.Function):
                _collect(expr)
            elif isinstance(expr, relax.Call) and isinstance(expr.op, relax.Function):
                _collect(expr.op)

        relax.analysis.post_order_visit(func.body, _visit)

    for f in mod.functions.values():
        if isinstance(f, relax.Function):
            _collect(f)
    return names


def _partition(mod, **kwargs):
    kwargs.setdefault("run_codegen", False)
    return partition_for_dnnl(mod, **kwargs)


# -------------------------------------------------------------------------
# Module builders
# -------------------------------------------------------------------------
def _make_conv2d_module(
    data_shape=(1, 3, 224, 224), weight_shape=(16, 3, 3, 3), dtype="float32", with_relu=False
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


def _make_conv_variant_module(op_fn, data_shape, weight_shape, dtype="float32", **kwargs):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    with builder.function("main", [data, weight]):
        with builder.dataflow():
            out = builder.emit(op_fn(data, weight, **kwargs))
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


def _make_conv2d_bias_module(
    data_shape=(1, 3, 32, 32),
    weight_shape=(4, 3, 3, 3),
    dtype="float32",
    activation=None,
    with_clip=False,
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    bias = relax.Var("bias", relax.TensorType((weight_shape[0], 1, 1), dtype))
    with builder.function("main", [data, weight, bias]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            out = builder.emit(relax.op.add(conv, bias))
            if activation is not None:
                out = builder.emit(activation(out))
            if with_clip:
                out = builder.emit(relax.op.clip(out, -1.0, 1.0))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_bias_module(m=8, k=16, n=32, dtype="float32", activation=None):
    builder = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorType((m, k), dtype))
    b = relax.Var("b", relax.TensorType((k, n), dtype))
    bias = relax.Var("bias", relax.TensorType((n,), dtype))
    with builder.function("main", [a, b, bias]):
        with builder.dataflow():
            out = builder.emit(relax.op.add(relax.op.matmul(a, b), bias))
            if activation is not None:
                out = builder.emit(activation(out))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_conv2d_bias_sum_module(
    data_shape=(1, 8, 16, 16), weight_shape=(8, 8, 3, 3), dtype="float32", with_relu=True
):
    oc = weight_shape[0]
    n, _, h, w = data_shape
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))
    bias = relax.Var("bias", relax.TensorType((oc, 1, 1), dtype))
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


def _make_mismatched_bias_sum_module():
    """bias shaped (oc, 1, w): broadcast-compatible but not a valid per-channel bias
    (two non-unit dims) -- should not fuse."""
    data_shape, weight_shape = (1, 8, 16, 16), (8, 8, 3, 3)
    n, oc, h, w = data_shape[0], weight_shape[0], data_shape[2], data_shape[3]
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    weight = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
    bad_bias = relax.Var("bias", relax.TensorType((oc, 1, w), "float32"))
    residual = relax.Var("residual", relax.TensorType((n, oc, h, w), "float32"))
    with builder.function("main", [data, weight, bad_bias, residual]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight, padding=[1, 1]))
            biased = builder.emit(relax.op.add(conv, bad_bias))
            summed = builder.emit(relax.op.add(biased, residual))
            out = builder.emit(relax.op.nn.relu(summed))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_bias_sum_module(m=16, k=32, n=64, dtype="float32"):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType((m, k), dtype))
    weight = relax.Var("weight", relax.TensorType((k, n), dtype))
    bias = relax.Var("bias", relax.TensorType((n,), dtype))
    residual = relax.Var("residual", relax.TensorType((m, n), dtype))
    with builder.function("main", [data, weight, bias, residual]):
        with builder.dataflow():
            biased = builder.emit(relax.op.add(relax.op.matmul(data, weight), bias))
            summed = builder.emit(relax.op.add(biased, residual))
            out = builder.emit_output(summed)
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
            deno = builder.emit(relax.op.sqrt(relax.op.add(var, relax.const(1e-5, dtype))))
            norm = builder.emit(relax.op.divide(diff, deno))
            scaled = builder.emit(relax.op.multiply(norm, gamma))
            out = builder.emit(relax.op.add(scaled, beta))
            out = builder.emit_output(out)
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
            erf = builder.emit(relax.op.erf(divisor))
            added_erf = builder.emit(relax.op.add(erf, relax.const(1.0, dtype)))
            mul1 = builder.emit(relax.op.multiply(added, added_erf))
            out = builder.emit(relax.op.multiply(mul1, relax.const(0.5, dtype)))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_standalone_elementwise_module(shape=(1, 64, 56, 56), dtype="float32"):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    y = relax.Var("y", relax.TensorType(shape, dtype))
    with builder.function("main", [x, y]):
        with builder.dataflow():
            out = builder.emit(relax.op.nn.relu(relax.op.add(x, y)))
            out = builder.emit_output(out)
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


def _make_pool_module(op_fn, shape=(1, 8, 16, 16), dtype="float32", **kwargs):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    with builder.function("main", [x]):
        with builder.dataflow():
            out = builder.emit(op_fn(x, **kwargs))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_standalone_op_module(op_fn, shape=(1, 16), dtype="float32"):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    with builder.function("main", [x]):
        with builder.dataflow():
            out = builder.emit(op_fn(x))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_resnet_downsample_module(
    data_shape=(1, 256, 56, 56),
    w1x1_shape=(64, 256, 1, 1),
    w3x3_shape=(64, 64, 3, 3),
    dtype="float32",
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    w1 = relax.Var("w_1x1", relax.TensorType(w1x1_shape, dtype))
    w2 = relax.Var("w_3x3", relax.TensorType(w3x3_shape, dtype))
    with builder.function("main", [data, w1, w2]):
        with builder.dataflow():
            conv1 = builder.emit(relax.op.nn.conv2d(data, w1, strides=(2, 2)))
            relu1 = builder.emit(relax.op.nn.relu(conv1))
            conv2 = builder.emit(relax.op.nn.conv2d(relu1, w2, strides=(1, 1), padding=(1, 1)))
            out = builder.emit_output(conv2)
        builder.emit_func_output(out)
    return builder.get()


def _make_pad_avgpool_module(
    shape=(1, 8, 16, 16), pad_value=0.0, pool_padding=(0, 0), dtype="float32"
):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    with builder.function("main", [x]):
        with builder.dataflow():
            padded = builder.emit(
                relax.op.nn.pad(
                    x, pad_width=[0, 0, 0, 0, 1, 1, 1, 1], pad_mode="constant", pad_value=pad_value
                )
            )
            out = builder.emit(
                relax.op.nn.avg_pool2d(padded, pool_size=(3, 3), padding=pool_padding)
            )
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_batch_norm_module(shape=(1, 8, 16, 16), dtype="float32"):
    c = shape[1]
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))
    gamma = relax.Var("gamma", relax.TensorType((c,), dtype))
    beta = relax.Var("beta", relax.TensorType((c,), dtype))
    mean = relax.Var("mean", relax.TensorType((c,), dtype))
    var = relax.Var("var", relax.TensorType((c,), dtype))
    with builder.function("main", [x, gamma, beta, mean, var]):
        with builder.dataflow():
            bn = builder.emit(relax.op.nn.batch_norm(x, gamma, beta, mean, var, axis=1))
            out = builder.emit_output(relax.TupleGetItem(bn, 0))
        builder.emit_func_output(out)
    return builder.get()


def _make_qnn_conv2d_module(dynamic_out_scale=False):
    data_shape = (1, 1, 2, 2)
    weight_q = relax.const([[[[1, 1], [1, 1]]]], "int8")
    data_scale, data_zp = relax.const(0.1, "float32"), relax.const(0, "int32")
    weight_scale, weight_zp = relax.const(0.05, "float32"), relax.const(0, "int32")
    out_zp = relax.const(0, "int32")

    builder = relax.BlockBuilder()
    data_q = relax.Var("data_q", relax.TensorType(data_shape, "int8"))
    params = [data_q]
    if dynamic_out_scale:
        out_scale = relax.Var("out_scale", relax.TensorType((), "float32"))
        params.append(out_scale)
    else:
        out_scale = relax.const(0.02, "float32")

    with builder.function("main", params):
        with builder.dataflow():
            dq_data = builder.emit(relax.op.dequantize(data_q, data_scale, data_zp))
            dq_weight = builder.emit(relax.op.dequantize(weight_q, weight_scale, weight_zp))
            conv = builder.emit(relax.op.nn.conv2d(dq_data, dq_weight))
            out = builder.emit(relax.op.quantize(conv, out_scale, out_zp, out_dtype="int8"))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_int64_module(kind):
    if kind == "conv2d":
        return _make_conv2d_module(dtype="int64")
    if kind == "matmul":
        return _make_matmul_module(dtype="int64")
    if kind == "eltwise":
        return _make_standalone_op_module(relax.op.abs, dtype="int64")
    raise ValueError(kind)


# -------------------------------------------------------------------------
# Attribute validation: reject dtypes/attrs/shapes the backend can't handle
# -------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["conv2d", "matmul", "eltwise"])
def test_int64_rejected(kind):
    mod = _make_int64_module(kind)
    partitioned = _partition(mod)
    assert len(_dnnl_regions(partitioned)) == 0


def test_general_adaptive_pool_rejected():
    """Only true global pooling (output_size == (1, 1)) is valid; oneDNN's pooling primitive
    can't express a general adaptive pool."""
    mod = _make_adaptive_avg_pool_module(output_size=(4, 4))
    partitioned = _partition(mod)
    assert "dnnl.global_avg_pool2d" not in _composite_names(partitioned)


def test_true_global_pool_accepted():
    mod = _make_adaptive_avg_pool_module(output_size=(1, 1))
    partitioned = _partition(mod)
    assert "dnnl.global_avg_pool2d" in _composite_names(partitioned)


def test_pooling_ceil_mode_rejected():
    mod = _make_pool_module(
        relax.op.nn.max_pool2d, pool_size=(2, 2), strides=(2, 2), ceil_mode=True
    )
    partitioned = _partition(mod)
    assert "dnnl.max_pool2d" not in _composite_names(partitioned)


def test_mismatched_channel_bias_rejected():
    """Bias shape broadcasts fine but isn't actually per-channel -- must not fuse."""
    mod = _make_mismatched_bias_sum_module()
    partitioned = _partition(mod, alter_layout=False)
    assert "dnnl.conv2d_bias_sum_relu" not in _composite_names(partitioned)


def test_qnn_dynamic_scale_rejected():
    """Requantize scale/zero_point must be compile-time constants."""
    mod = _make_qnn_conv2d_module(dynamic_out_scale=True)
    partitioned = _partition(mod)
    assert "dnnl.qnn.conv2d" not in _composite_names(partitioned)


def test_unsupported_standalone_elementwise_not_offloaded():
    mod = _make_standalone_elementwise_module()
    partitioned = _partition(mod)
    assert len(_dnnl_regions(partitioned)) == 0


# -------------------------------------------------------------------------
# Graph structure: fusion into a single offloaded region
# -------------------------------------------------------------------------
def test_conv2d_relu_fuses_to_one_region():
    mod = _make_conv2d_module(with_relu=True)
    partitioned = _partition(mod)
    regions = _dnnl_regions(partitioned)
    assert len(regions) == 1
    assert len(regions[0].params) == 2


def test_matmul_partitions():
    mod = _make_matmul_module()
    partitioned = _partition(mod)
    assert len(_dnnl_regions(partitioned)) == 1


@pytest.mark.parametrize(
    "op_fn,data_shape,weight_shape,composite",
    [
        pytest.param(relax.op.nn.conv1d, (1, 3, 32), (8, 3, 3), "dnnl.conv1d", id="conv1d"),
        pytest.param(
            relax.op.nn.conv3d, (1, 3, 8, 8, 8), (4, 3, 3, 3, 3), "dnnl.conv3d", id="conv3d"
        ),
        pytest.param(
            relax.op.nn.conv2d_transpose,
            (1, 8, 16, 16),
            (8, 4, 3, 3),
            "dnnl.conv2d_transpose",
            id="conv2d_transpose",
        ),
    ],
)
def test_conv_variant_partitions(op_fn, data_shape, weight_shape, composite):
    padding = [1] * (len(data_shape) - 2)
    mod = _make_conv_variant_module(op_fn, data_shape, weight_shape, padding=padding)
    partitioned = _partition(mod)
    assert composite in _composite_names(partitioned)


def test_conv2d_bias_fuses_without_activation():
    mod = _make_conv2d_bias_module(activation=None)
    partitioned = _partition(mod)
    assert "dnnl.conv2d_bias" in _composite_names(partitioned)


@pytest.mark.parametrize(
    "activation,suffix",
    [
        pytest.param(relax.op.nn.relu, "relu", id="relu"),
        pytest.param(relax.op.sigmoid, "sigmoid", id="sigmoid"),
        pytest.param(relax.op.nn.gelu, "gelu", id="gelu"),
        pytest.param(relax.op.tanh, "tanh", id="tanh"),
    ],
)
def test_conv2d_bias_activation_fuses(activation, suffix):
    mod = _make_conv2d_bias_module(activation=activation)
    partitioned = _partition(mod)
    assert f"dnnl.conv2d_bias_{suffix}" in _composite_names(partitioned)


def test_conv2d_bias_clip_fuses():
    mod = _make_conv2d_bias_module(activation=None, with_clip=True)
    partitioned = _partition(mod)
    assert "dnnl.conv2d_bias_clip" in _composite_names(partitioned)


def test_matmul_bias_relu_fuses():
    mod = _make_matmul_bias_module(activation=relax.op.nn.relu)
    partitioned = _partition(mod)
    assert "dnnl.matmul_bias_relu" in _composite_names(partitioned)


@pytest.mark.parametrize("with_relu", [True, False])
def test_conv2d_bias_sum_fuses(with_relu):
    mod = _make_conv2d_bias_sum_module(with_relu=with_relu)
    partitioned = _partition(mod, alter_layout=False)
    expected = "dnnl.conv2d_bias_sum" + ("_relu" if with_relu else "")
    assert expected in _composite_names(partitioned)


def test_matmul_bias_sum_fuses():
    mod = _make_matmul_bias_sum_module()
    partitioned = _partition(mod)
    assert "dnnl.matmul_bias_sum" in _composite_names(partitioned)


def test_layer_norm_rewrite_fuses_to_one_region():
    mod = _make_expanded_layernorm_module()
    partitioned = _partition(mod)
    assert len(_dnnl_regions(partitioned)) == 1


def test_dense_bias_gelu_rewrite_fuses():
    mod = _make_dense_bias_gelu_module()
    partitioned = _partition(mod)
    assert any("gelu" in n for n in _composite_names(partitioned))


def test_batch_norm_rewrite_fuses():
    mod = _make_batch_norm_module()
    partitioned = _partition(mod)
    assert "dnnl.batch_norm" in _composite_names(partitioned)


def test_pad_avg_pool_folds_and_still_offloads():
    mod = _make_pad_avgpool_module(pad_value=0.0, pool_padding=(0, 0))
    partitioned = _partition(mod)
    assert "dnnl.avg_pool2d" in _composite_names(partitioned)


def test_qnn_conv2d_fuses():
    mod = _make_qnn_conv2d_module()
    partitioned = _partition(mod)
    assert "dnnl.qnn.conv2d" in _composite_names(partitioned)


def test_resnet_downsample_stride_swap():
    mod = _make_resnet_downsample_module()
    partitioned = _partition(mod, alter_layout=False)

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
                    strides_by_kernel[weight_shape[-2:]] = tuple(int(s) for s in v.attrs.strides)

    for func in partitioned.functions.values():
        if isinstance(func, relax.Function):
            _walk(func)

    assert strides_by_kernel.get((1, 1)) == (1, 1)
    assert strides_by_kernel.get((3, 3)) == (2, 2)


def test_prune_subgraphs_demotes_light_ops():
    mod = _make_standalone_elementwise_module()
    partitioned = partition_for_dnnl(mod, prune_subgraphs=True, run_codegen=False)
    assert len(_dnnl_regions(partitioned)) == 0
