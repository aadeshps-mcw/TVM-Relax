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
#
# NOTE: coverage here is intentionally limited to conv1d/2d/3d, conv2d_transpose/conv3d_transpose,
# and avg/max pooling, since those are the only ops BuildEngine currently dispatches to in
# dnnl_json_runtime.cc. matmul/layer_norm/qnn patterns are registered in the pattern table but are
# not runtime-wired yet, so they are left out of this file for now.
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl
from tvm.script import relax as R

has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)

requires_dnnl_codegen = pytest.mark.skipif(
    not has_dnnl,
    reason="DNNL not enabled.",
)

pytestmark = [requires_dnnl_codegen]

torch = pytest.importorskip("torch")


def build_and_run(mod, inputs_np, target, legalize=False):
    with tvm.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = tvm.compile(mod, target)
    dev = tvm.device_from_target(target, 0)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs_np]
    return f(*inputs).numpy()


def _offload_and_compare(mod, params_np, data_np, rtol=1e-4, atol=1e-4):
    """Offload a single-op module to DNNL and compare against the LLVM reference.

    Each module here contains a single instance of the op under test, which both exercises the
    individual converter and avoids the structurally-identical-composite deduplication that would
    otherwise collapse repeated ops.
    """
    ref = build_and_run(mod, [data_np, *params_np.values()], "llvm", legalize=True)
    mod = relax.transform.BindParams("main", params_np)(mod)
    partitioned = partition_for_dnnl(mod)
    # Guard against a silent false pass: if no pattern matched, nothing is offloaded and the
    # comparison would trivially succeed via the TVM fallback without exercising DNNL at all.
    assert any(
        isinstance(fn, relax.Function)
        and fn.attrs is not None
        and fn.attrs.get("Codegen") == "dnnl"
        for fn in partitioned.functions.values()
    ), "expected the op under test to be offloaded to DNNL, but nothing was partitioned"
    offloaded = relax.transform.RunCodegen()(partitioned)
    # legalize=True here too: whatever didn't get offloaded (e.g. if only part of a graph fused
    # into a DNNL region) still needs lowering to run at all, not just the DNNL region itself.
    out = build_and_run(offloaded, [data_np], "llvm", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


def test_dnnl_conv1d():
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(data: R.Tensor((2, 8, 16), "float32"), weight: R.Tensor((4, 8, 3), "float32")):
            with R.dataflow():
                out = relax.op.nn.conv1d(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16).astype("float32")
    weight = np.random.randn(4, 8, 3).astype("float32")
    _offload_and_compare(Conv1d, {"weight": weight}, data)


def test_dnnl_conv3d():
    @tvm.script.ir_module
    class Conv3d:
        @R.function
        def main(
            data: R.Tensor((1, 4, 8, 8, 8), "float32"), weight: R.Tensor((6, 4, 3, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 4, 8, 8, 8).astype("float32")
    weight = np.random.randn(6, 4, 3, 3, 3).astype("float32")
    _offload_and_compare(Conv3d, {"weight": weight}, data)


def test_dnnl_conv2d_dilated():
    # Regression test: DNNL's dilation convention is "0 == no dilation", TVM's is "1 == no
    # dilation". BuildEngine subtracts 1 from every dilation value before calling into oneDNN;
    # get this wrong and results silently drift instead of erroring.
    @tvm.script.ir_module
    class DilatedConv2d:
        @R.function
        def main(
            data: R.Tensor((1, 4, 16, 16), "float32"), weight: R.Tensor((4, 4, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=2, dilation=2)
                R.output(out)
            return out

    data = np.random.randn(1, 4, 16, 16).astype("float32")
    weight = np.random.randn(4, 4, 3, 3).astype("float32")
    _offload_and_compare(DilatedConv2d, {"weight": weight}, data)


def test_dnnl_conv2d_grouped():
    # Exercises TensorRequisite::ApplyGroupWeightLayout: the weight is reshaped from the plain
    # (O, I/groups, kh, kw) layout the frontend produces into DNNL's grouped
    # (G, O/G, I/G, kh, kw) form before the convolution primitive is built.
    @tvm.script.ir_module
    class GroupedConv2d:
        @R.function
        def main(
            data: R.Tensor((1, 16, 12, 12), "float32"), weight: R.Tensor((16, 4, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=1, groups=4)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 12, 12).astype("float32")
    weight = np.random.randn(16, 4, 3, 3).astype("float32")
    _offload_and_compare(GroupedConv2d, {"weight": weight}, data)


def test_dnnl_conv2d_depthwise():
    # groups == in_channels == out_channels; the extreme case of the grouped-weight reshape,
    # where every group holds a single input and output channel.
    @tvm.script.ir_module
    class DepthwiseConv2d:
        @R.function
        def main(
            data: R.Tensor((1, 8, 12, 12), "float32"), weight: R.Tensor((8, 1, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=1, groups=8)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 12, 12).astype("float32")
    weight = np.random.randn(8, 1, 3, 3).astype("float32")
    _offload_and_compare(DepthwiseConv2d, {"weight": weight}, data)


def test_dnnl_conv2d_bias_relu():
    @tvm.script.ir_module
    class Conv2dBiasRelu:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = relax.op.nn.relu(biased)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 16, 16).astype("float32")
    weight = np.random.randn(16, 8, 3, 3).astype("float32")
    bias = np.random.randn(16, 1, 1).astype("float32")
    _offload_and_compare(Conv2dBiasRelu, {"weight": weight, "bias": bias}, data)


def test_dnnl_conv2d_bias_only():
    @tvm.script.ir_module
    class Conv2dBias:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                out = relax.op.add(conv, bias)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 16, 16).astype("float32")
    weight = np.random.randn(16, 8, 3, 3).astype("float32")
    bias = np.random.randn(16, 1, 1).astype("float32")
    _offload_and_compare(Conv2dBias, {"weight": weight, "bias": bias}, data)


@pytest.mark.parametrize(
    "activation_op",
    [relax.op.sigmoid, relax.op.nn.gelu, relax.op.tanh],
)
def test_dnnl_conv2d_bias_activation(activation_op):
    @tvm.script.ir_module
    class Conv2dBiasAct:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = activation_op(biased)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 16, 16).astype("float32")
    weight = np.random.randn(16, 8, 3, 3).astype("float32")
    bias = np.random.randn(16, 1, 1).astype("float32")
    _offload_and_compare(Conv2dBiasAct, {"weight": weight, "bias": bias}, data)


def test_dnnl_conv2d_bias_sum_relu():
    @tvm.script.ir_module
    class Conv2dBiasSumRelu:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
            residual: R.Tensor((1, 16, 16, 16), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                summed = relax.op.add(biased, residual)
                out = relax.op.nn.relu(summed)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 16, 16).astype("float32")
    weight = np.random.randn(16, 16, 3, 3).astype("float32")
    bias = np.random.randn(16, 1, 1).astype("float32")
    residual = np.random.randn(1, 16, 16, 16).astype("float32")
    _offload_and_compare(
        Conv2dBiasSumRelu, {"weight": weight, "bias": bias, "residual": residual}, data
    )


def test_dnnl_conv2d_bias_sum():
    @tvm.script.ir_module
    class Conv2dBiasSum:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
            residual: R.Tensor((1, 16, 16, 16), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = relax.op.add(biased, residual)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 16, 16).astype("float32")
    weight = np.random.randn(16, 16, 3, 3).astype("float32")
    bias = np.random.randn(16, 1, 1).astype("float32")
    residual = np.random.randn(1, 16, 16, 16).astype("float32")
    _offload_and_compare(
        Conv2dBiasSum, {"weight": weight, "bias": bias, "residual": residual}, data
    )


def test_dnnl_conv2d_transpose_bias_relu():
    # Default IOHW kernel layout ([in, out, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTransposeBiasRelu:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8), "float32"),
            weight: R.Tensor((8, 4, 3, 3), "float32"),
            bias: R.Tensor((4, 1, 1), "float32"),
        ):
            with R.dataflow():
                deconv = relax.op.nn.conv2d_transpose(data, weight, padding=1)
                biased = relax.op.add(deconv, bias)
                out = relax.op.nn.relu(biased)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 8, 8).astype("float32")
    weight = np.random.randn(8, 4, 3, 3).astype("float32")
    bias = np.random.randn(4, 1, 1).astype("float32")
    _offload_and_compare(ConvTransposeBiasRelu, {"weight": weight, "bias": bias}, data)


def test_dnnl_conv3d_transpose():
    # Default IODHW kernel layout ([in, out, d, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTranspose3d:
        @R.function
        def main(
            data: R.Tensor((1, 4, 6, 6, 6), "float32"), weight: R.Tensor((4, 2, 3, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 4, 6, 6, 6).astype("float32")
    weight = np.random.randn(4, 2, 3, 3, 3).astype("float32")
    _offload_and_compare(ConvTranspose3d, {"weight": weight}, data)


def test_dnnl_max_pool2d():
    @tvm.script.ir_module
    class MaxPool:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2))
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    _offload_and_compare(MaxPool, {}, data)


@pytest.mark.parametrize("count_include_pad", [True, False])
def test_dnnl_avg_pool2d_count_include_pad(count_include_pad):
    @tvm.script.ir_module
    class AvgPool:
        @R.function
        def main(data: R.Tensor((2, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.avg_pool2d(
                    data,
                    pool_size=(3, 3),
                    strides=(2, 2),
                    padding=1,
                    count_include_pad=count_include_pad,
                )
                R.output(out)
            return out

    data = np.random.randn(2, 8, 16, 16).astype("float32")
    _offload_and_compare(AvgPool, {}, data)


def test_dnnl_avg_pool2d_pad_fusion():
    # rewrite_pad_avg_pool2d folds a preceding zero-pad into avg_pool2d's own padding attribute;
    # check the fused form is numerically identical to the unfused one.
    @tvm.script.ir_module
    class PadThenPool:
        @R.function
        def main(data: R.Tensor((1, 4, 14, 14), "float32")):
            with R.dataflow():
                padded = relax.op.nn.pad(data, [0, 0, 0, 0, 1, 1, 1, 1])
                out = relax.op.nn.avg_pool2d(padded, pool_size=(3, 3), strides=(2, 2))
                R.output(out)
            return out

    data = np.random.randn(1, 4, 14, 14).astype("float32")
    _offload_and_compare(PadThenPool, {}, data)


def test_dnnl_resnet_basic_block():
    # A minimal ResNet-style basic block: conv+bias+relu, then conv+bias with an identity
    # shortcut added back in before the final relu. Ideally this offloads as two DNNL regions --
    # dnnl.conv2d_bias_relu for the first layer and dnnl.conv2d_bias_sum_relu for the second,
    # with the residual add fused into the second convolution's post-ops. We only assert that
    # *something* got offloaded and that the end-to-end numbers match here; see
    # test_dnnl_conv2d_bias_sum_relu_partition in test_backend_dnnl.py for the stricter check on
    # the exact composite that forms.
    @tvm.script.ir_module
    class Conv2dResNetBlock:
        @R.function
        def main(
            data: R.Tensor((1, 16, 32, 32), "float32"),
            weight1: R.Tensor((16, 16, 3, 3), "float32"),
            bias1: R.Tensor((16, 1, 1), "float32"),
            weight2: R.Tensor((16, 16, 3, 3), "float32"),
            bias2: R.Tensor((16, 1, 1), "float32"),
        ):
            with R.dataflow():
                conv1 = relax.op.nn.conv2d(data, weight1, padding=1)
                conv1 = relax.op.add(conv1, bias1)
                conv1 = relax.op.nn.relu(conv1)
                conv2 = relax.op.nn.conv2d(conv1, weight2, padding=1)
                conv2 = relax.op.add(conv2, bias2)
                summed = relax.op.add(conv2, data)
                out = relax.op.nn.relu(summed)
                R.output(out)
            return out

    data_np = np.random.randn(1, 16, 32, 32).astype("float32")
    weight1_np = np.random.randn(16, 16, 3, 3).astype("float32")
    bias1_np = np.random.randn(16, 1, 1).astype("float32")
    weight2_np = np.random.randn(16, 16, 3, 3).astype("float32")
    bias2_np = np.random.randn(16, 1, 1).astype("float32")
    inputs = [data_np, weight1_np, bias1_np, weight2_np, bias2_np]
    ref = build_and_run(Conv2dResNetBlock, inputs, "llvm", legalize=True)

    params_np = {
        "weight1": inputs[1],
        "bias1": inputs[2],
        "weight2": inputs[3],
        "bias2": inputs[4],
    }

    mod = relax.transform.BindParams("main", params_np)(Conv2dResNetBlock)
    partitioned = partition_for_dnnl(mod)
    assert any(
        isinstance(fn, relax.Function)
        and fn.attrs is not None
        and fn.attrs.get("Codegen") == "dnnl"
        for fn in partitioned.functions.values()
    ), "expected at least one layer to be offloaded to DNNL, but nothing was partitioned"

    offloaded = relax.transform.RunCodegen()(partitioned)
    out = build_and_run(offloaded, inputs[:1], "llvm", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_partition_for_dnnl():
    # End-to-end test of the partition_for_dnnl entry point: it should offload the conv2d -> relu
    # subgraph to DNNL with a single call.
    @tvm.script.ir_module
    class Model:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"), weight: R.Tensor((16, 8, 3, 3), "float32")
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                out = relax.op.nn.relu(conv)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 16, 16).astype("float32")
    weight = np.random.randn(16, 8, 3, 3).astype("float32")
    ref = build_and_run(Model, [data, weight], "llvm", legalize=True)

    mod = relax.transform.BindParams("main", {"weight": weight})(Model)
    mod = partition_for_dnnl(mod)
    assert any(
        isinstance(fn, relax.Function)
        and fn.attrs is not None
        and fn.attrs.get("Codegen") == "dnnl"
        for fn in mod.functions.values()
    ), "expected partition_for_dnnl to offload a subgraph to DNNL"

    mod = relax.transform.RunCodegen()(mod)
    out = build_and_run(mod, [data], "llvm", legalize=True)
    tvm.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
