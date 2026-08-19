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
# Regression tests for newly-added DNNL BYOC support of conv1d, conv2d,
# conv3d, conv2d_transpose and conv3d_transpose. Mirrors the structure of
# tests/python/relax/test_codegen_tensorrt.py: each op gets its own small,
# single-op IRModule so the converter/offload path for that op is exercised
# in isolation, and outputs are checked against a plain-TVM (LLVM) reference.
#
# In addition to the single-op cases, this file also runs a handful of full
# models (hand-built CNNs plus a pretrained-shape ResNet-18) end to end
# through torch.export -> Relax -> partition_for_dnnl, so the new ops are
# also exercised together, inside realistic graphs, and not just in
# isolation.
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
import torch.nn as nn  # noqa: E402

from tvm.relax.frontend.torch import from_exported_program  # noqa: E402


def _to_numpy(out):
    """Unwrap a VM call result into a numpy array.

    A fully DNNL-offloaded single-op graph returns a bare Tensor, but a graph with leftover
    non-offloaded ops (e.g. a full model with a residual permute/reshape around the DNNL
    subgraph) can come back wrapped in a Tuple/Array. Handle both.
    """
    if hasattr(out, "numpy"):
        return out.numpy()
    assert len(out) == 1, f"expected a single output, got {len(out)}"
    return _to_numpy(out[0])


def build_and_run(mod, inputs_np, legalize=False):
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    with tvm.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = tvm.compile(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm.runtime.tensor(inp, dev) for inp in inputs_np]
    return _to_numpy(f(*inputs))


def _offload_and_compare(mod, params_np, data_np, alter_layout=True, rtol=1e-4, atol=1e-4):
    """Offload a single-op module to DNNL and compare against the plain-LLVM reference.

    Each module here contains a single instance of the op under test, which both exercises the
    individual op's lowering in isolation and avoids ambiguity about which op produced a mismatch
    if one occurs.
    """
    ref = build_and_run(mod, [data_np, *params_np.values()], legalize=True)

    bound = relax.transform.BindParams("main", params_np)(mod)
    partitioned = partition_for_dnnl(bound, alter_layout=alter_layout)

    # Guard against a silent false pass: if nothing matched, the op under test never actually
    # reaches the DNNL codegen path and the comparison below would trivially succeed via the TVM
    # fallback without exercising anything.
    assert any(
        isinstance(fn, relax.Function)
        and fn.attrs is not None
        and fn.attrs.get("Codegen") == "dnnl"
        for fn in partitioned.functions.values()
    ), "expected the op under test to be offloaded to DNNL, but nothing was partitioned"

    with tvm.transform.PassContext(opt_level=3):
        offloaded = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    # legalize=True: any ops left outside the DNNL subgraph (e.g. a residual permute/reshape)
    # still need lowering to TIR before VM codegen; this is a no-op for graphs that end up
    # fully offloaded.
    out = build_and_run(offloaded, [data_np], legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# conv1d
# ---------------------------------------------------------------------------


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
    _offload_and_compare(Conv1d, {"weight": weight}, data, alter_layout=False)


def test_dnnl_conv1d_grouped():
    # groups > 1 is what exercises per-group weight cropping in the DNNL lowering.
    @tvm.script.ir_module
    class Conv1dGrouped:
        @R.function
        def main(data: R.Tensor((1, 16, 20), "float32"), weight: R.Tensor((32, 4, 3), "float32")):
            with R.dataflow():
                out = relax.op.nn.conv1d(data, weight, padding=1, groups=4)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 20).astype("float32")
    weight = np.random.randn(32, 4, 3).astype("float32")
    _offload_and_compare(Conv1dGrouped, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv1d_strided():
    @tvm.script.ir_module
    class Conv1dStrided:
        @R.function
        def main(data: R.Tensor((1, 8, 32), "float32"), weight: R.Tensor((16, 8, 3), "float32")):
            with R.dataflow():
                out = relax.op.nn.conv1d(data, weight, strides=2, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 8, 32).astype("float32")
    weight = np.random.randn(16, 8, 3).astype("float32")
    _offload_and_compare(Conv1dStrided, {"weight": weight}, data, alter_layout=True)


# ---------------------------------------------------------------------------
# conv2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alter_layout", [True, False], ids=["alter_layout", "plain_layout"])
def test_dnnl_conv2d(alter_layout):
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(
            data: R.Tensor((1, 16, 20, 20), "float32"), weight: R.Tensor((32, 16, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, strides=1, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 20, 20).astype("float32")
    weight = np.random.randn(32, 16, 3, 3).astype("float32")
    _offload_and_compare(Conv2d, {"weight": weight}, data, alter_layout=alter_layout)


@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        pytest.param(8, 16, 2, id="groups2"),
        pytest.param(16, 16, 16, id="depthwise"),
    ],
)
def test_dnnl_conv2d_grouped(in_channels, out_channels, groups):
    # Built with BlockBuilder rather than inline TVMScript: R.Tensor(...) shape annotations are
    # parsed by TVMScript's own evaluator and can't close over a plain Python variable like
    # `in_channels` coming from pytest.mark.parametrize (see test_tensorrt_layout_transform for
    # the same pattern, used there for a different reason -- an index_map lambda).
    weight_shape = (out_channels, in_channels // groups, 3, 3)

    bb = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType((1, in_channels, 16, 16), "float32"))
    weight = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
    with bb.function("main", [data, weight]):
        with bb.dataflow():
            out = bb.emit(relax.op.nn.conv2d(data, weight, padding=1, groups=groups))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    mod = bb.finalize()

    data_np = np.random.randn(1, in_channels, 16, 16).astype("float32")
    weight_np = np.random.randn(*weight_shape).astype("float32")
    _offload_and_compare(mod, {"weight": weight_np}, data_np, alter_layout=True)


def test_dnnl_conv2d_dilated():
    @tvm.script.ir_module
    class Conv2dDilated:
        @R.function
        def main(
            data: R.Tensor((1, 16, 24, 24), "float32"), weight: R.Tensor((32, 16, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=2, dilation=2)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 24, 24).astype("float32")
    weight = np.random.randn(32, 16, 3, 3).astype("float32")
    _offload_and_compare(Conv2dDilated, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv2d_relu_fused():
    # Sanity check that the op still offloads correctly as part of a larger dataflow block, not
    # only in a minimal single-op module.
    @tvm.script.ir_module
    class Conv2dRelu:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"), weight: R.Tensor((32, 16, 3, 3), "float32")
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                out = relax.op.nn.relu(conv)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 16, 16).astype("float32")
    weight = np.random.randn(32, 16, 3, 3).astype("float32")
    _offload_and_compare(Conv2dRelu, {"weight": weight}, data, alter_layout=True)


# ---------------------------------------------------------------------------
# conv3d
# ---------------------------------------------------------------------------


def test_dnnl_conv3d():
    @tvm.script.ir_module
    class Conv3d:
        @R.function
        def main(
            data: R.Tensor((1, 3, 8, 8, 8), "float32"),
            weight: R.Tensor((16, 3, 3, 3, 3), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 3, 8, 8, 8).astype("float32")
    weight = np.random.randn(16, 3, 3, 3, 3).astype("float32")
    _offload_and_compare(Conv3d, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv3d_grouped():
    @tvm.script.ir_module
    class Conv3dGrouped:
        @R.function
        def main(
            data: R.Tensor((1, 16, 8, 8, 8), "float32"),
            weight: R.Tensor((32, 8, 3, 3, 3), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d(data, weight, padding=1, groups=2)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 8, 8, 8).astype("float32")
    weight = np.random.randn(32, 8, 3, 3, 3).astype("float32")
    _offload_and_compare(Conv3dGrouped, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv3d_strided():
    @tvm.script.ir_module
    class Conv3dStrided:
        @R.function
        def main(
            data: R.Tensor((1, 3, 16, 16, 16), "float32"),
            weight: R.Tensor((8, 3, 3, 3, 3), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d(data, weight, strides=2, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 3, 16, 16, 16).astype("float32")
    weight = np.random.randn(8, 3, 3, 3, 3).astype("float32")
    _offload_and_compare(Conv3dStrided, {"weight": weight}, data, alter_layout=True)


# ---------------------------------------------------------------------------
# conv2d_transpose
# ---------------------------------------------------------------------------


def test_dnnl_conv2d_transpose():
    # Default IOHW kernel layout ([in, out, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTranspose2d:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"), weight: R.Tensor((16, 8, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 16, 16).astype("float32")
    weight = np.random.randn(16, 8, 3, 3).astype("float32")
    _offload_and_compare(ConvTranspose2d, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv2d_transpose_strided():
    @tvm.script.ir_module
    class ConvTranspose2dStrided:
        @R.function
        def main(
            data: R.Tensor((1, 16, 8, 8), "float32"), weight: R.Tensor((16, 8, 4, 4), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d_transpose(data, weight, strides=2, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 8, 8).astype("float32")
    weight = np.random.randn(16, 8, 4, 4).astype("float32")
    _offload_and_compare(ConvTranspose2dStrided, {"weight": weight}, data, alter_layout=True)


# ---------------------------------------------------------------------------
# conv3d_transpose
# ---------------------------------------------------------------------------


def test_dnnl_conv3d_transpose():
    # Default IODHW kernel layout ([in, out, d, h, w]); output channels are weight_shape[1].
    @tvm.script.ir_module
    class ConvTranspose3d:
        @R.function
        def main(
            data: R.Tensor((1, 16, 8, 8, 8), "float32"),
            weight: R.Tensor((16, 8, 3, 3, 3), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 8, 8, 8).astype("float32")
    weight = np.random.randn(16, 8, 3, 3, 3).astype("float32")
    _offload_and_compare(ConvTranspose3d, {"weight": weight}, data, alter_layout=True)


def test_dnnl_conv3d_transpose_strided():
    @tvm.script.ir_module
    class ConvTranspose3dStrided:
        @R.function
        def main(
            data: R.Tensor((1, 16, 4, 4, 4), "float32"),
            weight: R.Tensor((16, 8, 4, 4, 4), "float32"),
        ):
            with R.dataflow():
                out = relax.op.nn.conv3d_transpose(data, weight, strides=2, padding=1)
                R.output(out)
            return out

    data = np.random.randn(1, 16, 4, 4, 4).astype("float32")
    weight = np.random.randn(16, 8, 4, 4, 4).astype("float32")
    _offload_and_compare(ConvTranspose3dStrided, {"weight": weight}, data, alter_layout=True)


# ---------------------------------------------------------------------------
# End-to-end entry point test
# ---------------------------------------------------------------------------


def test_partition_for_dnnl_conv2d_stack():
    # End-to-end test of the partition_for_dnnl entry point on a small multi-conv2d stack: it
    # should offload the conv2d -> relu subgraphs to DNNL with a single call, exercising the
    # new op support the same way a real model would use it (rather than one op in isolation).
    @tvm.script.ir_module
    class ConvStack:
        @R.function
        def main(
            data: R.Tensor((1, 3, 32, 32), "float32"),
            weight1: R.Tensor((16, 3, 3, 3), "float32"),
            weight2: R.Tensor((32, 16, 3, 3), "float32"),
        ):
            with R.dataflow():
                conv1 = relax.op.nn.relu(relax.op.nn.conv2d(data, weight1, padding=1))
                conv2 = relax.op.nn.relu(relax.op.nn.conv2d(conv1, weight2, strides=2, padding=1))
                R.output(conv2)
            return conv2

    data_np = np.random.randn(1, 3, 32, 32).astype("float32")
    weight1_np = np.random.randn(16, 3, 3, 3).astype("float32")
    weight2_np = np.random.randn(32, 16, 3, 3).astype("float32")
    inputs = [data_np, weight1_np, weight2_np]
    ref = build_and_run(ConvStack, inputs, legalize=True)

    partitioned = partition_for_dnnl(ConvStack, alter_layout=True)
    assert any(
        isinstance(fn, relax.Function)
        and fn.attrs is not None
        and fn.attrs.get("Codegen") == "dnnl"
        for fn in partitioned.functions.values()
    ), "expected partition_for_dnnl to offload at least one subgraph to DNNL"

    with tvm.transform.PassContext(opt_level=3):
        offloaded = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    out = build_and_run(offloaded, inputs, legalize=True)

    tvm.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Full-model tests
# ---------------------------------------------------------------------------
# The single-op tests above isolate one converter at a time; the tests below
# instead build small torch.nn.Module models (plus a ResNet-18), export them
# through torch.export, convert to Relax, and offload the whole graph to
# DNNL -- checking against PyTorch's own eager output as ground truth. This
# is what actually exercises the new ops the way a real model would use
# them: chained with other ops, sharing buffers/layouts across layers, and
# (for the encoder/decoder models) mixing a conv with its transposed
# counterpart in the same graph.


def _torch_module_to_relax(torch_model, example_input):
    torch_model.eval()
    with torch.no_grad():
        exported = torch.export.export(torch_model, (example_input,))
    mod = from_exported_program(exported, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    return mod, params["main"]


def _compile_and_compare_model(
    torch_model, example_input, alter_layout=True, rtol=1e-3, atol=1e-3, min_dnnl_funcs=1
):
    """Same idea as _offload_and_compare, but for a full torch model instead of a hand-built
    single-op graph:
      1. Get PyTorch's own eager output -- this is the ground truth here.
      2. Convert to Relax, partition for DNNL, codegen, compile, run.
      3. Sanity-check that at least one DNNL function was actually created (i.e. something
         really got offloaded, not silently skipped).
      4. Compare numeric outputs.
    """
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(example_input).numpy()

    mod, params_np = _torch_module_to_relax(torch_model, example_input)

    partitioned = partition_for_dnnl(mod, alter_layout=alter_layout)
    dnnl_funcs = [
        gv.name_hint
        for gv, func in partitioned.functions.items()
        if func.attrs is not None and func.attrs.get("Codegen") == "dnnl"
    ]
    assert len(dnnl_funcs) >= min_dnnl_funcs, (
        f"expected at least {min_dnnl_funcs} DNNL-offloaded function(s), got {dnnl_funcs}"
    )

    with tvm.transform.PassContext(opt_level=3):
        offloaded = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)

    # legalize=True: torch-imported graphs commonly carry ordinary Relax ops (permute_dims for
    # weight-layout fixups, residual adds, etc.) alongside the DNNL subgraph(s); those still need
    # lowering to TIR before VM codegen.
    out = build_and_run(offloaded, [example_input.numpy(), *params_np], legalize=True)
    tvm.testing.assert_allclose(out, torch_out, rtol=rtol, atol=atol)
    return out, torch_out


class _TinyConvNet(nn.Module):
    # Two conv2d layers + a non-conv op (flatten), so there's something interesting for
    # partitioning to segregate.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return torch.flatten(x, 1)


class _TinyConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return torch.flatten(x, 1)


class _TinyConv1dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


class _TinyConv3dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)


class _TinyEncoderDecoder2d(nn.Module):
    # conv2d downsample followed by conv2d_transpose upsample, so both a conv and its transposed
    # counterpart appear in the same partitioned graph.
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.up = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.down(x))
        return self.up(x)


def test_dnnl_custom_convnet_offloaded_and_numerically_correct():
    model = _TinyConvNet()
    example_input = torch.randn(1, 3, 32, 32)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


def test_dnnl_custom_convnet3_offloaded_and_numerically_correct():
    model = _TinyConvNet3()
    example_input = torch.randn(1, 3, 32, 32)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


def test_dnnl_conv1d_model_offloaded_and_numerically_correct():
    model = _TinyConv1dNet()
    example_input = torch.randn(1, 3, 64)
    _compile_and_compare_model(model, example_input, alter_layout=False, min_dnnl_funcs=1)


def test_dnnl_conv3d_model_offloaded_and_numerically_correct():
    model = _TinyConv3dNet()
    example_input = torch.randn(1, 3, 8, 8, 8)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


def test_dnnl_conv2d_transpose_model_offloaded_and_numerically_correct():
    model = _TinyEncoderDecoder2d()
    example_input = torch.randn(1, 3, 16, 16)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


def test_dnnl_conv3d_transpose_model_offloaded_and_numerically_correct():
    # NOTE: unlike the other "_model" tests in this section, this one is hand-built via
    # TVMScript rather than going through torch.export -- the TVM torch frontend does not yet
    # support importing 5D (Conv3d) transposed convolution. It still exercises the property this
    # test section cares about: a conv3d downsample feeding a conv3d_transpose upsample in a
    # single graph, offloaded to DNNL together.
    @tvm.script.ir_module
    class EncoderDecoder3d:
        @R.function
        def main(
            data: R.Tensor((1, 3, 8, 8, 8), "float32"),
            down_weight: R.Tensor((8, 3, 3, 3, 3), "float32"),
            up_weight: R.Tensor((8, 3, 4, 4, 4), "float32"),
        ):
            with R.dataflow():
                down = relax.op.nn.relu(relax.op.nn.conv3d(data, down_weight, strides=2, padding=1))
                up = relax.op.nn.conv3d_transpose(down, up_weight, strides=2, padding=1)
                R.output(up)
            return up

    data_np = np.random.randn(1, 3, 8, 8, 8).astype("float32")
    down_weight_np = np.random.randn(8, 3, 3, 3, 3).astype("float32")
    up_weight_np = np.random.randn(8, 3, 4, 4, 4).astype("float32")
    _offload_and_compare(
        EncoderDecoder3d,
        {"down_weight": down_weight_np, "up_weight": up_weight_np},
        data_np,
        alter_layout=True,
    )


def test_dnnl_resnet18_offloaded_and_numerically_correct():
    # Pretrained-shape ResNet-18 (random weights are fine here -- only the graph shape and op mix
    # matter) exercises conv2d across many real shapes at once inside one realistic model: the
    # stem conv, stride-2 downsample convs, 1x1 shortcut convs, and varying channel counts.
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)
    model = nn.Sequential(*list(model.children())[:-2])
    example_input = torch.randn(1, 3, 224, 224)
    _compile_and_compare_model(
        model, example_input, alter_layout=True, rtol=1e-2, atol=1e-2, min_dnnl_funcs=1
    )


if __name__ == "__main__":
    tvm.testing.main()
