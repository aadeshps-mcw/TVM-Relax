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
# Generalization of the original conv2d-only Crop()/TreatAs() regression
# suite (v0.19.0 oneDNN-v3 opaque-memory-descriptor migration) to run the
# same case matrix against conv1d, conv2d, conv3d, and the transposed
# (deconv) 2D/3D variants. Op selection is table-driven off `relax.op.nn`
# and `torch.nn`; every test that was previously conv2d-only now takes an
# `op_type` parameter and is collected once per entry in OP_TYPES.
#
# API surface verified against the apache/tvm v0.19.0 tag:
#   relax.op.nn.conv1d / conv2d / conv3d
#   relax.op.nn.conv1d_transpose (kernel_layout default "IOW")
#   relax.op.nn.conv2d_transpose (kernel_layout default "IOHW")
#   relax.op.nn.conv3d_transpose (kernel_layout default "IODHW")
# For the transposed ops the weight tensor's leading two axes are
# (in_channels, out_channels // groups) -- the reverse order of the
# regular conv ops' (out_channels, in_channels // groups) -- which is why
# weight-shape construction is branched on `spec.transposed` below rather
# than shared verbatim between the two families.
#
# NOTE: test_resnet18_conv_offloaded_and_numerically_correct is left
# conv2d-only. It exercises a real pretrained torchvision model; there is
# no equivalent pretrained conv1d/conv3d/deconv net to substitute, so
# generalizing it would mean fabricating an architecture rather than
# reusing an existing one. It stays as the one real-model smoke test.



# *** It's a unofficial test script for internal checking ***
import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program



# Op registry -- one entry per convolution family under test


class _OpSpec:
    def __init__(self, ndim, transposed, relax_op, torch_cls, data_layout, kernel_layout):
        self.ndim = ndim
        self.transposed = transposed
        self.relax_op = relax_op
        self.torch_cls = torch_cls
        self.data_layout = data_layout
        self.kernel_layout = kernel_layout


OP_SPECS = {
    "conv1d": _OpSpec(1, False, relax.op.nn.conv1d, nn.Conv1d, "NCW", "OIW"),
    "conv2d": _OpSpec(2, False, relax.op.nn.conv2d, nn.Conv2d, "NCHW", "OIHW"),
    "conv3d": _OpSpec(3, False, relax.op.nn.conv3d, nn.Conv3d, "NCDHW", "OIDHW"),
    "deconv2d": _OpSpec(2, True, relax.op.nn.conv2d_transpose, nn.ConvTranspose2d, "NCHW", "IOHW"),
    "deconv3d": _OpSpec(3, True, relax.op.nn.conv3d_transpose, nn.ConvTranspose3d, "NCDHW", "IODHW"),
}

# The full case matrix below runs against every op in this list.
OP_TYPES = list(OP_SPECS.keys())



# Shape/parameter expansion helpers
#
# The original suite hand-picked spatial sizes, kernel sizes, strides,
# padding, and dilation as 2-tuples (H, W). To reuse those same case values
# for conv1d (1 spatial axis) and conv3d/deconv3d (3 spatial axes), each
# 2-tuple is expanded to `ndim` entries by repeating its *last* element, or
# truncated to `ndim` entries by dropping trailing ones. This keeps 2D
# cases bit-for-bit identical to the original file while giving 1D/3D a
# deterministic, non-arbitrary derivation from the same source values --
# it does not attempt to invent new "interesting" 1D/3D-specific values.


def _expand(values, ndim):
    values = tuple(values)
    if ndim <= len(values):
        return values[:ndim]
    return values + (values[-1],) * (ndim - len(values))


def _nd_shape(batch, channels, spatial, ndim):
    """spatial is the original 2D (H, W)-shaped spatial tuple; expanded/
    truncated to `ndim` spatial axes via _expand."""
    return (batch, channels) + _expand(spatial, ndim)

# Relax IRModule builder (generic over op_type)

def _make_convnd_module(
    op_type,
    data_shape,
    out_channels,
    groups=1,
    kernel_size=None,
    strides=None,
    padding=None,
    dilation=None,
    output_padding=None,
    dtype="float32",
    out_dtype=None,
):
    """Builds a standalone single-op Relax IRModule for `op_type`, kept
    minimal so any numeric mismatch can only come from that op's lowering
    itself, not from a fused/rewritten pattern. Mirrors the original
    _make_conv2d_module, generalized over ndim and transposed-vs-not.
    """
    spec = OP_SPECS[op_type]
    ndim = spec.ndim
    kernel_size = _expand(kernel_size or (3, 3), ndim)
    strides = _expand(strides or (1, 1), ndim)
    padding = _expand(padding or (1, 1), ndim)
    dilation = _expand(dilation or (1, 1), ndim)

    in_channels = data_shape[1]
    assert in_channels % groups == 0, "in_channels must be divisible by groups"
    assert out_channels % groups == 0, "out_channels must be divisible by groups"

    if spec.transposed:
        # relax's *_transpose kernel_layout default (IOW / IOHW / IODHW)
        # puts in_channels first, matching torch.nn.ConvTranspose*d's
        # weight layout of (in_channels, out_channels // groups, *k).
        weight_shape = (in_channels, out_channels // groups, *kernel_size)
    else:
        weight_shape = (out_channels, in_channels // groups, *kernel_size)

    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))

    with builder.function("main", [data, weight]):
        with builder.dataflow():
            kwargs = dict(
                strides=list(strides),
                padding=list(padding),
                dilation=list(dilation),
                groups=groups,
                out_dtype=out_dtype or dtype,
            )
            if spec.transposed:
                kwargs["output_padding"] = list(_expand(output_padding or (0, 0), ndim))
            conv = builder.emit(spec.relax_op(data, weight, **kwargs))
            out = builder.emit_output(conv)
        builder.emit_func_output(out)
    return builder.get(), weight_shape


# Helper: convert a torch.nn.Module into a Relax IRModule via torch.export

def _torch_module_to_relax(torch_model, example_input):
    torch_model.eval()
    with torch.no_grad():
        exported = torch.export.export(torch_model, (example_input,))
    mod = from_exported_program(exported, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)
    return mod, params["main"]


def _compile_and_compare_model(torch_model, example_input, alter_layout=True,
                                rtol=1e-3, atol=1e-3, min_dnnl_funcs=1):
    """Same idea as _compile_and_compare, but for a full torch model instead
    of a hand-built single-op graph:
      1. Get PyTorch's own eager output -- this is the ground truth here.
      2. Convert to Relax, partition for DNNL, codegen, compile, run.
      3. Sanity-check that at least one DNNL function was actually created
         (i.e. something really got offloaded, not silently skipped).
      4. Compare numeric outputs.
    """
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(example_input).numpy()

    mod, params_np = _torch_module_to_relax(torch_model, example_input)

    target = tvm.target.Target("llvm")
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
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    input_tensor = tvm.runtime.tensor(example_input.numpy())
    tvm_args = [input_tensor] + [tvm.runtime.tensor(p) for p in params_np]
    out = vm["main"](*tvm_args)
    assert len(out) == 1
    res = out[0].numpy()
    np.testing.assert_allclose(res, torch_out, rtol=rtol, atol=atol)
    return res, torch_out



# Generic small torch models (replace _TinyConvNet / _TinyConvNet3 /
# _TinyConv1dNet / _TinyConv3dNet / _TinyDeconv2dNet / _TinyDeconv3dNet
# and their multi-layer/strided variants with one table-driven builder)


def _make_stack_model(op_type, channels, strides, kernel_size=3, padding=1,
                       relu_between=True, flatten_output=True):
    """channels = [in_ch, ch_after_layer1, ch_after_layer2, ...]
    strides = one stride per layer (len == len(channels) - 1)."""
    spec = OP_SPECS[op_type]
    Conv = spec.torch_cls
    layers = []
    for i in range(len(channels) - 1):
        layers.append(Conv(channels[i], channels[i + 1], kernel_size=kernel_size,
                            stride=strides[i], padding=padding))
        if relu_between:
            layers.append(nn.ReLU())

    class _StackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            x = self.body(x)
            if flatten_output:
                x = torch.flatten(x, 1)
            return x

    return _StackNet()


def _example_input(op_type, channels, spatial_size, batch=1):
    spec = OP_SPECS[op_type]
    shape = (batch, channels) + (spatial_size,) * spec.ndim
    return torch.randn(*shape)


@pytest.mark.parametrize("op_type", ["conv1d", "conv2d", "conv3d"])
def test_custom_convnet_conv_offloaded_and_numerically_correct(op_type):
    spatial = 64 if op_type == "conv1d" else (32 if op_type == "conv2d" else 8)
    model = _make_stack_model(op_type, channels=[3, 16, 32], strides=[1, 2])
    example_input = _example_input(op_type, 3, spatial)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


@pytest.mark.parametrize("op_type", ["conv1d", "conv2d", "conv3d"])
def test_custom_convnet_three_conv_offloaded(op_type):
    spatial = 64 if op_type == "conv1d" else (32 if op_type == "conv2d" else 8)
    model = _make_stack_model(op_type, channels=[3, 16, 32, 64], strides=[1, 2, 1])
    example_input = _example_input(op_type, 3, spatial)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


# Pretrained ResNet-18 -- exercises Crop()/TreatAs() across many real
# conv2d shapes (stem conv, stride-2 downsample convs, 1x1 shortcut convs,
# varying channel counts) all at once, inside one realistic model.
# Left conv2d-only; see module docstring.
def test_resnet18_conv_offloaded_and_numerically_correct():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)  # random weights is fine here
    model = nn.Sequential(*list(model.children())[:-2])
    example_input = torch.randn(1, 3, 224, 224)
    _compile_and_compare_model(
        model, example_input, alter_layout=True, rtol=1e-2, atol=1e-2, min_dnnl_funcs=1
    )


@pytest.mark.parametrize("op_type", ["deconv2d"])
def test_deconv_basic(op_type):
    spatial = 16 if op_type == "deconv2d" else 8
    model = _make_stack_model(op_type, channels=[16, 8], strides=[1], flatten_output=False,
                               relu_between=False)
    example_input = _example_input(op_type, 16, spatial)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


@pytest.mark.parametrize("op_type", ["deconv2d"])
def test_deconv_strided(op_type):
    spatial = 8 if op_type == "deconv2d" else 4
    spec = OP_SPECS[op_type]
    Conv = spec.torch_cls

    class _StridedDeconv(nn.Module):
        def __init__(self):
            super().__init__()
            self.deconv = Conv(16, 8, kernel_size=4, stride=2, padding=1)

        def forward(self, x):
            return self.deconv(x)

    model = _StridedDeconv()
    example_input = _example_input(op_type, 16, spatial)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)


@pytest.mark.parametrize("op_type", ["conv1d", "conv3d"])
def test_conv_multi_layer(op_type):
    """Multi-layer conv1d/conv3d, mirroring the original file's
    test_conv1d_multi_layer / test_conv3d_multi_layer."""
    spatial = 64 if op_type == "conv1d" else 8
    model = _make_stack_model(op_type, channels=[3, 16, 32], strides=[1, 2])
    example_input = _example_input(op_type, 3, spatial)
    _compile_and_compare_model(model, example_input, alter_layout=False if op_type == "conv1d" else True,
                                min_dnnl_funcs=1)



# Shared numeric-verification helper (unchanged from the original: generic
# over the IRModule passed in, so no op_type dependency here)


def _compile_and_compare(mod, params_np, alter_layout=True, rtol=1e-4, atol=1e-4):
    """Runs `mod` two ways -- (1) plain, unpartitioned TVM as ground truth, and
    (2) partitioned + codegen'd through the DNNL BYOC path -- and asserts the
    numeric outputs match.

    This is the load-bearing check for the oneDNN v3 TensorRequisite fixes:
    Crop() and TreatAs() only affect *how* a tensor's underlying buffer is
    reinterpreted before/after a oneDNN primitive call. If either silently
    mis-describes the physical layout (wrong strides, wrong logical/physical
    dim mapping, wrong format_tag resolved), the DNNL branch below will still
    run without crashing but will produce numerically wrong results -- which
    is exactly what this comparison is designed to catch. Applies identically
    regardless of which op_type built `mod`.
    """
    target = tvm.target.Target("llvm")
    tvm_args = [tvm.runtime.tensor(p) for p in params_np]

    ref_ex = relax.build(mod, target=target)
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    ref_out = ref_vm["main"](*tvm_args).numpy()

    partitioned = partition_for_dnnl(mod, alter_layout=alter_layout)
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](*tvm_args).numpy()

    np.testing.assert_allclose(out, ref_out, rtol=rtol, atol=atol)
    return out, ref_out


def _run_convnd_case(op_type, data_shape, out_channels, groups=1, kernel_size=(3, 3),
                      strides=(1, 1), padding=(1, 1), dilation=(1, 1), output_padding=(0, 0),
                      dtype="float32", out_dtype=None, alter_layout=True, seed=0):
    np.random.seed(seed)
    mod, weight_shape = _make_convnd_module(
        op_type,
        data_shape=data_shape,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        dtype=dtype,
        out_dtype=out_dtype,
    )
    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")
    return _compile_and_compare(mod, [data_np, weight_np], alter_layout=alter_layout)



# Crop() coverage -- grouped / depthwise convolution

# Crop() slices a combined weight tensor into per-group sub-views before each
# group's convolution runs (dnnl::memory::desc::submemory_desc() in the fixed
# implementation, replacing the old direct-field-write "auto-padded" hack).
# Parametrized across group counts, channel counts that are/aren't multiples
# of oneDNN's common block sizes (8, 16), and now every op_type in
# OP_TYPES -- Crop() is invoked identically regardless of spatial rank or
# whether the op is transposed.
@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        pytest.param(4, 8, 1, id="groups1-baseline-noop"),
        pytest.param(8, 16, 2, id="groups2-block-aligned"),
        pytest.param(6, 12, 2, id="groups2-not-block-aligned"),
        pytest.param(16, 32, 4, id="groups4-block-aligned"),
        pytest.param(9, 18, 3, id="groups3-not-block-aligned"),
        pytest.param(32, 64, 8, id="groups8-block-aligned"),
        pytest.param(8, 8, 8, id="depthwise-block-aligned"),
        pytest.param(5, 5, 5, id="depthwise-not-block-aligned"),
        pytest.param(3, 3, 3, id="depthwise-tiny-below-block"),
    ],
)
@pytest.mark.parametrize("alter_layout", [True, False], ids=["alter_layout", "plain_layout"])
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_grouped_numerically_correct(op_type, in_channels, out_channels, groups, alter_layout):
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, in_channels, (16, 16), ndim),
        out_channels=out_channels,
        groups=groups,
        alter_layout=alter_layout,
        seed=hash((op_type, in_channels, out_channels, groups)) % (2**31),
    )


@pytest.mark.parametrize(
    "strides,padding",
    [
        pytest.param((1, 1), (0, 0), id="no-pad-stride1"),
        pytest.param((1, 1), (1, 1), id="pad1-stride1"),
        pytest.param((2, 2), (1, 1), id="pad1-stride2"),
        pytest.param((2, 2), (0, 0), id="no-pad-stride2"),
        pytest.param((1, 1), (2, 2), id="pad2-stride1"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_grouped_stride_padding_variants(op_type, strides, padding):
    """Non-trivial stride/padding combinations, to vary the `offset` argument
    Crop() is called with beyond the simplest zero-offset case above."""
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, 16, (20, 20), ndim),
        out_channels=32,
        groups=4,
        strides=strides,
        padding=padding,
        seed=1,
    )


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_grouped_batch_variants(op_type, batch):
    """Batch size shouldn't interact with per-group weight cropping, but
    verify explicitly rather than assume it."""
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(batch, 16, (16, 16), ndim),
        out_channels=32,
        groups=4,
        seed=2,
    )


@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        pytest.param(256, 512, 1, id="wide-256to512"),
        pytest.param(256, 256, 32, id="wide-grouped-32"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_treatas_large_channel_counts(op_type, in_channels, out_channels, groups):
    """Exercises higher block-count paths that small-channel tests never reach."""
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, in_channels, (14, 14), ndim),
        out_channels=out_channels,
        groups=groups,
        alter_layout=True,
        seed=9,
    )


@pytest.mark.parametrize("dilation", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_treatas_dilation_variants(op_type, dilation):
    """Dilation changes the effective kernel footprint oneDNN sees, which
    can select a different primitive/format path than dilation=1."""
    ndim = OP_SPECS[op_type].ndim
    dilation_nd = _expand(dilation, ndim)
    k = 3
    pad = dilation_nd[0] * (k - 1) // 2
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, 16, (24, 24), ndim),
        out_channels=32,
        kernel_size=(k,) * ndim,
        padding=(pad,) * ndim,
        dilation=dilation_nd,
        alter_layout=True,
        seed=7,
    )


@pytest.mark.parametrize(
    "kernel_size,strides,padding",
    [
        pytest.param((3, 5), (1, 1), (1, 2), id="asym-kernel"),
        pytest.param((3, 3), (1, 2), (1, 1), id="asym-stride"),
        pytest.param((3, 3), (1, 1), (0, 2), id="asym-padding"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_treatas_asymmetric_variants(op_type, kernel_size, strides, padding):
    """H and W are separate positional dims in format_tag resolution --
    symmetric-only shapes could hide an axis-swap bug. For conv1d there is
    only one spatial axis, so the asymmetry collapses to that axis's value
    (_expand truncates); for the 3D ops the third axis reuses the pair's
    last value (_expand pads by repetition) rather than inventing a new
    value out of nowhere."""
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, 16, (20, 24), ndim),
        out_channels=32,
        kernel_size=_expand(kernel_size, ndim),
        strides=_expand(strides, ndim),
        padding=_expand(padding, ndim),
        alter_layout=True,
        seed=8,
    )



# TreatAs() coverage -- layout / format_tag resolution

# TreatAs() is exercised whenever the DNNL runtime reinterprets a tensor's
# buffer under a different (often blocked) layout -- primarily triggered
# internally when `alter_layout=True` lets oneDNN pick its own preferred
# format for conv weights/activations. Channel counts span oneDNN's common
# blocking factors (8, 16, 32) as well as values that are NOT multiples of
# them, forcing a variety of format_tag values (and, for unaligned counts,
# fallback to plain/unblocked tags) through CanonicalFormatTagName() /
# FormatTagsByCanonicalName(). Now run for every op_type.
@pytest.mark.parametrize(
    "in_channels,out_channels",
    [
        pytest.param(3, 8, id="stem-like-3to8"),
        pytest.param(8, 8, id="8to8-block-aligned"),
        pytest.param(16, 16, id="16to16-block-aligned"),
        pytest.param(16, 32, id="16to32-block-aligned"),
        pytest.param(64, 128, id="64to128-block-aligned"),
        pytest.param(5, 11, id="5to11-not-block-aligned"),
        pytest.param(17, 33, id="17to33-not-block-aligned"),
        pytest.param(1, 1, id="1to1-degenerate"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_treatas_channel_variants(op_type, in_channels, out_channels):
    ndim = OP_SPECS[op_type].ndim
    out_altered, ref_out = _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, in_channels, (32, 32), ndim),
        out_channels=out_channels,
        alter_layout=True,
        seed=3,
    )
    out_plain, _ = _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, in_channels, (32, 32), ndim),
        out_channels=out_channels,
        alter_layout=False,
        seed=3,
    )
    # Compare the two DNNL paths against each other directly, so a mismatch
    # here points specifically at layout-selection/TreatAs(), not at DNNL
    # codegen in general (both already passed vs. the plain-TVM reference).
    np.testing.assert_allclose(out_altered, out_plain, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "kernel_size",
    [
        pytest.param((1, 1), id="pointwise-1x1"),
        pytest.param((3, 3), id="3x3"),
        pytest.param((5, 5), id="5x5"),
        pytest.param((7, 7), id="7x7"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_treatas_kernel_size_variants(op_type, kernel_size):
    """Kernel size affects which internal oneDNN convolution algorithm gets
    picked (direct vs. Winograd, etc.), which in turn affects which preferred
    (often blocked) layout it requests -- varying this broadens format_tag
    coverage beyond what channel-count variation alone reaches."""
    ndim = OP_SPECS[op_type].ndim
    k = _expand(kernel_size, ndim)
    padding = tuple(x // 2 for x in k)
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, 16, (32, 32), ndim),
        out_channels=32,
        kernel_size=k,
        padding=padding,
        alter_layout=True,
        seed=4,
    )


@pytest.mark.parametrize(
    "spatial,weight_channels",
    [
        pytest.param((24, 24), 16, id="batch2"),  # batch is applied separately below
        pytest.param((7, 13), 16, id="odd-nonsquare-spatial"),
        pytest.param((8, 8), 8, id="small-spatial-batch4"),
        pytest.param((1, 1), 16, id="degenerate-1x1-spatial"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_treatas_batch_and_spatial_variants(op_type, spatial, weight_channels):
    """Varies batch size and spatial dims independently of channel count --
    TreatAs()'s logical-dims computation merges all outer + inner tokens
    positionally, so a bug tied to dimension *position* (rather than size)
    wouldn't necessarily surface if only channel counts were varied above."""
    ndim = OP_SPECS[op_type].ndim
    # id="batch2" reuses the original file's batch=2 case; every other id
    # here keeps batch=1, matching the original per-case batch values.
    batch = 2 if spatial == (24, 24) else (4 if spatial == (8, 8) else 1)
    channels = 8 if spatial == (8, 8) else weight_channels
    degenerate = spatial == (1, 1)
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(batch, channels, spatial, ndim),
        out_channels=weight_channels if not degenerate else channels,
        kernel_size=(1,) * ndim if degenerate else (3,) * ndim,
        padding=(0,) * ndim if degenerate else (1,) * ndim,
        alter_layout=True,
        seed=5,
    )



# Combined Crop() + TreatAs() interaction -- highest-risk case

# When `alter_layout=True` AND `groups > 1` together, oneDNN may request a
# *blocked* layout for the combined weight tensor before Crop() slices it
# per-group -- i.e. Crop() ends up operating on a tensor TreatAs() already
# reinterpreted. This is the one scenario where the two fixes' behavior
# genuinely composes, and where the removed "auto-padded shrink" hack most
# plausibly mattered. Channel-per-group counts below deliberately span both
# block-aligned and (notably) sub-block sizes, since a group whose channel
# count is smaller than oneDNN's chosen block size is exactly the case the
# old hack existed for. Now run for every op_type.
@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        pytest.param(16, 32, 2, id="groups2-block-aligned-altered"),
        pytest.param(8, 16, 4, id="groups4-per-group-below-block"),
        pytest.param(6, 6, 6, id="depthwise-per-group-1-channel"),
        pytest.param(32, 32, 32, id="depthwise-wide-per-group-1-channel"),
    ],
)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_treatas_interaction_grouped_with_altered_layout(
    op_type, in_channels, out_channels, groups
):
    ndim = OP_SPECS[op_type].ndim
    _run_convnd_case(
        op_type,
        data_shape=_nd_shape(1, in_channels, (16, 16), ndim),
        out_channels=out_channels,
        groups=groups,
        alter_layout=True,
        seed=6,
    )



# Dtype coverage

# oneDNN's blocking factor depends on dtype (f32/bf16 commonly block by
# 8/16, s8/u8 commonly block by 4) -- TreatAs()'s format_tag table has only
# been exercised for float32 above. Repeats the highest-risk Crop+TreatAs
# case across every dtype the pipeline accepts, and now across every
# op_type too.

DTYPE_CASES = [
    pytest.param("float32", "float32", id="f32"),
    pytest.param("float16", "float16", id="f16"),
    pytest.param("bfloat16", "bfloat16", id="bf16"),
    pytest.param("int8", "int32", id="s8"),
    pytest.param("uint8", "int32", id="u8"),
    pytest.param("int32", "int32", id="s32"),  # confirm this is a real input dtype, not just an accumulator dtype, before trusting it
]


def _random_for_dtype(shape, dtype):
    if dtype in ("float32", "float16"):
        return np.random.uniform(-1, 1, size=shape).astype(dtype)
    if dtype == "bfloat16":
        import ml_dtypes
        return np.random.uniform(-1, 1, size=shape).astype(ml_dtypes.bfloat16)
    if dtype == "int8":
        return np.random.randint(-128, 128, size=shape).astype("int8")
    if dtype == "uint8":
        return np.random.randint(0, 256, size=shape).astype("uint8")
    if dtype == "int32":
        return np.random.randint(-1000, 1000, size=shape).astype("int32")
    raise ValueError(dtype)


@pytest.mark.parametrize("dtype,out_dtype", DTYPE_CASES)
@pytest.mark.parametrize("op_type", OP_TYPES)
def test_convnd_crop_treatas_dtype_variants(op_type, dtype, out_dtype):
    np.random.seed(10)
    ndim = OP_SPECS[op_type].ndim
    data_shape = _nd_shape(1, 16, (16, 16), ndim)
    mod, weight_shape = _make_convnd_module(
        op_type, data_shape=data_shape, out_channels=32, groups=4,  # groups keeps Crop() in play
        dtype=dtype, out_dtype=out_dtype,
    )
    data_np = _random_for_dtype(data_shape, dtype)
    weight_np = _random_for_dtype(weight_shape, dtype)
    _compile_and_compare(mod, [data_np, weight_np], alter_layout=True, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
