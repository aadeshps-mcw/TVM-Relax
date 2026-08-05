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
# NOTE: This file intentionally covers *only* Conv2D. Its goal is narrow:
# verify that the oneDNN v3 opaque-memory-descriptor migration of
# TensorRequisite::Crop() and TensorRequisite::TreatAs() (formerly hand-built
# via direct dnnl_memory_desc_t field writes, now built via
# submemory_desc()/dnnl::memory::desc(dims, dtype, format_tag) respectively)
# did not introduce numerical regressions. It is not a partitioning/pattern-
# matching test suite -- see the main DNNL BYOC test file for that.

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl



# Module builder

def _make_conv2d_module(
    data_shape,
    out_channels,
    groups=1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    dtype="float32",
    out_dtype=None,
):
    """Builds a standalone Conv2D-only Relax IRModule (no ReLU/bias/etc. --
    kept minimal so any numeric mismatch can only come from the conv2d
    lowering itself, not from a fused/rewritten pattern).

    `groups` controls how many independent convolution groups the weight
    tensor is split across -- groups > 1 is what exercises Crop() in this
    codebase (per-group weight slicing); `groups == in_channels` gives a
    depthwise convolution.
    """
    in_channels = data_shape[1]
    assert in_channels % groups == 0, "in_channels must be divisible by groups"
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    weight_shape = (out_channels, in_channels // groups, *kernel_size)

    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, dtype))
    weight = relax.Var("weight", relax.TensorType(weight_shape, dtype))

    with builder.function("main", [data, weight]):
        with builder.dataflow():
            conv = builder.emit(
                relax.op.nn.conv2d(
                    data, weight, strides=list(strides), padding=list(padding), dilation=list(dilation), groups=groups, out_dtype=out_dtype or dtype,
                )
            )
            out = builder.emit_output(conv)
        builder.emit_func_output(out)
    return builder.get(), weight_shape
import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program



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
    np.testing.assert_allclose(res, torch_out, rtol=rtol, atol = atol)

    # np.testing.assert_allclose(out, torch_out, rtol=rtol, atol=atol)
    return res, torch_out

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
        x = torch.flatten(x, 1)
        return x

# Custom small CNN -- multiple conv2d layers + pooling + a non-conv op,
# so segregation actually has something interesting to segregate

class _TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = torch.flatten(x, 1)
        return x


def test_custom_convnet_conv_offloaded_and_numerically_correct():
    model = _TinyConvNet()
    example_input = torch.randn(1, 3, 32, 32)
    _compile_and_compare_model(model, example_input, alter_layout=True, min_dnnl_funcs=1)

def test_custom_convnet_three_conv_offloaded():
    model = _TinyConvNet3()
    example_input = torch.randn(1, 3, 32, 32)

    _compile_and_compare_model(
        model,
        example_input,
        alter_layout=True,
        min_dnnl_funcs=1,
    )

# Pretrained ResNet-18 -- exercises Crop()/TreatAs() across many real
# conv2d shapes (stem conv, stride-2 downsample convs, 1x1 shortcut convs,
# varying channel counts) all at once, inside one realistic model.

def test_resnet18_conv_offloaded_and_numerically_correct():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)  # random weights is fine here
    model = nn.Sequential(*list(model.children())[:-2])
    example_input = torch.randn(1, 3, 224, 224)
    _compile_and_compare_model(
        model, example_input, alter_layout=True, rtol=1e-2, atol=1e-2, min_dnnl_funcs=1
    )


# Shared numeric-verification helper

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
    is exactly what this comparison is designed to catch.
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


def _run_conv2d_case(data_shape, out_channels, groups=1, kernel_size=(3, 3),
                      strides=(1, 1), padding=(1, 1), dilation=(1,1), dtype="float32", out_dtype = None, alter_layout=True, seed=0):
    np.random.seed(seed)
    mod, weight_shape = _make_conv2d_module(
        data_shape=data_shape,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        dtype = dtype,
        out_dtype=out_dtype,
    )
    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")
    return _compile_and_compare(mod, [data_np, weight_np], alter_layout=alter_layout)



# Crop() coverage -- grouped / depthwise convolution

# Crop() slices a combined weight tensor into per-group sub-views before each
# group's convolution runs (dnnl::memory::desc::submemory_desc() in the fixed
# implementation, replacing the old direct-field-write "auto-padded" hack).
# Parametrized across group counts and, within each, channel counts that
# are/aren't multiples of oneDNN's common block sizes (8, 16) -- that
# alignment boundary is exactly what the removed hack used to special-case,
# and what now either succeeds via submemory_desc() or intentionally throws
# rather than silently mis-describing the layout.
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
def test_conv2d_crop_grouped_numerically_correct(in_channels, out_channels, groups, alter_layout):
    _run_conv2d_case(
        data_shape=(1, in_channels, 16, 16),
        out_channels=out_channels,
        groups=groups,
        alter_layout=alter_layout,
        seed=hash((in_channels, out_channels, groups)) % (2**31),
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
def test_conv2d_crop_grouped_stride_padding_variants(strides, padding):
    """Non-trivial stride/padding combinations, to vary the `offset` argument
    Crop() is called with beyond the simplest zero-offset case above."""
    _run_conv2d_case(
        data_shape=(1, 16, 20, 20),
        out_channels=32,
        groups=4,
        strides=strides,
        padding=padding,
        seed=1,
    )


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_conv2d_crop_grouped_batch_variants(batch):
    """Batch size shouldn't interact with per-group weight cropping, but
    verify explicitly rather than assume it."""
    _run_conv2d_case(
        data_shape=(batch, 16, 16, 16),
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
def test_conv2d_crop_treatas_large_channel_counts(in_channels, out_channels, groups):
    """Exercises higher block-count paths that small-channel tests never reach."""
    _run_conv2d_case(
        data_shape=(1, in_channels, 14, 14),
        out_channels=out_channels,
        groups=groups,
        alter_layout=True,
        seed=9,
    )

@pytest.mark.parametrize("dilation", [(1, 1), (2, 2), (3, 3)])
def test_conv2d_treatas_dilation_variants(dilation):
    """Dilation changes the effective kernel footprint oneDNN sees, which
    can select a different primitive/format path than dilation=1."""
    k = 3
    pad = dilation[0] * (k - 1) // 2
    _run_conv2d_case(
        data_shape=(1, 16, 24, 24),
        out_channels=32,
        kernel_size=(k, k),
        padding=(pad, pad),
        dilation=dilation,
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
def test_conv2d_treatas_asymmetric_variants(kernel_size, strides, padding):
    """H and W are separate positional dims in format_tag resolution --
    symmetric-only shapes could hide an axis-swap bug."""
    _run_conv2d_case(
        data_shape=(1, 16, 20, 24),
        out_channels=32,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
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
# FormatTagsByCanonicalName().
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
def test_conv2d_treatas_channel_variants(in_channels, out_channels):
    out_altered, ref_out = _run_conv2d_case(
        data_shape=(1, in_channels, 32, 32),
        out_channels=out_channels,
        alter_layout=True,
        seed=3,
    )
    out_plain, _ = _run_conv2d_case(
        data_shape=(1, in_channels, 32, 32),
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
def test_conv2d_treatas_kernel_size_variants(kernel_size):
    """Kernel size affects which internal oneDNN convolution algorithm gets
    picked (direct vs. Winograd, etc.), which in turn affects which preferred
    (often blocked) layout it requests -- varying this broadens format_tag
    coverage beyond what channel-count variation alone reaches."""
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    _run_conv2d_case(
        data_shape=(1, 16, 32, 32),
        out_channels=32,
        kernel_size=kernel_size,
        padding=padding,
        alter_layout=True,
        seed=4,
    )


@pytest.mark.parametrize(
    "data_shape,weight_channels",
    [
        pytest.param((2, 16, 24, 24), 16, id="batch2"),
        pytest.param((1, 16, 7, 13), 16, id="odd-nonsquare-spatial"),
        pytest.param((4, 8, 8, 8), 8, id="small-spatial-batch4"),
        pytest.param((1, 16, 1, 1), 16, id="degenerate-1x1-spatial"),
    ],
)
def test_conv2d_treatas_batch_and_spatial_variants(data_shape, weight_channels):
    """Varies batch size and spatial dims independently of channel count --
    TreatAs()'s logical-dims computation merges all outer + inner tokens
    positionally, so a bug tied to dimension *position* (rather than size)
    wouldn't necessarily surface if only channel counts were varied above."""
    _run_conv2d_case(
        data_shape=data_shape,
        out_channels=weight_channels,
        kernel_size=(1, 1) if data_shape[-1] == 1 else (3, 3),
        padding=(0, 0) if data_shape[-1] == 1 else (1, 1),
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
# old hack existed for.
@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        pytest.param(16, 32, 2, id="groups2-block-aligned-altered"),
        pytest.param(8, 16, 4, id="groups4-per-group-below-block"),
        pytest.param(6, 6, 6, id="depthwise-per-group-1-channel"),
        pytest.param(32, 32, 32, id="depthwise-wide-per-group-1-channel"),
    ],
)
def test_conv2d_crop_treatas_interaction_grouped_with_altered_layout(
    in_channels, out_channels, groups
):
    _run_conv2d_case(
        data_shape=(1, in_channels, 16, 16),
        out_channels=out_channels,
        groups=groups,
        alter_layout=True,
        seed=6,
    )

# ---------------------------------------------------------------------------
# Dtype coverage
# ---------------------------------------------------------------------------
# oneDNN's blocking factor depends on dtype (f32/bf16 commonly block by
# 8/16, s8/u8 commonly block by 4) -- TreatAs()'s format_tag table has only
# been exercised for float32 above. Repeats the highest-risk Crop+TreatAs
# case across every dtype the pipeline accepts.

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
def test_conv2d_crop_treatas_dtype_variants(dtype, out_dtype):
    np.random.seed(10)
    data_shape = (1, 16, 16, 16)
    mod, weight_shape = _make_conv2d_module(
        data_shape=data_shape, out_channels=32, groups=4,  # groups keeps Crop() in play
        dtype=dtype, out_dtype=out_dtype,
    )
    data_np = _random_for_dtype(data_shape, dtype)
    weight_np = _random_for_dtype(weight_shape, dtype)
    _compile_and_compare(mod, [data_np, weight_np], alter_layout=True, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

