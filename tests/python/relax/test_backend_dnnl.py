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
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import (
    partition_for_dnnl,
    prune_dnnl_subgraphs,
    rewrite_pad_avg_pool2d,
)
from tvm.relax.expr_functor import visitor
from tvm.script import relax as R


def _dnnl_regions(mod):
    return [
        func
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
    ]


def _composite_names(mod):
    # MergeCompositeFunctions nests the Composite-tagged function as a local Function literal
    # bound to a dataflow var inside the outer Codegen-tagged function's body, not as a separate
    # entry in mod.functions -- mirror the same walk codegen.cc's DNNLJSONSerializer does over
    # each function's bindings to find it.
    names = []
    for func in mod.functions.values():
        if not isinstance(func, relax.Function):
            continue
        seq = func.body
        if not isinstance(seq, relax.SeqExpr):
            continue
        for block in seq.blocks:
            for binding in block.bindings:
                value = binding.value
                if (
                    isinstance(value, relax.Function)
                    and value.attrs is not None
                    and "Composite" in value.attrs
                ):
                    names.append(value.attrs["Composite"])
    return sorted(names)


def _has_op_call(func, op_name):
    found = [False]

    @visitor
    class _Finder(relax.PyExprVisitor):
        def visit_call_(self, call):
            if isinstance(call.op, tvm.ir.Op) and call.op.name == op_name:
                found[0] = True
            super().visit_call_(call)

    _Finder().visit_expr(func)
    return found[0]


def test_dnnl_conv2d_partition():
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"), weight: R.Tensor((16, 8, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=1)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2d)
    assert len(_dnnl_regions(mod)) == 1
    assert _composite_names(mod) == ["dnnl.conv2d"]


def test_dnnl_conv1d_partition():
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(data: R.Tensor((2, 8, 16), "float32"), weight: R.Tensor((4, 8, 3), "float32")):
            with R.dataflow():
                out = relax.op.nn.conv1d(data, weight, padding=1)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv1d)
    assert _composite_names(mod) == ["dnnl.conv1d"]


def test_dnnl_conv3d_partition():
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

    mod = partition_for_dnnl(Conv3d)
    assert _composite_names(mod) == ["dnnl.conv3d"]


def test_dnnl_conv2d_transpose_partition():
    @tvm.script.ir_module
    class ConvTranspose:
        @R.function
        def main(
            data: R.Tensor((1, 8, 8, 8), "float32"), weight: R.Tensor((8, 4, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d_transpose(data, weight, padding=1)
                R.output(out)
            return out

    mod = partition_for_dnnl(ConvTranspose)
    assert _composite_names(mod) == ["dnnl.conv2d_transpose"]


def test_dnnl_conv3d_transpose_partition():
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

    mod = partition_for_dnnl(ConvTranspose3d)
    assert _composite_names(mod) == ["dnnl.conv3d_transpose"]


def test_dnnl_max_pool2d_partition():
    @tvm.script.ir_module
    class MaxPool:
        @R.function
        def main(data: R.Tensor((1, 8, 16, 16), "float32")):
            with R.dataflow():
                out = relax.op.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2))
                R.output(out)
            return out

    mod = partition_for_dnnl(MaxPool)
    assert _composite_names(mod) == ["dnnl.max_pool2d"]


def test_dnnl_avg_pool3d_partition():
    @tvm.script.ir_module
    class AvgPool3d:
        @R.function
        def main(data: R.Tensor((1, 8, 8, 8, 8), "float32")):
            with R.dataflow():
                out = relax.op.nn.avg_pool3d(data, pool_size=(2, 2, 2), strides=(2, 2, 2))
                R.output(out)
            return out

    mod = partition_for_dnnl(AvgPool3d)
    assert _composite_names(mod) == ["dnnl.avg_pool3d"]


def test_dnnl_conv2d_grouped_partition():
    # groups > 1 exercises TensorRequisite::ApplyGroupWeightLayout in the runtime; at the
    # partition level it should still collapse into a single dnnl.conv2d region like the
    # ungrouped case, since grouping is purely an attribute on the existing conv2d op.
    @tvm.script.ir_module
    class GroupedConv2d:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"), weight: R.Tensor((16, 4, 3, 3), "float32")
        ):
            with R.dataflow():
                out = relax.op.nn.conv2d(data, weight, padding=1, groups=4)
                R.output(out)
            return out

    mod = partition_for_dnnl(GroupedConv2d)
    assert _composite_names(mod) == ["dnnl.conv2d"]


def test_dnnl_conv2d_bias_relu_partition():
    @tvm.script.ir_module
    class Conv2dBiasRelu:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16,), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = relax.op.nn.relu(biased)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBiasRelu)
    assert _composite_names(mod) == ["dnnl.conv2d_bias_relu"]


def test_dnnl_conv2d_bias_only_partition():
    @tvm.script.ir_module
    class Conv2dBias:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16,), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                out = relax.op.add(conv, bias)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBias)
    assert _composite_names(mod) == ["dnnl.conv2d_bias"]


@pytest.mark.parametrize(
    "activation_op, expected_suffix",
    [
        (relax.op.nn.relu, "relu"),
        (relax.op.sigmoid, "sigmoid"),
        (relax.op.nn.gelu, "gelu"),
        (relax.op.tanh, "tanh"),
    ],
)
def test_dnnl_conv2d_bias_activation_partition(activation_op, expected_suffix):
    @tvm.script.ir_module
    class Conv2dBiasAct:
        @R.function
        def main(
            data: R.Tensor((1, 8, 16, 16), "float32"),
            weight: R.Tensor((16, 8, 3, 3), "float32"),
            bias: R.Tensor((16,), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = activation_op(biased)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBiasAct)
    assert _composite_names(mod) == [f"dnnl.conv2d_bias_{expected_suffix}"]


def test_dnnl_conv2d_bias_sum_relu_partition():
    @tvm.script.ir_module
    class Conv2dBiasSumRelu:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((16,), "float32"),
            residual: R.Tensor((1, 16, 16, 16), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                summed = relax.op.add(biased, residual)
                out = relax.op.nn.relu(summed)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBiasSumRelu)
    assert _composite_names(mod) == ["dnnl.conv2d_bias_sum_relu"]


def test_dnnl_conv2d_bias_sum_partition():
    @tvm.script.ir_module
    class Conv2dBiasSum:
        @R.function
        def main(
            data: R.Tensor((1, 16, 16, 16), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((16,), "float32"),
            residual: R.Tensor((1, 16, 16, 16), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                out = relax.op.add(biased, residual)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBiasSum)
    assert _composite_names(mod) == ["dnnl.conv2d_bias_sum"]


def test_dnnl_sum_pattern_rejects_bias_channel_mismatch():
    # bias here is sized to match the width dim (10), not the conv's channel dim (16), so the
    # add still broadcasts but bias isn't actually a per-channel bias DNNL's post-op can
    # represent. The checker in _sum_pattern should reject the "_sum" fusion; the bias-add can
    # still fold into a plain dnnl.conv2d_bias, just not the residual-sum composite.
    @tvm.script.ir_module
    class Conv2dBadBiasSum:
        @R.function
        def main(
            data: R.Tensor((1, 16, 8, 10), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((10,), "float32"),
            residual: R.Tensor((1, 16, 8, 10), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                summed = relax.op.add(biased, residual)
                out = relax.op.nn.relu(summed)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBadBiasSum)
    assert not any("_sum" in name for name in _composite_names(mod))


def test_dnnl_sum_pattern_rejects_residual_shape_mismatch():
    # residual broadcasts against the conv output (1 channel vs 16) but isn't a true elementwise
    # match, so it can't be represented as a DNNL sum post-op.
    @tvm.script.ir_module
    class Conv2dBadResidualSum:
        @R.function
        def main(
            data: R.Tensor((1, 16, 8, 8), "float32"),
            weight: R.Tensor((16, 16, 3, 3), "float32"),
            bias: R.Tensor((16, 1, 1), "float32"),
            residual: R.Tensor((1, 1, 8, 8), "float32"),
        ):
            with R.dataflow():
                conv = relax.op.nn.conv2d(data, weight, padding=1)
                biased = relax.op.add(conv, bias)
                summed = relax.op.add(biased, residual)
                out = relax.op.nn.relu(summed)
                R.output(out)
            return out

    mod = partition_for_dnnl(Conv2dBadResidualSum)
    assert not any("_sum" in name for name in _composite_names(mod))


def test_dnnl_pad_avg_pool2d_fold_supported():
    @tvm.script.ir_module
    class PadAvgPool:
        @R.function
        def main(data: R.Tensor((1, 4, 8, 8), "float32")):
            with R.dataflow():
                padded = relax.op.nn.pad(data, [0, 0, 0, 0, 1, 1, 1, 1])
                out = relax.op.nn.avg_pool2d(padded, pool_size=(2, 2), strides=(2, 2))
                R.output(out)
            return out

    rewritten = rewrite_pad_avg_pool2d(PadAvgPool)
    assert not _has_op_call(rewritten["main"], "relax.nn.pad")


@pytest.mark.parametrize(
    "pad_width, pad_mode, pad_value, pool_padding",
    [
        # non-zero pad value can't be folded into avg_pool2d's zero-implicit padding
        ([0, 0, 0, 0, 1, 1, 1, 1], "constant", 1.0, (0, 0, 0, 0)),
        # non-constant pad mode has no equivalent padding attribute on avg_pool2d
        ([0, 0, 0, 0, 1, 1, 1, 1], "reflect", 0.0, (0, 0, 0, 0)),
        # padding on the batch axis has no meaning for avg_pool2d's spatial padding
        ([0, 1, 0, 0, 1, 1, 1, 1], "constant", 0.0, (0, 0, 0, 0)),
        # avg_pool2d already carries its own padding, so the two can't be merged blindly
        ([0, 0, 0, 0, 1, 1, 1, 1], "constant", 0.0, (1, 1, 1, 1)),
    ],
)
def test_dnnl_pad_avg_pool2d_fold_fallback(pad_width, pad_mode, pad_value, pool_padding):
    @tvm.script.ir_module
    class PadAvgPool:
        @R.function
        def main(data: R.Tensor((1, 4, 8, 8), "float32")):
            with R.dataflow():
                padded = relax.op.nn.pad(data, pad_width, pad_mode, pad_value)
                out = relax.op.nn.avg_pool2d(
                    padded, pool_size=(2, 2), strides=(2, 2), padding=pool_padding
                )
                R.output(out)
            return out

    rewritten = rewrite_pad_avg_pool2d(PadAvgPool)
    assert _has_op_call(rewritten["main"], "relax.nn.pad")


def test_dnnl_pad_avg_pool2d_fold_rejects_multi_consumer_pad():
    # the padded tensor also feeds a second consumer, so folding the pad into avg_pool2d would
    # change what the other consumer sees.
    @tvm.script.ir_module
    class PadAvgPool:
        @R.function
        def main(data: R.Tensor((1, 4, 8, 8), "float32")):
            with R.dataflow():
                padded = relax.op.nn.pad(data, [0, 0, 0, 0, 1, 1, 1, 1])
                pooled = relax.op.nn.avg_pool2d(padded, pool_size=(2, 2), strides=(2, 2))
                extra = relax.op.sum(padded, axis=[2, 3], keepdims=True)
                out = relax.op.add(pooled, extra)
                R.output(out)
            return out

    rewritten = rewrite_pad_avg_pool2d(PadAvgPool)
    assert _has_op_call(rewritten["main"], "relax.nn.pad")


def test_dnnl_prune_removes_empty_compute_subgraph():
    # Hand-construct a module where a subgraph is tagged for DNNL offload but its body contains
    # no actual DNNL-computable op -- prune_dnnl_subgraphs should demote it back to a normal
    # function instead of handing an empty region to the DNNL codegen.
    #
    # Built with BlockBuilder + an explicit GlobalVar call rather than TVMScript, since calling a
    # sibling function by its bare name from within the same ir_module class isn't resolved by
    # the parser here.
    @tvm.script.ir_module
    class EmptyDnnlRegion:
        @R.function
        def dnnl_region(x: R.Tensor((1, 4, 8, 8), "float32")) -> R.Tensor((1, 4, 8, 8), "float32"):
            R.func_attr({"Codegen": "dnnl", "global_symbol": "dnnl_region"})
            with R.dataflow():
                out = relax.op.nn.relu(x)
                R.output(out)
            return out

    region_gv = EmptyDnnlRegion.get_global_var("dnnl_region")
    bb = relax.BlockBuilder(EmptyDnnlRegion)
    x2 = relax.Var("x2", relax.TensorType((1, 4, 8, 8), "float32"))
    with bb.function("main", [x2]):
        with bb.dataflow():
            out = bb.emit_output(relax.Call(region_gv, [x2]))
        bb.emit_func_output(out)
    mod = bb.finalize()

    pruned = prune_dnnl_subgraphs(mod)
    assert not any(
        isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
        for func in pruned.functions.values()
    )


def test_partition_for_dnnl_entry_point():
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

    mod = partition_for_dnnl(Model)
    assert _dnnl_regions(mod)
    assert _composite_names(mod) == ["dnnl.conv2d_relu"]


if __name__ == "__main__":
    tvm.testing.main()
