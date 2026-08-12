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

"""Tests for DNNL residual/sum fusion patterns (dnnl.conv2d_bias_sum[_relu])."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl


def _get_composite_names(mod: tvm.IRModule) -> list[str]:
    """Collect the 'Composite' attr of every composite function offloaded to DNNL.

    MergeCompositeFunctions nests the "Composite"-tagged function *inside* the
    body of the top-level "Codegen: dnnl" function (bound as a local Function
    value via a VarBinding), rather than leaving it as its own entry in
    mod.functions. So we have to walk into each function's SeqExpr bindings,
    not just scan mod.functions.items() directly.
    """
    names = []

    def _scan(expr):
        if isinstance(expr, relax.Function):
            if expr.attrs is not None:
                composite = expr.attrs.get("Composite")
                if composite is not None:
                    names.append(str(composite))
            _scan(expr.body)
        elif isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    if hasattr(binding, "value"):
                        _scan(binding.value)

    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _scan(func)

    return names


def _build_conv2d_bias_residual_mod(with_relu: bool) -> tvm.IRModule:
    """data --conv2d--> +bias --add(residual)--> [relu] --> output

    `residual` is a second external input of the same shape as the conv output,
    forcing the sum pattern (not just the plain bias pattern) to be the only
    thing that can match the whole chain.
    """
    data_shape = (1, 3, 8, 8)
    weight_shape = (4, 3, 3, 3)
    out_shape = (1, 4, 6, 6)  # no padding, stride 1, kernel 3 -> 8-3+1=6
    bias_shape = (4,)

    bb = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    weight = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
    bias = relax.Var("bias", relax.TensorType(bias_shape, "float32"))
    residual = relax.Var("residual", relax.TensorType(out_shape, "float32"))

    with bb.function("main", [data, weight, bias, residual]):
        with bb.dataflow():
            conv_out = bb.emit(relax.op.nn.conv2d(data, weight))
            # reshape bias to broadcast over channel dim, matching how conv2d_bias
            # composites are normally built elsewhere in this file.
            bias_r = bb.emit(relax.op.reshape(bias, (1, 4, 1, 1)))
            biased = bb.emit(relax.op.add(conv_out, bias_r))
            summed = bb.emit(relax.op.add(biased, residual))
            out = bb.emit(relax.op.nn.relu(summed)) if with_relu else summed
            out = bb.emit_output(out)
        bb.emit_func_output(out)

    return bb.get()


@pytest.mark.parametrize("with_relu", [False, True])
def test_dnnl_conv2d_bias_sum_pattern_matches(with_relu):
    mod = _build_conv2d_bias_residual_mod(with_relu)
    partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

    composite_names = _get_composite_names(partitioned)
    expected = "dnnl.conv2d_bias_sum_relu" if with_relu else "dnnl.conv2d_bias_sum"

    assert expected in composite_names, (
        f"Expected composite '{expected}' in offloaded functions, got: {composite_names}"
    )
    # Guard against the smaller dnnl.conv2d_bias pattern greedily claiming
    # part of the chain instead of the full residual fusion.
    assert "dnnl.conv2d_bias" not in composite_names
    assert "dnnl.conv2d_bias_relu" not in composite_names


def test_dnnl_conv2d_bias_sum_end_to_end():
    """Full numerical check: partition, build, run, compare against numpy reference."""
    with_relu = True
    mod = _build_conv2d_bias_residual_mod(with_relu)

    np.random.seed(0)
    data_np = np.random.uniform(-1, 1, (1, 3, 8, 8)).astype("float32")
    weight_np = np.random.uniform(-1, 1, (4, 3, 3, 3)).astype("float32")
    bias_np = np.random.uniform(-1, 1, (4,)).astype("float32")
    residual_np = np.random.uniform(-1, 1, (1, 4, 6, 6)).astype("float32")

    # Reference: plain conv2d + bias + residual add + relu via the relax VM
    # without DNNL offload, using the same mod before partitioning.
    ref_target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        ref_ex = relax.build(mod, target=ref_target)
    ref_vm = relax.VirtualMachine(ref_ex, tvm.cpu())
    ref_out = ref_vm["main"](
        tvm.runtime.tensor(data_np),
        tvm.runtime.tensor(weight_np),
        tvm.runtime.tensor(bias_np),
        tvm.runtime.tensor(residual_np),
    ).numpy()

    # DNNL-offloaded path. partition_for_dnnl() leaves the DNNL subgraph as a
    # nested/local Relax function tagged Codegen="dnnl" -- RunCodegen() is the
    # pass that actually invokes our registered relax.ext.dnnl (DNNLCompiler in
    # codegen.cc) on it, replacing it with a call into the compiled external
    # module and erasing the local function. Without this, relax.build's
    # VMShapeLower stage chokes on the leftover local function.
    partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
    with tvm.transform.PassContext(opt_level=3):
        partitioned = relax.transform.RunCodegen()(partitioned)
        dnnl_ex = relax.build(partitioned, target=ref_target)
    dnnl_vm = relax.VirtualMachine(dnnl_ex, tvm.cpu())
    dnnl_out = dnnl_vm["main"](
        tvm.runtime.tensor(data_np),
        tvm.runtime.tensor(weight_np),
        tvm.runtime.tensor(bias_np),
        tvm.runtime.tensor(residual_np),
    ).numpy()

    tvm.testing.assert_allclose(ref_out, dnnl_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
