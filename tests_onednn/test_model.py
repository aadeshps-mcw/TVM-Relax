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
Custom-model test for the DNNL BYOC backend, covering three fusion paths
in one module:

  1. conv2d -> +bias -> +residual -> relu   (expect: dnnl.conv2d_bias_sum_relu)
  2. matmul -> +bias -> relu                (expect: dnnl.matmul_bias_relu)
  3. standalone sigmoid                     (expect: dnnl.sigmoid)

API used here is lifted from this fork's own tests/python/relax/test_backend_dnnl.py
and test_dnnl_matmul_patterns.py rather than upstream Relax:
  - relax.Var(name, relax.TensorType(shape, dtype))   -- NOT TensorStructInfo
  - tvm.runtime.tensor(...)                           -- NOT tvm.nd.array
  - partition_for_dnnl(mod, run_codegen=False) to inspect composite names,
    then relax.transform.RunCodegen()(mod) separately to actually compile --
    partition_for_dnnl alone only tags subgraphs, it doesn't lift/compile them.
  - Composite-tagged functions live *nested inside* the Codegen="dnnl"
    function's body after MergeCompositeFunctions, not as top-level
    mod.functions entries -- must walk recursively (see
    _all_composite_names_in_mod below) or the check silently finds nothing.

Run with:
    pytest test_dnnl_custom_model.py -v
"""

import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

requires_dnnl_runtime = pytest.mark.skipif(
    tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", allow_missing=True) is None,
    reason="DNNL JSON runtime not registered in this build",
)


def _to_tensor(np_array, dev):
    """tvm.runtime.tensor(...) call signature, tolerant of device being
    positional vs keyword (unconfirmed which this build expects)."""
    try:
        return tvm.runtime.tensor(np_array, device=dev)
    except TypeError:
        return tvm.runtime.tensor(np_array, dev)


def _all_composite_names_in_mod(mod: tvm.IRModule) -> list:
    """Recursively collect every `Composite` attr string in the module,
    including composite functions bound *locally* inside another
    function's body (where they actually live post-MergeCompositeFunctions)."""
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


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _make_conv2d_bias_sum_relu_module(
    data_shape=(1, 8, 16, 16),
    weight_shape=(8, 8, 3, 3),
    dtype="float32",
):
    """conv2d -> +bias -> +residual -> relu. Bias is pre-shaped to
    (out_channels, 1, 1) -- broadcasts against NCHW output on the trailing
    dims -- matching this repo's own _make_conv2d_bias_relu_module helper,
    so no reshape op sits between the conv and the bias add."""
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
            out = builder.emit(relax.op.nn.relu(summed))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_matmul_bias_relu_module(m=16, k=32, n=64, dtype="float32"):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType((m, k), dtype))
    weight = relax.Var("weight", relax.TensorType((k, n), dtype))
    bias = relax.Var("bias", relax.TensorType((n,), dtype))

    with builder.function("main", [data, weight, bias]):
        with builder.dataflow():
            mm = builder.emit(relax.op.matmul(data, weight))
            biased = builder.emit(relax.op.add(mm, bias))
            out = builder.emit(relax.op.nn.relu(biased))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


def _make_standalone_sigmoid_module(shape=(1, 16), dtype="float32"):
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(shape, dtype))

    with builder.function("main", [x]):
        with builder.dataflow():
            out = builder.emit(relax.op.sigmoid(x))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    return builder.get()


# ---------------------------------------------------------------------------
# Section 1: pattern matching only (no DNNL runtime required)
# ---------------------------------------------------------------------------


class TestCustomModelPatternMatching:
    def test_conv2d_bias_sum_relu_selects_expected_composite(self):
        mod = _make_conv2d_bias_sum_relu_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.conv2d_bias_sum_relu" in names, f"got {names}"

    def test_matmul_bias_relu_selects_expected_composite(self):
        mod = _make_matmul_bias_relu_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul_bias_relu" in names, f"got {names}"

    def test_standalone_sigmoid_selects_expected_composite(self):
        mod = _make_standalone_sigmoid_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.sigmoid" in names, f"got {names}"


# ---------------------------------------------------------------------------
# Section 2: codegen + numeric correctness (requires DNNL runtime registered)
# ---------------------------------------------------------------------------


@requires_dnnl_runtime
class TestCustomModelCodegenDispatch:
    def _compile(self, mod: tvm.IRModule):
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        partitioned = relax.transform.RunCodegen()(partitioned)
        target = tvm.target.Target("llvm")
        return relax.build(partitioned, target=target)

    def _vm_for(self, mod: tvm.IRModule):
        # Go through VM init (not just relax.build) -- oneDNN primitive
        # descriptors are only actually constructed at vm_initialization,
        # so build() alone can pass even with a malformed desc.
        ex = self._compile(mod)
        return relax.VirtualMachine(ex, tvm.cpu())

    def test_conv2d_bias_sum_relu_matches_reference(self):
        data_shape = (1, 8, 16, 16)
        weight_shape = (8, 8, 3, 3)
        oc = weight_shape[0]
        n, _, h, w = data_shape
        dtype = "float32"

        mod = _make_conv2d_bias_sum_relu_module(data_shape, weight_shape, dtype)

        rng = np.random.default_rng(0)
        data_np = rng.standard_normal(data_shape).astype(dtype)
        weight_np = (rng.standard_normal(weight_shape) * 0.1).astype(dtype)
        bias_np = (rng.standard_normal((oc, 1, 1)) * 0.1).astype(dtype)
        residual_np = rng.standard_normal((n, oc, h, w)).astype(dtype)

        dev = tvm.cpu()
        ref_ex = relax.build(mod, target="llvm")
        ref_vm = relax.VirtualMachine(ref_ex, dev)
        ref_out = ref_vm["main"](
            _to_tensor(data_np, dev),
            _to_tensor(weight_np, dev),
            _to_tensor(bias_np, dev),
            _to_tensor(residual_np, dev),
        ).numpy()

        vm = self._vm_for(mod)
        out = vm["main"](
            _to_tensor(data_np, dev),
            _to_tensor(weight_np, dev),
            _to_tensor(bias_np, dev),
            _to_tensor(residual_np, dev),
        ).numpy()

        np.testing.assert_allclose(ref_out, out, rtol=1e-4, atol=1e-4)

    def test_matmul_bias_relu_matches_reference(self):
        m, k, n = 16, 32, 64
        dtype = "float32"
        mod = _make_matmul_bias_relu_module(m, k, n, dtype)

        rng = np.random.default_rng(1)
        data_np = rng.standard_normal((m, k)).astype(dtype)
        weight_np = (rng.standard_normal((k, n)) * 0.1).astype(dtype)
        bias_np = (rng.standard_normal((n,)) * 0.1).astype(dtype)

        dev = tvm.cpu()
        ref_ex = relax.build(mod, target="llvm")
        ref_vm = relax.VirtualMachine(ref_ex, dev)
        ref_out = ref_vm["main"](
            _to_tensor(data_np, dev), _to_tensor(weight_np, dev), _to_tensor(bias_np, dev)
        ).numpy()

        vm = self._vm_for(mod)
        out = vm["main"](
            _to_tensor(data_np, dev), _to_tensor(weight_np, dev), _to_tensor(bias_np, dev)
        ).numpy()

        np.testing.assert_allclose(ref_out, out, rtol=1e-4, atol=1e-4)

    def test_standalone_sigmoid_matches_reference(self):
        shape = (1, 16)
        dtype = "float32"
        mod = _make_standalone_sigmoid_module(shape, dtype)

        rng = np.random.default_rng(2)
        x_np = rng.standard_normal(shape).astype(dtype)

        dev = tvm.cpu()
        ref_ex = relax.build(mod, target="llvm")
        ref_vm = relax.VirtualMachine(ref_ex, dev)
        ref_out = ref_vm["main"](_to_tensor(x_np, dev)).numpy()

        vm = self._vm_for(mod)
        out = vm["main"](_to_tensor(x_np, dev)).numpy()

        np.testing.assert_allclose(ref_out, out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
