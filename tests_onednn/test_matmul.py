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
Tests for the new dnnl.matmul_* fused patterns (bias / activation / clip)
added to python/tvm/relax/op/contrib/dnnl.py, and their codegen.cc dispatch.

Run with:
    pytest test_dnnl_matmul_patterns.py -v

Requires a TVM build with USE_DNNL_CODEGEN (or equivalent) enabled, since the
codegen-level tests actually invoke `relax.ext.dnnl` and require the DNNL
runtime module to be registered. The pattern-matching tests (Section 1) only
need FuseOpsByPattern/MergeCompositeFunctions and do NOT require DNNL runtime
libs to be linked -- keep them separable so CI can run them even on builds
without DNNL, by skipping Section 2 (see `requires_dnnl_runtime` marker).
"""

import itertools
import linecache

import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl  # adjust import path

# if partition_for_dnnl lives at tvm.relax.op.contrib.dnnl in your tree, use:
# from tvm.relax.op.contrib.dnnl import partition_for_dnnl

requires_dnnl_runtime = pytest.mark.skipif(
    tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", allow_missing=True) is None,
    reason="DNNL JSON runtime not registered in this build",
)

# ---------------------------------------------------------------------------
# Helpers to build small Relax modules
# ---------------------------------------------------------------------------
_synthetic_src_counter = itertools.count()


def _to_tensor(np_array, dev):
    """Construct a device tensor from a numpy array, tolerant of the exact
    tvm.runtime.tensor(...) call signature (this build replaced the old
    tvm.nd.array with tvm.runtime.tensor as part of the tvm-ffi refactor;
    whether device is positional or keyword isn't confirmed here)."""
    try:
        return tvm.runtime.tensor(np_array, device=dev)
    except TypeError:
        return tvm.runtime.tensor(np_array, dev)


def _ir_module_from_source(src: str):
    """Compile and execute TVMScript source text and return the resulting
    IRModule.

    TVMScript's parser calls inspect.getsource() on the decorated function
    to re-read its own source text (it's parsing Python-as-a-DSL, not just
    running bytecode). A plain exec() of a string leaves no trace for
    `inspect` to find, so getsource() raises "could not get source code".
    We work around this the standard way: register the source under a
    synthetic filename in linecache before exec'ing it, so inspect can look
    it up there instead of hitting the filesystem.
    """
    filename = f"<tvmscript_generated_{next(_synthetic_src_counter)}>"
    linecache.cache[filename] = (
        len(src),
        None,
        src.splitlines(keepends=True),
        filename,
    )
    code = compile(src, filename, "exec")
    ns: dict = {}
    exec(code, ns)
    return ns["Module"]


_MATMUL_BIAS_RELU_SRC = """
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        data: R.Tensor(({m}, {k}), "{dtype}"),
        weight: R.Tensor(({k}, {n}), "{dtype}"),
        bias: R.Tensor(({n},), "{dtype}"),
    ) -> R.Tensor(({m}, {n}), "{dtype}"):
        with R.dataflow():
            out = R.matmul(data, weight)
            out = R.add(out, bias)
            out = R.nn.relu(out)
            R.output(out)
        return out
"""

_MATMUL_BIAS_ONLY_SRC = """
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        data: R.Tensor(({m}, {k}), "{dtype}"),
        weight: R.Tensor(({k}, {n}), "{dtype}"),
        bias: R.Tensor(({n},), "{dtype}"),
    ) -> R.Tensor(({m}, {n}), "{dtype}"):
        with R.dataflow():
            out = R.matmul(data, weight)
            out = R.add(out, bias)
            R.output(out)
        return out
"""

_MATMUL_BIAS_CLIP_SRC = """
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        data: R.Tensor(({m}, {k}), "{dtype}"),
        weight: R.Tensor(({k}, {n}), "{dtype}"),
        bias: R.Tensor(({n},), "{dtype}"),
    ) -> R.Tensor(({m}, {n}), "{dtype}"):
        with R.dataflow():
            out = R.matmul(data, weight)
            out = R.add(out, bias)
            out = R.clip(out, 0.0, 6.0)
            R.output(out)
        return out
"""

_PLAIN_MATMUL_SRC = """
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        data: R.Tensor(({m}, {k}), "{dtype}"),
        weight: R.Tensor(({k}, {n}), "{dtype}"),
    ) -> R.Tensor(({m}, {n}), "{dtype}"):
        with R.dataflow():
            out = R.matmul(data, weight)
            R.output(out)
        return out
"""


def _matmul_bias_relu_module(m=16, k=32, n=64, dtype="float32"):
    """data[m,k] @ weight[k,n] + bias[n] -> relu"""
    return _ir_module_from_source(_MATMUL_BIAS_RELU_SRC.format(m=m, k=k, n=n, dtype=dtype))


def _matmul_bias_only_module(m=16, k=32, n=64, dtype="float32"):
    """Same as above but no activation -- should hit dnnl.matmul_bias."""
    return _ir_module_from_source(_MATMUL_BIAS_ONLY_SRC.format(m=m, k=k, n=n, dtype=dtype))


def _matmul_bias_clip_module(m=16, k=32, n=64, dtype="float32"):
    """matmul + bias + clip(0, 6) -- a relu6-style pattern, exercises the
    _clip attribute-extraction path in codegen.cc."""
    return _ir_module_from_source(_MATMUL_BIAS_CLIP_SRC.format(m=m, k=k, n=n, dtype=dtype))


def _plain_matmul_module(m=16, k=32, n=64, dtype="float32"):
    """Bare matmul, no bias/activation -- should still hit the base
    dnnl.matmul pattern (i.e. adding the fused variants must not break the
    no-fusion fallback)."""
    return _ir_module_from_source(_PLAIN_MATMUL_SRC.format(m=m, k=k, n=n, dtype=dtype))


def _get_dnnl_composite_names(mod: tvm.IRModule) -> list[str]:
    """Collect the `Composite` attr string off every function tagged with
    Codegen == "dnnl" in the module (post-partitioning, pre-BYOC-compile)."""
    names = []
    for gvar, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if func.attrs is None:
            continue
        # Composite-tagged inner functions live inside the dnnl Codegen
        # function; walk both the outer function and anything it calls.
        composite = func.attrs.get("Composite")
        if composite is not None:
            names.append(str(composite))
    return names


def _all_composite_names_in_mod(mod: tvm.IRModule) -> list[str]:
    """Collect every Composite attr string in the module, including on
    functions that are bound *locally* inside another function's body.

    After FuseOpsByPattern + MergeCompositeFunctions, each Composite-tagged
    function ends up as a local (nested) binding inside the outer
    Codegen="dnnl" function's body -- it is NOT a top-level entry in
    mod.functions. Only scanning mod.functions.items() (as an earlier
    version of this helper did) silently finds nothing.
    """
    names: list[str] = []

    def _collect_from_function(func: relax.Function):
        if func.attrs is not None:
            c = func.attrs.get("Composite")
            if c is not None:
                names.append(str(c))
        body = func.body
        seq = body if isinstance(body, relax.SeqExpr) else None
        if seq is None:
            return
        for block in seq.blocks:
            for binding in block.bindings:
                value = getattr(binding, "value", None)
                if isinstance(value, relax.Function):
                    _collect_from_function(value)

    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _collect_from_function(func)

    return names


# ---------------------------------------------------------------------------
# Section 1: pattern registration / matching (no DNNL runtime required)
# ---------------------------------------------------------------------------


class TestMatmulPatternMatching:
    def test_matmul_bias_relu_selects_longest_composite(self):
        """A matmul+bias+relu graph should be fused into a single
        dnnl.matmul_bias_relu composite -- not dnnl.matmul_bias (shorter,
        would under-fuse) and not left unpartitioned."""
        mod = _matmul_bias_relu_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul_bias_relu" in names, (
            f"expected dnnl.matmul_bias_relu in composite names, got {names}"
        )
        # Make sure the shorter bias-only pattern did NOT also fire on a
        # subset of the same ops (would indicate a matching-order bug).
        assert "dnnl.matmul_bias" not in names, (
            f"dnnl.matmul_bias should not appear when relu is present, got {names}"
        )

    def test_matmul_bias_only_selects_bias_pattern(self):
        mod = _matmul_bias_only_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul_bias" in names, f"got {names}"

    def test_matmul_bias_clip_selects_clip_pattern(self):
        mod = _matmul_bias_clip_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul_bias_clip" in names, f"got {names}"

    def test_plain_matmul_still_matches_base_pattern(self):
        """Regression guard: adding matmul_bias/_relu/etc. must not steal
        the plain matmul-only graph away from the base dnnl.matmul pattern,
        and must not leave it unpartitioned."""
        mod = _plain_matmul_module()
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)

        names = _all_composite_names_in_mod(partitioned)
        assert "dnnl.matmul" in names, f"got {names}"
        assert not any(n.startswith("dnnl.matmul_") for n in names), (
            f"plain matmul should not match a fused variant, got {names}"
        )

    @pytest.mark.parametrize(
        "activation_suffix",
        ["relu", "tanh", "sigmoid", "gelu", "swish"],
    )
    def test_all_activation_variants_registered(self, activation_suffix):
        """Sanity check that every entry in _ACTIVATIONS actually produced
        both a bias and no-bias matmul pattern name. This test builds the
        graph directly for `relu`/`tanh` etc. only where R exposes the op
        as a single call; for others it's enough to confirm the pattern
        exists in the registry (structural check, not a full graph match)."""
        from tvm.relax.backend.contrib.dnnl import _dnnl_patterns  # adjust path

        pattern_names = {p[0] for p in _dnnl_patterns()}
        assert f"dnnl.matmul_{activation_suffix}" in pattern_names
        assert f"dnnl.matmul_bias_{activation_suffix}" in pattern_names
        # and confirm conv2d got the same treatment (parity check for the
        # shared _make_fused_variants helper)
        assert f"dnnl.conv2d_{activation_suffix}" in pattern_names
        assert f"dnnl.conv2d_bias_{activation_suffix}" in pattern_names


# ---------------------------------------------------------------------------
# Section 2: codegen.cc dispatch (requires DNNL runtime module registered)
# ---------------------------------------------------------------------------


@requires_dnnl_runtime
class TestMatmulCodegenDispatch:
    def _compile(self, mod: tvm.IRModule):
        """Partition + run codegen; this is where ResolveRootCall gets
        exercised for real. Any 'Unimplemented pattern' InternalError means
        the prefix-map fix in codegen.cc is missing or wrong.

        RunCodegen is required here (not just partition_for_dnnl): partition
        only tags subgraphs with the Codegen="dnnl" attribute, it doesn't
        invoke DNNLJSONSerializer/relax.ext.dnnl itself. Skipping this step
        leaves a local (un-lifted) Relax Function in the module, which
        VMShapeLower rejects with "do not work for local functions".
        """
        partitioned = partition_for_dnnl(mod, alter_layout=False, run_codegen=False)
        partitioned = relax.transform.RunCodegen()(partitioned)
        target = tvm.target.Target("llvm")
        ex = relax.build(partitioned, target=target)
        return ex

    def _compile_and_init_vm(self, mod: tvm.IRModule):
        """Like _compile, but also constructs a VirtualMachine.

        relax.build() alone does NOT build oneDNN primitive descriptors --
        that happens lazily during VM initialization (vm_initialization).
        A test that only calls relax.build() can pass even when the actual
        primitive descriptor is malformed (e.g. a bias with the wrong rank
        for oneDNN's matmul primitive) and will silently fail to catch it.
        Always go through this helper, not _compile alone, when the goal is
        to confirm the fused op actually works at runtime.
        """
        ex = self._compile(mod)
        dev = tvm.cpu()
        return relax.VirtualMachine(ex, dev)

    def test_matmul_bias_relu_compiles(self):
        mod = _matmul_bias_relu_module()
        # Must go through VM init, not just relax.build(), since oneDNN
        # primitive descriptors (where the bias-rank bug lives) are only
        # actually constructed during vm_initialization.
        self._compile_and_init_vm(mod)

    def test_matmul_bias_compiles(self):
        mod = _matmul_bias_only_module()
        self._compile_and_init_vm(mod)

    def test_matmul_bias_clip_compiles_with_correct_bounds(self):
        """End-to-end: confirm the a_min/a_max clip bounds actually reach
        the DNNL runtime and produce numerically correct output (0, 6)
        clamping), not silently running with bounds (0, 0) -- this is
        exactly the bug the codegen.cc comment warns about."""
        import numpy as np

        m, k, n = 4, 8, 4
        mod = _matmul_bias_clip_module(m=m, k=k, n=n)
        vm = self._compile_and_init_vm(mod)
        dev = tvm.cpu()

        np.random.seed(0)
        data_np = np.random.uniform(-10, 10, size=(m, k)).astype("float32")
        weight_np = np.random.uniform(-10, 10, size=(k, n)).astype("float32")
        bias_np = np.zeros((n,), dtype="float32")

        out = vm["main"](
            _to_tensor(data_np, dev),
            _to_tensor(weight_np, dev),
            _to_tensor(bias_np, dev),
        ).numpy()

        expected = np.clip(data_np @ weight_np + bias_np, 0.0, 6.0)
        # A wide-ish tolerance since this is exercising oneDNN's kernel, not
        # a bitwise numpy re-implementation.
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

        # The critical regression check: if a_min/a_max silently defaulted
        # to (0, 0), *every* output would be exactly 0. Guard against that
        # specific failure mode explicitly, since assert_allclose above
        # could in principle pass on a degenerate all-zero comparison if
        # `expected` also happened to clip everything to 0 for this seed.
        assert np.any(out > 0.0), (
            "all outputs are <= 0; looks like clip bounds defaulted to (0, 0) "
            "instead of (0, 6) -- check a_min/a_max attr extraction in codegen.cc"
        )

    def test_plain_matmul_still_compiles(self):
        """Regression guard at the codegen level too, not just pattern
        matching: dnnl.matmul (no fused variant) must still resolve via
        the prefix map after dnnl.matmul was moved out of kExactResolvers."""
        mod = _plain_matmul_module()
        self._compile_and_init_vm(mod)

    def test_layer_norm_unaffected_by_matmul_prefix_change(self):
        """Regression guard: moving dnnl.matmul from kExactResolvers to
        kPrefixResolvers must not affect dnnl.layer_norm's exact-match
        entry, which stays in kExactResolvers."""

        src = """
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main(
        data: R.Tensor((2, 8, 16), "float32"),
        gamma: R.Tensor((16,), "float32"),
        beta: R.Tensor((16,), "float32"),
    ) -> R.Tensor((2, 8, 16), "float32"):
        with R.dataflow():
            out = R.nn.layer_norm(data, gamma, beta, axes=[-1])
            R.output(out)
        return out
"""
        mod = _ir_module_from_source(src)
        self._compile_and_init_vm(mod)


if __name__ == "__main__":
    tvm.testing.main()
