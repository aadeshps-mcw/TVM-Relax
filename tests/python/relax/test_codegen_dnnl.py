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

"""End-to-end frontend (pattern partitioning) and codegen (DNNL JSON runtime) tests for the DNNL
BYOC backend.

Each case builds a small Relax graph, partitions it with partition_for_dnnl, asserts that a DNNL
subgraph was actually produced (so a broken pattern or predicate can't silently fall back to
native TVM and pass by accident), runs RunCodegen, and compares the DNNL-offloaded result against
a plain TVM build of the same graph. This exercises codegen.cc's serialization and the DNNL JSON
runtime end to end; it isn't a numerics stress test, so shapes stay small and values are plain
uniform random floats.

Deliberately out of scope:
  - ResNetV1Rewrite. It unconditionally rewrites a 1x1-stride-2 -> relu -> 3x3-stride-1 conv
    chain into a 1x1-stride-1 -> relu -> 3x3-stride-2 chain, which is not a numerically
    transparent rewrite (it changes which conv does the downsampling). None of the graphs below
    use that stride/kernel-size combination, so this rewrite never fires here; validating the
    rewrite itself would need a reference computed against the rewritten graph, not the original,
    which is a different kind of test than codegen correctness.
  - swish and mish. dnnl.py's module docstring notes these aren't wired into
    _FUSABLE_ACTIVATIONS yet.
  - QNN fusion beyond the bare qnn.conv2d / qnn.matmul base case (e.g. QNN + bias), which isn't
    registered as a pattern.

Not covered here, and left to test_backend: constant-folding of BindParams-bound weights, runtime
scratchpad reuse, and multi-subgraph module composition.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl
from tvm.relax.expr_functor import visitor

has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)
pytestmark = [pytest.mark.skipif(not has_dnnl, reason="DNNL not enabled.")]

np.random.seed(0)


# Shared helpers


def _rand(shape, low=-1.0, high=1.0, dtype="float32"):
    return np.random.uniform(low, high, size=shape).astype(dtype)


def _var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorType(shape, dtype))


def _build_module(param_specs, body_fn):
    """param_specs: list of (name, shape, dtype). body_fn(bb, *params) returns the output expr.
    Args passed to _build_and_run must be in this same order, since nothing here looks a param's
    name back up after construction."""
    params = [_var(name, shape, dtype) for name, shape, dtype in param_specs]
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = body_fn(bb, *params)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.get()


def _to_tensor(np_array, dev):
    try:
        return tvm.runtime.tensor(np_array, device=dev)
    except TypeError:
        return tvm.runtime.tensor(np_array, dev)


def _build_and_run(mod, args):
    dev = tvm.cpu()
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, dev)
    return vm["main"](*[_to_tensor(a, dev) for a in args]).numpy()


def _run_codegen(mod):
    with tvm.transform.PassContext(opt_level=3):
        return relax.transform.RunCodegen()(mod)


def _has_dnnl_subgraph(mod):
    return any(
        isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
        for func in mod.functions.values()
    )


def _dnnl_composite_names(mod):
    """Names of every DNNL composite function reachable from mod's top-level functions. Uses a
    proper PyExprVisitor rather than a hand-rolled block walk, since MergeCompositeFunctions can
    embed the composite Function directly as a Call's op (TVMScript's printer synthesizes the
    "define then call by name" rendering purely for readability) rather than binding it to a
    separate Var first -- a plain VarBinding-only walk misses that case entirely."""
    names = []

    @visitor
    class _CompositeCollector(relax.PyExprVisitor):
        def visit_function_(self, f):
            if f.attrs is not None and f.attrs.get("Composite") is not None:
                names.append(str(f.attrs.get("Composite")))
            super().visit_function_(f)

    collector = _CompositeCollector()
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            collector.visit_expr(func)
    return names


def _check(mod, args, rtol=1e-4, atol=1e-4, expect_offload=True, prune_subgraphs=True):
    ref = _build_and_run(mod, args)

    partitioned = partition_for_dnnl(mod, run_codegen=False, prune_subgraphs=prune_subgraphs)
    offloaded = _has_dnnl_subgraph(partitioned)
    if expect_offload:
        assert offloaded, "expected at least one subgraph to be offloaded to DNNL"
    else:
        assert not offloaded, "expected no subgraph to be offloaded to DNNL"

    compiled = _run_codegen(partitioned)
    got = _build_and_run(compiled, args)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)


# Bare conv / matmul / layer_norm, across every rank and the transpose variants


def test_dnnl_conv1d():
    data_shape, weight_shape = (1, 3, 16), (8, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv1d(data, weight, padding=1))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_conv2d():
    data_shape, weight_shape = (1, 3, 16, 16), (8, 3, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_conv3d():
    data_shape, weight_shape = (1, 3, 8, 8, 8), (8, 3, 3, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv3d(data, weight, padding=1))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_conv2d_transpose():
    # Weight is in the default IOHW layout (in_channels first). This is a direct regression test
    # for the transpose-conv fallback weight layout fix (dnnl.py previously assumed OIHW, which
    # is regular conv's convention, not deconv's).
    data_shape, weight_shape = (1, 8, 8, 8), (8, 4, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv2d_transpose(data, weight, padding=1))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_conv3d_transpose():
    data_shape, weight_shape = (1, 4, 8, 8, 8), (4, 2, 3, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv3d_transpose(data, weight, padding=1))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_matmul():
    data_shape, weight_shape = (4, 16), (16, 8)

    def body(bb, data, weight):
        return bb.emit(relax.op.matmul(data, weight))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_layer_norm():
    data_shape, gamma_shape, beta_shape = (2, 8, 16), (16,), (16,)

    def body(bb, data, gamma, beta):
        return bb.emit(relax.op.nn.layer_norm(data, gamma, beta, axes=[-1]))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("gamma", gamma_shape, "float32"),
            ("beta", beta_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(gamma_shape), _rand(beta_shape)]
    _check(mod, args)


# Standalone eltwise, clip, and pooling patterns


_ELTWISE_UNARY_OPS = {
    "abs": (relax.op.abs, (-1.0, 1.0)),
    "exp": (relax.op.exp, (-1.0, 1.0)),
    "log": (relax.op.log, (0.1, 2.0)),
    "sqrt": (relax.op.sqrt, (0.1, 2.0)),
    "round": (relax.op.round, (-4.0, 4.0)),
    "relu": (relax.op.nn.relu, (-1.0, 1.0)),
    "tanh": (relax.op.tanh, (-1.0, 1.0)),
    "sigmoid": (relax.op.sigmoid, (-1.0, 1.0)),
}


@pytest.mark.parametrize("op_name", list(_ELTWISE_UNARY_OPS.keys()))
def test_dnnl_eltwise(op_name):
    # prune_subgraphs=False: every op tested here is individually listed in dnnl.py's
    # _DNNL_COMPUTE_OPS, so prune_dnnl_subgraphs would never demote a single-op composite of one
    # of these anyway under the list's current contents -- but that's incidental to what this
    # test is actually checking (pattern match + codegen + runtime correctness for the op), not
    # something it should depend on. Disabling pruning here keeps this test's pass/fail tied only
    # to the pattern/codegen path, not to whether _DNNL_COMPUTE_OPS happens to still list the op.
    op_fn, (low, high) = _ELTWISE_UNARY_OPS[op_name]
    shape = (2, 8)

    def body(bb, data):
        return bb.emit(op_fn(data))

    mod = _build_module([("data", shape, "float32")], body)
    args = [_rand(shape, low, high)]
    _check(mod, args, prune_subgraphs=False)


def test_dnnl_leaky_relu():
    # Has its own alpha attribute, so it's kept separate from the plain-unary loop above.
    # prune_subgraphs=False for the same reason as test_dnnl_eltwise above.
    shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.leakyrelu(data, alpha=0.1))

    mod = _build_module([("data", shape, "float32")], body)
    args = [_rand(shape)]
    _check(mod, args, prune_subgraphs=False)


def test_dnnl_clip():
    # prune_subgraphs=False for the same reason as test_dnnl_eltwise above.
    shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.clip(data, -0.5, 0.5))

    mod = _build_module([("data", shape, "float32")], body)
    args = [_rand(shape, -2.0, 2.0)]
    _check(mod, args, prune_subgraphs=False)


_POOL_CASES = {
    "max_pool1d": (relax.op.nn.max_pool1d, (1, 4, 16)),
    "max_pool2d": (relax.op.nn.max_pool2d, (1, 4, 16, 16)),
    "max_pool3d": (relax.op.nn.max_pool3d, (1, 4, 8, 8, 8)),
    "avg_pool1d": (relax.op.nn.avg_pool1d, (1, 4, 16)),
    "avg_pool2d": (relax.op.nn.avg_pool2d, (1, 4, 16, 16)),
    "avg_pool3d": (relax.op.nn.avg_pool3d, (1, 4, 8, 8, 8)),
}


@pytest.mark.parametrize("op_name", list(_POOL_CASES.keys()))
def test_dnnl_pooling(op_name):
    op_fn, data_shape = _POOL_CASES[op_name]
    rank = len(data_shape) - 2
    pool_size = (2,) * rank
    strides = (2,) * rank

    def body(bb, data):
        return bb.emit(op_fn(data, pool_size=pool_size, strides=strides))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args)


@pytest.mark.parametrize("count_include_pad", [True, False])
def test_dnnl_avg_pool2d_count_include_pad(count_include_pad):
    # Regression test: dnnl_json_runtime.cc's avg-pool dispatch previously hardcoded
    # pooling_avg_exclude_padding regardless of what Relax's count_include_pad attribute said.
    # Padding must be non-zero here for the two algorithms to actually diverge -- with zero
    # padding no window ever touches a padded cell, so include vs exclude give identical results
    # either way and this wouldn't exercise the bug at all.
    data_shape = (1, 4, 8, 8)

    def body(bb, data):
        return bb.emit(
            relax.op.nn.avg_pool2d(
                data,
                pool_size=(3, 3),
                strides=(2, 2),
                padding=(1, 1),
                count_include_pad=count_include_pad,
            )
        )

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args)


def test_dnnl_pad_avg_pool2d_fusion():
    # Exercises rewrite_pad_avg_pool2d: a separate zero-constant nn.pad immediately followed by an
    # avg_pool2d (with its own padding=0) gets folded into a single avg_pool2d call with the pad
    # width moved into its own padding attribute. The rewritten call is built with
    # count_include_pad=True specifically to reproduce the original two-step graph's implicit
    # "pad first, then pool with padding=0" divisor -- this is the exact case that depends on
    # count_include_pad actually being honored by the runtime.
    data_shape = (1, 4, 8, 8)

    def body(bb, data):
        padded = bb.emit(
            relax.op.nn.pad(
                data, pad_width=[0, 0, 0, 0, 1, 1, 1, 1], pad_mode="constant", pad_value=0.0
            )
        )
        return bb.emit(relax.op.nn.avg_pool2d(padded, pool_size=(3, 3), strides=(2, 2)))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]

    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert "dnnl.avg_pool2d" in _dnnl_composite_names(partitioned)

    _check(mod, args)


# Remaining standalone patterns: softmax, add, multiply, batch_norm, and true global average pool


def test_dnnl_softmax():
    data_shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.softmax(data, axis=-1))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args)


def test_dnnl_add():
    # prune_subgraphs=False for the same reason as test_dnnl_eltwise above.
    shape = (2, 8)

    def body(bb, a, b):
        return bb.emit(relax.op.add(a, b))

    mod = _build_module([("a", shape, "float32"), ("b", shape, "float32")], body)
    args = [_rand(shape), _rand(shape)]
    _check(mod, args, prune_subgraphs=False)


def test_dnnl_multiply():
    # prune_subgraphs=False for the same reason as test_dnnl_eltwise above.
    shape = (2, 8)

    def body(bb, a, b):
        return bb.emit(relax.op.multiply(a, b))

    mod = _build_module([("a", shape, "float32"), ("b", shape, "float32")], body)
    args = [_rand(shape), _rand(shape)]
    _check(mod, args, prune_subgraphs=False)


def test_dnnl_batch_norm():
    # training=False is required: relax.nn.batch_norm defaults to training=True, under which the
    # op normalizes using live batch statistics of the data and only uses mean/var to produce
    # updated running averages -- a fundamentally different computation than what the DNNL
    # runtime executes (batch_normalization_forward built with use_global_stats, which always
    # normalizes using the supplied mean/var directly). With training=True this test would still
    # get offloaded (the bare dnnl.batch_norm pattern has no checker rejecting it) but would fail
    # numerically, since native and DNNL would be computing two different things while looking
    # like a same-graph comparison.
    #
    # With training=False, _unwrap_batch_norm_tuple_output still applies regardless of whether
    # DecomposeOpsForInference decomposes this call at all: relax.nn.batch_norm returns a tuple,
    # FuseOpsByPattern can't anchor a match on the TupleGetItem that follows the call, so the
    # composite needs retyping to return element 0 directly either way.
    c = 4
    data_shape = (1, c, 8, 8)
    param_shape = (c,)

    def body(bb, data, gamma, beta, mean, var):
        bn = bb.emit(relax.op.nn.batch_norm(data, gamma, beta, mean, var, axis=1, training=False))
        return bb.emit(bn[0])

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("gamma", param_shape, "float32"),
            ("beta", param_shape, "float32"),
            ("mean", param_shape, "float32"),
            ("var", param_shape, "float32"),
        ],
        body,
    )
    args = [
        _rand(data_shape),
        _rand(param_shape, 0.5, 1.5),
        _rand(param_shape, -0.2, 0.2),
        _rand(param_shape, -0.1, 0.1),
        _rand(param_shape, 0.5, 1.5),
    ]
    _check(mod, args)


def test_dnnl_global_avg_pool2d():
    data_shape = (1, 4, 8, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.adaptive_avg_pool2d(data, output_size=(1, 1)))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args)


# Fused bias + activation, for every op in _FUSABLE_OPS x every entry in _FUSABLE_ACTIVATIONS.
# Sampled on conv2d and matmul; the other _FUSABLE_OPS ranks already get bare-pattern coverage
# above; the interesting question this section answers is whether bias/activation fusion itself
# (as opposed to which conv rank it's attached to) is correct.


_FUSABLE_ACTIVATION_CASES = {
    "none": None,
    "relu": relax.op.nn.relu,
    "sigmoid": relax.op.sigmoid,
    "gelu": relax.op.nn.gelu,
    "tanh": relax.op.tanh,
}


@pytest.mark.parametrize("activation_name", list(_FUSABLE_ACTIVATION_CASES.keys()))
def test_dnnl_conv2d_bias_activation(activation_name):
    activation_fn = _FUSABLE_ACTIVATION_CASES[activation_name]
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)

    def body(bb, data, weight, bias):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        return bb.emit(activation_fn(biased)) if activation_fn is not None else biased

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape)]
    _check(mod, args)


@pytest.mark.parametrize("activation_name", list(_FUSABLE_ACTIVATION_CASES.keys()))
def test_dnnl_matmul_bias_activation(activation_name):
    activation_fn = _FUSABLE_ACTIVATION_CASES[activation_name]
    data_shape, weight_shape, bias_shape = (4, 16), (16, 8), (8,)

    def body(bb, data, weight, bias):
        mm = bb.emit(relax.op.matmul(data, weight))
        biased = bb.emit(relax.op.add(mm, bias))
        return bb.emit(activation_fn(biased)) if activation_fn is not None else biased

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape)]
    _check(mod, args)


# Fused clip, with and without bias


def test_dnnl_conv2d_clip():
    data_shape, weight_shape = (1, 4, 16, 16), (8, 4, 3, 3)

    def body(bb, data, weight):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        return bb.emit(relax.op.clip(conv, -1.0, 1.0))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    _check(mod, args)


def test_dnnl_conv2d_bias_clip():
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)

    def body(bb, data, weight, bias):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        return bb.emit(relax.op.clip(biased, 0.0, 6.0))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape)]
    _check(mod, args)


# Bias plus residual-sum fusion, with and without a trailing relu


def test_dnnl_conv2d_bias_sum():
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)
    residual_shape = (1, 8, 16, 16)

    def body(bb, data, weight, bias, residual):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        return bb.emit(relax.op.add(biased, residual))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
            ("residual", residual_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape), _rand(residual_shape)]
    _check(mod, args)


def test_dnnl_conv2d_bias_sum_relu():
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)
    residual_shape = (1, 8, 16, 16)

    def body(bb, data, weight, bias, residual):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        summed = bb.emit(relax.op.add(biased, residual))
        return bb.emit(relax.op.nn.relu(summed))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
            ("residual", residual_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape), _rand(residual_shape)]
    _check(mod, args)


def test_dnnl_matmul_bias_sum():
    # matmul only gets the no-relu sum variant; see _sum_patterns' docstring in dnnl.py.
    data_shape, weight_shape, bias_shape, residual_shape = (4, 16), (16, 8), (8,), (4, 8)

    def body(bb, data, weight, bias, residual):
        mm = bb.emit(relax.op.matmul(data, weight))
        biased = bb.emit(relax.op.add(mm, bias))
        return bb.emit(relax.op.add(biased, residual))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
            ("residual", residual_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape), _rand(residual_shape)]
    _check(mod, args)


def test_dnnl_conv2d_bias_sum_relu_pattern_priority():
    # Regression test for a pattern-priority bug: dnnl.conv2d_bias_sum (the shorter, less
    # specific composite name) previously won FuseOpsByPattern's greedy match over
    # dnnl.conv2d_bias_sum_relu even when the graph has the trailing relu that the longer pattern
    # requires. _ordered_dnnl_patterns sorts by composite-name length to fix this; verify the
    # larger, more specific composite is the one actually selected for this graph.
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)
    residual_shape = (1, 8, 16, 16)

    def body(bb, data, weight, bias, residual):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        summed = bb.emit(relax.op.add(biased, residual))
        return bb.emit(relax.op.nn.relu(summed))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
            ("residual", residual_shape, "float32"),
        ],
        body,
    )

    partitioned = partition_for_dnnl(mod, run_codegen=False)
    composite_names = _dnnl_composite_names(partitioned)
    assert "dnnl.conv2d_bias_sum_relu" in composite_names
    assert "dnnl.conv2d_bias_sum" not in composite_names


# Negative / rejection paths: these must NOT be offloaded to DNNL, and the graph must still
# produce a correct result via the native TVM fallback.


def test_dnnl_int64_matmul_not_offloaded():
    # dnnl_conv_checker's shared int64-rejection guard, exercised on the bare matmul pattern.
    data_shape, weight_shape = (4, 16), (16, 8)

    def body(bb, data, weight):
        return bb.emit(relax.op.matmul(data, weight))

    mod = _build_module([("data", data_shape, "int64"), ("weight", weight_shape, "int64")], body)
    args = [
        np.random.randint(-8, 8, size=data_shape).astype("int64"),
        np.random.randint(-8, 8, size=weight_shape).astype("int64"),
    ]
    _check(mod, args, expect_offload=False)


def test_dnnl_int64_eltwise_not_offloaded():
    # dnnl_eltwise_checker's int64 rejection, exercised on the standalone relu pattern.
    shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.relu(data))

    mod = _build_module([("data", shape, "int64")], body)
    args = [np.random.randint(-8, 8, size=shape).astype("int64")]
    _check(mod, args, expect_offload=False)


def test_dnnl_max_pool2d_ceil_mode_not_offloaded():
    # dnnl_pooling_checker explicitly rejects ceil_mode=True, since oneDNN's pooling primitive
    # doesn't expose the same output-size rounding TVM's ceil_mode does.
    data_shape = (1, 4, 15, 15)

    def body(bb, data):
        return bb.emit(
            relax.op.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2), ceil_mode=True)
        )

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args, expect_offload=False)


def test_dnnl_adaptive_avg_pool2d_non_global_not_offloaded():
    # dnnl_global_avg_pool2d_checker only accepts output_size == (1, 1); a genuinely adaptive pool
    # with a larger output can't be expressed by oneDNN's fixed-window pooling primitive, so it
    # must fall back to TVM's native implementation.
    data_shape = (1, 4, 8, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.adaptive_avg_pool2d(data, output_size=(4, 4)))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    _check(mod, args, expect_offload=False)


# QNN base case. legalize_qnn_op_for_dnnl algebraically rewrites the terminal
# quantize(conv_out, out_scale, out_zp) into dequantize(conv_out, 1/out_scale, 0), so a plain
# native run of the ORIGINAL graph (which still has a real int8 quantize op) isn't a valid
# reference for the DNNL-offloaded path -- they're different computations, not just different
# backends for the same one. Instead the reference below is built by hand-applying that exact
# rewrite as its own small float32-only Relax graph and running it natively; this checks DNNL's
# actual numeric output against the same algebra the legalization pass is supposed to implement,
# without reimplementing convolution or matmul in numpy.


def _qnn_conv2d_case():
    data_shape, weight_shape = (1, 4, 8, 8), (8, 4, 3, 3)
    data_q_np = np.random.randint(-32, 32, size=data_shape).astype("int8")
    weight_q_np = np.random.randint(-32, 32, size=weight_shape).astype("int8")
    data_scale_np = np.array(0.05, dtype="float32")
    data_zp_np = np.array(0, dtype="int8")
    weight_scale_np = np.array(0.02, dtype="float32")
    weight_zp_np = np.array(0, dtype="int8")
    out_scale_np = np.array(0.1, dtype="float32")
    out_zp_np = np.array(0, dtype="int8")  # must be exactly 0: see _try_requantize_consts

    def body(bb, data_q):
        data_dq = bb.emit(
            relax.op.dequantize(
                data_q,
                relax.const(data_scale_np),
                relax.const(data_zp_np),
                axis=1,
                out_dtype="float32",
            )
        )
        weight_dq = bb.emit(
            relax.op.dequantize(
                relax.const(weight_q_np),
                relax.const(weight_scale_np),
                relax.const(weight_zp_np),
                axis=0,
                out_dtype="float32",
            )
        )
        conv = bb.emit(relax.op.nn.conv2d(data_dq, weight_dq, padding=(1, 1)))
        return bb.emit(
            relax.op.quantize(
                conv, relax.const(out_scale_np), relax.const(out_zp_np), axis=1, out_dtype="int8"
            )
        )

    mod = _build_module([("data_q", data_shape, "int8")], body)

    def ref_body(bb, data_q):
        data_dq = bb.emit(
            relax.op.dequantize(
                data_q,
                relax.const(data_scale_np),
                relax.const(data_zp_np),
                axis=1,
                out_dtype="float32",
            )
        )
        weight_fp_np = weight_scale_np * (weight_q_np.astype("float32") - weight_zp_np)
        conv = bb.emit(relax.op.nn.conv2d(data_dq, relax.const(weight_fp_np), padding=(1, 1)))
        return bb.emit(relax.op.multiply(conv, relax.const(np.float32(1.0 / out_scale_np))))

    ref_mod = _build_module([("data_q", data_shape, "int8")], ref_body)
    return mod, ref_mod, [data_q_np]


def _qnn_matmul_case():
    data_shape, weight_shape = (4, 16), (16, 8)
    data_q_np = np.random.randint(-32, 32, size=data_shape).astype("int8")
    weight_q_np = np.random.randint(-32, 32, size=weight_shape).astype("int8")
    data_scale_np = np.array(0.05, dtype="float32")
    data_zp_np = np.array(0, dtype="int8")
    weight_scale_np = np.array(0.02, dtype="float32")
    weight_zp_np = np.array(0, dtype="int8")
    out_scale_np = np.array(0.1, dtype="float32")
    out_zp_np = np.array(0, dtype="int8")

    def body(bb, data_q):
        data_dq = bb.emit(
            relax.op.dequantize(
                data_q,
                relax.const(data_scale_np),
                relax.const(data_zp_np),
                axis=-1,
                out_dtype="float32",
            )
        )
        weight_dq = bb.emit(
            relax.op.dequantize(
                relax.const(weight_q_np),
                relax.const(weight_scale_np),
                relax.const(weight_zp_np),
                axis=-1,
                out_dtype="float32",
            )
        )
        mm = bb.emit(relax.op.matmul(data_dq, weight_dq))
        return bb.emit(
            relax.op.quantize(
                mm, relax.const(out_scale_np), relax.const(out_zp_np), axis=-1, out_dtype="int8"
            )
        )

    mod = _build_module([("data_q", data_shape, "int8")], body)

    def ref_body(bb, data_q):
        data_dq = bb.emit(
            relax.op.dequantize(
                data_q,
                relax.const(data_scale_np),
                relax.const(data_zp_np),
                axis=-1,
                out_dtype="float32",
            )
        )
        weight_fp_np = weight_scale_np * (weight_q_np.astype("float32") - weight_zp_np)
        mm = bb.emit(relax.op.matmul(data_dq, relax.const(weight_fp_np)))
        return bb.emit(relax.op.multiply(mm, relax.const(np.float32(1.0 / out_scale_np))))

    ref_mod = _build_module([("data_q", data_shape, "int8")], ref_body)
    return mod, ref_mod, [data_q_np]


def test_dnnl_qnn_conv2d():
    mod, ref_mod, args = _qnn_conv2d_case()

    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert "dnnl.qnn.conv2d" in _dnnl_composite_names(partitioned)

    compiled = _run_codegen(partitioned)
    got = _build_and_run(compiled, args)
    ref = _build_and_run(ref_mod, args)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_dnnl_qnn_matmul():
    mod, ref_mod, args = _qnn_matmul_case()

    partitioned = partition_for_dnnl(mod, run_codegen=False)
    assert "dnnl.qnn.matmul" in _dnnl_composite_names(partitioned)

    compiled = _run_codegen(partitioned)
    got = _build_and_run(compiled, args)
    ref = _build_and_run(ref_mod, args)
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)


# Sanity check of the public one-call entry point (alter_layout, prune_subgraphs, and
# run_codegen all left at their defaults), independent of the two-step split _check() uses
# internally to assert offload happened.


def test_partition_for_dnnl_default_entry_point():
    data_shape, weight_shape, bias_shape = (1, 4, 16, 16), (8, 4, 3, 3), (8, 1, 1)

    def body(bb, data, weight, bias):
        conv = bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))
        biased = bb.emit(relax.op.add(conv, bias))
        return bb.emit(relax.op.nn.relu(biased))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape)]

    ref = _build_and_run(mod, args)
    compiled = partition_for_dnnl(mod)
    got = _build_and_run(compiled, args)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
