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
Smoke test for the new DNNL "standalone op" patterns added to
_standalone_patterns() in dnnl_pattern.py (relu, sigmoid, tanh, abs, exp,
log, sqrt, round, leaky_relu, clip, softmax, add, multiply, pooling,
batch_norm).

For each op this:
  1. Builds a tiny Relax module containing just that op.
  2. Runs it un-partitioned (pure TVM) as a reference.
  3. Runs partition_for_dnnl() on it and asserts a "dnnl.<op>" composite
     function actually got created -- this is the check that catches a
     pattern silently failing to match (dead code path bug class).
  4. Builds + runs the DNNL-partitioned module and compares numerics
     against the un-partitioned run.
  5. Where a trivial NumPy reference exists, also checks against that.

Run with:
    python test_dnnl_standalone_ops.py

Requires a TVM build with the DNNL BYOC backend compiled in (the same
build you just fixed dnnl_json_runtime.cc / codegen.cc for).
"""

import numpy as np

import tvm
import tvm.runtime
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl


def build_and_run(mod, args, target="llvm", dev=None):
    dev = dev or tvm.runtime.cpu()
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)
    out = vm["main"](*[tvm.runtime.tensor(a, device=dev) for a in args])
    if hasattr(out, "numpy"):
        return out.numpy()
    # Multi-output case: assume an iterable of tensor-like objects, each with .numpy()
    return [o.numpy() for o in out]


def has_dnnl_composite(mod, expected_name_substr):
    from tvm.relax.expr_functor import visitor

    found = {"flag": False}

    @visitor
    class _CompositeFinder(relax.PyExprVisitor):
        def visit_function_(self, func):
            if func.attrs is not None:
                composite = func.attrs.get("Composite")
                if composite is not None and expected_name_substr in str(composite):
                    found["flag"] = True
            super().visit_function_(func)

    for gvar, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        _CompositeFinder().visit_expr(func)
        if found["flag"]:
            return True
    return False


def check_op(name, build_mod_fn, np_args, np_ref_fn=None, rtol=1e-4, atol=1e-4):
    print(f"--- {name} ---")
    mod = build_mod_fn()

    ref = build_and_run(mod, np_args)

    # Partition WITHOUT codegen first, so the Composite-tagged function is
    # still present in the IR to check against.
    dnnl_mod = partition_for_dnnl(mod, run_codegen=False)
    assert has_dnnl_composite(dnnl_mod, f"dnnl.{name}"), (
        f"{name}: no 'dnnl.{name}' composite function found after partitioning -- "
        f"the pattern did not match (check _standalone_patterns() / ResolveRootCall)"
    )

    # Now actually run codegen to get a buildable module.
    with tvm.transform.PassContext(opt_level=3):
        built_mod = relax.transform.RunCodegen()(dnnl_mod)

    got = build_and_run(built_mod, np_args)

    if isinstance(ref, list):
        for r, g in zip(ref, got):
            np.testing.assert_allclose(g, r, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)

    if np_ref_fn is not None:
        expected = np_ref_fn(*np_args)
        actual = got[0] if isinstance(got, list) else got
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    print(f"{name}: OK")


def make_unary_mod(op_fn, shape, dtype="float32"):
    def _build():
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorType(shape, dtype))
        with bb.function("main", [x]):
            with bb.dataflow():
                y = bb.emit(op_fn(x))
                out = bb.emit_output(y)
            bb.emit_func_output(out)
        return bb.get()

    return _build


def make_binary_mod(op_fn, shape, dtype="float32"):
    def _build():
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorType(shape, dtype))
        y = relax.Var("y", relax.TensorType(shape, dtype))
        with bb.function("main", [x, y]):
            with bb.dataflow():
                z = bb.emit(op_fn(x, y))
                out = bb.emit_output(z)
            bb.emit_func_output(out)
        return bb.get()

    return _build


def main():
    shape = (2, 8)
    x = np.random.uniform(-2, 2, size=shape).astype("float32")
    x_pos = np.random.uniform(0.1, 2, size=shape).astype("float32")  # for log/sqrt domains
    y = np.random.uniform(-2, 2, size=shape).astype("float32")

    # ---------------- elementwise ----------------
    check_op("relu", make_unary_mod(relax.op.nn.relu, shape), [x], lambda a: np.maximum(a, 0))
    check_op(
        "sigmoid", make_unary_mod(relax.op.sigmoid, shape), [x], lambda a: 1 / (1 + np.exp(-a))
    )
    check_op("tanh", make_unary_mod(relax.op.tanh, shape), [x], lambda a: np.tanh(a))
    check_op("abs", make_unary_mod(relax.op.abs, shape), [x], lambda a: np.abs(a))
    check_op("exp", make_unary_mod(relax.op.exp, shape), [x], lambda a: np.exp(a))
    check_op("log", make_unary_mod(relax.op.log, shape), [x_pos], lambda a: np.log(a))
    check_op("sqrt", make_unary_mod(relax.op.sqrt, shape), [x_pos], lambda a: np.sqrt(a))
    check_op("round", make_unary_mod(relax.op.round, shape), [x], lambda a: np.round(a))

    def _leaky_relu(x_):
        return relax.op.nn.leakyrelu(x_, alpha=0.1)

    check_op(
        "leaky_relu", make_unary_mod(_leaky_relu, shape), [x], lambda a: np.where(a > 0, a, 0.1 * a)
    )

    def _clip(x_):
        return relax.op.clip(x_, -1.0, 1.0)

    check_op("clip", make_unary_mod(_clip, shape), [x], lambda a: np.clip(a, -1.0, 1.0))

    def _softmax(x_):
        return relax.op.nn.softmax(x_, axis=-1)

    def _softmax_ref(a):
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    check_op("softmax", make_unary_mod(_softmax, shape), [x], _softmax_ref)

    # ---------------- binary ----------------
    check_op("add", make_binary_mod(relax.op.add, shape), [x, y], lambda a, b: a + b)
    check_op("multiply", make_binary_mod(relax.op.multiply, shape), [x, y], lambda a, b: a * b)

    # ---------------- pooling (NCHW) ----------------
    pool_shape = (1, 4, 8, 8)
    xp = np.random.uniform(-1, 1, size=pool_shape).astype("float32")

    def _max_pool_mod():
        bb = relax.BlockBuilder()
        xin = relax.Var("x", relax.TensorType(pool_shape, "float32"))
        with bb.function("main", [xin]):
            with bb.dataflow():
                y_ = bb.emit(relax.op.nn.max_pool2d(xin, pool_size=(2, 2), strides=(2, 2)))
                out = bb.emit_output(y_)
            bb.emit_func_output(out)
        return bb.get()

    check_op("max_pool2d", _max_pool_mod, [xp])

    def _avg_pool_mod():
        bb = relax.BlockBuilder()
        xin = relax.Var("x", relax.TensorType(pool_shape, "float32"))
        with bb.function("main", [xin]):
            with bb.dataflow():
                y_ = bb.emit(relax.op.nn.avg_pool2d(xin, pool_size=(2, 2), strides=(2, 2)))
                out = bb.emit_output(y_)
            bb.emit_func_output(out)
        return bb.get()

    check_op("avg_pool2d", _avg_pool_mod, [xp])

    # ---------------- batch norm ----------------
    # NOTE: this is the op flagged as a caveat -- relax.nn.batch_norm returns a
    # 3-tuple (out, running_mean, running_var) and the composite's JSON node is
    # currently hardcoded to num_outputs_ = 1 in codegen.cc. Extracting index 0
    # via TupleGetItem below is the case that's expected to work; if this test
    # fails here specifically, that's the tuple-output issue to go dig into.
    bn_shape = (1, 4, 4, 4)
    xb = np.random.uniform(-1, 1, size=bn_shape).astype("float32")
    gamma = np.random.uniform(0.5, 1.5, size=(4,)).astype("float32")
    beta = np.random.uniform(-0.5, 0.5, size=(4,)).astype("float32")
    mean = np.random.uniform(-0.5, 0.5, size=(4,)).astype("float32")
    var = np.random.uniform(0.5, 1.5, size=(4,)).astype("float32")

    def _bn_mod():
        bb = relax.BlockBuilder()
        xin = relax.Var("x", relax.TensorType(bn_shape, "float32"))
        g = relax.Var("gamma", relax.TensorType((4,), "float32"))
        b = relax.Var("beta", relax.TensorType((4,), "float32"))
        m = relax.Var("mean", relax.TensorType((4,), "float32"))
        v = relax.Var("var", relax.TensorType((4,), "float32"))
        with bb.function("main", [xin, g, b, m, v]):
            with bb.dataflow():
                bn_out = bb.emit(relax.op.nn.batch_norm(xin, g, b, m, v, axis=1, training=False))
                y0 = bb.emit(relax.TupleGetItem(bn_out, 0))
                out = bb.emit_output(y0)
            bb.emit_func_output(out)
        return bb.get()

    check_op("batch_norm", _bn_mod, [xb, gamma, beta, mean, var])

    print("\nAll standalone DNNL op tests passed.")


if __name__ == "__main__":
    main()
