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

Each case builds a small Relax graph covering one of dnnl.py's normal, non-edge-case patterns,
runs it twice, once through plain TVM and once through partition_for_dnnl -> RunCodegen (which
exercises codegen.cc and the DNNL runtime), and checks the two outputs match. This is a smoke
test for the whole pipeline, not a numerics stress test: shapes are small, values are plain
uniform random floats, and there's no attempt to cover every op variant or dtype.

Not covered here, worth adding separately: conv1d/conv3d and the transpose conv variants, the
QNN base case (a fair native-vs-DNNL comparison for it isn't a small addition, since
legalize_qnn_op_for_dnnl's terminal quantize -> dequantize rewrite changes the traced graph's
output dtype between the two paths), and swish/mish since dnnl.py doesn't wire those up yet.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

has_dnnl = tvm.get_global_func("relax.ext.dnnl", True)
pytestmark = [pytest.mark.skipif(not has_dnnl, reason="DNNL not enabled.")]

np.random.seed(0)


def _rand(shape, low=-1.0, high=1.0, dtype="float32"):
    return np.random.uniform(low, high, size=shape).astype(dtype)


def _var(name, shape, dtype="float32"):
    return relax.Var(name, relax.TensorType(shape, dtype))


def _build_module(param_specs, body_fn):
    """param_specs: list of (name, shape, dtype). body_fn(bb, *params) returns the output expr.
    Args passed to _run must be in this same order -- nothing here looks a param's name back up
    after construction."""
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


def _run(mod, args, use_dnnl):
    if use_dnnl:
        mod = partition_for_dnnl(mod)
    dev = tvm.cpu()
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, dev)
    return vm["main"](*[_to_tensor(a, dev) for a in args]).numpy()


def _check(mod, args, rtol=1e-4, atol=1e-4):
    ref = _run(mod, args, use_dnnl=False)
    got = _run(mod, args, use_dnnl=True)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)


# Case builders. Each returns (mod, args), where args is a plain list of numpy arrays in the
# SAME order the params were declared in the case's _build_module call -- no name lookup.
# Conv bias shapes are (oc, 1, 1): NCHW broadcasting aligns from the right, so (oc, 1, 1)
# broadcasts correctly against a (n, oc, h, w) conv output, matching the convention already
# used elsewhere in this codebase.


def case_conv2d():
    data_shape, weight_shape = (1, 3, 16, 16), (8, 3, 3, 3)

    def body(bb, data, weight):
        return bb.emit(relax.op.nn.conv2d(data, weight, padding=(1, 1)))

    mod = _build_module(
        [("data", data_shape, "float32"), ("weight", weight_shape, "float32")], body
    )
    args = [_rand(data_shape), _rand(weight_shape)]
    return mod, args


def case_conv2d_bias_relu():
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
    return mod, args


def case_conv2d_bias_sum_relu():
    # Residual fusion: matches dnnl.conv2d_bias_sum_relu. This is the pattern behind the
    # "with_relu=False listed before with_relu=True" priority fix, so it's worth keeping in the
    # suite even as a normal case.
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
    return mod, args


def case_conv2d_bias_clip():
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
    return mod, args


def case_matmul_bias():
    data_shape, weight_shape, bias_shape = (4, 16), (16, 8), (8,)

    def body(bb, data, weight, bias):
        mm = bb.emit(relax.op.matmul(data, weight))
        return bb.emit(relax.op.add(mm, bias))

    mod = _build_module(
        [
            ("data", data_shape, "float32"),
            ("weight", weight_shape, "float32"),
            ("bias", bias_shape, "float32"),
        ],
        body,
    )
    args = [_rand(data_shape), _rand(weight_shape), _rand(bias_shape)]
    return mod, args


def case_matmul_bias_sum():
    # Residual matmul: matches dnnl.matmul_bias_sum, the no-relu-only variant per _sum_patterns'
    # scope. Also the exact composite shape that exercises MatMul()'s residual-sum wiring in
    # dnnl_json_runtime.cc.
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
    return mod, args


def case_max_pool2d():
    data_shape = (1, 4, 16, 16)

    def body(bb, data):
        return bb.emit(relax.op.nn.max_pool2d(data, pool_size=(2, 2), strides=(2, 2)))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    return mod, args


def case_avg_pool2d():
    data_shape = (1, 4, 16, 16)

    def body(bb, data):
        return bb.emit(relax.op.nn.avg_pool2d(data, pool_size=(2, 2), strides=(2, 2)))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    return mod, args


def case_relu():
    data_shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.relu(data))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    return mod, args


def case_softmax():
    data_shape = (2, 8)

    def body(bb, data):
        return bb.emit(relax.op.nn.softmax(data, axis=-1))

    mod = _build_module([("data", data_shape, "float32")], body)
    args = [_rand(data_shape)]
    return mod, args


def case_layer_norm():
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
    return mod, args


CASES = {
    "conv2d": case_conv2d,
    "conv2d_bias_relu": case_conv2d_bias_relu,
    "conv2d_bias_sum_relu": case_conv2d_bias_sum_relu,
    "conv2d_bias_clip": case_conv2d_bias_clip,
    "matmul_bias": case_matmul_bias,
    "matmul_bias_sum": case_matmul_bias_sum,
    "max_pool2d": case_max_pool2d,
    "avg_pool2d": case_avg_pool2d,
    "relu": case_relu,
    "softmax": case_softmax,
    "layer_norm": case_layer_norm,
}


@pytest.mark.parametrize("case_name", list(CASES.keys()))
def test_dnnl_matches_native(case_name):
    mod, args = CASES[case_name]()
    _check(mod, args)


if __name__ == "__main__":
    tvm.testing.main()
