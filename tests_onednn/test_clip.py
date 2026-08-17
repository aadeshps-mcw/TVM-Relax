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

import numpy as np

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl


def _run_conv2d_clip_test(name, with_bias, clip_min=-0.3, clip_max=0.3, atol=1e-4, rtol=1e-4):
    data_shape = (1, 3, 8, 8)
    weight_shape = (4, 3, 3, 3)

    weight_np = np.random.uniform(size=weight_shape).astype("float32")
    bias_np = np.arange(4).astype("float32") - 1.5

    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    weight_const = relax.const(weight_np, "float32")

    with builder.function("main", [data]):
        with builder.dataflow():
            conv = builder.emit(relax.op.nn.conv2d(data, weight_const, padding=(1, 1)))
            cur = conv
            if with_bias:
                bias_const = relax.const(bias_np, "float32")
                bias_rs = builder.emit(relax.op.reshape(bias_const, (1, 4, 1, 1)))
                cur = builder.emit(relax.op.add(cur, bias_rs))
            cur = builder.emit(relax.op.clip(cur, clip_min, clip_max))
            out = builder.emit_output(cur)
        builder.emit_func_output(out)
    mod = builder.get()

    data_np = np.random.uniform(low=-3.0, high=3.0, size=data_shape).astype("float32")
    tvm_args = [tvm.runtime.tensor(data_np)]

    target = tvm.target.Target("llvm")
    ref_ex = relax.build(mod, target=target)
    ref_out = relax.VirtualMachine(ref_ex, tvm.cpu())["main"](*tvm_args).numpy()

    assert np.isclose(ref_out.min(), clip_min, atol=1e-3), (
        f"Reference output min {ref_out.min()} did not hit clip_min {clip_min}"
    )
    assert np.isclose(ref_out.max(), clip_max, atol=1e-3), (
        f"Reference output max {ref_out.max()} did not hit clip_max {clip_max}"
    )

    partitioned = partition_for_dnnl(mod)
    print(partitioned)

    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    out = relax.VirtualMachine(ex, tvm.cpu())["main"](*tvm_args).numpy()

    np.testing.assert_allclose(out, ref_out, rtol=rtol, atol=atol)
    print(f"{name} OK")


def test_conv2d_clip():
    _run_conv2d_clip_test("conv2d_clip", with_bias=False)


def test_conv2d_bias_clip():
    _run_conv2d_clip_test("conv2d_bias_clip", with_bias=True)


if __name__ == "__main__":
    test_conv2d_clip()
    test_conv2d_bias_clip()
