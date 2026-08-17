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
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl

_requires_dnnl = (
    tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", allow_missing=True) is not None
)
pytestmark = pytest.mark.skipif(
    not _requires_dnnl, reason="DNNL JSON runtime not registered in this build"
)


def test_matmul():
    data_shape = (4, 16)
    weight_shape = (16, 8)

    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    weight = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
    with builder.function("main", [data, weight]):
        with builder.dataflow():
            out = builder.emit(relax.op.matmul(data, weight))
            out = builder.emit_output(out)
        builder.emit_func_output(out)
    mod = builder.get()

    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")
    tvm_args = [tvm.runtime.tensor(data_np), tvm.runtime.tensor(weight_np)]

    target = tvm.target.Target("llvm")
    ref_ex = relax.build(mod, target=target)
    ref_out = relax.VirtualMachine(ref_ex, tvm.cpu())["main"](*tvm_args).numpy()

    partitioned = partition_for_dnnl(mod)
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)
    ex = relax.build(codegen_mod, target=target)
    out = relax.VirtualMachine(ex, tvm.cpu())["main"](*tvm_args).numpy()

    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    print("matmul OK")


if __name__ == "__main__":
    test_matmul()
