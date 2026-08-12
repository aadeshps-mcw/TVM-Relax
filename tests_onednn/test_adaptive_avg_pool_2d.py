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
import tvm.testing
from tvm import relax
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl


def _compile_and_compare(mod, inputs_np, rtol=1e-5, atol=1e-5):
    """
    1. Runs the native TVM CPU reference.
    2. Partitions for DNNL and verifies at least one function was offloaded.
    3. Builds the DNNL graph and runs it.
    4. Compares the outputs.
    """
    target = tvm.target.Target("llvm")
    tvm_args = [tvm.runtime.tensor(x) for x in inputs_np]

    # 1. Native TVM Reference
    ex_ref = relax.build(mod, target=target)
    vm_ref = relax.VirtualMachine(ex_ref, tvm.cpu())
    out_ref = vm_ref["main"](*tvm_args).numpy()

    # 2. Partition and verify offload happened
    partitioned = partition_for_dnnl(mod, run_codegen=False)

    dnnl_funcs = [
        gv.name_hint
        for gv, func in partitioned.functions.items()
        if func.attrs is not None and func.attrs.get("Codegen") == "dnnl"
    ]
    assert len(dnnl_funcs) >= 1, (
        "Expected ops to be offloaded, but got 0 subgraphs. "
        "Check that _DNNL_COMPUTE_OPS contains the op!"
    )

    # 3. Lower and Build for DNNL
    with tvm.transform.PassContext(opt_level=3):
        codegen_mod = relax.transform.RunCodegen(target_options={"dnnl": {}})(partitioned)

    ex_dnnl = relax.build(codegen_mod, target=target)
    vm_dnnl = relax.VirtualMachine(ex_dnnl, tvm.cpu())
    out_dnnl = vm_dnnl["main"](*tvm_args).numpy()

    # 4. Compare numerics
    np.testing.assert_allclose(out_dnnl, out_ref, rtol=rtol, atol=atol)


def test_global_avg_pool2d():
    """Verify adaptive_avg_pool2d (global pooling) avoids pruning and computes correctly."""
    builder = relax.BlockBuilder()
    data_shape = (1, 16, 8, 8)

    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    with builder.function("main", [data]):
        with builder.dataflow():
            # Use the correct TVM Relax operator for global pooling
            gv = builder.emit(relax.op.nn.adaptive_avg_pool2d(data, output_size=(1, 1)))
            out = builder.emit_output(gv)
        builder.emit_func_output(out)

    mod = builder.get()

    # Generate random test data
    np.random.seed(0)
    data_np = np.random.uniform(size=data_shape).astype("float32")

    _compile_and_compare(mod, [data_np])


def test_batch_matmul():
    """Verify batch_matmul avoids pruning and computes correctly."""
    builder = relax.BlockBuilder()

    # Base shapes for batched matmul (Batch=2, M=16, K=32, N=64)
    B, M, K, N = 2, 16, 32, 64

    shape_a = (B, M, K)
    shape_b = (B, K, N)

    data_a = relax.Var("a", relax.TensorType(shape_a, "float32"))
    data_b = relax.Var("b", relax.TensorType(shape_b, "float32"))

    try:
        matmul_op = relax.op.nn.batch_matmul
    except AttributeError:
        matmul_op = relax.op.matmul

    with builder.function("main", [data_a, data_b]):
        with builder.dataflow():
            gv = builder.emit(matmul_op(data_a, data_b, out_dtype="float32"))
            out = builder.emit_output(gv)
        builder.emit_func_output(out)

    mod = builder.get()

    np.random.seed(1)
    a_np = np.random.uniform(-1, 1, size=shape_a).astype("float32")
    b_np = np.random.uniform(-1, 1, size=shape_b).astype("float32")

    _compile_and_compare(mod, [a_np, b_np], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
