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
Numerical-correctness test for the DNNL BYOC partitioning pipeline.

Goal
----
Isolate whether numerical mismatches observed on a full model come from the
Relax -> DNNL frontend pipeline (`partition_for_dnnl` and the transform
passes it runs: DecomposeOpsForInference, FoldConstant,
FoldBatchnormToConv2D, CanonicalizeBindings, EliminateCommonSubexpr,
ConvertLayout, rewrite_layer_norm, rewrite_dense_bias_gelu_reshape_last,
FuseOpsByPattern/MergeCompositeFunctions, prune_dnnl_subgraphs), rather
than from codegen.cc / the DNNL runtime, or from the torch importer itself.

Strategy
--------
For the same input tensor, produce three outputs and diff them pairwise:

  1. torch_ref   - PyTorch eager-mode output (ground truth)
  2. tvm_native  - Relax module built WITHOUT partition_for_dnnl
                   (exercises only the importer + native TVM ops)
  3. tvm_dnnl    - Relax module built WITH partition_for_dnnl
                   (exercises the full frontend pipeline + DNNL runtime)

  torch_ref ~= tvm_native  but  tvm_native != tvm_dnnl
      -> the mismatch is introduced inside partition_for_dnnl
         (pattern matching / ConvertLayout / attr extraction in codegen.cc)
         and NOT the base importer.

  torch_ref != tvm_native
      -> the bug predates partitioning entirely (importer bug); fix that
         first, the DNNL comparison downstream is meaningless until then.

`alter_layout` and `prune_subgraphs` are parametrized so a failure can be
bisected to a specific stage of the pipeline (e.g. if only
alter_layout=True fails, suspect ConvertLayout).

NOTE: adjust the two `tvm.relax...` import paths below if your tree lays
them out differently (`from_exported_program` and `partition_for_dnnl`).
"""

import numpy as np
import pytest
import torch
from torch import nn

import tvm
from tvm import relax

# Adjust this import path if partition_for_dnnl lives elsewhere in your tree.
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl
from tvm.relax.frontend.torch import from_exported_program

SEED = 0
ATOL = 1e-4
RTOL = 1e-4


class _TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


def _build_model_and_input():
    torch.manual_seed(SEED)
    model = _TinyConvNet().eval()
    example_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    return model, example_input


def _import_to_relax(model, example_input):
    """PyTorch -> Relax IRModule with weights bound as constants (not
    inputs), so partition_for_dnnl's FoldConstant/pattern-matching sees
    weights as constants exactly like it would for a real deployed model."""
    exported = torch.export.export(model, (example_input,))
    mod = from_exported_program(exported, keep_params_as_input=False)
    return mod


def _run_torch(model, example_input):
    with torch.no_grad():
        out = model(example_input)
    return out.numpy()


def _build_and_run(mod, input_np, target_str="llvm", run_codegen=False):
    """
    run_codegen=True must be used for a module that went through
    partition_for_dnnl(). partition_for_dnnl() only *tags* subgraphs with
    Codegen="dnnl" (a local/nested Relax function) -- it doesn't lower them.
    relax.transform.RunCodegen() is the pass that actually invokes the
    registered relax.ext.dnnl compiler and replaces the tagged local
    function with a call into the compiled external module. Skipping this
    for a DNNL-partitioned module leaves a dangling local function that
    VMShapeLower chokes on ("VMShapeLower do not work for local functions").
    """
    if run_codegen:
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.RunCodegen()(mod)

    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target)
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, dev)
    tvm_input = tvm.runtime.tensor(input_np)
    out = vm["main"](tvm_input)
    if isinstance(out, (tvm.ir.Array | tuple | list)):
        out = out[0]
    return out.numpy()


def _offloaded_dnnl_functions(mod_dnnl):
    """Names of functions actually offloaded to the dnnl codegen, so we can
    confirm the test is exercising the BYOC path and not silently no-op'ing."""
    return [
        gvar.name_hint
        for gvar, func in mod_dnnl.functions.items()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get("Codegen") == "dnnl"
    ]


@pytest.mark.parametrize("alter_layout", [False, True])
@pytest.mark.parametrize("prune_subgraphs", [False, True])
def test_dnnl_partition_numerics(alter_layout, prune_subgraphs):
    model, example_input = _build_model_and_input()
    input_np = example_input.numpy()

    # 1. Ground truth.
    torch_ref = _run_torch(model, example_input)

    # 2. Native TVM (no DNNL partitioning at all) - sanity check on the importer.
    mod_native = _import_to_relax(model, example_input)
    tvm_native = _build_and_run(mod_native, input_np)

    np.testing.assert_allclose(
        torch_ref,
        tvm_native,
        rtol=RTOL,
        atol=ATOL,
        err_msg=(
            "Mismatch between PyTorch and native TVM (importer bug, "
            "unrelated to partition_for_dnnl). Fix this before trusting "
            "any DNNL comparison below."
        ),
    )

    # 3. DNNL-partitioned pipeline - the thing under test.
    mod_dnnl = _import_to_relax(model, example_input)
    mod_dnnl = partition_for_dnnl(
        mod_dnnl, alter_layout=alter_layout, prune_subgraphs=prune_subgraphs
    )

    offloaded = _offloaded_dnnl_functions(mod_dnnl)
    assert offloaded, (
        f"No functions were offloaded to DNNL (alter_layout={alter_layout}, "
        f"prune_subgraphs={prune_subgraphs}); this test is not exercising "
        f"the BYOC codegen path at all."
    )

    tvm_dnnl = _build_and_run(mod_dnnl, input_np, run_codegen=True)

    np.testing.assert_allclose(
        tvm_native,
        tvm_dnnl,
        rtol=RTOL,
        atol=ATOL,
        err_msg=(
            f"Native TVM and DNNL-partitioned outputs diverge "
            f"(alter_layout={alter_layout}, prune_subgraphs={prune_subgraphs}). "
            f"Offloaded subgraphs: {offloaded}. This points at "
            f"partition_for_dnnl's transform pipeline (pattern matching / "
            f"ConvertLayout / codegen attr extraction) rather than the "
            f"base importer."
        ),
    )

    np.testing.assert_allclose(
        torch_ref,
        tvm_dnnl,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Mismatch between PyTorch reference and DNNL-partitioned TVM output.",
    )


if __name__ == "__main__":
    # Quick standalone run without pytest, useful for a first manual check
    # or for bisecting which (alter_layout, prune_subgraphs) combo fails.
    for alter_layout in (False, True):
        for prune_subgraphs in (False, True):
            print(f"=== alter_layout={alter_layout} prune_subgraphs={prune_subgraphs} ===")
            try:
                test_dnnl_partition_numerics(alter_layout, prune_subgraphs)
                print("PASS")
            except AssertionError as e:
                print("FAIL:", e)
