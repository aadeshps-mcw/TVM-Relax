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

"""Pattern table and partitioning for the DNNL BYOC backend."""

import tvm
from tvm import relax
from tvm.relax.dpl import (
    is_expr,
    is_op,
    make_fused_bias_activation_pattern,
    rewrite_call,
    wildcard,
)
from tvm.relax.expr_functor import visitor
from tvm.relax.transform import (
    FuseOpsByPattern,
    MergeCompositeFunctions,
)

from ..pattern_registry import Pattern, register_patterns


def _op_pattern(composite_name: str, op_name: str, num_args: int) -> Pattern:
    """A pattern matching a single op called with ``num_args`` wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    return (composite_name, is_op(op_name)(*args), {})


def _make_conv2d_clip_pattern(with_bias: bool):
    lhs = wildcard()
    rhs = wildcard()
    out = is_op("relax.nn.conv2d")(lhs, rhs)
    if with_bias:
        bias = wildcard()
        out = is_op("relax.add")(out, bias)
    clip_min = wildcard()
    clip_max = wildcard()
    out = is_op("relax.clip")(out, clip_min, clip_max)
    return out


def _ordered_dnnl_patterns() -> list[Pattern]:
    """Patterns in match-priority order: most specific (largest fused subgraph) first,
    so FuseOpsByPattern's greedy matching doesn't let a smaller generic pattern
    (e.g. dnnl.conv2d_bias) pre-empt a larger one (e.g. dnnl.conv2d_bias_relu)
    that shares the same conv2d+bias prefix."""
    all_patterns = _dnnl_patterns()
    # Longer composite name generally implies a more specific / larger pattern here
    # since we build names as dnnl.<base>[_bias][_<activation>].
    return sorted(all_patterns, key=lambda p: -len(p[0]))


def _dnnl_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    patterns.append(_op_pattern("dnnl.conv2d", "relax.nn.conv2d", 2))
    patterns.append(_op_pattern("dnnl.matmul", "relax.matmul", 2))
    patterns.append(_op_pattern("dnnl.layer_norm", "relax.nn.layer_norm", 3))

    # 2. Fused Ops
    _ACTIVATIONS = [
        ("relu", "relax.nn.relu"),
        ("tanh", "relax.tanh"),
        ("sigmoid", "relax.sigmoid"),
        ("gelu", "relax.nn.gelu"),
        (
            "swish",
            "relax.nn.silu",
        ),  # silu == swish(beta=1); DNNL runtime keys off "_swish"
    ]

    for suffix, act_op in _ACTIVATIONS:
        pat_bias = make_fused_bias_activation_pattern(
            "relax.nn.conv2d", with_bias=True, activation=act_op
        )
        patterns.append((f"dnnl.conv2d_bias_{suffix}", pat_bias))

        pat_nobias = make_fused_bias_activation_pattern(
            "relax.nn.conv2d", with_bias=False, activation=act_op
        )
        patterns.append((f"dnnl.conv2d_{suffix}", pat_nobias))

    patterns.append(("dnnl.conv2d_clip", _make_conv2d_clip_pattern(with_bias=False)))
    patterns.append(("dnnl.conv2d_bias_clip", _make_conv2d_clip_pattern(with_bias=True)))

    # Bias-only, no activation.
    pat_conv_bias = make_fused_bias_activation_pattern(
        "relax.nn.conv2d", with_bias=True, activation=None
    )
    patterns.append(("dnnl.conv2d_bias", pat_conv_bias))

    return patterns


register_patterns(sorted(_dnnl_patterns(), key=lambda p: len(p[0])))


def partition_for_dnnl(
    mod: tvm.IRModule,
    params: dict[str, tvm.runtime.Tensor] | None = None,
    alter_layout: bool = True,
    prune_subgraphs: bool = True,
) -> tvm.IRModule:
    if params:
        mod = relax.transform.BindParams("main", params)(mod)

    pre_seq = tvm.transform.Sequential(
        [
            relax.transform.DecomposeOpsForInference(),
            relax.transform.FoldConstant(),
            relax.transform.FoldBatchnormToConv2D(),
            relax.transform.CanonicalizeBindings(),
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = pre_seq(mod)

    if alter_layout:
        desired_layouts = {
            "relax.nn.conv2d": ["NCHW", "OIHW"],
            "relax.nn.conv2d_transpose": ["NCHW", "OIHW"],
        }
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.ConvertLayout(desired_layouts)(mod)
            mod = relax.transform.FoldConstant()(mod)

    mod = rewrite_layer_norm(mod)
    mod = rewrite_dense_bias_gelu_reshape_last(mod)

    dnnl_patterns = _ordered_dnnl_patterns()

    byoc_seq = tvm.transform.Sequential(
        [
            FuseOpsByPattern(dnnl_patterns),
            MergeCompositeFunctions(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)

    if prune_subgraphs:
        mod = prune_dnnl_subgraphs(mod)

    return mod


_DNNL_COMPUTE_OPS = {
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
    "relax.nn.layer_norm",
}


def _count_compute_ops(mod: tvm.IRModule, func: relax.Function) -> int:
    count = 0
    seen_globals = set()

    @visitor
    class _Counter(relax.PyExprVisitor):
        def visit_call_(self, call: relax.Call):
            nonlocal count
            if isinstance(call.op, tvm.ir.Op) and call.op.name in _DNNL_COMPUTE_OPS:
                count += 1
            elif isinstance(call.op, relax.GlobalVar):
                name = call.op.name_hint
                if name not in seen_globals and name in mod.global_var_map_:
                    seen_globals.add(name)
                    self.visit_expr(mod[name])
            super().visit_call_(call)

    _Counter().visit_expr(func)
    return count


def prune_dnnl_subgraphs(mod: tvm.IRModule) -> tvm.IRModule:
    to_demote = []
    for gvar, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if func.attrs is None or func.attrs.get("Codegen") != "dnnl":
            continue
        if _count_compute_ops(mod, func) == 0:
            to_demote.append(gvar)

    if not to_demote:
        return mod

    new_mod = mod.clone() if hasattr(mod, "clone") else tvm.IRModule(mod.functions)
    for gvar in to_demote:
        func = new_mod[gvar]
        if func.attrs is not None and "Codegen" in func.attrs:
            func = func.without_attr("Codegen")
        if func.attrs is not None and "global_symbol" in func.attrs:
            func = func.without_attr("global_symbol")
        new_mod[gvar] = func

    with tvm.transform.PassContext(opt_level=3):
        new_mod = relax.transform.InlinePrivateFunctions()(new_mod)
        new_mod = relax.transform.DeadCodeElimination(["main"])(new_mod)

    return new_mod


def rewrite_layer_norm(mod: tvm.IRModule) -> tvm.IRModule:
    """Rewrite multiple operators into a TVM native layer normalization."""
    data_pat = wildcard()
    gamma_pat = wildcard()
    beta_pat = wildcard()
    mu = is_op("relax.mean")(data_pat)
    diff = is_op("relax.subtract")(data_pat, mu)
    cdiff = diff | is_op("relax.astype")(diff)
    const_two = is_expr(relax.const(2)) | is_expr(relax.const(2.0))
    p1 = is_op("relax.power")(cdiff, const_two)
    mp1 = is_op("relax.mean")(p1) | is_op("relax.variance")(data_pat, mu)
    eps = is_expr(relax.const(1e-5)) | is_expr(relax.const(1e-6))
    added_eps = is_op("relax.add")(mp1, eps)
    deno = is_op("relax.sqrt")(added_eps)
    div_out = is_op("relax.divide")(diff, deno)
    div_out2 = is_op("relax.multiply")(diff, is_op("relax.rsqrt")(added_eps))
    weighted = is_op("relax.multiply")(div_out | div_out2, gamma_pat)
    added_bias = is_op("relax.add")(weighted, beta_pat)

    pattern = added_bias

    def rewriter(expr, matches):
        data = matches[data_pat]
        gamma = matches[gamma_pat]
        beta = matches[beta_pat]
        return relax.op.nn.layer_norm(data=data, gamma=gamma, beta=beta, axes=[-1])

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            new_mod[gvar] = rewrite_call(pattern, rewriter, func)
    return new_mod


def rewrite_dense_bias_gelu_reshape_last(mod: tvm.IRModule) -> tvm.IRModule:
    """Reorder reshape operators for dense_bias_gelu/dense_bias fusion."""

    def _apply(func: relax.Function, has_gelu: bool) -> relax.Function:
        data_pat = wildcard()
        weight_pat = wildcard()
        bias_pat = wildcard()
        const1 = wildcard()
        const2 = wildcard()
        const3 = wildcard()

        den = is_op("relax.matmul")(data_pat, weight_pat)
        re_den = is_op("relax.reshape")(den)
        added = is_op("relax.add")(bias_pat, re_den)

        if has_gelu:
            divisor = is_op("relax.divide")(added, const1)
            val_erf = is_op("relax.erf")(divisor)
            added_erf = is_op("relax.add")(val_erf, const2)
            mul1 = is_op("relax.multiply")(added, added_erf)
            mul2 = is_op("relax.multiply")(mul1, const3)
            pattern = mul2
        else:
            pattern = added

        def rewriter(expr, matches):
            reshape_expr = matches[re_den]
            shape = reshape_expr.args[1]

            data = matches[data_pat]
            weight = matches[weight_pat]
            bias = matches[bias_pat]

            den_new = relax.op.matmul(data, weight)
            added_new = relax.op.add(bias, den_new)

            if not has_gelu:
                return relax.op.reshape(added_new, shape)

            c1 = matches[const1]
            c2 = matches[const2]
            c3 = matches[const3]

            divisor_new = relax.op.divide(added_new, c1)
            val_erf_new = relax.op.erf(divisor_new)
            added_erf_new = relax.op.add(val_erf_new, c2)
            mul1_new = relax.op.multiply(added_new, added_erf_new)
            mul2_new = relax.op.multiply(mul1_new, c3)

            return relax.op.reshape(mul2_new, shape)

        return rewrite_call(pattern, rewriter, func)

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            f1 = _apply(func, has_gelu=True)
            f2 = _apply(f1, has_gelu=False)
            new_mod[gvar] = f2

    return new_mod
