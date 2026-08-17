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

"""Pattern table and partitioning for the DNNL BYOC backend.

Scope note (post cross-check against TVM 0.19's Relay DNNL pattern table):
this file covers bare ops, bias+activation fusion, and conv2d/matmul bias+sum(+relu) residual
fusion -- the parts of the legacy Relay pattern table that could be verified against this
codebase's actual Relax dpl API. NOT ported, deliberately, pending separate verification:
  - QNN/int8 quantized fusion patterns (qnn.conv2d/qnn.dense + requantize + sum) -- a materially
    different, more involved pattern shape than anything below.
  - swish/mish/gelu as decomposed multi-op activation patterns -- the legacy Relay version builds
    these from primitive ops (e.g. swish = sigmoid -> multiply) because Relay had no fused op for
    them; whether Relax's make_fused_bias_activation_pattern needs the same treatment or already
    has native ops for these depends on the current op registry and hasn't been checked here.
  - ResNetV1Rewrite (downsample-reorder optimization) -- a separate graph rewrite, not a BYOC
    pattern; out of scope for this file.
"""

import itertools

import tvm
from tvm import relax, tirx
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
    PatternCheckContext,
)

from ..pattern_registry import Pattern, get_patterns_with_prefix, register_patterns


def _op_pattern(composite_name: str, op_name: str, num_args: int) -> Pattern:
    """A pattern matching a single op called with ``num_args`` wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    return (composite_name, is_op(op_name)(*args), {})


# Bias + activation fusion (generated, not hand-listed -- see _fused_patterns()).

# Ops for which DNNL implements a fused bias-add + activation post-op chain. layer_norm is
# deliberately excluded -- it isn't followed by a DNNL-fusable activation the way conv/matmul are.
_FUSABLE_OPS: list[str] = [
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
]

# Activations DNNL can fuse as a post-op. ``None`` means "bias-only, no activation" and is a valid
# combination in its own right (e.g. "dnnl.conv2d_bias"). Extend this list -- and the runtime's
# op-name -> dnnl::algorithm table in ParseAttrs -- together; codegen.cc needs no change either way.
# NOTE: verify these op names against the actual relax op registry before relying on this list --
# in particular confirm "relax.nn.gelu" vs "relax.gelu" for your TVM revision, and see the module
# docstring re: gelu/swish/mish possibly needing decomposed multi-op patterns instead.
_FUSABLE_ACTIVATIONS: list[str | None] = [
    None,
    "relax.nn.relu",
    "relax.sigmoid",
    "relax.nn.gelu",
    "relax.tanh",
]


def _composite_name_for(op_name: str, with_bias: bool, activation: str | None) -> str:
    """e.g. ("relax.nn.conv2d", True, "relax.nn.relu") -> "dnnl.conv2d_bias_relu"."""
    base = op_name.rsplit(".", 1)[-1]  # "relax.nn.conv2d" -> "conv2d"
    parts = [f"dnnl.{base}"]
    if with_bias:
        parts.append("bias")
    if activation is not None:
        parts.append(activation.rsplit(".", 1)[-1])
    return "_".join(parts)


def _fused_patterns() -> list[Pattern]:
    """Every (op, with_bias, activation) combination, generated instead of hand-listed.

    Extending fusion coverage (a new activation, a new fusable op) means adding one entry to
    _FUSABLE_OPS or _FUSABLE_ACTIVATIONS above -- nothing else in this function changes.
    """
    patterns: list[Pattern] = []
    for op_name, with_bias in itertools.product(_FUSABLE_OPS, (False, True)):
        for activation in _FUSABLE_ACTIVATIONS:
            if not with_bias and activation is None:
                continue  # already covered by the bare _op_pattern registration
            pat = make_fused_bias_activation_pattern(
                op_name, with_bias=with_bias, activation=activation
            )
            name = _composite_name_for(op_name, with_bias, activation)
            patterns.append((name, pat))
    return patterns


# Bias + residual-sum (+ activation) fusion. Ported from TVM 0.19's Relay
# make_conv_bias_sum_relu_pattern / make_dense_bias_sum_pattern, matched 1:1 in scope (conv2d gets
# both the relu and no-relu sum variant; matmul gets only the no-relu sum variant, exactly as the
# legacy pattern table registers it -- not generalized beyond what was actually validated upstream).
# Unlike the bias+activation loop above, this needs its own builder: the residual operand is a
# second full tensor input, not a scalar/activation choice, so it isn't expressible as another
# axis of the same loop.


def _sum_pattern(op_name: str, channel_axis: int, with_relu: bool) -> Pattern:
    """<op>(data, weight) + bias + residual, optionally followed by relu.

    channel_axis is where the op's output channel dimension lives (1 for NCHW-style conv output,
    -1 for matmul's last-dim output) -- used by the predicate below to check the bias is actually
    shaped like a per-channel bias, not some other broadcastable-but-wrong-length tensor.
    """
    data1 = wildcard()
    weight = wildcard()
    bias = wildcard()
    data2 = wildcard()

    op = is_op(op_name)(data1, weight)
    biased = is_op("relax.add")(op, bias)
    summed = is_op("relax.add")(biased, data2)
    root = is_op("relax.nn.relu")(summed) if with_relu else summed

    base = op_name.rsplit(".", 1)[-1]
    name = f"dnnl.{base}_bias_sum" + ("_relu" if with_relu else "")

    def check(context: PatternCheckContext) -> bool:
        op_expr = context.annotated_expr["op"]
        bias_expr = context.annotated_expr["bias"]
        data2_expr = context.annotated_expr["data2"]

        # NOTE: verify .ty / .ty.shape access against your TVM build -- grounded in the existing
        # tensorrt resize2d predicate's use of `.ty.ndim` / `.ty.dtype` / `.ty.shape`, but this
        # predicate additionally indexes into per-dimension values, which that example did not do.
        if op_expr.ty.shape is None or data2_expr.ty.shape is None:
            return False
        out_dims = list(op_expr.ty.shape.values)
        sum_dims = list(data2_expr.ty.shape.values)

        # Residual add must be a true elementwise match against the op's own output shape.
        if len(out_dims) != len(sum_dims):
            return False
        for a, b in zip(out_dims, sum_dims):
            if isinstance(a, tirx.IntImm) and isinstance(b, tirx.IntImm) and a.value != b.value:
                return False

        # Bias must be a scalar or a 1D tensor sized to the op's channel dimension -- catches a
        # bias that merely happens to be broadcastable but isn't actually a per-channel bias.
        if bias_expr.ty.shape is not None:
            bias_dims = list(bias_expr.ty.shape.values)
            if len(bias_dims) not in (0, 1):
                return False
            if len(bias_dims) == 1:
                channel_dim = out_dims[channel_axis]
                bias_dim = bias_dims[0]
                if (
                    isinstance(bias_dim, tirx.IntImm)
                    and isinstance(channel_dim, tirx.IntImm)
                    and bias_dim.value != channel_dim.value
                ):
                    return False

        return True

    return (
        name,
        root,
        {"op": op, "bias": bias, "data2": data2, "root": root},
        check,
    )


def _sum_patterns() -> list[Pattern]:
    return [
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=True),
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=False),
        _sum_pattern("relax.matmul", channel_axis=-1, with_relu=False),
    ]


def _dnnl_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []
    for composite, op in [
        ("dnnl.conv1d", "relax.nn.conv1d"),
        ("dnnl.conv2d", "relax.nn.conv2d"),
        ("dnnl.conv3d", "relax.nn.conv3d"),
        ("dnnl.conv2d_transpose", "relax.nn.conv2d_transpose"),
        ("dnnl.conv3d_transpose", "relax.nn.conv3d_transpose"),
        ("dnnl.matmul", "relax.matmul"),
    ]:
        patterns.append(_op_pattern(composite, op, 2))
    patterns.append(_op_pattern("dnnl.layer_norm", "relax.nn.layer_norm", 3))

    # Fused ops: bias-add and/or activation, for every (op, with_bias, activation) combination.
    # See _fused_patterns() docstring to extend.
    patterns.extend(_fused_patterns())

    # Bias + residual-sum (+ optional relu) fusion. See _sum_patterns() docstring to extend.
    patterns.extend(_sum_patterns())

    return patterns


register_patterns(_dnnl_patterns())


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
            "relax.nn.conv3d": ["NCDHW", "OIDHW"],
            "relax.nn.conv3d_transpose": ["NCDHW", "OIDHW"],
        }
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.ConvertLayout(desired_layouts)(mod)
            mod = relax.transform.FoldConstant()(mod)

    mod = rewrite_layer_norm(mod)
    mod = rewrite_dense_bias_gelu_reshape_last(mod)

    dnnl_patterns = get_patterns_with_prefix("dnnl")

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
