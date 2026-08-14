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
  - QNN/int8 quantized fusion patterns (qnn.conv2d/qnn.dense + requantize + sum): the BASE case
    (bare quantized conv2d/dense -- no bias, no activation, no residual-sum) is now covered by
    make_qnn_conv2d_pattern() / make_qnn_dense_pattern() below, registered as dnnl.qnn.conv2d /
    dnnl.qnn.matmul. These deliberately mirror the "run the op in float, fold the output-side
    affine rescale into DNNL's o_scl/dst_zp post-op" design (see ParseAttrs in
    dnnl_json_runtime.cc) so the two composite names stay drop-in-compatible with whatever
    equivalent patterns land from a parallel branch -- do not rename dnnl.qnn.conv2d /
    dnnl.qnn.matmul without checking for a collision there first.
    Bias/activation/residual-sum fusion ON TOP of the quantized form is still NOT ported at the
    PATTERN-MATCHING level -- same caveat as the bias+activation loop above, now applying to the
    quantized variants too. legalize_qnn_op_for_dnnl() below DOES legalize the general
    "qnn.conv2d/qnn.dense + bias + requantize" chain (weight/bias constant-folding, requantize
    rewritten to an equivalent dequantize) ahead of pattern matching -- but without a registered
    dnnl.qnn.conv2d_bias-style pattern, a biased chain still only gets PARTIALLY fused (conv+bias
    as one dnnl.conv2d_bias/dnnl.matmul_bias composite, rescale left unfused right after it).
    See legalize_qnn_op_for_dnnl()'s own docstring for the fusion-boundary detail.
  - swish/mish/gelu as decomposed multi-op activation patterns -- the legacy Relay version builds
    these from primitive ops (e.g. swish = sigmoid -> multiply) because Relay had no fused op for
    them; whether Relax's make_fused_bias_activation_pattern needs the same treatment or already
    has native ops for these depends on the current op registry and hasn't been checked here.
  - ResNetV1Rewrite (downsample-reorder optimization) -- a separate graph rewrite, not a BYOC
    pattern; out of scope for this file.
"""

import itertools

import numpy as np

import tvm
from tvm import relax, tirx
from tvm.relax.dpl import (
    is_const,
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
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=False),
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=True),
        _sum_pattern("relax.matmul", channel_axis=-1, with_relu=False),
    ]


def _reject_int64_like(*exprs) -> bool:
    """oneDNN does not natively support int64."""
    for expr in exprs:
        ty = getattr(expr, "ty", None)
        dtype = getattr(ty, "dtype", None)
        if dtype == "int64":
            return False
    return True


def _is_scalar_or_size1_const(expr) -> bool:
    """True if `expr`'s shape is 0-d, or 1-d with exactly one element. only scalar (per-tensor) output rescale is accepted."""
    ty = getattr(expr, "ty", None)
    shape = getattr(ty, "shape", None)
    if shape is None:
        return True  # can't prove it's non-scalar; let it through, runtime will ICHECK
    dims = list(shape.values)
    if len(dims) == 0:
        return True
    if len(dims) == 1 and isinstance(dims[0], tirx.IntImm) and dims[0].value == 1:
        return True
    return False


def dnnl_qnn_checker(context: PatternCheckContext) -> bool:
    """Shared checker for dnnl.qnn.conv2d / dnnl.qnn.matmul base patterns."""
    op_expr = context.annotated_expr["op"]
    scale_expr = context.annotated_expr["scale"]
    zp_expr = context.annotated_expr["zp"]

    if not _reject_int64_like(op_expr, scale_expr, zp_expr):
        return False

    if not _is_scalar_or_size1_const(scale_expr):
        return False
    if not _is_scalar_or_size1_const(zp_expr):
        return False

    return True


def make_qnn_conv2d_pattern() -> Pattern:
    """Base quantized-conv2d pattern: relax.nn.conv2d(data, weight) -> relax.dequantize(out,
    scale, zp).

    Returns
    -------
    pattern : Pattern
        ("dnnl.qnn.conv2d", DFPattern, annotation_map, dnnl_qnn_checker) -- ready to hand to
        register_patterns() directly, matching the calling convention _op_pattern() etc. already
        use in this file (see _dnnl_patterns() below).
    """
    data = wildcard()
    weight = wildcard()
    op = is_op("relax.nn.conv2d")(data, weight)

    scale = is_const()
    zp = is_const()
    out = is_op("relax.dequantize")(op, scale, zp)

    ann = {"data": data, "weight": weight, "op": op, "scale": scale, "zp": zp}
    return ("dnnl.qnn.conv2d", out, ann, dnnl_qnn_checker)


def make_qnn_dense_pattern() -> Pattern:
    """Base quantized-dense pattern: relax.matmul(data, weight) -> relax.dequantize(out, scale,
    zp). "Dense" here means relax.matmul, matching how the rest of this file (dnnl.matmul,
    _FUSABLE_OPS, _sum_patterns) already treats relax.matmul as the dense/fully-connected op --
    there is no separate relax.nn.dense composite family here.

    Returns
    -------
    pattern : Pattern
        ("dnnl.qnn.matmul", DFPattern, annotation_map, dnnl_qnn_checker).
    """
    data = wildcard()
    weight = wildcard()
    op = is_op("relax.matmul")(data, weight)

    scale = is_const()
    zp = is_const()
    out = is_op("relax.dequantize")(op, scale, zp)

    ann = {"data": data, "weight": weight, "op": op, "scale": scale, "zp": zp}
    return ("dnnl.qnn.matmul", out, ann, dnnl_qnn_checker)


def _dnnl_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []
    # 1-input ops
    for composite, op in [
        ("dnnl.avg_pool1d", "relax.nn.avg_pool1d"),
        ("dnnl.avg_pool2d", "relax.nn.avg_pool2d"),
        ("dnnl.avg_pool3d", "relax.nn.avg_pool3d"),
        ("dnnl.max_pool1d", "relax.nn.max_pool1d"),
        ("dnnl.max_pool2d", "relax.nn.max_pool2d"),
        ("dnnl.max_pool3d", "relax.nn.max_pool3d"),
    ]:
        patterns.append(_op_pattern(composite, op, 1))
    # 2-input ops
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
    patterns.append(make_qnn_conv2d_pattern())
    patterns.append(make_qnn_dense_pattern())

    return patterns


register_patterns(_dnnl_patterns())

_CONV_LAYOUT_QUERY_SPECS: dict[str, tuple[int, bool, list[str]]] = {
    "relax.nn.conv1d": (1, False, ["NCW", "OIW"]),
    "relax.nn.conv2d": (2, False, ["NCHW", "OIHW"]),
    "relax.nn.conv3d": (3, False, ["NCDHW", "OIDHW"]),
    "relax.nn.conv2d_transpose": (2, True, ["NCHW", "IOHW"]),
    "relax.nn.conv3d_transpose": (3, True, ["NCDHW", "IODHW"]),
}


def _query_one_conv2d_layout(query_fn, call: relax.Call) -> list[str] | None:
    """Extract shape/attrs from a single ungrouped conv2d call and query oneDNN for its
    preferred layout via the minimal 7-arg FFI function. Returns None (caller falls back to the
    hardcoded default) on any non-static shape, missing type info, a blocked-format rejection,
    or an oneDNN/argument-marshalling failure -- all legitimate "can't answer" outcomes, not bugs.
    """
    src_ty = call.args[0].ty
    wgh_ty = call.args[1].ty
    if src_ty is None or wgh_ty is None:
        return None

    src_shape = src_ty.shape
    wgh_shape = wgh_ty.shape
    if src_shape is None or wgh_shape is None:
        return None

    if not (
        all(isinstance(d, tvm.tirx.IntImm) for d in src_shape.values)
        and all(isinstance(d, tvm.tirx.IntImm) for d in wgh_shape.values)
    ):
        return None

    src_vals = [int(d) for d in src_shape.values]
    wgh_vals = [int(d) for d in wgh_shape.values]
    if len(src_vals) != 4 or len(wgh_vals) != 4:
        return None

    attrs = call.attrs

    try:
        src_layout, wgh_layout = query_fn(
            src_vals,
            wgh_vals,
            [int(v) for v in attrs.strides],
            [int(v) for v in attrs.dilation],
            [int(v) for v in attrs.padding],
            int(attrs.groups),
            str(src_ty.dtype),
        )
        src_layout, wgh_layout = str(src_layout), str(wgh_layout)
    except (tvm.error.TVMError, TypeError, ValueError):
        # TypeError/ValueError guard against a future arg-count/type drift between this Python
        # call site and the C++ signature failing loudly across the whole partition pass instead
        # of degrading to "use the plain default", same spirit as the TVMError case.
        return None

    is_blocked = any(c.isdigit() for c in src_layout + wgh_layout)
    if is_blocked:
        return None

    return [src_layout, wgh_layout]


def _query_dnnl_conv_layouts(mod: tvm.IRModule) -> dict[str, list[str]]:
    """Ask oneDNN what layout it would choose for each offloadable conv op's activation/weight
    tensors -- conv1d/2d/3d and their transposed forms -- using one representative call per op
    found in the module, instead of hardcoding plain NCHW/OIHW-style defaults for everything but
    conv2d. Single module traversal, table-driven via _CONV_LAYOUT_QUERY_SPECS above."""
    query_fn = tvm.get_global_func(
        "runtime.contrib.dnnl.query_optimal_conv2d_layout", allow_missing=True
    )
    if query_fn is None:
        return {}

    found: dict[str, list[str]] = {}

    @visitor
    class _FirstConv2dFinder(relax.PyExprVisitor):
        def visit_call_(self, call: relax.Call):
            if (
                "relax.nn.conv2d" not in found
                and isinstance(call.op, tvm.ir.Op)
                and call.op.name == "relax.nn.conv2d"
            ):
                layout = _query_one_conv2d_layout(query_fn, call)
                if layout is not None:
                    found["relax.nn.conv2d"] = layout
            super().visit_call_(call)

    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            _FirstConv2dFinder().visit_expr(func)
            if "relax.nn.conv2d" in found:
                break

    return found


def _try_fold_qdq_constant(q_expr, scale_expr, zp_expr) -> np.ndarray | None:
    """If q/scale/zp are all compile-time relax.Constant and scale/zp are per-tensor (size 1),
    return the dequantized float32 numpy array; else None -- caller must leave the chain alone,
    never guess. Shared by weight-folding and opportunistic bias-folding below.
    """
    if not (
        isinstance(q_expr, relax.Constant)
        and isinstance(scale_expr, relax.Constant)
        and isinstance(zp_expr, relax.Constant)
    ):
        return None

    q_np = q_expr.data.numpy()
    scale_np = scale_expr.data.numpy().astype("float32")
    zp_np = zp_expr.data.numpy().astype("float32")

    # Base scope: per-tensor quantization only (matches the restriction this function already had for weight-folding).
    if scale_np.size != 1 or zp_np.size != 1:
        return None

    return scale_np * (q_np.astype("float32") - zp_np)


def _try_requantize_consts(
    out_scale_expr, out_zp_expr
) -> tuple[relax.Constant, relax.Constant] | None:
    if not (isinstance(out_scale_expr, relax.Constant) and isinstance(out_zp_expr, relax.Constant)):
        return None

    out_scale_np = out_scale_expr.data.numpy().astype("float32")
    out_zp_np = out_zp_expr.data.numpy()

    if out_scale_np.size != 1 or out_zp_np.size != 1:
        return None

    # Output scale must be floating point.
    dq_scale_np = 1.0 / out_scale_np

    # Relax dequantize does not accept float32 zero_point.
    # For the current legalization scope, only support zero_point == 0.
    if float(out_zp_np.reshape(-1)[0]) != 0.0:
        return None

    dq_zp_np = np.array(0, dtype="int32")

    return (
        relax.const(dq_scale_np, "float32"),
        relax.const(dq_zp_np, "int32"),
    )


def legalize_qnn_op_for_dnnl(mod: tvm.IRModule) -> tvm.IRModule:
    """LegalizeQnnOpForDnnl: rewrite a quantized conv2d/matmul chain --

        dequantize(data_q) + dequantize(weight_q)
            -> nn.conv2d/matmul
            -> [add(bias)]                        # optional -- "qnn.conv2d/qnn.dense + bias"
            -> quantize(out, out_scale, out_zp)    # "+ requantize" -- relax has no dedicated
                                                    # requantize op, so this IS how the ticket's
                                                    # "qnn.conv2d + bias + requantize" chain is
                                                    # actually spelled in Relax's QDQ-only IR.

      1. weight-side dequantize -> folded to a float32 constant when compile-time-foldable.
      2. bias (if present) is left as a plain `add`. It is opportunistically folded to a float32
         constant too when it happens to ALSO be a compile-time dequantize(int32_const, ...)
         chain (the common shape for a real quantized bias, scale = data_scale * weight_scale);
         anything else (already float, or a dynamic Var) is passed through untouched --
         make_fused_bias_activation_pattern's bias slot is a bare wildcard, so either form
         already matches dnnl.conv2d_bias / dnnl.matmul_bias with no further rewriting needed.
      3. terminal quantize(out, out_scale, out_zp) -> rewritten into the algebraically
         equivalent dequantize(out, scale', zp'), so downstream pattern matching only ever needs
         to recognize ONE terminal-node shape -- the same one make_qnn_conv2d_pattern /
         make_qnn_dense_pattern already look for. See _try_requantize_consts() for the exact
         derivation and its rounding-mode caveat.

    Data-side dequantize is deliberately left untouched, same as before: whatever consumes the
    legalized chain treats `data`/`weight` as opaque wildcards, so it doesn't matter whether the
    data-side dequantize has been folded or still runs standalone ahead of the offloaded region.
    """
    data_q = wildcard()
    weight_q = wildcard()
    data_scale, data_zp = wildcard(), wildcard()
    weight_scale, weight_zp = wildcard(), wildcard()
    bias = wildcard()
    out_scale = is_const()
    out_zp = is_const()

    dq_data = is_op("relax.dequantize")(data_q, data_scale, data_zp)
    dq_weight = is_op("relax.dequantize")(weight_q, weight_scale, weight_zp)

    def _make_rewriter(op_out_pat, has_bias: bool):
        def rewriter(expr, matches):
            # `expr` is the matched root: the terminal relax.quantize(...) call.
            weight_np = _try_fold_qdq_constant(
                matches[weight_q], matches[weight_scale], matches[weight_zp]
            )
            if weight_np is None:
                return expr  # weight not compile-time-foldable -> never guess, leave alone

            requant_consts = _try_requantize_consts(matches[out_scale], matches[out_zp])
            if requant_consts is None:
                return expr  # dynamic requantize params -> leave alone
            dq_scale_const, dq_zp_const = requant_consts

            op_call = matches[op_out_pat]
            new_weight = relax.const(weight_np, "float32")
            new_op_call = relax.Call(
                op_call.op, [matches[dq_data], new_weight], attrs=op_call.attrs
            )

            if has_bias:
                bias_expr = matches[bias]
                # Opportunistic bias fold: only when bias is ITSELF a fully-constant
                # dequantize(...) chain. Anything else is passed through unchanged.
                if (
                    isinstance(bias_expr, relax.Call)
                    and isinstance(bias_expr.op, tvm.ir.Op)
                    and bias_expr.op.name == "relax.dequantize"
                    and len(bias_expr.args) == 3
                ):
                    bias_np = _try_fold_qdq_constant(*bias_expr.args)
                    if bias_np is not None:
                        bias_expr = relax.const(bias_np, "float32")
                new_inner = relax.op.add(new_op_call, bias_expr)
            else:
                new_inner = new_op_call

            return relax.op.dequantize(
                new_inner,
                dq_scale_const,
                dq_zp_const,
                axis=expr.attrs.axis,
                out_dtype="float32",
            )

        return rewriter

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue

        f = func
        for op_name in ("relax.nn.conv2d", "relax.matmul"):
            op_out = is_op(op_name)(dq_data, dq_weight)
            biased = is_op("relax.add")(op_out, bias)

            root_no_bias = is_op("relax.quantize")(op_out, out_scale, out_zp)
            root_bias = is_op("relax.quantize")(biased, out_scale, out_zp)

            f = rewrite_call(root_no_bias, _make_rewriter(op_out, has_bias=False), f)
            f = rewrite_call(root_bias, _make_rewriter(op_out, has_bias=True), f)

        new_mod[gvar] = f
    return new_mod


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
        queried_layouts = _query_dnnl_conv_layouts(mod)
        desired_layouts = {
            op_name: queried_layouts.get(op_name, default_layout)
            for op_name, (_rank, is_transpose, default_layout) in _CONV_LAYOUT_QUERY_SPECS.items()
            if not is_transpose
        }
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.ConvertLayout(desired_layouts)(mod)
            mod = relax.transform.FoldConstant()(mod)

    mod = rewrite_layer_norm(mod)
    mod = rewrite_dense_bias_gelu_reshape_last(mod)
    mod = rewrite_pad_avg_pool2d(mod)
    mod = legalize_qnn_op_for_dnnl(mod)
    mod = relax.transform.FoldConstant()(mod)

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
    "relax.matmul",
    "relax.nn.dense",
    "relax.nn.layer_norm",
    "relax.nn.max_pool2d",
    "relax.nn.avg_pool2d",
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


def rewrite_pad_avg_pool2d(mod: tvm.IRModule) -> tvm.IRModule:
    """Fold a directly-preceding constant zero-pad into avg_pool2d's own padding attribute.

    Only safe when:
      - the pad is a constant pad of value 0.0
      - pad widths on the N and C axes are 0 (spatial-only padding)
      - avg_pool2d does not already carry non-zero padding of its own
      - the pad has exactly one consumer (this avg_pool2d call) -- otherwise removing it
        would change another consumer's input.
    Non-matching pad+avg_pool2d pairs are left alone and will simply run as two separate,
    non-offloaded ops (correct, just not fused) -- this pass never guesses.
    """
    data_pat = wildcard()
    pad_pat = is_op("relax.nn.pad")(data_pat)
    pool_pat = is_op("relax.nn.avg_pool2d")(pad_pat)

    def _checked_rewriter(func: relax.Function):
        # Count consumers of each var so we can reject fan-out pads.
        use_count: dict[int, int] = {}

        @visitor
        class _UseCounter(relax.PyExprVisitor):
            def visit_var_(self, var):
                use_count[id(var)] = use_count.get(id(var), 0) + 1

        _UseCounter().visit_expr(func)

        def rewriter(expr, matches):
            pad_call = matches[pad_pat]
            pool_call = matches[pool_pat]
            data = matches[data_pat]

            pad_attrs = pad_call.attrs
            pool_attrs = pool_call.attrs

            # 1) pad value must be exactly 0, mode must be constant.
            if str(pad_attrs.pad_mode) != "constant":
                return expr
            pad_value = float(pad_attrs.pad_value)
            if pad_value != 0.0:
                return expr

            # 2) pad_width is a flat list of (before, after) per axis, NCHW order assumed
            #    (matches avg_pool2d's default layout attr -- see note below for NHWC).
            pad_width = [int(v) for v in pad_attrs.pad_width]

            # pad_width is stored as pairs; axis 0 = N, axis 1 = C, axes 2,3 = H, W
            n_before, n_after = pad_width[0], pad_width[1]
            c_before, c_after = pad_width[2], pad_width[3]
            h_before, h_after = pad_width[4], pad_width[5]
            w_before, w_after = pad_width[6], pad_width[7]
            if (n_before, n_after, c_before, c_after) != (0, 0, 0, 0):
                return expr  # never fold padding on batch/channel axes

            # 3) avg_pool2d must not already have non-zero padding.
            existing_padding = [int(v) for v in pool_attrs.padding]
            if any(p != 0 for p in existing_padding):
                return expr

            # 4) pad must have a single consumer (this pool call).
            if use_count.get(id(matches[pad_pat]), 0) > 1:
                return expr

            merged_padding = [h_before, w_before, h_after, w_after]  # (top, left, bottom, right)

            return relax.op.nn.avg_pool2d(
                data,
                pool_size=pool_attrs.pool_size,
                strides=pool_attrs.strides,
                padding=merged_padding,
                dilation=pool_attrs.dilation,
                ceil_mode=pool_attrs.ceil_mode,
                count_include_pad=True,
                layout=pool_attrs.layout,
                out_layout=pool_attrs.out_layout,
            )

        return rewriter

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            new_mod[gvar] = rewrite_call(pool_pat, _checked_rewriter(func), func)
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
