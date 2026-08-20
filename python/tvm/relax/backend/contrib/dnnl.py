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

This file covers bare ops, bias/activation/clip fusion, and conv2d/matmul bias+sum(+relu)
residual fusion for the DNNL backend.

Scope notes:
  - QNN/int8 fusion beyond the base case. Bare quantized conv2d/dense (no bias, activation, or
    sum) is covered by make_qnn_conv2d_pattern() and make_qnn_dense_pattern(), which produce the
    dnnl.qnn.conv2d and dnnl.qnn.matmul composites. Keep these two names stable, and check for
    collisions before renaming either one.
  - swish and mish are not wired into _FUSABLE_ACTIVATIONS below; they aren't offered as fused
    activation post-ops in this file. See the TODO near _FUSABLE_ACTIVATIONS for more.

Clip fusion (dnnl.<op>_clip, dnnl.<op>_bias_clip) is generated generically for every op in
_FUSABLE_OPS, the same way bias and activation fusion are; see _fused_patterns().
"""

import functools

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
from tvm.relax.expr_functor import PyExprMutator, mutator, visitor
from tvm.relax.transform import (
    FuseOpsByPattern,
    MergeCompositeFunctions,
    PatternCheckContext,
)

from ..pattern_registry import Pattern, register_patterns

# Constants

SUPPORTED_ELTWISE = {
    "abs",
    "exp",
    "log",
    "sqrt",
    "round",
    "relu",
    "nn.relu",
    "leakyrelu",
    "nn.leakyrelu",
    "tanh",
    "sigmoid",
    "clip",
    "gelu_erf",
    "gelu",
    "silu",
}

# Ops for which DNNL implements a fused bias-add, activation, and clip post-op chain. layer_norm
# is excluded here since it isn't followed by a DNNL-fusable activation the way conv/matmul are.
_FUSABLE_OPS: list[str] = [
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
]

# Activations DNNL can fuse as a post-op. None means bias-only, no activation, which is a valid
# combination on its own (e.g. "dnnl.conv2d_bias"). Extend this list and the runtime's op-name to
# dnnl::algorithm table in ParseAttrs together; codegen.cc doesn't need to change either way.
# These op names need to match the current relax op registry, gelu especially, since it may be
# named differently across TVM revisions.
#
# TODO: swish and mish aren't wired in here. The legacy Relay version built them from
# decomposed primitive ops (e.g. swish as sigmoid followed by multiply) because Relay had no
# fused op for them -- that decomposed-pattern approach hasn't been ported to Relax yet. gelu
# *is* wired in below even though the same "does Relax have a native op for this" question
# applies to it equally; resolve this inconsistency one way or the other (either port the
# swish/mish decomposed-pattern approach, or drop the implicit assumption that gelu is special).
_FUSABLE_ACTIVATIONS: list[str | None] = [
    None,
    "relax.nn.relu",
    "relax.sigmoid",
    "relax.nn.gelu",
    "relax.tanh",
]

# Rank, is_transpose, and (data_layout, kernel_layout) for every conv variant this backend
# handles. This is the single source of truth for ConvertLayout's desired_layouts in
# partition_for_dnnl. The kernel layout is intentionally IO-swapped (IOHW/IODHW) for the
# transpose variants, since a deconv weight tensor's channel-axis convention differs from a
# regular conv weight's.
_CONV_LAYOUT_QUERY_SPECS: dict[str, tuple[int, bool, list[str]]] = {
    "relax.nn.conv1d": (1, False, ["NCW", "OIW"]),
    "relax.nn.conv2d": (2, False, ["NCHW", "OIHW"]),
    "relax.nn.conv3d": (3, False, ["NCDHW", "OIDHW"]),
    "relax.nn.conv2d_transpose": (2, True, ["NCHW", "IOHW"]),
    "relax.nn.conv3d_transpose": (3, True, ["NCDHW", "IODHW"]),
}

# Ops that count as real DNNL compute for prune_dnnl_subgraphs (see _count_compute_ops).
# Deliberately excludes bare elementwise/add/multiply -- a standalone region built from
# only those isn't worth offloading; see test_prune_subgraphs_demotes_light_ops.

_DNNL_COMPUTE_OPS = {
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
    "relax.nn.batch_matmul",
    "relax.nn.layer_norm",
    "relax.nn.batch_norm",
    "relax.nn.adaptive_avg_pool2d",
    "relax.nn.max_pool1d",
    "relax.nn.max_pool2d",
    "relax.nn.max_pool3d",
    "relax.nn.avg_pool1d",
    "relax.nn.avg_pool2d",
    "relax.nn.avg_pool3d",
    "relax.nn.softmax",
}


# Shared helpers
@mutator
class _QnnTerminalRewriter(PyExprMutator):
    def visit_call_(self, call: relax.Call):
        call = super().visit_call_(call)  # rewrites args first (post-order)

        # Only target terminal quantize nodes
        if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "relax.quantize"):
            return call

        # call.args[0] is now the (already-rewritten) conv2d/matmul-or-add expr
        inner = call.args[0]
        out_scale, out_zp = call.args[1], call.args[2]

        requant_consts = _try_requantize_consts(out_scale, out_zp)
        if requant_consts is None:
            return call

        dq_scale_const, dq_zp_const = requant_consts

        # emit via self.builder so a fresh, correctly-typed Var is created
        return relax.op.dequantize(
            inner,
            dq_scale_const,
            dq_zp_const,
            axis=call.attrs.axis,
            out_dtype="float32",
        )


def _get_dtype(node) -> str | None:
    ty = getattr(node, "ty", None)
    if ty is None:
        return None
    return getattr(ty, "dtype", None)


def _has_int64(*exprs) -> bool:
    """Returns True if any of exprs has an int64 dtype."""
    return any(_get_dtype(e) == "int64" for e in exprs)


def _strip_relax_prefix(op_name: str) -> str:
    """Turns "relax.nn.relu" or "relax.sigmoid" into "relu"/"sigmoid" -- the short form used as
    both SUPPORTED_ELTWISE keys and DNNL runtime activation-post-op names."""
    return op_name.replace("relax.", "").replace("nn.", "")


def _is_valid_qnn_scale(scale_expr, op_expr, axis: int) -> bool:
    """True if scale_expr is a scalar/size-1 constant (per-tensor), or a 1D constant sized to
    op_expr's output channel dimension along `axis` (per-channel). This mirrors what the DNNL
    runtime's ParseAttrs actually supports for o_scl_tr -- both cases are handled through
    post-ops (a scalar eltwise_linear fold, or a binary_mul against a reshaped per-channel
    tensor keyed off dst_axis). This is deliberately more permissive than dst_zp's check below,
    since ParseAttrs enforces (via an explicit IsScalar() ICHECK) that a per-channel
    zero_point is NOT supported -- only scale can vary per channel.
    """
    if _is_scalar_or_size1_const(scale_expr):
        return True

    ty = getattr(scale_expr, "ty", None)
    shape = getattr(ty, "shape", None)
    if shape is None:
        return False
    dims = list(shape.values)
    if len(dims) != 1:
        return False

    out_ty = getattr(op_expr, "ty", None)
    out_shape = getattr(out_ty, "shape", None)
    if out_shape is None:
        return False
    out_dims = list(out_shape.values)
    ax = axis if axis >= 0 else axis + len(out_dims)
    if ax < 0 or ax >= len(out_dims):
        return False

    channel_dim = out_dims[ax]
    if isinstance(dims[0], tirx.IntImm) and isinstance(channel_dim, tirx.IntImm):
        return dims[0].value == channel_dim.value
    return True  # dynamic dimension, can't rule it out statically


def _reject_int64(call) -> bool:
    """oneDNN doesn't natively support int64, so this rejects a call if its output dtype, any of
    its argument dtypes, or an explicit out_dtype attr (for ops that carry one) is int64."""
    if _has_int64(call, *call.args):
        return False
    if call.attrs is not None and hasattr(call.attrs, "out_dtype"):
        if str(call.attrs.out_dtype) == "int64":
            return False
    return True


def _is_valid_broadcast_bias(bias_dims, channel_dim) -> bool:
    """True if bias_dims describes a scalar, or a tensor with exactly one non-1-sized dimension
    that matches the op's channel dimension. This covers both the 1D (oc,) convention and the
    (oc, 1, 1)-style NCHW-broadcast convention; any rank works as long as there's only one real
    axis."""
    if len(bias_dims) == 0:
        return True

    non_unit = [d for d in bias_dims if not (isinstance(d, tirx.IntImm) and d.value == 1)]
    if len(non_unit) == 0:
        return True  # all ones, trivially broadcastable
    if len(non_unit) > 1:
        return False  # more than one real axis, not a simple per-channel bias

    dim = non_unit[0]
    if isinstance(dim, tirx.IntImm) and isinstance(channel_dim, tirx.IntImm):
        return dim.value == channel_dim.value
    return True  # dynamic dimension, can't rule it out statically, so don't block the match


def _is_scalar_or_size1_const(expr) -> bool:
    """True if expr's shape is 0-d, or 1-d with exactly one element. Only a scalar, per-tensor
    output rescale is accepted."""
    ty = getattr(expr, "ty", None)
    shape = getattr(ty, "shape", None)
    if shape is None:
        return True  # can't prove it isn't scalar, so let it through and let the runtime check
    dims = list(shape.values)
    if len(dims) == 0:
        return True
    if len(dims) == 1 and isinstance(dims[0], tirx.IntImm) and dims[0].value == 1:
        return True
    return False


def _validate_eltwise_op_name(op_name: str) -> None:
    """Raises instead of silently excluding, for op names that reach DNNL codegen through a path
    that should always be a known, closed set. This is different from dnnl_eltwise_checker's
    pattern-matching context, where returning False just means "don't offload this", which makes
    sense for runtime-shaped inputs but not for the fixed, compile-time-known set of activation
    post-ops used here."""
    if op_name not in SUPPORTED_ELTWISE:
        raise ValueError(
            f"Unsupported DNNL eltwise/activation post-op: {op_name!r}. "
            f"Supported: {sorted(SUPPORTED_ELTWISE)}"
        )


# Pattern-match checkers


def dnnl_pooling_checker(ctx) -> bool:
    call = ctx.matched_expr
    if hasattr(call.attrs, "ceil_mode") and call.attrs.ceil_mode:
        return False
    return _reject_int64(call)


def dnnl_global_avg_pool2d_checker(ctx) -> bool:
    """Only true global average pooling, where output_size is (1, 1), is valid here. A general
    adaptive_avg_pool2d with a larger output_size is a materially different computation, since
    oneDNN's pooling_forward primitive can't express per-window variable kernel sizes. Offloading
    it would produce a malformed descriptor at runtime. Adaptive pools with any other output size
    fall through to TVM's native implementation instead.
    """
    call = ctx.matched_expr
    output_size = tuple(int(v) for v in call.attrs.output_size)
    if output_size != (1, 1):
        return False
    return _reject_int64(call)


def dnnl_conv_checker(ctx) -> bool:
    """Shared int64-rejection guard for every bare conv/matmul base pattern: conv1d, conv2d,
    conv3d, their transposed variants, and matmul."""
    return _reject_int64(ctx.matched_expr)


def dnnl_eltwise_checker(ctx) -> bool:
    call = ctx.matched_expr
    if not isinstance(call.op, tvm.ir.Op):
        return True

    op_name = _strip_relax_prefix(call.op.name)
    if op_name not in SUPPORTED_ELTWISE:
        return False
    return _reject_int64(call)


def dnnl_qnn_checker(context: PatternCheckContext) -> bool:
    """Shared checker for the dnnl.qnn.conv2d and dnnl.qnn.matmul base patterns."""
    op_expr = context.annotated_expr["op"]
    scale_expr = context.annotated_expr["scale"]
    zp_expr = context.annotated_expr["zp"]

    if _has_int64(op_expr, scale_expr, zp_expr):
        return False

    axis = int(context.matched_expr.attrs.axis)
    if not _is_valid_qnn_scale(scale_expr, op_expr, axis):
        return False

    # dst_zp must stay strictly scalar/size-1 -- ParseAttrs' IsScalar() check on dst_zp_tr
    # in dnnl_json_runtime.cc has no per-channel path, unlike scale.
    if not _is_scalar_or_size1_const(zp_expr):
        return False
    return True


# Pattern builders


def _op_pattern(composite_name: str, op_name: str, num_args: int, checker=None) -> Pattern:
    """A pattern matching a single op called with num_args wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    pat = is_op(op_name)(*args)
    # The empty dictionary {} is required as the 3rd element for annotations.
    return (composite_name, pat, {}, checker) if checker else (composite_name, pat, {})


def _base_name(op_name: str) -> str:
    """Turns "relax.nn.conv2d" into "dnnl.conv2d"."""
    return f"dnnl.{op_name.rsplit('.', 1)[-1]}"


def _composite_name_for(op_name: str, with_bias: bool, activation: str | None) -> str:
    """Builds a composite name, e.g. ("relax.nn.conv2d", True, "relax.nn.relu") becomes
    "dnnl.conv2d_bias_relu"."""
    parts = [_base_name(op_name)]
    if with_bias:
        parts.append("bias")
    if activation is not None:
        parts.append(activation.rsplit(".", 1)[-1])
    return "_".join(parts)


def _make_clip_pattern(op_name: str, with_bias: bool):
    lhs = wildcard()
    rhs = wildcard()
    out = is_op(op_name)(lhs, rhs)
    if with_bias:
        bias = wildcard()
        out = is_op("relax.add")(out, bias)
    clip_min = wildcard()
    clip_max = wildcard()
    out = is_op("relax.clip")(out, clip_min, clip_max)
    return out


def _fused_patterns() -> list[Pattern]:
    """Generates every (op, with_bias, activation) combination instead of hand-listing them,
    plus clip fusion for each op. Clip is built separately since its bounds aren't expressible
    through make_fused_bias_activation_pattern's activation argument; see _make_clip_pattern.
    These fused and clip patterns don't reject int64 inputs the way the bare op patterns do.

    To extend fusion coverage with a new activation or a new fusable op, add one entry to
    _FUSABLE_OPS or _FUSABLE_ACTIVATIONS above. Nothing else in this function needs to change.
    """
    patterns: list[Pattern] = []
    for op_name in _FUSABLE_OPS:
        for with_bias in (False, True):
            for activation in _FUSABLE_ACTIVATIONS:
                if not with_bias and activation is None:
                    continue  # already covered by the bare pattern in _dnnl_patterns()
                if activation is not None:
                    _validate_eltwise_op_name(_strip_relax_prefix(activation))
                pat = make_fused_bias_activation_pattern(
                    op_name, with_bias=with_bias, activation=activation
                )
                patterns.append((_composite_name_for(op_name, with_bias, activation), pat))

        patterns.append(
            (f"{_base_name(op_name)}_clip", _make_clip_pattern(op_name, with_bias=False))
        )
        patterns.append(
            (f"{_base_name(op_name)}_bias_clip", _make_clip_pattern(op_name, with_bias=True))
        )

    return patterns


# Bias plus residual-sum, with an optional activation, fusion.
#
# This matches the legacy pattern table's scope exactly: conv2d gets both the relu and no-relu
# sum variant, matmul gets only the no-relu sum variant. It isn't generalized beyond what was
# actually validated upstream.
#
# Unlike the bias and activation loop above, this needs its own builder. The residual operand is
# a second full tensor input rather than a scalar or activation choice, so it can't be expressed
# as another axis of that same loop.


def _sum_pattern(op_name: str, channel_axis: int, with_relu: bool) -> Pattern:
    """Matches op(data, weight) plus bias plus a residual, optionally followed by relu.

    channel_axis is where the op's output channel dimension lives: 1 for NCHW-style conv output,
    -1 for matmul's last-dim output. The predicate below uses it to check that the bias is
    actually shaped like a per-channel bias, rather than some other tensor that just happens to
    be broadcastable.

    This is distinct from bias-only fusion: it matches a second add whose other operand is an
    external residual tensor, not the bias.
    """
    data1 = wildcard()
    weight = wildcard()
    bias = wildcard()
    data2 = wildcard()

    op = is_op(op_name)(data1, weight)
    biased = is_op("relax.add")(op, bias)
    summed = is_op("relax.add")(biased, data2)
    root = is_op("relax.nn.relu")(summed) if with_relu else summed

    name = f"{_base_name(op_name)}_bias_sum" + ("_relu" if with_relu else "")

    def check(context: PatternCheckContext) -> bool:
        op_expr = context.annotated_expr["op"]
        bias_expr = context.annotated_expr["bias"]
        data2_expr = context.annotated_expr["data2"]

        # This follows the same .ty / .ty.shape access pattern as the tensorrt resize2d
        # predicate, but also indexes into individual dimension values, which that one doesn't.
        if op_expr.ty.shape is None or data2_expr.ty.shape is None:
            return False
        out_dims = list(op_expr.ty.shape.values)
        sum_dims = list(data2_expr.ty.shape.values)

        # The residual add must be a true elementwise match against the op's own output shape.
        if len(out_dims) != len(sum_dims):
            return False
        for a, b in zip(out_dims, sum_dims):
            if isinstance(a, tirx.IntImm) and isinstance(b, tirx.IntImm) and a.value != b.value:
                return False

        # The bias must be a scalar or a 1D tensor sized to the op's channel dimension, which
        # catches a bias that merely happens to be broadcastable but isn't actually per-channel.
        if bias_expr.ty.shape is not None:
            bias_dims = list(bias_expr.ty.shape.values)
            channel_dim = out_dims[channel_axis]
            if not _is_valid_broadcast_bias(bias_dims, channel_dim):
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


def _standalone_patterns() -> list[Pattern]:
    """Bare, unfused op patterns: elementwise ops, clip, pooling, softmax, add, multiply,
    batch_norm, and true global average pooling. See _dnnl_patterns() for where these get
    registered."""
    patterns: list[Pattern] = []

    _ELTWISE_OPS = [
        ("abs", "relax.abs"),
        ("exp", "relax.exp"),
        ("log", "relax.log"),
        ("sqrt", "relax.sqrt"),
        ("round", "relax.round"),
        ("relu", "relax.nn.relu"),
        ("leaky_relu", "relax.nn.leakyrelu"),
        ("tanh", "relax.tanh"),
        ("sigmoid", "relax.sigmoid"),
    ]
    for suffix, op_name in _ELTWISE_OPS:
        patterns.append(_op_pattern(f"dnnl.{suffix}", op_name, 1, dnnl_eltwise_checker))

    # Clip as a standalone op has no preceding conv or matmul, so it needs its own small pattern
    # instead of reusing _make_clip_pattern's conv/matmul-rooted shape.
    patterns.append(_op_pattern("dnnl.clip", "relax.clip", 3, dnnl_eltwise_checker))

    for suffix, op_name in [
        ("max_pool1d", "relax.nn.max_pool1d"),
        ("max_pool2d", "relax.nn.max_pool2d"),
        ("max_pool3d", "relax.nn.max_pool3d"),
        ("avg_pool1d", "relax.nn.avg_pool1d"),
        ("avg_pool2d", "relax.nn.avg_pool2d"),
        ("avg_pool3d", "relax.nn.avg_pool3d"),
    ]:
        patterns.append(_op_pattern(f"dnnl.{suffix}", op_name, 1, dnnl_pooling_checker))

    patterns.append(_op_pattern("dnnl.softmax", "relax.nn.softmax", 1))
    patterns.append(_op_pattern("dnnl.add", "relax.add", 2))
    patterns.append(_op_pattern("dnnl.multiply", "relax.multiply", 2))
    patterns.append(_op_pattern("dnnl.batch_norm", "relax.nn.batch_norm", 5))
    patterns.append(
        _op_pattern(
            "dnnl.global_avg_pool2d",
            "relax.nn.adaptive_avg_pool2d",
            1,
            dnnl_global_avg_pool2d_checker,
        )
    )
    return patterns


def make_qnn_conv2d_pattern() -> Pattern:
    """Base quantized-conv2d pattern: relax.nn.conv2d(data, weight) followed by
    relax.dequantize(out, scale, zp).

    Returns
    -------
    pattern : Pattern
        ("dnnl.qnn.conv2d", DFPattern, annotation_map, dnnl_qnn_checker), ready to hand to
        register_patterns() directly, matching the calling convention _op_pattern() and the
        other pattern builders in this file use. See _dnnl_patterns() below.
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
    """Base quantized-dense pattern: relax.matmul(data, weight) followed by relax.dequantize(out,
    scale, zp). "Dense" here means relax.matmul, matching how the rest of this file (dnnl.matmul,
    _FUSABLE_OPS, _sum_patterns) already treats relax.matmul as the dense op. There's no separate
    relax.nn.dense composite family here.

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

    # Bare conv/matmul base patterns, guarded by dnnl_conv_checker's int64 rejection, shared
    # across every rank and transpose variant plus matmul.
    for composite, op in [
        ("dnnl.conv1d", "relax.nn.conv1d"),
        ("dnnl.conv2d", "relax.nn.conv2d"),
        ("dnnl.conv3d", "relax.nn.conv3d"),
        ("dnnl.conv2d_transpose", "relax.nn.conv2d_transpose"),
        ("dnnl.conv3d_transpose", "relax.nn.conv3d_transpose"),
        ("dnnl.matmul", "relax.matmul"),
    ]:
        patterns.append(_op_pattern(composite, op, 2, dnnl_conv_checker))
    patterns.append(_op_pattern("dnnl.layer_norm", "relax.nn.layer_norm", 3))

    # Bare pooling, eltwise, softmax, add, multiply, batch_norm, and global average pool patterns.
    patterns.extend(_standalone_patterns())

    # Fused ops: bias-add, activation, and clip, for every (op, with_bias, activation)
    # combination. See _fused_patterns() to extend.
    patterns.extend(_fused_patterns())

    # Bias plus residual-sum, with an optional relu, fusion. See _sum_patterns() to extend.
    patterns.extend(_sum_patterns())

    # Base QNN (int8) patterns.
    patterns.append(make_qnn_conv2d_pattern())
    patterns.append(make_qnn_dense_pattern())

    return patterns


@functools.lru_cache(maxsize=1)
def _ordered_dnnl_patterns() -> list[Pattern]:
    """Returns patterns in match-priority order, with the most specific and largest fused
    subgraphs first. This keeps FuseOpsByPattern's greedy matching from letting a smaller,
    generic pattern like dnnl.conv2d_bias pre-empt a larger one like dnnl.conv2d_bias_relu that
    shares the same conv2d+bias prefix."""
    all_patterns = _dnnl_patterns()
    # A longer composite name generally means a more specific, larger pattern here, since names
    # are built as dnnl.<base>, optionally followed by _bias and then _<activation> or _clip.
    return sorted(all_patterns, key=lambda p: -len(p[0]))


register_patterns(_dnnl_patterns())


# Graph rewrites


def _unwrap_batch_norm_tuple_output(mod: tvm.IRModule) -> tvm.IRModule:
    """FuseOpsByPattern can only anchor a match on a CallNode binding, so the dnnl.batch_norm
    composite absorbs the bare batch_norm call but not the TupleGetItem(call, 0) that follows it
    at each call site. This rewrites the composite's Codegen function to return element 0 only,
    fixing its return type, then collapses the downstream TupleGetItem at every call site into
    the call result directly, reconstructing the call with an explicit ret_ty rather than reusing
    the stale, tuple-typed node.
    """
    targets: set[str] = set()
    for gvar, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if func.attrs is None or func.attrs.get("Codegen") != "dnnl":
            continue
        is_bn = {"flag": False}

        @visitor
        class _Finder(relax.PyExprVisitor):
            def visit_function_(self, f):
                if f.attrs is not None and f.attrs.get("Composite") == "dnnl.batch_norm":
                    is_bn["flag"] = True
                super().visit_function_(f)

        _Finder().visit_expr(func)
        if is_bn["flag"]:
            targets.add(gvar.name_hint)

    if not targets:
        return mod

    new_mod = tvm.IRModule(mod.functions)

    # Step 1: fix each target Codegen function to return element 0, and remember its new
    # (single-tensor) return type for step 2.
    target_ret_tys: dict[str, tvm.ir.Type] = {}
    for name in targets:
        gvar = new_mod.get_global_var(name)
        func = new_mod[gvar]
        seq = func.body
        assert isinstance(seq, relax.SeqExpr), f"{name}: expected SeqExpr body"

        ret_expr = seq.body  # the Var the function currently returns (the raw tuple)
        new_body = relax.SeqExpr(seq.blocks, relax.TupleGetItem(ret_expr, 0))

        old_ret_ty = func.ret_ty
        assert isinstance(old_ret_ty, relax.TupleType), (
            f"{name}: expected composite to return a tuple type, got {old_ret_ty}"
        )
        new_ret_ty = old_ret_ty.fields[0]
        target_ret_tys[name] = new_ret_ty

        new_mod[gvar] = relax.Function(func.params, new_body, new_ret_ty, func.is_pure, func.attrs)

    # Step 2: retype call sites and collapse the downstream TupleGetItem.
    # visit_call_ retypes the call, and this mutator's base var-remap then automatically
    # propagates that new type onto any bound Var before visit_tuple_getitem_ sees it. So by the
    # time visit_tuple_getitem_ resolves tuple_value, it may already be a plain tensor-typed var
    # rather than a tuple. We can't call super().visit_tuple_getitem_() here, since it assumes
    # tuple_value stays a TupleType and throws otherwise. Instead we resolve tuple_value
    # ourselves and short-circuit to it directly once it's no longer a tuple.
    @mutator
    class _CallSiteFixer(relax.PyExprMutator):
        def visit_call_(self, call: relax.Call):
            call = super().visit_call_(call)
            if isinstance(call.op, relax.GlobalVar) and call.op.name_hint in target_ret_tys:
                new_ret_ty = target_ret_tys[call.op.name_hint]
                return relax.Call(call.op, call.args, call.attrs, call.ty_args, ret_ty=new_ret_ty)
            return call

        def visit_tuple_getitem_(self, node: relax.TupleGetItem):
            new_tuple_value = self.visit_expr(node.tuple_value)
            if not isinstance(new_tuple_value.ty, relax.TupleType):
                assert node.index == 0, (
                    f"expected index 0 after batch_norm retype, got {node.index}"
                )
                return new_tuple_value
            return relax.TupleGetItem(new_tuple_value, node.index)

    fixer = _CallSiteFixer(new_mod)
    for gvar, func in list(new_mod.functions.items()):
        if isinstance(func, relax.Function) and gvar.name_hint not in targets:
            new_mod[gvar] = fixer.visit_expr(func)

    with tvm.transform.PassContext(opt_level=3):
        new_mod = relax.transform.Normalize()(new_mod)

    return new_mod


def rewrite_resnet_downsample(func: relax.Function) -> relax.Function:
    data = wildcard()
    weight_1x1 = wildcard()
    weight_3x3 = wildcard()

    conv1 = is_op("relax.nn.conv2d")(data, weight_1x1)
    relu1 = is_op("relax.nn.relu")(conv1)
    conv2 = is_op("relax.nn.conv2d")(relu1, weight_3x3)

    def _rewriter(expr, matches):
        matched_conv1 = matches[conv1]
        matched_conv2 = matches[conv2]

        if list(matched_conv1.attrs.strides) != [2, 2] or list(matched_conv2.attrs.strides) != [
            1,
            1,
        ]:
            return expr

        new_conv1 = relax.op.nn.conv2d(
            matches[data],
            matches[weight_1x1],
            strides=(1, 1),
            padding=matched_conv1.attrs.padding,
            dilation=matched_conv1.attrs.dilation,
            groups=matched_conv1.attrs.groups,
            data_layout=matched_conv1.attrs.data_layout,
            kernel_layout=matched_conv1.attrs.kernel_layout,
            out_layout=matched_conv1.attrs.out_layout,
            out_dtype=matched_conv1.attrs.out_dtype,
        )
        new_relu = relax.op.nn.relu(new_conv1)
        new_conv2 = relax.op.nn.conv2d(
            new_relu,
            matches[weight_3x3],
            strides=(2, 2),
            padding=matched_conv2.attrs.padding,
            dilation=matched_conv2.attrs.dilation,
            groups=matched_conv2.attrs.groups,
            data_layout=matched_conv2.attrs.data_layout,
            kernel_layout=matched_conv2.attrs.kernel_layout,
            out_layout=matched_conv2.attrs.out_layout,
            out_dtype=matched_conv2.attrs.out_dtype,
        )
        return new_conv2

    return rewrite_call(conv2, _rewriter, func)


@tvm.transform.module_pass(opt_level=3, name="ResNetV1Rewrite")
class ResNetV1Rewrite:
    def transform_module(self, mod, ctx):
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                mod[gv] = rewrite_resnet_downsample(func)
        return mod


def rewrite_pad_avg_pool2d(mod: tvm.IRModule) -> tvm.IRModule:
    """Folds a directly-preceding constant zero-pad into avg_pool2d's own padding attribute.

    This only happens when all of the following hold:
      - the pad is a constant pad of value 0.0
      - pad widths on the N and C axes are 0, meaning the padding is spatial only
      - avg_pool2d doesn't already carry non-zero padding of its own
      - the pad has exactly one consumer, this avg_pool2d call, since removing it otherwise
        would change another consumer's input

    Pad and avg_pool2d pairs that don't match these conditions are left alone and simply run as
    two separate, non-offloaded ops. That's still correct, just not fused, and this pass never
    guesses.

    The pad_width indexing below assumes NCHW axis order, matching avg_pool2d's default layout
    attr. An NHWC layout would need different axis indices, which this doesn't handle yet.
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

            # 2) pad_width is a flat list of (before, after) per axis, NCHW order assumed.
            pad_width = [int(v) for v in pad_attrs.pad_width]

            # pad_width is stored as (before, after) pairs: axis 0 is N, axis 1 is C, axes 2
            # and 3 are H and W.
            n_before, n_after = pad_width[0], pad_width[1]
            c_before, c_after = pad_width[2], pad_width[3]
            h_before, h_after = pad_width[4], pad_width[5]
            w_before, w_after = pad_width[6], pad_width[7]
            if (n_before, n_after, c_before, c_after) != (0, 0, 0, 0):
                return expr  # never fold padding on the batch or channel axes

            # 3) avg_pool2d must not already have non-zero padding.
            existing_padding = [int(v) for v in pool_attrs.padding]
            if any(p != 0 for p in existing_padding):
                return expr

            # 4) pad must have a single consumer, this pool call.
            padded_arg = pool_call.args[0]
            if not isinstance(padded_arg, relax.Var) or use_count.get(id(padded_arg), 0) > 1:
                return expr

            merged_padding = [h_before, w_before, h_after, w_after]  # top, left, bottom, right

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
    """Rewrites multiple operators into a TVM native layer normalization."""
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


def rewrite_batch_norm(mod: tvm.IRModule) -> tvm.IRModule:
    """Rewrites the decomposed batch_norm primitive chain, produced by
    DecomposeOpsForInference, back into a single relax.nn.batch_norm call.

    The decomposed shape is:
        (x - mean) / sqrt(var + eps) * gamma + beta
    with mean, var, gamma, and beta each pre-broadcast via expand_dims([0, 2, 3]), i.e. NCHW with
    channel axis 1. Without this rewrite, no batch_norm call survives for the dnnl.batch_norm
    standalone pattern to match against.
    """
    x_pat = wildcard()
    mean_pat = wildcard()
    var_pat = wildcard()
    gamma_pat = wildcard()
    beta_pat = wildcard()
    eps_pat = wildcard()

    mean_exp = is_op("relax.expand_dims")(mean_pat)
    diff = is_op("relax.subtract")(x_pat, mean_exp)

    var_exp = is_op("relax.expand_dims")(var_pat)
    added_eps = is_op("relax.add")(var_exp, eps_pat)
    deno = is_op("relax.sqrt")(added_eps)
    normed = is_op("relax.divide")(diff, deno)

    gamma_exp = is_op("relax.expand_dims")(gamma_pat)
    scaled = is_op("relax.multiply")(normed, gamma_exp)

    beta_exp = is_op("relax.expand_dims")(beta_pat)
    shifted = is_op("relax.add")(scaled, beta_exp)

    pattern = shifted

    def rewriter(expr, matches):
        x = matches[x_pat]
        mean = matches[mean_pat]
        var = matches[var_pat]
        gamma = matches[gamma_pat]
        beta = matches[beta_pat]
        bn_out = relax.op.nn.batch_norm(x, gamma, beta, mean, var, axis=1)
        return relax.TupleGetItem(bn_out, 0)

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            new_mod[gvar] = rewrite_call(pattern, rewriter, func)
    return new_mod


def rewrite_dense_bias_gelu_reshape_last(mod: tvm.IRModule) -> tvm.IRModule:
    """Reorders reshape operators for dense_bias_gelu and dense_bias fusion."""

    def _apply(func: relax.Function, has_gelu: bool) -> relax.Function:
        data_pat = wildcard()
        weight_pat = wildcard()
        bias_pat = wildcard()
        const1 = wildcard()
        const2 = wildcard()
        const3 = wildcard()

        den = is_op("relax.matmul")(data_pat, weight_pat)
        re_den = is_op("relax.reshape")(den, wildcard())
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
            added_new = relax.op.add(den_new, bias)

            if not has_gelu:
                return relax.op.reshape(added_new, shape)

            gelu_new = relax.op.nn.gelu(added_new)
            return relax.op.reshape(gelu_new, shape)

        return rewrite_call(pattern, rewriter, func)

    new_mod = tvm.IRModule(mod.functions)
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            f1 = _apply(func, has_gelu=True)
            f2 = _apply(f1, has_gelu=False)
            new_mod[gvar] = f2

    return new_mod


# QNN legalization


def _try_fold_qdq_constant(q_expr, scale_expr, zp_expr) -> np.ndarray | None:
    """If q, scale, and zp are all compile-time relax.Constant values and scale/zp are per-tensor
    (size 1), returns the dequantized float32 numpy array. Otherwise returns None, and the caller
    should leave the chain alone rather than guess. Shared by weight-folding and opportunistic
    bias-folding below.
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

    # Relax's dequantize doesn't accept a float32 zero point, so for now this only supports
    # zero_point equal to 0.
    if float(out_zp_np.reshape(-1)[0]) != 0.0:
        return None

    dq_zp_np = np.array(0, dtype="int32")

    return (
        relax.const(dq_scale_np, "float32"),
        relax.const(dq_zp_np, "int32"),
    )


def legalize_qnn_op_for_dnnl(mod: tvm.IRModule) -> tvm.IRModule:
    """LegalizeQnnOpForDnnl rewrites a quantized conv2d/matmul chain:

        dequantize(data_q) and dequantize(weight_q)
            -> nn.conv2d/matmul
            -> [add(bias)]                        # optional: "qnn.conv2d/qnn.dense + bias"
            -> quantize(out, out_scale, out_zp)    # the "+ requantize" step. Relax has no
                                                    # dedicated requantize op, so this is how the
                                                    # "qnn.conv2d + bias + requantize" chain is
                                                    # actually spelled in Relax's QDQ-only IR.

    It does three things:

      1. Folds the weight-side dequantize to a float32 constant whenever it's compile-time
         foldable.
      2. Leaves the bias, if present, as a plain add. It opportunistically folds it to a float32
         constant too when it happens to also be a compile-time dequantize(int32_const, ...)
         chain, the common shape for a real quantized bias where scale is data_scale times
         weight_scale. Anything else, whether already float or a dynamic Var, passes through
         untouched, since make_fused_bias_activation_pattern's bias slot is a bare wildcard and
         either form already matches dnnl.conv2d_bias or dnnl.matmul_bias with no further
         rewriting needed.
      3. Rewrites the terminal quantize(out, out_scale, out_zp) into the algebraically equivalent
         dequantize(out, scale', zp'), so downstream pattern matching only ever needs to
         recognize one terminal-node shape, the same one make_qnn_conv2d_pattern and
         make_qnn_dense_pattern already look for. See _try_requantize_consts() for the exact
         derivation and its rounding-mode caveat.

    The data-side dequantize is deliberately left untouched, same as before: whatever consumes
    the legalized chain treats data and weight as opaque wildcards, so it doesn't matter whether
    the data-side dequantize has been folded or still runs standalone ahead of the offloaded
    region.
    """
    data_q = wildcard()
    weight_q = wildcard()
    data_scale, data_zp = wildcard(), wildcard()
    weight_scale, weight_zp = wildcard(), wildcard()
    bias = wildcard()
    out_scale = is_const()
    out_zp = is_const()

    dq_data = is_op("relax.dequantize")(data_q, data_scale, data_zp)
    folded_weight = is_const()
    dq_weight = is_op("relax.dequantize")(weight_q, weight_scale, weight_zp) | folded_weight

    def _make_rewriter(op_out_pat, has_bias: bool):
        def rewriter(expr, matches):
            # expr is the matched root, the terminal relax.quantize(...) call.
            if folded_weight in matches:
                new_weight = matches[folded_weight]
            else:
                weight_np = _try_fold_qdq_constant(
                    matches[weight_q], matches[weight_scale], matches[weight_zp]
                )
                if weight_np is None:
                    return expr  # weight isn't compile-time-foldable, leave the chain alone

            op_call = matches[op_out_pat]
            new_op_call = relax.Call(
                op_call.op, [matches[dq_data], new_weight], attrs=op_call.attrs
            )

            if has_bias:
                bias_expr = matches[bias]
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

            # Keep it wrapped in the ORIGINAL relax.quantize call — type-preserving,
            # so rewrite_call is safe here. The terminal quantize->dequantize swap
            # happens afterward in _QnnTerminalRewriter.
            return relax.op.quantize(
                new_inner,
                expr.args[1],
                expr.args[2],
                out_dtype=expr.attrs.out_dtype,
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

    # Terminal quantize -> dequantize swap changes dtype, so it must go through
    # a proper mutator/BlockBuilder emit rather than rewrite_call, or the
    # rebound Var (and, if it's the return value, the function's
    # ret_struct_info) keeps its stale int-dtype struct info and fails the
    # well-formedness check.
    fixer = _QnnTerminalRewriter(new_mod)
    for gvar, func in list(new_mod.functions.items()):
        if isinstance(func, relax.Function):
            new_mod[gvar] = fixer.visit_expr(func)

    with tvm.transform.PassContext(opt_level=3):
        new_mod = relax.transform.Normalize()(new_mod)

    return new_mod


def partition_for_dnnl(
    mod: tvm.IRModule,
    params: dict[str, tvm.runtime.Tensor] | None = None,
    alter_layout: bool = True,
    prune_subgraphs: bool = True,
    run_codegen: bool = True,
) -> tvm.IRModule:
    # Apply the downsample rewrite before any partitioning begins.
    mod = ResNetV1Rewrite()(mod)

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
        # This is derived entirely from _CONV_LAYOUT_QUERY_SPECS, the single source of truth for
        # conv layouts, including the IO-swapped kernel layout for the transpose variants.
        desired_layouts = {
            op_name: default_layout
            for op_name, (_rank, _is_transpose, default_layout) in _CONV_LAYOUT_QUERY_SPECS.items()
        }
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.ConvertLayout(desired_layouts)(mod)
            mod = relax.transform.FoldConstant()(mod)

    mod = rewrite_layer_norm(mod)
    mod = rewrite_dense_bias_gelu_reshape_last(mod)
    mod = rewrite_batch_norm(mod)
    mod = rewrite_pad_avg_pool2d(mod)
    mod = legalize_qnn_op_for_dnnl(mod)
    mod = relax.transform.FoldConstant()(mod)

    dnnl_patterns = _ordered_dnnl_patterns()
    byoc_seq = tvm.transform.Sequential(
        [
            FuseOpsByPattern(dnnl_patterns, annotate_codegen=True),
            MergeCompositeFunctions(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)

    mod = _unwrap_batch_norm_tuple_output(mod)

    if prune_subgraphs:
        mod = prune_dnnl_subgraphs(mod)

    if run_codegen:
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.RunCodegen()(mod)

    return mod


# Subgraph pruning


def _count_compute_ops(mod: tvm.IRModule, func: relax.Function) -> int:
    count = 0
    seen_globals: set = set()

    @visitor
    class _Counter(relax.PyExprVisitor):
        def visit_call_(self, call: relax.Call):
            nonlocal count
            if isinstance(call.op, tvm.ir.Op) and call.op.name in _DNNL_COMPUTE_OPS:
                count += 1
            elif isinstance(call.op, relax.GlobalVar) and call.op not in seen_globals:
                gv = call.op
                seen_globals.add(gv)
                if gv in mod.functions:
                    self.visit_expr(mod[gv])
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
        new_mod = relax.transform.LambdaLift()(new_mod)
        new_mod = relax.transform.InlinePrivateFunctions()(new_mod)
        new_mod = relax.transform.DeadCodeElimination(["main"])(new_mod)

    return new_mod
