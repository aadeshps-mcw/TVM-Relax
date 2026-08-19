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

Still worth independently re-verifying against the current relax op registry:
  - QNN/int8 quantized fusion patterns (kept from the "broad" revision, but
    its own docstring never claimed these were cross-checked against Relay).
  - Exact op names for gelu ("relax.nn.gelu" vs "relax.gelu") and swish/mish
    on your TVM revision.
"""

import tvm
from tvm import relax, tirx
from tvm.relax.backend.patterns import make_matmul_dequantize_pattern
from tvm.relax.dpl import (
    is_const,
    is_expr,
    is_op,
    make_fused_bias_activation_pattern,
    rewrite_call,
    wildcard,
)
from tvm.relax.expr_functor import mutator, visitor
from tvm.relax.transform import (
    FuseOpsByPattern,
    MergeCompositeFunctions,
    PatternCheckContext,
)

from ..pattern_registry import Pattern, register_patterns

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

# Ops for which DNNL implements a fused bias-add + activation post-op chain.
# layer_norm is deliberately excluded -- it isn't followed by a DNNL-fusable
# activation the way conv/matmul are.
_FUSABLE_OPS: list[str] = [
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
]

# Activations DNNL can fuse as a post-op, plus the runtime op name each maps to
# for _validate_eltwise_op_name / the eltwise checker below. "swish" is kept in
# this generalized list (rather than hand-special-cased to conv2d only) since
# silu == swish(beta=1) and there's no reason bias+swish fusion is conv2d-specific.
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


def _get_dtype(node):
    ty = getattr(node, "ty", None)
    if ty is None:
        return None
    return getattr(ty, "dtype", None)


def _reject_int64(call):
    """Safety guard: oneDNN does not natively support int64."""
    out_dtype = _get_dtype(call)
    if out_dtype == "int64":
        return False

    for arg in call.args:
        if _get_dtype(arg) == "int64":
            return False

    if call.attrs is not None and hasattr(call.attrs, "out_dtype"):
        if str(call.attrs.out_dtype) == "int64":
            return False

    return True


def dnnl_pooling_checker(ctx) -> bool:
    call = ctx.matched_expr
    if hasattr(call.attrs, "ceil_mode") and call.attrs.ceil_mode:
        return False
    return _reject_int64(call)


def dnnl_global_avg_pool2d_checker(ctx) -> bool:
    """Only true global average pooling (output_size == (1, 1)) is valid here.
    A general adaptive_avg_pool2d with a larger output_size is a materially
    different computation -- oneDNN's pooling_forward primitive can't express
    per-window variable kernel sizes, so offloading it produces a malformed
    descriptor at runtime ('could not create a descriptor for a pooling
    forward propagation primitive'). Non-(1,1) adaptive pools correctly fall
    through to TVM's native implementation instead.
    """
    call = ctx.matched_expr
    output_size = tuple(int(v) for v in call.attrs.output_size)
    if output_size != (1, 1):
        return False
    return _reject_int64(call)


def dnnl_conv_checker(ctx) -> bool:
    """int64-rejection guard shared by every bare conv/matmul base pattern
    (conv1d/2d/3d, transposed variants, matmul)."""
    return _reject_int64(ctx.matched_expr)


def _is_valid_broadcast_bias(bias_dims, channel_axis, channel_dim) -> bool:
    """True if bias_dims describes a scalar, or a tensor with exactly one
    non-1-sized dimension that matches the op's channel dimension (covers
    both the 1D (oc,) convention and the (oc, 1, 1)-style NCHW-broadcast
    convention -- any rank is fine as long as there's only one real axis)."""
    if len(bias_dims) == 0:
        return True

    non_unit = [d for d in bias_dims if not (isinstance(d, tirx.IntImm) and d.value == 1)]
    if len(non_unit) == 0:
        return True  # all-1s, trivially broadcastable
    if len(non_unit) > 1:
        return False  # more than one "real" axis -- not a simple per-channel bias

    dim = non_unit[0]
    if isinstance(dim, tirx.IntImm) and isinstance(channel_dim, tirx.IntImm):
        return dim.value == channel_dim.value
    return True  # dynamic dim -- can't statically rule out, don't block the match


def dnnl_eltwise_checker(ctx) -> bool:
    call = ctx.matched_expr
    if not isinstance(call.op, tvm.ir.Op):
        return True

    op_name = call.op.name.replace("relax.", "").replace("nn.", "")
    if op_name not in SUPPORTED_ELTWISE:
        return False
    return _reject_int64(call)


def _validate_eltwise_op_name(op_name: str) -> None:
    """Strict guard: raises rather than silently excluding, for op names that
    reach DNNL codegen through a path that should always be a known, closed set
    (as opposed to dnnl_eltwise_checker's pattern-matching context, where
    returning False just means 'don't offload this' -- appropriate for
    runtime-shaped inputs, but not for the fixed, compile-time-known set of
    activation post-ops below)."""
    if op_name not in SUPPORTED_ELTWISE:
        raise ValueError(
            f"Unsupported DNNL eltwise/activation post-op: {op_name!r}. "
            f"Supported: {sorted(SUPPORTED_ELTWISE)}"
        )


def _op_pattern(composite_name: str, op_name: str, num_args: int, checker=None) -> Pattern:
    """A pattern matching a single op called with ``num_args`` wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    pat = is_op(op_name)(*args)
    # The empty dictionary {} is required as the 3rd element for annotations
    return (composite_name, pat, {}, checker) if checker else (composite_name, pat, {})


def _make_conv2d_dequantize_pattern():
    """conv2d(data, weight) -> dequantize(out, scale, zp). Runs conv2d in float;
    scale/zp are compile-time constants folded into a DNNL output rescale post-op
    (reuses the existing o_scl_idx/dst_zp_idx runtime machinery -- see ParseAttrs
    in dnnl_json_runtime.cc). This does NOT execute the convolution itself in int8."""
    data = wildcard()
    weight = wildcard()
    out = is_op("relax.nn.conv2d")(data, weight)

    scale = is_const()
    zp = is_const()
    out = is_op("relax.dequantize")(out, scale, zp)

    return out, {"data": data, "weight": weight, "scale": scale, "zp": zp}


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


def _sum_pattern(op_name: str, channel_axis: int, with_relu: bool) -> Pattern:
    """<op>(data, weight) + bias + residual, optionally followed by relu.

    channel_axis is where the op's output channel dimension lives (1 for
    NCHW-style conv output, -1 for matmul's last-dim output) -- used by the
    predicate below to check the bias is actually shaped like a per-channel
    bias, not some other broadcastable-but-wrong-length tensor.

    Distinct from bias-only fusion: this matches a *second* add whose other
    operand is an external residual tensor, not the bias.
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

        # NOTE: verify .ty / .ty.shape access against your TVM build -- grounded
        # in the existing tensorrt resize2d predicate's use of `.ty.ndim` /
        # `.ty.dtype` / `.ty.shape`, but this predicate additionally indexes
        # into per-dimension values, which that example did not do.
        if op_expr.ty.shape is None or data2_expr.ty.shape is None:
            return False
        out_dims = list(op_expr.ty.shape.values)
        sum_dims = list(data2_expr.ty.shape.values)

        # Residual add must be a true elementwise match against the op's own
        # output shape.
        if len(out_dims) != len(sum_dims):
            return False
        for a, b in zip(out_dims, sum_dims):
            if isinstance(a, tirx.IntImm) and isinstance(b, tirx.IntImm) and a.value != b.value:
                return False

        # Bias must be a scalar or a 1D tensor sized to the op's channel
        # dimension -- catches a bias that merely happens to be broadcastable
        # but isn't actually a per-channel bias.
        if bias_expr.ty.shape is not None:
            bias_dims = list(bias_expr.ty.shape.values)
            channel_dim = out_dims[channel_axis]
            if not _is_valid_broadcast_bias(bias_dims, channel_axis, channel_dim):
                return False

        return True

    return (
        name,
        root,
        {"op": op, "bias": bias, "data2": data2, "root": root},
        check,
    )


def _sum_patterns() -> list[Pattern]:
    """Bias + residual-sum (+ activation) fusion. Scope matched 1:1 against the
    legacy Relay pattern table: conv2d gets both the relu and no-relu sum
    variant; matmul gets only the no-relu sum variant."""
    return [
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=True),
        _sum_pattern("relax.nn.conv2d", channel_axis=1, with_relu=False),
        _sum_pattern("relax.matmul", channel_axis=-1, with_relu=False),
    ]


def _make_fused_variants(prefix: str, op_name: str) -> list[Pattern]:
    """Build the full bias/activation/clip pattern family for a single base op.

    e.g. prefix="dnnl.conv2d", op_name="relax.nn.conv2d" produces
    dnnl.conv2d_relu, dnnl.conv2d_bias_relu, ..., dnnl.conv2d_clip,
    dnnl.conv2d_bias_clip, dnnl.conv2d_bias.

    Extending fusion coverage (a new activation, a new fusable op) means
    adding one entry to _ACTIVATIONS or _FUSABLE_OPS -- nothing else in this
    function or its caller changes.
    """
    patterns: list[Pattern] = []

    for suffix, act_op in _ACTIVATIONS:
        act_short = act_op.replace("relax.", "").replace("nn.", "")
        _validate_eltwise_op_name(act_short)

        pat_bias = make_fused_bias_activation_pattern(op_name, with_bias=True, activation=act_op)
        patterns.append((f"{prefix}_bias_{suffix}", pat_bias))

        pat_nobias = make_fused_bias_activation_pattern(op_name, with_bias=False, activation=act_op)
        patterns.append((f"{prefix}_{suffix}", pat_nobias))

    # clip fusion isn't expressible via make_fused_bias_activation_pattern's
    # activation arg, so it's built separately.
    patterns.append((f"{prefix}_clip", _make_clip_pattern(op_name, with_bias=False)))
    patterns.append((f"{prefix}_bias_clip", _make_clip_pattern(op_name, with_bias=True)))

    # Bias-only, no activation.
    pat_bias_only = make_fused_bias_activation_pattern(op_name, with_bias=True, activation=None)
    patterns.append((f"{prefix}_bias", pat_bias_only))

    return patterns


def _standalone_patterns() -> list[Pattern]:
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

    # clip as a standalone op has no preceding conv/matmul, so this needs its
    # own tiny pattern rather than reusing _make_clip_pattern's conv/matmul
    # -rooted shape.
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


def _dnnl_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    # 1. Base ops (no-fusion fallback): every bare conv variant + matmul +
    # layer_norm, each guarded against int64 (layer_norm excluded from the
    # guard family since it was never observed producing int64 issues, matching
    # the original scope).
    _BASE_OPS = [
        ("dnnl.conv1d", "relax.nn.conv1d", 2),
        ("dnnl.conv2d", "relax.nn.conv2d", 2),
        ("dnnl.conv3d", "relax.nn.conv3d", 2),
        ("dnnl.conv2d_transpose", "relax.nn.conv2d_transpose", 2),
        ("dnnl.conv3d_transpose", "relax.nn.conv3d_transpose", 2),
        ("dnnl.matmul", "relax.matmul", 2),
    ]
    for composite, op_name, nargs in _BASE_OPS:
        patterns.append(_op_pattern(composite, op_name, nargs, dnnl_conv_checker))
    patterns.append(_op_pattern("dnnl.layer_norm", "relax.nn.layer_norm", 3))

    # 2. Fused variants (bias / activation / clip), generated across every
    # fusable op via itertools-style enumeration (see _FUSABLE_OPS).
    for op_name in _FUSABLE_OPS:
        base = op_name.rsplit(".", 1)[-1]
        prefix = f"dnnl.{base}"
        patterns.extend(_make_fused_variants(prefix, op_name))

    # 3. Bias + residual-sum (+ optional relu) fusion, with real shape
    # validation (see _sum_pattern's predicate).
    patterns.extend(_sum_patterns())

    # 4. QNN / dequantize fusion (int8-adjacent: runs in float, folds
    # scale/zp into a DNNL output-rescale post-op). Kept from the broader
    # revision -- still worth an independent re-check against Relay's legacy
    # qnn.conv2d/qnn.dense + requantize + sum patterns before relying on it
    # for anything beyond simple dequantize folding.
    out, ann = _make_conv2d_dequantize_pattern()
    patterns.append(("dnnl.qnn.conv2d", out, ann))
    out, ann = make_matmul_dequantize_pattern(transposed_rhs=False)
    patterns.append(("dnnl.qnn.matmul", out, ann))

    # 5. Standalone ops (no preceding conv/matmul to fuse with).
    patterns.extend(_standalone_patterns())

    return patterns


def _ordered_dnnl_patterns() -> list[Pattern]:
    """Patterns in match-priority order: most specific (largest fused subgraph)
    first, so FuseOpsByPattern's greedy matching doesn't let a smaller generic
    pattern (e.g. dnnl.conv2d_bias) pre-empt a larger one (e.g.
    dnnl.conv2d_bias_relu) that shares the same conv2d+bias prefix."""
    all_patterns = _dnnl_patterns()
    # Longer composite name generally implies a more specific / larger pattern
    # here since we build names as dnnl.<base>[_bias][_<activation>].
    return sorted(all_patterns, key=lambda p: -len(p[0]))


register_patterns(_dnnl_patterns())


def _unwrap_batch_norm_tuple_output(mod: tvm.IRModule) -> tvm.IRModule:
    """FuseOpsByPattern can only anchor a match on a CallNode binding, so the
    dnnl.batch_norm composite absorbs the bare batch_norm call but not the
    TupleGetItem(call, 0) that follows it at each call site. This rewrites
    the composite's Codegen function to return element 0 only (fixing its
    return type), then collapses the downstream TupleGetItem at every
    call site into the call result directly -- reconstructing the call with
    an explicit ret_ty rather than reusing the stale (3-tuple-typed) node.
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

    # Step 1: fix each target Codegen function to return element 0, and
    # remember its new (single-tensor) return type for step 2.
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
    # NOTE: visit_call_ retypes the call; this mutator's base var-remap then
    # automatically propagates that new type onto any Var (e.g. `lv`) bound
    # to the call, *before* visit_tuple_getitem_ sees it. So by the time
    # visit_tuple_getitem_ resolves tuple_value, it may already be a plain
    # tensor-typed var rather than a tuple -- we must NOT call
    # super().visit_tuple_getitem_() (it assumes tuple_value stays a
    # TupleType and throws otherwise). Instead resolve tuple_value ourselves
    # and short-circuit to it directly when it's no longer a tuple.
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


def partition_for_dnnl(
    mod: tvm.IRModule,
    params: dict[str, tvm.runtime.Tensor] | None = None,
    alter_layout: bool = True,
    prune_subgraphs: bool = True,
    run_codegen: bool = True,
) -> tvm.IRModule:
    # Apply the downsample rewrite before any partitioning begins
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
    mod = rewrite_batch_norm(mod)

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


_DNNL_COMPUTE_OPS = {
    "relax.nn.conv1d",
    "relax.nn.conv2d",
    "relax.nn.conv3d",
    "relax.nn.conv2d_transpose",
    "relax.nn.conv3d_transpose",
    "relax.matmul",
    "relax.nn.batch_matmul",
    "relax.nn.layer_norm",
    # Standalone ops: for these, the op itself IS what DNNL dispatches on, so
    # a subgraph containing only one of these is not "trivial" the way e.g. a
    # lone leftover bias-add would be. Without these, every standalone
    # composite gets demoted by prune_dnnl_subgraphs even though the
    # composite/codegen promotion worked correctly.
    "relax.abs",
    "relax.exp",
    "relax.log",
    "relax.sqrt",
    "relax.round",
    "relax.tanh",
    "relax.sigmoid",
    "relax.clip",
    "relax.nn.adaptive_avg_pool2d",
    "relax.nn.max_pool1d",
    "relax.nn.max_pool2d",
    "relax.nn.max_pool3d",
    "relax.nn.avg_pool1d",
    "relax.nn.avg_pool2d",
    "relax.nn.avg_pool3d",
    "relax.nn.softmax",
    "relax.nn.batch_norm",
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
                gv = call.op
                if gv not in seen_globals and gv in mod.functions:
                    seen_globals.add(gv)
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


def rewrite_batch_norm(mod: tvm.IRModule) -> tvm.IRModule:
    """Rewrite the decomposed batch_norm primitive chain (produced by
    DecomposeOpsForInference) back into a single relax.nn.batch_norm call.

    Decomposed shape:
        (x - mean) / sqrt(var + eps) * gamma + beta
    with mean/var/gamma/beta each pre-broadcast via expand_dims([0, 2, 3])
    (i.e. NCHW, channel axis = 1). Without this rewrite, no batch_norm call
    survives for the dnnl.batch_norm standalone pattern to match against.
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
