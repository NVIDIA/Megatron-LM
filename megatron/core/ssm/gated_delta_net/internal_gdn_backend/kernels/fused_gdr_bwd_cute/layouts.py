"""Trace-time layout contracts for the SM100 fused GDR backward kernel."""

import hashlib
from dataclasses import dataclass
from typing import Any

import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05


@dataclass(frozen=True)
class MmaVariantSpec:
    """One tcgen05 orientation variant within a logical MMA family."""

    name: str
    family: str
    tile: tuple[int, int, int]
    a_major: str
    b_major: str


@dataclass(frozen=True)
class MmaOperationSpec:
    """One scheduled MMA mapped to a variant and physical SMEM views."""

    name: str
    stage: str
    variant: str
    a_view: str
    b_view: str


@dataclass(frozen=True)
class BuiltMmaVariant:
    """A variant and the exact staged operand layouts built for it."""

    spec: MmaVariantSpec
    tiled_mma: Any
    smem_a_staged: Any
    smem_b_staged: Any

    @property
    def smem_a(self):
        return _mma_operand_view(self.smem_a_staged)

    @property
    def smem_b(self):
        return _mma_operand_view(self.smem_b_staged)


@dataclass(frozen=True)
class MmaVariantBindings:
    """The ten variants, named so device wiring never uses tuple indices."""

    mm_64_64_128_k_k: BuiltMmaVariant
    mm_64_128_64_mn_mn: BuiltMmaVariant
    mm_64_128_64_k_mn: BuiltMmaVariant
    mm_128_64_64_mn_mn: BuiltMmaVariant
    mm_128_64_128_mn_k: BuiltMmaVariant
    mm_128_64_128_k_k: BuiltMmaVariant
    mm_64_64_64_mn_mn: BuiltMmaVariant
    mm_64_64_64_k_k: BuiltMmaVariant
    mm_64_128_128_k_mn: BuiltMmaVariant
    mm_64_128_128_k_k: BuiltMmaVariant

    def __getitem__(self, name: str) -> BuiltMmaVariant:
        return getattr(self, name)


@dataclass(frozen=True)
class CanonicalLayouts:
    """The six canonical physical SMEM mappings."""

    token_direct: Any
    token_transposed: Any
    square_direct: Any
    square_transposed: Any
    state_direct: Any
    state_transposed: Any

    def __getitem__(self, name: str):
        return getattr(self, name)


@dataclass(frozen=True)
class MmaOperandLayouts:
    """Exact A/B layouts consumed by one oriented MMA variant."""

    a_staged: Any
    b_staged: Any
    a: Any
    b: Any


@dataclass(frozen=True)
class PackedMmaLayouts:
    """Named aliases whose 64x128 N mode is explicitly packed."""

    mm_64_128_64_mn_mn: MmaOperandLayouts
    mm_64_128_64_k_mn: MmaOperandLayouts
    mm_64_128_128_k_mn: MmaOperandLayouts
    mm_64_128_128_k_k: MmaOperandLayouts
    mm_64_64_128_k_k: MmaOperandLayouts
    mm_64_64_64_mn_mn: MmaOperandLayouts
    mm_64_64_64_k_k: MmaOperandLayouts

    def __getitem__(self, name: str) -> MmaOperandLayouts:
        return getattr(self, name)


@dataclass(frozen=True)
class MmaOperationBinding:
    """Resolved production wiring for one logical MMA operation."""

    spec: MmaOperationSpec
    variant: BuiltMmaVariant
    operands: MmaOperandLayouts
    canonical_a: Any
    canonical_b: Any

    @property
    def tiled_mma(self):
        return self.variant.tiled_mma


@dataclass(frozen=True)
class MmaOperationBindings:
    """All nineteen logical operations in schedule-name form."""

    p: MmaOperationBinding
    dvprime_state: MmaOperationBinding
    dvprime_pg: MmaOperationBinding
    u_state: MmaOperationBinding
    dv: MmaOperationBinding
    vprime: MmaOperationBinding
    dag: MmaOperationBinding
    dpg: MmaOperationBinding
    dk_state: MmaOperationBinding
    dq_state: MmaOperationBinding
    dk_state_g: MmaOperationBinding
    dq_dp: MmaOperationBinding
    dh_k: MmaOperationBinding
    dk_dp: MmaOperationBinding
    da_left: MmaOperationBinding
    at: MmaOperationBinding
    da_right: MmaOperationBinding
    dh_q: MmaOperationBinding
    dk_da: MmaOperationBinding

    def __getitem__(self, name: str) -> MmaOperationBinding:
        return getattr(self, name)


@dataclass(frozen=True)
class BackwardLayoutPlan:
    """Named trace-time-only layout plan; never pass it to ``@cute.kernel``."""

    variants: MmaVariantBindings
    canonical: CanonicalLayouts
    canonical_staged: CanonicalLayouts
    operations: MmaOperationBindings
    packed: PackedMmaLayouts
    physical_columns_by_variant: dict[str, int]


@dataclass(frozen=True)
class TmaDescriptorBundle:
    """Tensor-bound descriptors kept only for the duration of one trace."""

    q: Any
    k: Any
    v: Any
    g: Any
    h: Any
    a: Any
    do: Any
    beta: Any
    dq_store: Any


MMA_VARIANT_SPECS = (
    MmaVariantSpec("mm_64_64_128_k_k", "mm_64_64_128", (64, 64, 128), "K", "K"),
    MmaVariantSpec("mm_64_128_64_mn_mn", "mm_64_128_64", (64, 128, 64), "MN", "MN"),
    MmaVariantSpec("mm_64_128_64_k_mn", "mm_64_128_64", (64, 128, 64), "K", "MN"),
    MmaVariantSpec("mm_128_64_64_mn_mn", "mm_128_64_64", (128, 64, 64), "MN", "MN"),
    MmaVariantSpec("mm_128_64_128_mn_k", "mm_128_64_128", (128, 64, 128), "MN", "K"),
    MmaVariantSpec("mm_128_64_128_k_k", "mm_128_64_128", (128, 64, 128), "K", "K"),
    MmaVariantSpec("mm_64_64_64_mn_mn", "mm_64_64_64", (64, 64, 64), "MN", "MN"),
    MmaVariantSpec("mm_64_64_64_k_k", "mm_64_64_64", (64, 64, 64), "K", "K"),
    MmaVariantSpec("mm_64_128_128_k_mn", "mm_64_128_128", (64, 128, 128), "K", "MN"),
    MmaVariantSpec("mm_64_128_128_k_k", "mm_64_128_128", (64, 128, 128), "K", "K"),
)


MMA_OPERATION_SPECS = (
    MmaOperationSpec("p", "00", "mm_64_64_128_k_k", "token_direct", "token_direct"),
    MmaOperationSpec("dvprime_state", "01", "mm_64_128_128_k_mn", "token_direct", "state_transposed"),
    MmaOperationSpec("dvprime_pg", "02", "mm_64_128_64_mn_mn", "square_transposed", "token_transposed"),
    MmaOperationSpec("u_state", "03", "mm_64_128_128_k_mn", "token_direct", "state_transposed"),
    MmaOperationSpec("dv", "04", "mm_64_128_64_mn_mn", "square_transposed", "token_transposed"),
    MmaOperationSpec("vprime", "05a", "mm_64_128_64_k_mn", "square_direct", "token_transposed"),
    MmaOperationSpec("dag", "05b", "mm_64_64_128_k_k", "token_direct", "token_direct"),
    MmaOperationSpec("dpg", "06", "mm_64_64_128_k_k", "token_direct", "token_direct"),
    MmaOperationSpec("dk_state", "07", "mm_64_128_128_k_k", "token_direct", "state_direct"),
    MmaOperationSpec("dq_state", "08", "mm_64_128_128_k_k", "token_direct", "state_direct"),
    MmaOperationSpec("dk_state_g", "09", "mm_64_128_128_k_k", "token_direct", "state_direct"),
    MmaOperationSpec("dq_dp", "10", "mm_64_128_64_k_mn", "square_direct", "token_transposed"),
    MmaOperationSpec("dh_k", "11", "mm_128_64_64_mn_mn", "token_transposed", "square_transposed"),
    MmaOperationSpec("dk_dp", "12", "mm_64_128_64_mn_mn", "square_transposed", "token_transposed"),
    MmaOperationSpec("da_left", "13a", "mm_64_64_64_mn_mn", "square_transposed", "square_transposed"),
    MmaOperationSpec("at", "13c", "mm_64_64_64_k_k", "square_direct", "square_direct"),
    MmaOperationSpec("da_right", "13b", "mm_64_64_64_k_k", "square_direct", "square_direct"),
    MmaOperationSpec("dh_q", "14", "mm_128_64_64_mn_mn", "token_transposed", "square_transposed"),
    MmaOperationSpec("dk_da", "15", "mm_64_128_64_k_mn", "square_direct", "token_transposed"),
)


_CANONICAL_LAYOUT_SOURCES = {
    "token_direct": ("mm_64_64_128_k_k", "a"),
    "token_transposed": ("mm_128_64_64_mn_mn", "a"),
    "square_direct": ("mm_64_64_64_k_k", "a"),
    "square_transposed": ("mm_64_64_64_mn_mn", "a"),
    "state_direct": ("mm_128_64_128_k_k", "a"),
    "state_transposed": ("mm_128_64_128_mn_k", "a"),
}

_CANONICAL_LAYOUT_SIZES = {
    "token_direct": 64 * 128,
    "token_transposed": 64 * 128,
    "square_direct": 64 * 64,
    "square_transposed": 64 * 64,
    "state_direct": 128 * 128,
    "state_transposed": 128 * 128,
}

_PACKED_VARIANT_NAMES = (
    "mm_64_128_64_mn_mn",
    "mm_64_128_64_k_mn",
    "mm_64_128_128_k_mn",
    "mm_64_128_128_k_k",
    "mm_64_64_128_k_k",
    "mm_64_64_64_mn_mn",
    "mm_64_64_64_k_k",
)


def _major_mode(name: str):
    return {"K": OperandMajorMode.K, "MN": OperandMajorMode.MN}[name]


def _make_tiled_mma(io_dtype, acc_dtype, spec: MmaVariantSpec):
    # Preserve the baseline N=64 instruction plus two-atom permutation for
    # 64x128 outputs. It is what produces their packed 64-column TMEM view.
    pack_n = spec.tile[0] == 64 and spec.tile[1] == 128
    instruction_n = spec.tile[1] // 2 if pack_n else spec.tile[1]
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        (spec.tile[0], instruction_n, 16),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        _major_mode(spec.a_major),
        _major_mode(spec.b_major),
    )
    if pack_n:
        return cute.make_tiled_mma(op, permutation_mnk=spec.tile)
    return cute.make_tiled_mma(op)


def _mma_operand_view(staged_layout):
    """Drop only the singleton stage mode from an MMA SMEM layout."""

    return cute.select(staged_layout, mode=[0, 1, 2])


def _operand_layouts(variant: BuiltMmaVariant) -> MmaOperandLayouts:
    return MmaOperandLayouts(
        variant.smem_a_staged,
        variant.smem_b_staged,
        variant.smem_a,
        variant.smem_b,
    )


def _linear_address_mapping(layout):
    """Regroup any static domain into one linear coordinate domain."""

    linear_domain = cute.make_layout(cute.size(layout))
    return cute.composition(layout, linear_domain)


def _packed_regroup(canonical_name: str):
    """Map the two permuted N atoms into the canonical logical coordinates."""

    if canonical_name == "token_transposed":
        return cute.make_layout(
            ((64, 16), 2, 4),
            stride=((1, 128), 64, 2048),
        )
    if canonical_name == "state_transposed":
        return cute.make_layout(
            ((64, 16), 2, 8),
            stride=((1, 128), 64, 2048),
        )
    if canonical_name == "state_direct":
        return cute.make_layout(
            ((64, 16), 2, (4, 2)),
            stride=((1, 128), 64, (2048, 8192)),
        )
    raise AssertionError(
        f"no packed regroup is defined for {canonical_name}"
    )


def _validate_packed_mapping(
    actual,
    canonical,
    operation: str,
    operand: str,
    canonical_name: str,
):
    """Prove equal coordinate-to-address maps after explicit regrouping."""

    assert cute.size(actual) == cute.size(canonical)
    assert cute.cosize(actual) == cute.cosize(canonical)
    assert actual.inner == canonical.inner
    actual_mapping = _linear_address_mapping(actual)
    canonical_mapping = _linear_address_mapping(canonical)
    if actual_mapping != canonical_mapping:
        regrouped = cute.composition(
            canonical, _packed_regroup(canonical_name)
        )
        canonical_mapping = _linear_address_mapping(regrouped)
    assert actual_mapping == canonical_mapping, (
        f"{operation}: packed {operand} layout does not map to canonical "
        f"{canonical_name} physical addresses"
    )


def _validate_accumulator_layouts(variants: MmaVariantBindings) -> dict[str, int]:
    direct = variants.mm_64_128_64_mn_mn.tiled_mma
    direct_shape = direct.partition_shape_C((64, 128))
    direct_layout = direct.make_fragment_C(cute.append(direct_shape, 1)).layout
    for name in (
        "mm_64_128_64_k_mn",
        "mm_64_128_128_k_mn",
        "mm_64_128_128_k_k",
    ):
        mma = variants[name].tiled_mma
        shape = mma.partition_shape_C((64, 128))
        assert mma.make_fragment_C(cute.append(shape, 1)).layout == direct_layout

    square = variants.mm_64_64_128_k_k.tiled_mma
    square_shape = square.partition_shape_C((64, 64))
    square_layout = square.make_fragment_C(cute.append(square_shape, 1)).layout
    for name in ("mm_64_64_64_mn_mn", "mm_64_64_64_k_k"):
        mma = variants[name].tiled_mma
        shape = mma.partition_shape_C((64, 64))
        assert mma.make_fragment_C(cute.append(shape, 1)).layout == square_layout

    physical_columns = {}
    for spec in MMA_VARIANT_SPECS:
        mma = variants[spec.name].tiled_mma
        shape = mma.partition_shape_C(spec.tile[:2])
        fragment = mma.make_fragment_C(cute.append(shape, 1))
        columns = tcgen05.find_tmem_tensor_col_offset(fragment)
        expected = (
            spec.tile[1] // 2
            if spec.tile[0] == 64 and spec.tile[1] == 128
            else spec.tile[1]
        )
        assert columns == expected, (
            f"{spec.name}: expected {expected} TMEM columns, got {columns}"
        )
        physical_columns[spec.name] = columns
    return physical_columns


def _validate_specs() -> None:
    variant_names = tuple(spec.name for spec in MMA_VARIANT_SPECS)
    operation_names = tuple(spec.name for spec in MMA_OPERATION_SPECS)
    canonical_names = set(_CANONICAL_LAYOUT_SOURCES)
    assert len(variant_names) == len(set(variant_names)) == 10
    assert len(operation_names) == len(set(operation_names)) == 19
    assert canonical_names == set(CanonicalLayouts.__dataclass_fields__)
    for operation in MMA_OPERATION_SPECS:
        assert operation.variant in variant_names
        assert operation.a_view in canonical_names
        assert operation.b_view in canonical_names


def build_backward_layout_plan(io_dtype, acc_dtype) -> BackwardLayoutPlan:
    """Build and validate the trace-time-only named backward layout plan."""

    _validate_specs()
    built = {}
    for spec in MMA_VARIANT_SPECS:
        tiled_mma = _make_tiled_mma(io_dtype, acc_dtype, spec)
        built[spec.name] = BuiltMmaVariant(
            spec=spec,
            tiled_mma=tiled_mma,
            smem_a_staged=sm100_utils.make_smem_layout_a(
                tiled_mma,
                spec.tile,
                io_dtype,
                1,
                is_k_major=spec.a_major == "K",
            ),
            smem_b_staged=sm100_utils.make_smem_layout_b(
                tiled_mma,
                spec.tile,
                io_dtype,
                1,
                is_k_major=spec.b_major == "K",
            ),
        )
    variants = MmaVariantBindings(**built)

    staged_values = {}
    for name, (variant_name, operand) in _CANONICAL_LAYOUT_SOURCES.items():
        staged_values[name] = getattr(variants[variant_name], f"smem_{operand}_staged")
    canonical_staged = CanonicalLayouts(**staged_values)
    canonical = CanonicalLayouts(
        **{name: _mma_operand_view(layout) for name, layout in staged_values.items()}
    )

    operation_values = {}
    for spec in MMA_OPERATION_SPECS:
        variant = variants[spec.variant]
        assert variant.tiled_mma.op.a_major_mode == _major_mode(variant.spec.a_major)
        assert variant.tiled_mma.op.b_major_mode == _major_mode(variant.spec.b_major)
        operands = _operand_layouts(variant)
        canonical_a = canonical[spec.a_view]
        canonical_b = canonical[spec.b_view]
        if variant.spec.tile[0] == 64 and variant.spec.tile[1] == 128:
            _validate_packed_mapping(
                operands.a,
                canonical_a,
                spec.name,
                "A",
                spec.a_view,
            )
            _validate_packed_mapping(
                operands.b,
                canonical_b,
                spec.name,
                "B",
                spec.b_view,
            )
        else:
            assert operands.a == canonical_a, (
                f"{spec.name}: A view {spec.a_view} is not the MMA physical mapping"
            )
            assert operands.b == canonical_b, (
                f"{spec.name}: B view {spec.b_view} is not the MMA physical mapping"
            )
        operation_values[spec.name] = MmaOperationBinding(
            spec, variant, operands, canonical_a, canonical_b
        )
    operations = MmaOperationBindings(**operation_values)

    packed = PackedMmaLayouts(
        **{name: _operand_layouts(variants[name]) for name in _PACKED_VARIANT_NAMES}
    )
    physical_columns = _validate_accumulator_layouts(variants)

    for name, expected in _CANONICAL_LAYOUT_SIZES.items():
        staged = canonical_staged[name]
        view = canonical[name]
        assert _mma_operand_view(staged) == view
        assert cute.cosize(staged) == expected
        assert cute.cosize(view) == expected

    return BackwardLayoutPlan(
        variants,
        canonical,
        canonical_staged,
        operations,
        packed,
        physical_columns,
    )


def _layout_snapshot(layout) -> tuple[str, ...]:
    """Return a stable structural description for baseline comparisons."""

    return (
        str(layout),
        str(cute.size(layout)),
        str(cute.cosize(layout)),
        str(getattr(layout, "inner", None)),
        str(getattr(layout, "offset", None)),
        str(getattr(layout, "outer", None)),
    )


def layout_plan_snapshot(plan: BackwardLayoutPlan) -> tuple:
    """Serialize every physical layout contract without tensor descriptors."""

    variants = tuple(
        (
            spec.name,
            spec.family,
            spec.tile,
            spec.a_major,
            spec.b_major,
            _layout_snapshot(plan.variants[spec.name].smem_a_staged),
            _layout_snapshot(plan.variants[spec.name].smem_b_staged),
            plan.physical_columns_by_variant[spec.name],
        )
        for spec in MMA_VARIANT_SPECS
    )
    canonical = tuple(
        (
            name,
            _layout_snapshot(plan.canonical[name]),
            _layout_snapshot(plan.canonical_staged[name]),
        )
        for name in CanonicalLayouts.__dataclass_fields__
    )
    operations = tuple(
        (
            spec.name,
            spec.stage,
            spec.variant,
            spec.a_view,
            spec.b_view,
            _layout_snapshot(plan.operations[spec.name].operands.a),
            _layout_snapshot(plan.operations[spec.name].operands.b),
        )
        for spec in MMA_OPERATION_SPECS
    )
    return variants, canonical, operations


def layout_plan_fingerprint(plan: BackwardLayoutPlan) -> str:
    """Hash the deterministic snapshot for concise IR/test reporting."""

    snapshot = repr(layout_plan_snapshot(plan)).encode("utf-8")
    return hashlib.sha256(snapshot).hexdigest()


__all__ = [
    "BackwardLayoutPlan",
    "BuiltMmaVariant",
    "CanonicalLayouts",
    "MMA_OPERATION_SPECS",
    "MMA_VARIANT_SPECS",
    "MmaOperationBinding",
    "MmaOperationBindings",
    "MmaOperationSpec",
    "MmaOperandLayouts",
    "MmaVariantBindings",
    "MmaVariantSpec",
    "PackedMmaLayouts",
    "TmaDescriptorBundle",
    "build_backward_layout_plan",
    "layout_plan_fingerprint",
    "layout_plan_snapshot",
]
