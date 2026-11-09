"""Trace-time layout contracts for the SM100 fused KDA backward kernel.

The KDA schedule reuses the GDR backward's oriented MMA variants, canonical
SMEM mappings and packed-operand validation (``fused_gdr_bwd_cute.layouts``)
and adds one variant for the intra-chunk lower products.  Operation names
follow the per-chunk math in the ``kernel`` module docstring (for example
``x_state = k*Gamma @ S`` and ``dvn_state = k*2^(G_last - G) @ dS``).
"""

from dataclasses import dataclass
from typing import Any

import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils

from ..fused_gdr_bwd_cute.layouts import (
    _CANONICAL_LAYOUT_SIZES,
    _CANONICAL_LAYOUT_SOURCES,
    BuiltMmaVariant,
    MmaOperandLayouts,
    MmaOperationSpec,
    MmaVariantSpec,
    _major_mode,
    _make_tiled_mma,
    _mma_operand_view,
    _operand_layouts,
    _validate_packed_mapping,
)

MMA_VARIANT_SPECS = (
    MmaVariantSpec("mm_64_64_128_k_k", "mm_64_64_128", (64, 64, 128), "K", "K"),
    MmaVariantSpec("mm_64_128_64_mn_mn", "mm_64_128_64", (64, 128, 64), "MN", "MN"),
    MmaVariantSpec("mm_64_128_64_k_mn", "mm_64_128_64", (64, 128, 64), "K", "MN"),
    MmaVariantSpec("mm_128_64_64_mn_mn", "mm_128_64_64", (128, 64, 64), "MN", "MN"),
    MmaVariantSpec("mm_128_64_64_mn_k", "mm_128_64_64", (128, 64, 64), "MN", "K"),
    MmaVariantSpec("mm_128_64_128_mn_k", "mm_128_64_128", (128, 64, 128), "MN", "K"),
    MmaVariantSpec("mm_128_64_128_k_k", "mm_128_64_128", (128, 64, 128), "K", "K"),
    MmaVariantSpec("mm_64_64_64_mn_mn", "mm_64_64_64", (64, 64, 64), "MN", "MN"),
    MmaVariantSpec("mm_64_64_64_k_k", "mm_64_64_64", (64, 64, 64), "K", "K"),
    MmaVariantSpec("mm_64_128_128_k_mn", "mm_64_128_128", (64, 128, 128), "K", "MN"),
)

# (name, stage, variant, A view, B view).  "ch-major" results put the 128 key
# channels on the M (TMEM lane) axis; "token" results put 64 tokens there.
MMA_OPERATION_SPECS = (
    # Token-major value-side products.
    MmaOperationSpec("x_state", "01", "mm_64_128_128_k_mn", "token_direct", "state_transposed"),
    MmaOperationSpec("dvn_state", "02", "mm_64_128_128_k_mn", "token_direct", "state_transposed"),
    MmaOperationSpec(
        "dvn_aqk", "03", "mm_64_128_64_mn_mn", "square_transposed", "token_transposed"
    ),
    MmaOperationSpec("vn", "04", "mm_64_128_64_k_mn", "square_direct", "token_transposed"),
    MmaOperationSpec("d", "05", "mm_64_128_64_mn_mn", "square_transposed", "token_transposed"),
    # 64x64 score gradients and the inverse backward.
    MmaOperationSpec("dakk", "06", "mm_64_64_128_k_k", "token_direct", "token_direct"),
    MmaOperationSpec("daqk", "07", "mm_64_64_128_k_k", "token_direct", "token_direct"),
    MmaOperationSpec(
        "inv_left", "08a", "mm_64_64_64_mn_mn", "square_transposed", "square_transposed"
    ),
    MmaOperationSpec("inv_right", "08b", "mm_64_64_64_k_k", "square_direct", "square_direct"),
    # Channel-major state-path token gradients.
    MmaOperationSpec("dq_state", "09", "mm_128_64_128_k_k", "state_direct", "token_direct"),
    MmaOperationSpec("dk_state", "10", "mm_128_64_128_k_k", "state_direct", "token_direct"),
    MmaOperationSpec("dk_w", "11", "mm_128_64_128_k_k", "state_direct", "token_direct"),
    # State recurrence: gate on the key side (A), two 64-column value halves.
    MmaOperationSpec("dh_k", "12", "mm_128_64_64_mn_mn", "token_transposed", "square_transposed"),
    MmaOperationSpec("dh_q", "13", "mm_128_64_64_mn_mn", "token_transposed", "square_transposed"),
    # Intra-chunk products, one 16-token k-phase per anchor block.
    MmaOperationSpec("intra_lower", "14", "mm_128_64_64_mn_k", "token_transposed", "square_direct"),
    MmaOperationSpec(
        "intra_upper", "15", "mm_128_64_64_mn_mn", "token_transposed", "square_transposed"
    ),
)


@dataclass(frozen=True)
class KdaVariantBindings:
    """Variants by name; device wiring never uses tuple indices."""

    values: dict

    def __getitem__(self, name: str) -> BuiltMmaVariant:
        return self.values[name]


@dataclass(frozen=True)
class KdaCanonicalLayouts:
    """The six canonical physical SMEM mappings shared with GDR."""

    token_direct: Any
    token_transposed: Any
    square_direct: Any
    square_transposed: Any
    state_direct: Any
    state_transposed: Any

    def __getitem__(self, name: str):
        return getattr(self, name)


@dataclass(frozen=True)
class KdaOperationBinding:
    """Resolved wiring for one logical MMA operation."""

    spec: MmaOperationSpec
    variant: BuiltMmaVariant
    operands: MmaOperandLayouts

    @property
    def tiled_mma(self):
        return self.variant.tiled_mma


@dataclass(frozen=True)
class KdaLayoutPlan:
    """Named trace-time-only layout plan; never pass it to ``@cute.kernel``."""

    variants: KdaVariantBindings
    canonical: KdaCanonicalLayouts
    canonical_staged: KdaCanonicalLayouts
    operations: dict
    physical_columns_by_variant: dict


def _validate_specs() -> None:
    variant_names = tuple(spec.name for spec in MMA_VARIANT_SPECS)
    operation_names = tuple(spec.name for spec in MMA_OPERATION_SPECS)
    assert len(variant_names) == len(set(variant_names))
    assert len(operation_names) == len(set(operation_names))
    for operation in MMA_OPERATION_SPECS:
        assert operation.variant in variant_names, operation.name
        assert operation.a_view in _CANONICAL_LAYOUT_SOURCES, operation.name
        assert operation.b_view in _CANONICAL_LAYOUT_SOURCES, operation.name


_validate_specs()


def _physical_columns(tiled_mma, tile) -> int:
    shape = tiled_mma.partition_shape_C(tile[:2])
    fragment = tiled_mma.make_fragment_C(cute.append(shape, 1))
    return cute.nvgpu.tcgen05.find_tmem_tensor_col_offset(fragment)


def build_kda_layout_plan(io_dtype, acc_dtype) -> KdaLayoutPlan:
    """Build and validate the named KDA layout plan."""

    built = {}
    for spec in MMA_VARIANT_SPECS:
        tiled_mma = _make_tiled_mma(io_dtype, acc_dtype, spec)
        built[spec.name] = BuiltMmaVariant(
            spec=spec,
            tiled_mma=tiled_mma,
            smem_a_staged=sm100_utils.make_smem_layout_a(
                tiled_mma, spec.tile, io_dtype, 1, is_k_major=spec.a_major == "K"
            ),
            smem_b_staged=sm100_utils.make_smem_layout_b(
                tiled_mma, spec.tile, io_dtype, 1, is_k_major=spec.b_major == "K"
            ),
        )
    variants = KdaVariantBindings(built)

    staged = {
        name: getattr(variants[variant_name], f"smem_{operand}_staged")
        for name, (variant_name, operand) in _CANONICAL_LAYOUT_SOURCES.items()
    }
    canonical_staged = KdaCanonicalLayouts(**staged)
    canonical = KdaCanonicalLayouts(
        **{name: _mma_operand_view(layout) for name, layout in staged.items()}
    )
    for name, expected in _CANONICAL_LAYOUT_SIZES.items():
        assert cute.cosize(canonical[name]) == expected, name

    operations = {}
    for spec in MMA_OPERATION_SPECS:
        variant = variants[spec.variant]
        assert variant.tiled_mma.op.a_major_mode == _major_mode(variant.spec.a_major)
        assert variant.tiled_mma.op.b_major_mode == _major_mode(variant.spec.b_major)
        operands = _operand_layouts(variant)
        if variant.spec.tile[0] == 64 and variant.spec.tile[1] == 128:
            _validate_packed_mapping(
                operands.a, canonical[spec.a_view], spec.name, "A", spec.a_view
            )
            _validate_packed_mapping(
                operands.b, canonical[spec.b_view], spec.name, "B", spec.b_view
            )
        else:
            assert operands.a == canonical[spec.a_view], f"{spec.name}: A is not {spec.a_view}"
            assert operands.b == canonical[spec.b_view], f"{spec.name}: B is not {spec.b_view}"
        operations[spec.name] = KdaOperationBinding(spec, variant, operands)

    physical_columns = {
        spec.name: _physical_columns(variants[spec.name].tiled_mma, spec.tile)
        for spec in MMA_VARIANT_SPECS
    }
    return KdaLayoutPlan(variants, canonical, canonical_staged, operations, physical_columns)


__all__ = ["KdaLayoutPlan", "MMA_OPERATION_SPECS", "MMA_VARIANT_SPECS", "build_kda_layout_plan"]
