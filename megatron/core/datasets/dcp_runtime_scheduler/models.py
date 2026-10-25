# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""In-tree, CPU-only factories for the frozen attention and mixed-layer models.

These are provisional timing predictions, not native/eager calibration claims.
The imported kernels only perform CPU arithmetic; their names identify the
calibration regime and do not execute CUDA Graphs.
"""

import math
from functools import lru_cache

from .attention.backward import BackwardModel
from .attention.dispatch import dispatch_backend
from .attention.predictor import ForwardModel, load_parameters


class CorePredictor:
    """Compact execution of the original frozen forward/backward formulas."""

    def __init__(
        self,
        *,
        hardware='gb200',
        query_heads=32,
        kv_heads=4,
        head_dim=128,
        tokens_per_rank=16384,
        dataset_alignment=64,
        kernel_backend='python',
    ):
        if kernel_backend not in ('python', 'numba'):
            raise ValueError('kernel_backend must be python or numba')
        if any(
            type(value) is not int or value <= 0
            for value in (query_heads, kv_heads, head_dim, tokens_per_rank, dataset_alignment)
        ):
            raise ValueError('Prediction geometry requires positive integers')
        self.hardware = hardware
        self.geometry = (query_heads, kv_heads, head_dim)
        self.tokens_per_rank = tokens_per_rank
        self.dataset_alignment = dataset_alignment
        self.kernel_backend = kernel_backend
        self.models = {'forward': ForwardModel(hardware), 'backward': BackwardModel(hardware)}
        for model in self.models.values():
            model.geometry_profile(*self.geometry)

    def __call__(self, lengths, cp, *, ranks=None):
        if type(cp) is not int or cp < 1 or not lengths or len(lengths) > 64:
            raise ValueError('A positive CP size and one to 64 sequences are required')
        if any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError('Logical sequence lengths must be positive integers')
        if ranks is not None:
            ranks = tuple(ranks)
            if (
                len(ranks) != cp
                or not ranks
                or ranks[0] < 0
                or ranks != tuple(range(ranks[0], ranks[0] + cp))
            ):
                raise ValueError('Frozen runtime models require contiguous rank intervals')
        with dispatch_backend(self.kernel_backend):
            result = {}
            options = dict(
                query_heads=self.geometry[0],
                kv_heads=self.geometry[1],
                head_dim=self.geometry[2],
                storage_tokens=self.tokens_per_rank,
                sequence_alignment=2 * cp,
                dataset_alignment=self.dataset_alignment,
                clock_law='two_term',
                use_production_default_clock=True,
            )
            for phase, model in self.models.items():
                result[phase + '_ms'] = float(model.predict_time_ms(lengths, cp, **options))
        if any(not math.isfinite(value) or value <= 0 for value in result.values()):
            raise ValueError('Invalid frozen attention prediction')
        return result

    def metadata(self):
        return dict(
            hardware=self.hardware,
            geometry=list(self.geometry),
            launch_mode='cuda_graph',
            clock_input='production_default',
            heads={p: m.head for p, m in self.models.items()},
            model_sha256={p: m.data_sha256 for p, m in self.models.items()},
            runtime_parameters_sha256={p: m.parameter_sha256 for p, m in self.models.items()},
            head_status={p: m.head_status for p, m in self.models.items()},
            known_limits={p: m.known_limits for p, m in self.models.items()},
            kernel_backend=self.kernel_backend,
            provisional=True,
            scalar_execution='same_frozen_formula_as_full_trace',
        )


class AttentionCostModel:
    """Frozen FA core costs plus caller-supplied non-attention layer times.

    Only an in-domain NVL72 placement is accepted for GB200/GB300. These core
    costs omit projections and are not a calibrated full-model prediction unless
    the caller supplies appropriately measured remainder costs.
    """

    name = 'calibrated_attention'
    placement_invariant = True

    def __init__(
        self,
        geometry,
        *,
        hardware='gb200',
        query_heads=32,
        kv_heads=4,
        head_dim=128,
        layer_count=1,
        forward_non_attention_ms=0.0,
        backward_non_attention_ms=0.0,
        kernel_backend='python',
        allow_provisional=False,
    ):
        if allow_provisional is not True:
            raise ValueError('Frozen timing models require explicit allow_provisional=True')
        if hardware not in ('gb200', 'gb300') or geometry.world_size > 72:
            raise ValueError('The in-tree full F/B model covers one GB200/GB300 NVL72 domain')
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError('layer_count must be a positive integer')
        if any(
            not math.isfinite(v) or v < 0
            for v in (forward_non_attention_ms, backward_non_attention_ms)
        ):
            raise ValueError('Non-attention times must be finite and nonnegative')
        self.geometry = geometry
        self.layer_count = layer_count
        self.forward_non_attention_ms = forward_non_attention_ms
        self.backward_non_attention_ms = backward_non_attention_ms
        self.core = CorePredictor(
            hardware=hardware,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            tokens_per_rank=geometry.tokens_per_rank,
            dataset_alignment=geometry.dataset_alignment,
            kernel_backend=kernel_backend,
        )
        self.source_hashes = self.core.metadata()['model_sha256']

    @lru_cache(maxsize=65536)
    def _costs(self, lengths, cp):
        result = self.core(lengths, cp)
        return result['forward_ms'], result['backward_ms']

    def forward(self, lengths, cp, *, ranks=None):
        return self._costs(tuple(lengths), cp)[0]

    def backward(self, lengths, cp, *, ranks=None):
        return self._costs(tuple(lengths), cp)[1]

    def program_costs(self, lengths, cp, *, ranks=None):
        fwd, bwd = self._costs(tuple(lengths), cp)
        return dict(att_f=fwd, att_b=bwd)

    guide_program_costs = program_costs

    def timing(self):
        return dict(
            execution_program=[
                dict(
                    layer_count=self.layer_count,
                    forward=[
                        operation
                        for _ in range(self.layer_count)
                        for operation in (
                            dict(sync='cp', cost='att_f'),
                            dict(sync='ep', duration_ms=self.forward_non_attention_ms),
                        )
                    ],
                    backward=[
                        operation
                        for _ in range(self.layer_count)
                        for operation in (
                            dict(sync='cp', cost='att_b'),
                            dict(sync='ep', duration_ms=self.backward_non_attention_ms),
                        )
                    ],
                )
            ]
        )

    def clear(self):
        self._costs.cache_clear()

    def check_domain(self, geometry, config=None):
        if geometry != self.geometry:
            raise ValueError('Cost model and runtime planner geometry differ')

    def metadata(self):
        return dict(
            core=self.core.metadata(),
            provisional=True,
            scope='FA core plus supplied remainders; not a measured native/eager E2E model',
        )


def qwen35_new_core_model(*, fa_remainder=None, non_attention=None, kernel_backend='python'):
    """The exact a55 Qwen3.5 model, with its paired frozen inputs in this package."""
    from .qwen35 import Qwen35CoreCostModel

    if fa_remainder is None:
        fa_remainder = load_parameters('qwen35.json')[0]['fa_remainder']
    predictor = CorePredictor(
        query_heads=32, kv_heads=2, head_dim=256, kernel_backend=kernel_backend
    )
    return Qwen35CoreCostModel(predictor, fa_remainder=fa_remainder, non_attention=non_attention)


def make_model(name, geometry, **options):
    """Resolve an in-tree model name without importing the Megatron package."""
    if name == 'calibrated_attention':
        return AttentionCostModel(geometry, **options)
    if name == 'usp_calib_v1':
        from .usp_calib_v1 import USPCalibV1CostModel

        return USPCalibV1CostModel(geometry, **options)
    if name in ('qwen35_gb200', 'dcp_scheduler.adapters:qwen35_new_core_model'):
        return qwen35_new_core_model(**options)
    raise ValueError(f'Unknown in-tree DCP cost model: {name}')
