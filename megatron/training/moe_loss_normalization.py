# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent input-token normalization of MoE losses for masked text training."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def prepare_moe_loss_normalization(
    data_iterator: Iterator | list[Iterator | None] | None,
    num_microbatches: int,
    *,
    pad_token_id: int | None,
    tp_group: torch.distributed.ProcessGroup,
    dp_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup | None = None,
) -> tuple[Iterator | list[Iterator | None] | None, torch.Tensor]:
    """Buffer output-stage batches and return the global scored/input-token ratio.

    Call on every TP/DP/CP/PP rank before executing the training schedule. Only
    TP rank zero on the logical output stage looks ahead in its iterator. Batches
    precede CP slicing: CP replicas see the same complete sequences, so reduce
    across DP, not DP x CP.
    When padding is present, the caller must pass ``tokens == pad_token_id``
    as the router padding mask; ``None`` counts all input positions.
    Buffering CPU batch references does not retain GPU activations.

    A step-wide ratio is required: per-microbatch ratios incorrectly weight
    auxiliary losses when scored-token fractions differ between microbatches.
    """
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive.")
    if tp_group is None or dp_group is None:
        raise ValueError("Explicit TP and DP process groups are required.")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    last_stage = pp_group is None or pp_group.rank() == pp_group.size() - 1
    prepared = data_iterator
    if last_stage:
        if torch.distributed.get_rank(tp_group) == 0:
            # Only the logical output stage needs to look ahead. Other physical
            # and virtual stages keep their original iterators and consumption.
            source = data_iterator[-1] if isinstance(data_iterator, list) else data_iterator
            if source is None:
                raise ValueError("MoE normalization requires the output-stage data iterator.")
            buffered = [next(source) for _ in range(num_microbatches)]
            local_counts = torch.stack(
                [
                    torch.stack(
                        [
                            batch["loss_mask"].sum().to(torch.int64),
                            (
                                batch["tokens"].ne(pad_token_id).sum()
                                if pad_token_id is not None
                                else batch["tokens"].new_tensor(batch["tokens"].numel())
                            ),
                        ]
                    )
                    for batch in buffered
                ]
            ).sum(dim=0)
            counts.copy_(local_counts)
            # get_batch replaces CPU dict entries with CUDA tensors. Yield shallow
            # copies so the lookahead/replay buffers retain only their CPU records.
            prepared_batches = (dict(batch) for batch in buffered)
            if isinstance(data_iterator, list):
                prepared = [*data_iterator[:-1], prepared_batches]
            else:
                prepared = prepared_batches
        torch.distributed.broadcast(
            counts, src=torch.distributed.get_global_rank(tp_group, 0), group=tp_group
        )
        torch.distributed.all_reduce(counts, group=dp_group)
    if pp_group is not None and pp_group.size() > 1:
        torch.distributed.broadcast(
            counts,
            src=torch.distributed.get_global_rank(pp_group, pp_group.size() - 1),
            group=pp_group,
        )
    # finalize_model_grads also clamps an empty scored-token denominator to one.
    ratio = counts[0].clamp_min(1).float() / counts[1].clamp_min(1)
    return prepared, ratio


@contextmanager
def input_token_moe_loss_scaling(
    config: TransformerConfig | list[TransformerConfig], scored_to_input_ratio: torch.Tensor
) -> Iterator[None]:
    """Apply the correction with AMP, preserving explicit full-override MoE hooks.

    Router backward seeds a summed input-token auxiliary gradient. Finalization
    divides *all* gradients by the global scored-token count. Multiplying only
    the auxiliary seed by scored/input cancels that shared denominator while
    leaving the main-loss gradient unchanged. Keep the ratio constant throughout
    the schedule, including interleaved forward/backward and recomputation.
    """
    if isinstance(config, list):
        with ExitStack() as stack:
            seen = set()
            for chunk_config in config:
                if id(chunk_config) not in seen:
                    stack.enter_context(
                        input_token_moe_loss_scaling(chunk_config, scored_to_input_ratio)
                    )
                    seen.add(id(chunk_config))
            yield
        return
    original = config.moe_grad_scale_func
    # The public hook is a full override (e.g. an RL caller may already supply
    # its own normalization). Preserve it rather than applying the ratio twice.
    if original is not None:
        yield
        return
    # Captured graphs must read a stable address even when target fractions
    # change between steps. Replacing this tensor would replay a stale ratio.
    ratio = getattr(config, "_moe_input_token_ratio", None)
    if ratio is None:
        ratio = scored_to_input_ratio.clone()
        config._moe_input_token_ratio = ratio
    else:
        ratio.copy_(scored_to_input_ratio)

    def scale() -> torch.Tensor:
        if config.grad_scale_func is not None:
            return config.grad_scale_func(torch.ones_like(ratio)) * ratio
        return ratio

    config.moe_grad_scale_func = scale
    try:
        yield
    finally:
        config.moe_grad_scale_func = original


def uses_moe_auxiliary_losses(config: TransformerConfig) -> bool:
    """Avoid normalization lookahead when no auxiliary gradient is configured."""
    if getattr(config, "num_moe_experts", None) is None:
        return False
    coefficients = config.moe_aux_loss_coeff
    if not isinstance(coefficients, list):
        coefficients = [coefficients]
    return any(coefficient != 0 for coefficient in coefficients) or bool(config.moe_z_loss_coeff)
