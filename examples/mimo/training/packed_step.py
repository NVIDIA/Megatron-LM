# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Thin PP1 Megatron training adapter for the public MIMO packing/runtime APIs."""

from functools import partial

import torch
import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.runtime import LazyMimoStepRuntime, MimoStepRuntime
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.utils import get_attr_wrapped_model


class PackedMimoStep:
    """Adapt a replicated source-global-batch iterator to the native PP1 schedule.

    ``prepare_source(source)`` is supplied by the model/data adapter and returns
    ``(samples, feature_slices, encoder_inputs)`` for this rank. ``encoder_inputs``
    can be a callable accepting ordered media IDs to materialize on first use,
    or an already prepared input dict for whole-step encoding. The source owns
    media processing and assignment; packing and graph lifetimes belong to core.
    One adapter handles one named modality.

    The constructor does not change config. Call ``install`` explicitly after
    model construction. Ordinary decoder backward, DDP, optimizer and checkpoint
    behavior remain in the host training loop.
    """

    def __init__(
        self,
        model,
        pg_collection,
        prepare_source,
        cp_groups=None,
        *,
        modality_name,
        tp_cp_groups=None,
    ) -> None:
        config = model.config
        if pg_collection.tp.size() != 1 or pg_collection.pp.size() != 1:
            raise ValueError("Colocated MIMO packing currently requires TP=PP=1")
        if config.gtp_weight_remat_size != 1 or config.expert_gtp_weight_remat_size != 1:
            raise ValueError("Colocated MIMO packing does not support GTP or expert GTP")
        if not config.calculate_per_token_loss or not config.sequence_packing_scheduler:
            raise ValueError("Colocated MIMO requires packed scheduling and per-token loss")
        if cp_groups is None:
            cp_groups = (
                pg_collection.dynamic_cp
                if config.dynamic_context_parallel
                else {pg_collection.cp.size(): pg_collection.cp}
            )
        self.model, self.pg = model, pg_collection
        self.prepare_source = prepare_source
        self.cp_groups, self.tp_cp_groups = cp_groups, tp_cp_groups
        self.modality_name = modality_name
        self.runtime = MimoStepRuntime(pg_collection.dp_cp)
        self.plan = None

    def install(
        self, *, overlap_grad_reduce=False, overlap_param_gather=False, rerun_mode="disabled"
    ):
        """Register the host hooks after validating the supported execution mode."""
        if overlap_grad_reduce or overlap_param_gather:
            raise ValueError("Packed encoder backward requires non-overlapped DDP")
        if getattr(self.model.config, "overlap_moe_expert_parallel_comm", False):
            raise ValueError("Packed MIMO requires the sequential PP1 decoder schedule")
        if rerun_mode != "disabled":
            raise ValueError("Colocated MIMO currently requires rerun mode disabled")
        self.model.mimo_step_runtime = self.runtime
        self.model.mimo_packed_step = self
        self.model.config.sequence_packing_data_adapter = self.prepare
        self.model.config.finalize_model_grads_func = self.finalize

    def prepare(self, data_iterator, num_microbatches, scheduler, pg_collection):
        """Prepare one source step; the dataset owns its source-global-batch size."""
        if self.runtime.active:
            raise RuntimeError("The previous MIMO source step has not finished")
        samples, slices, encoder_inputs = self.prepare_source(next(data_iterator))
        device = torch.device("cuda", torch.cuda.current_device())
        self.plan = prepare_mimo_step(
            samples,
            slices,
            scheduler=scheduler,
            domain_ranks=dist.get_process_group_ranks(self.pg.dp_cp),
            cp_groups=self.cp_groups,
            tp_cp_groups=self.tp_cp_groups,
            config=self.model.config,
            device=device,
        )
        # The host explicitly sets model.train()/eval(). Do not infer the mode
        # from the grad context in which data preparation happened to run.
        forward_only = not self.model.training
        empty = torch.empty(
            (0, self.model.config.hidden_size), device=device, dtype=self.model.config.params_dtype
        )
        runtime_type = LazyMimoStepRuntime if callable(encoder_inputs) else MimoStepRuntime
        if type(self.runtime) is not runtime_type:
            self.runtime = runtime_type(self.pg.dp_cp)
            self.model.mimo_step_runtime = self.runtime
        if callable(encoder_inputs):
            # Encoding moves inside the decoder's forward callback. Preserve the
            # preparation context rather than inheriting decoder-only autocast.
            encoder_autocast = torch.is_autocast_enabled("cuda")
            encoder_autocast_dtype = torch.get_autocast_dtype("cuda")

            def encode(media_ids):
                with torch.autocast("cuda", enabled=encoder_autocast, dtype=encoder_autocast_dtype):
                    return self.model.encode_modalities(encoder_inputs(media_ids)).get(
                        self.modality_name, empty
                    )

            self.runtime.begin_lazy_step(
                [item.feature_plans for item in self.plan.rounds],
                slices,
                encode,
                empty,
                forward_only=forward_only,
            )
        else:
            with torch.set_grad_enabled(not forward_only):
                features = self.model.encode_modalities(encoder_inputs).get(
                    self.modality_name, empty
                )
            expected = max(
                (
                    piece.offset + piece.length
                    for piece in slices.values()
                    if piece.producer_rank == dist.get_rank()
                ),
                default=0,
            )
            if features.shape[0] != expected:
                raise ValueError(f"Encoder emitted {features.shape[0]} rows, expected {expected}")
            self.runtime.begin_step(features, len(self.plan.rounds), forward_only=forward_only)
        return (
            RerunDataIterator(iter(self.plan.rounds)),
            len(self.plan.rounds),
            float(self.plan.source_tokens),
            float(self.plan.source_tokens_squared),
        )

    def forward(self, item, wrapped_model):
        """The PP1 schedule completes the previous backward before this callback."""
        if self.runtime.pending is not None:
            with torch.autocast("cuda", enabled=False):
                self.runtime.complete_round(self.runtime.pending)
        transfer = self.runtime.route(item.feature_plans, item.cp_group)
        output, mask = wrapped_model(
            **item.model_inputs, modality_embeddings={self.modality_name: transfer.features}
        )
        if self.runtime.forward_only:
            self.runtime.complete_round(transfer)
            if self.runtime.completed_rounds == self.runtime.num_rounds:
                self.runtime.finish_forward()
                self.plan = None
        return output, partial(self.loss, mask)

    @staticmethod
    def loss(mask, output):
        """Use the native token-weighted three-value loss contract."""
        raw = (output.float().view(-1) * mask.float().view(-1)).sum()
        count = mask.sum().detach().to(torch.int)
        return raw, count, {"lm loss": torch.stack((raw.detach(), count))}

    def finalize(self, model, num_tokens=None, **kwargs):
        """Complete encoder backward before native gradient reduction and scaling."""
        if self.runtime.pending is not None:
            self.runtime.complete_round(self.runtime.pending)
        self.runtime.finish_backward()
        self.plan = None
        finalize_model_grads(model, num_tokens, **kwargs)


def forward_step(data_iterator, model):
    """Forward callback for the existing native Megatron training schedule."""
    adapter = get_attr_wrapped_model(model, "mimo_packed_step")
    return adapter.forward(next(data_iterator), model)
