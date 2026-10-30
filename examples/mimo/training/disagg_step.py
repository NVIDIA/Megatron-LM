# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Synchronous disjoint-role adapter for native packed MIMO training."""

import torch
import torch.distributed as dist

from examples.mimo.training.packed_step import PackedMimoStep
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.packing import prepare_mimo_step
from megatron.core.models.mimo.runtime import MimoStepRuntime
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.rerun_state_machine import RerunDataIterator


class PackedDisaggMimoStep(PackedMimoStep):
    """Reuse the decoder's native PP1 schedule, with explicit encoder participants.

    All roles prepare one source batch. Only decoder ranks pack it; their leader
    publishes the immutable per-round routes. Encoder ranks execute those routes
    without a decoder loss, accumulate returned feature gradients, and backpropagate
    once before the existing heterogeneous finalizer and optimizer run.
    """

    def __init__(self, model, topology, prepare_source, *, modality_name):
        self.is_decoder = model.role.has_language_module
        name = MIMO_LANGUAGE_MODULE_KEY if self.is_decoder else modality_name
        super().__init__(
            model,
            topology.module_pgs[name],
            prepare_source,
            topology.cp_groups,
            modality_name=modality_name,
        )
        self.decoder_ranks = tuple(
            sorted(
                rank
                for line in topology.grids[MIMO_LANGUAGE_MODULE_KEY].get_rank_enum("pp")
                for rank in line
            )
        )
        producers = tuple(
            sorted(
                rank for line in topology.grids[modality_name].get_rank_enum("pp") for rank in line
            )
        )
        self.runtime = MimoStepRuntime(
            dist.group.WORLD, producer_ranks=producers, decoder_ranks=self.decoder_ranks
        )
        self._finalize_grads = model.config.finalize_model_grads_func
        self.routes = None

    def install(self, **kwargs):
        super().install(**kwargs)
        self.model.config.forward_backward_func = self.run
        if not self.is_decoder:
            # The host constructs an unused scheduler before calling the data adapter.
            # Its encoder-local size must not constrain the decoder's minimum CP.
            self.model.config.min_dynamic_context_parallel_size = 1

    def prepare(self, data_iterator, num_microbatches, scheduler, pg_collection):
        if self.runtime.active:
            raise RuntimeError("The previous MIMO source step has not finished")
        samples, slices, inputs = self.prepare_source(next(data_iterator))
        device = torch.device("cuda", torch.cuda.current_device())
        self.plan = (
            prepare_mimo_step(
                samples,
                slices,
                scheduler=scheduler,
                domain_ranks=self.decoder_ranks,
                cp_groups=self.cp_groups,
                config=self.model.config,
                device=device,
            )
            if self.is_decoder
            else None
        )
        # Only control metadata crosses the whole job. Features use directed P2P.
        message = [None]
        if dist.get_rank() == self.decoder_ranks[0]:
            message[0] = (
                tuple(item.feature_plans for item in self.plan.rounds),
                self.plan.source_tokens,
                self.plan.source_tokens_squared,
                not self.model.training,
            )
        dist.broadcast_object_list(message, src=self.decoder_ranks[0])
        self.routes, tokens, squared, forward_only = message[0]
        if forward_only != (not self.model.training):
            raise RuntimeError("Encoder and decoder execution modes disagree")
        with torch.set_grad_enabled(not forward_only):
            encoded = self.model.encode_modalities(inputs) if not self.is_decoder else {}
        features = encoded.get(self.modality_name)
        if features is None:
            features = torch.empty(
                (0, self.model.config.hidden_size),
                device=device,
                dtype=self.model.config.params_dtype,
            )
        expected = max(
            (
                piece.offset + piece.length
                for piece in slices.values()
                if piece.producer_rank == dist.get_rank()
            ),
            default=0,
        )
        invalid = torch.tensor(
            int(
                features.shape != (expected, self.model.config.hidden_size)
                or features.dtype != self.model.config.params_dtype
                or features.device != device
            ),
            device=device,
        )
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        if invalid.item():
            raise ValueError(
                "Encoder output shape, dtype or device disagrees with the media contract"
            )
        self.runtime.begin_step(features, len(self.routes), forward_only=forward_only)
        items = self.plan.rounds if self.is_decoder else self.routes
        return RerunDataIterator(iter(items)), len(self.routes), float(tokens), float(squared)

    def run(self, *, model, data_iterator, num_microbatches, forward_only=False, **kwargs):
        """Keep the native decoder schedule; encoder-only ranks service its rounds."""
        if not self.runtime.active or num_microbatches != self.runtime.num_rounds:
            raise RuntimeError("Prepare the complete source step before executing its rounds")
        if forward_only != self.runtime.forward_only:
            raise RuntimeError("Prepared source step and schedule execution modes disagree")
        if self.is_decoder:
            kwargs.update(pg_collection=self.pg, p2p_communicator=None)
            losses = forward_backward_no_pipelining(
                model=model,
                data_iterator=data_iterator,
                num_microbatches=num_microbatches,
                forward_only=forward_only,
                **kwargs,
            )
            if forward_only:
                self.routes = None
            return losses
        timers = self.model.config.timers
        if timers is not None:
            timers('forward-backward', log_level=1).start(
                barrier=self.model.config.barrier_with_L1_time
            )
        for _ in range(num_microbatches):
            transfer = self.runtime.route(next(data_iterator))
            self.runtime.complete_round(transfer)
        if forward_only:
            self.runtime.finish_forward()
            self.routes = None
        else:
            self.finalize(
                model,
                torch.zeros((), dtype=torch.int, device="cuda"),
                force_all_reduce=kwargs.get("force_all_reduce", False),
            )
        if timers is not None:
            timers('forward-backward').stop()
        return []

    def finalize(self, model, num_tokens=None, **kwargs):
        if self.runtime.pending is not None:
            self.runtime.complete_round(self.runtime.pending)
        self.runtime.finish_backward()
        self.plan = self.routes = None
        self._finalize_grads(model, num_tokens, **kwargs)
