# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real PP/VPP schedule parity against all-to-all with native DDP/DistOpt.

Fixed expert identities avoid route flips amplifying the intended BF16/MXFP8
arithmetic difference; scores and their router gradients remain differentiable.
Every chunk contains multiple TransformerLayers. No shared expert or combined
1F1B communication overlap is enabled. Run with a world size divisible by eight;
16 ranks additionally exercise expert-DP reduce-scatter and parameter all-gather.
"""

import gc
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import (
    copy_tensor_to_quantized_param,
    get_fp8_context,
    get_grouped_tensor_members,
    is_grouped_mxfp8tensor,
    is_grouped_tensor,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class PipelineCase:
    """Small numerical case; topology is initialized by the caller."""

    backend: str = "dist_moe"
    precision: str = "bf16"
    pp: int = 2
    ep: int = 4
    vpp: int = 1
    layers_per_chunk: int = 2
    hidden: int = 256
    ffn: int = 256
    experts: int = 8
    topk: int = 2
    tokens: int = 256
    microbatches: int = 6
    steps: int = 3
    learning_rate: float = 1e-4
    adam_eps: float = 1e-8
    seed: int = 1234
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    activation_slot_capacity_factor: float | None = None
    cuda_graph: bool = False
    graph_recapture: bool = False


def _config(case):
    options = {}
    if case.activation_slot_capacity_factor is not None:
        options["activation_slot_capacity_factor"] = case.activation_slot_capacity_factor
    return TransformerConfig(
        num_layers=case.pp * case.vpp * case.layers_per_chunk,
        hidden_size=case.hidden,
        ffn_hidden_size=case.ffn,
        num_attention_heads=case.hidden // 64,
        pipeline_model_parallel_size=case.pp,
        virtual_pipeline_model_parallel_size=case.vpp if case.vpp > 1 else None,
        microbatch_group_size_per_vp_stage=case.pp,
        pipeline_dtype=torch.bfloat16,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="local" if case.cuda_graph else "none",
        cuda_graph_modules=[],
        cuda_graph_warmup_steps=2,
        expert_model_parallel_size=case.ep,
        num_moe_experts=case.experts,
        moe_ffn_hidden_size=case.ffn,
        moe_router_topk=case.topk,
        moe_router_dtype="fp32",
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_fusion=False,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=True,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_overlap=False,
        overlap_moe_expert_parallel_comm=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        fp8="e4m3" if case.precision == "mxfp8" else None,
        fp8_recipe="mxfp8" if case.precision == "mxfp8" else "delayed",
        fp8_param=case.precision == "mxfp8",
        moe_megakernel_backend="dist_moe" if case.backend == "dist_moe" else None,
        moe_megakernel_backend_config=options,
    )


def _plain(parameter):
    if is_grouped_tensor(parameter):
        members = get_grouped_tensor_members(parameter, create_if_missing=True)
        return torch.stack([p.dequantize() if hasattr(p, "dequantize") else p for p in members])
    return parameter.dequantize() if hasattr(parameter, "dequantize") else parameter


@torch.no_grad()
def _initialize(layer, case):
    """Layer/expert identities, rather than global process rank, determine weights."""
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    experts = layer.mlp.experts
    owner_ids = {id(p) for p in experts.parameters()}

    def random(name, shape):
        identity = f"layer{layer.layer_number}/{name}"
        seed = case.seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16) * 0.02

    def assign(parameter, source):
        if is_grouped_tensor(parameter):
            _plain(parameter)
        copy_tensor_to_quantized_param(parameter, source.reshape(parameter.shape))
        if hasattr(parameter, "_high_precision_init_val"):
            parameter._high_precision_init_val.copy_(source.reshape(parameter.shape))

    for name, parameter in layer.named_parameters():
        if id(parameter) in owner_ids:
            continue
        value = (
            torch.full_like(parameter, 0 if name.endswith("bias") else 1)
            if "layer_norm" in name or "layernorm" in name
            else random(name, parameter.shape)
        )
        assign(parameter, value)
    for name in ("linear_fc1", "linear_fc2"):
        linear = getattr(experts, name)
        assert linear.single_grouped_weight
        assert is_grouped_mxfp8tensor(linear.weight) == (case.precision == "mxfp8")
        shape = (2 * case.ffn, case.hidden) if name == "linear_fc1" else (case.hidden, case.ffn)
        local = case.experts // case.ep
        assign(
            linear.weight,
            torch.stack(
                [random(f"{name}/expert{ep_rank * local + i}", shape) for i in range(local)]
            ),
        )


class _TransformerChunk(MegatronModule):
    def __init__(self, config, case, chunk_id, snapshots):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules,
        )
        from megatron.core.transformer.cuda_graphs import annotate_first_last_layer
        from megatron.core.transformer.transformer_layer import TransformerLayer

        super().__init__(config)
        self.case, self.snapshots = case, snapshots
        self.vp_stage = chunk_id if case.vpp > 1 else None
        self.model_type = ModelType.encoder_or_decoder
        self.share_embeddings_and_output_weights = False
        self.pre_process = parallel_state.get_pipeline_model_parallel_rank() == 0 and chunk_id == 0
        self.post_process = (
            parallel_state.get_pipeline_model_parallel_rank() == case.pp - 1
            and chunk_id == case.vpp - 1
        )
        self.input_tensor = None
        self.phase = "initial"
        self.microbatch = None
        self.owner_completions = {}
        self.pending_update = None
        self.phase_forwards = 0
        self.capture_observations = None
        self.route_offsets = {}
        spec = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=case.experts, moe_grouped_gemm=True
        )
        with get_fp8_context(config, is_init=True):
            self.layers = torch.nn.ModuleList(
                [
                    TransformerLayer(config, spec, layer_number=i + 1, vp_stage=self.vp_stage)
                    for i in range(case.layers_per_chunk)
                ]
            )
        annotate_first_last_layer(self.layers)
        self.cuda().to(dtype=torch.bfloat16)
        for layer in self.layers:
            _initialize(layer, case)
            self._install_observers(layer)
            self._fix_routing(layer)

    def _save(self, name, value):
        if self.capture_observations is not None:
            # These clones become graph nodes. Read them only outside capture/replay.
            self.capture_observations[name.split("/", 2)[2]] = value.detach().float().clone()
            return
        assert name not in self.snapshots, f"Duplicate observation: {name}"
        self.snapshots[name] = value.detach().float().cpu().clone()

    def _install_observers(self, layer):
        def observe(module, inputs, output, boundary):
            prefix = f"{self.phase}/mb{self.microbatch}/layer{layer.layer_number}/{boundary}"
            self._save(f"{prefix}Y", output[0])
            if boundary == "" and self.pending_update is not None and self.microbatch == 0:
                # This hook runs after the layer's native DDP pre-hooks completed
                # param gather. Observe refreshed weights without forcing gather
                # ahead of the scheduler (especially VPP chunk >= 2).
                for name, parameter in layer.named_parameters():
                    self._save(
                        f"train{self.pending_update}/updated/layer{layer.layer_number}/{name}",
                        _plain(parameter),
                    )
            if torch.is_grad_enabled() and inputs[0].requires_grad:
                inputs[0].register_hook(lambda grad: self._save(f"{prefix}dgrad", grad))

        layer.register_forward_hook(
            lambda module, inputs, output: observe(module, inputs, output, "")
        )
        layer.mlp.register_forward_hook(
            lambda module, inputs, output: observe(module, inputs, output, "moe_")
        )

        def observe_router(module, inputs, output):
            prefix = f"{self.phase}/mb{self.microbatch}/layer{layer.layer_number}"
            scores, routes = output
            self._save(f"{prefix}/routing_map", routes)
            self._save(f"{prefix}/router_scores", scores)
            if torch.is_grad_enabled() and scores.requires_grad:
                scores.register_hook(lambda grad: self._save(f"{prefix}/router_score_grad", grad))

        layer.mlp.router.register_forward_hook(observe_router)
        for name, owner in layer.mlp.experts.named_parameters():
            key = f"layer{layer.layer_number}/{name}"

            def count_completion(grad, key=key):
                if self.capture_observations is not None:
                    return
                identity = f"{self.phase}/{key}"
                self.owner_completions[identity] = self.owner_completions.get(identity, 0) + 1

            owner.register_hook(count_completion)

    def _fix_routing(self, layer):
        offset = torch.zeros((), device="cuda", dtype=torch.int64)
        self.route_offsets[layer.layer_number] = offset

        def routing(logits, **_kwargs):
            logits = logits.reshape(-1, self.case.experts)
            # The device scalar is updated outside the captured layer on every MB.
            ids = (
                torch.arange(logits.shape[0] * self.case.topk, device=logits.device).reshape(
                    -1, self.case.topk
                )
                + offset
            ) % self.case.experts
            routes = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ids, True)
            scores = torch.softmax(logits.gather(1, ids).float(), dim=-1)
            return (torch.zeros_like(logits).scatter(1, ids, scores.to(logits.dtype)), routes)

        layer.mlp.router.routing = routing

    def set_input_tensor(self, value):
        assert len(value) == 1
        self.input_tensor = value[0]

    def sample(self, microbatch, target=False):
        dp_rank = parallel_state.get_data_parallel_rank()
        phase_seed = (
            100_000 if self.phase == "eval" else int(self.phase.removeprefix("train")) * 1000
        )
        seed = self.case.seed + phase_seed + dp_rank * 31 + microbatch * 7 + int(target)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(
            (self.case.tokens, 1, self.case.hidden),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    def forward(self, microbatch):
        self.microbatch = microbatch
        self.phase_forwards += 1
        value = (
            self.sample(microbatch).requires_grad_(torch.is_grad_enabled())
            if self.pre_process
            else self.input_tensor
        )
        with get_fp8_context(self.config):
            for layer in self.layers:
                self.route_offsets[layer.layer_number].fill_(
                    microbatch + layer.layer_number + (17 if self.phase == "eval" else 0)
                )
                value, _ = layer(value, attention_mask=None)
        return value


def _parameter_snapshot(models, snapshots, prefix, gradients=False):
    for model in models:
        for layer in model.module.layers:
            for name, parameter in layer.named_parameters():
                value = parameter.main_grad if gradients else _plain(parameter)
                snapshots[f"{prefix}/layer{layer.layer_number}/{name}"] = (
                    value.detach().float().cpu().clone()
                )


def _master_snapshot(optimizer, snapshots, prefix):
    for chain, opt in enumerate(getattr(optimizer, "chained_optimizers", [optimizer])):
        for attribute in ("shard_fp32_from_float16_groups", "shard_fp32_groups"):
            for group, parameters in enumerate(getattr(opt, attribute, [])):
                for index, parameter in enumerate(parameters):
                    snapshots[f"{prefix}/master{chain}/{attribute}/{group}/{index}"] = (
                        parameter.detach().float().cpu().clone()
                    )


def run_pipeline_case(case):
    """Run independent optimizer trajectories and train/eval/train on initialized groups."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.megakernel.dist_moe import (
        close_dist_moe_runtime,
        initialize_dist_moe_runtime,
    )

    assert case.steps >= 2 and case.learning_rate > 0
    assert not case.graph_recapture or case.steps >= 3
    assert case.microbatches >= case.pp and case.microbatches % case.pp == 0
    config = _config(case)
    torch.manual_seed(case.seed)
    model_parallel_cuda_manual_seed(
        case.seed,
        te_rng_tracker=case.cuda_graph,
        use_cudagraphable_rng=case.cuda_graph,
        force_reset_rng=case.cuda_graph,
    )
    snapshots = {}
    quantized = case.precision == "mxfp8"
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=True,
        overlap_grad_reduce=case.overlap_grad_reduce,
        overlap_param_gather=case.overlap_param_gather,
        align_param_gather=case.overlap_param_gather,
        fp8_param_gather=quantized,
        reuse_grad_buf_for_mxfp8_param_ag=quantized,
    )
    models = [
        DistributedDataParallel(
            config, ddp_config, _TransformerChunk(config, case, chunk, snapshots)
        )
        for chunk in range(case.vpp)
    ]
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=case.learning_rate,
            min_lr=case.learning_rate,
            adam_eps=case.adam_eps,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=0.0,
            weight_decay=0.0,
            overlap_param_gather=case.overlap_param_gather,
            fp8_recipe="mxfp8" if quantized else None,
            reuse_grad_buf_for_mxfp8_param_ag=quantized,
        ),
        models,
        use_gloo_process_groups=False,
    )
    config.no_sync_func = [model.no_sync for model in models] if case.vpp > 1 else models[0].no_sync
    if case.overlap_grad_reduce:
        config.grad_sync_func = (
            [model.start_grad_sync for model in models]
            if case.vpp > 1
            else models[0].start_grad_sync
        )
    param_sync_events = []
    if case.overlap_param_gather:

        def param_sync(chunk_id):
            def sync(*parameters):
                chunk = models[chunk_id].module
                param_sync_events.append(
                    dict(phase=chunk.phase, chunk=chunk_id, forwards_seen=chunk.phase_forwards)
                )
                models[chunk_id].start_param_sync(*parameters)

            return sync

        callbacks = [param_sync(chunk) for chunk in range(case.vpp)]
        config.param_sync_func = callbacks if case.vpp > 1 else callbacks[0]
    config.finalize_model_grads_func = finalize_model_grads
    if case.backend == "dist_moe":
        initialize_dist_moe_runtime(
            models, ddp_config=ddp_config, max_num_local_input_tokens=case.tokens
        )
    _parameter_snapshot(models, snapshots, "initial")
    _master_snapshot(optimizer, snapshots, "initial_master")
    graph_audit = None
    if case.cuda_graph:
        from tests.unit_tests.transformer.moe.test_dist_moe_pipeline_graph_numeric import (
            PipelineGraphAudit,
        )

        assert case.backend == "dist_moe"
        graph_audit = PipelineGraphAudit(models, case)

    def forward_step(iterator, model):
        microbatch = next(iterator)
        value = model(microbatch)

        def loss_fn(output):
            loss = F.mse_loss(output.float(), model.module.sample(microbatch, target=True).float())
            model.module._save(f"{model.module.phase}/mb{microbatch}/loss", loss)
            return loss, {"loss": loss.detach()}

        return value, loss_fn

    def schedule(phase, forward_only=False):
        for model in models:
            model.train(not forward_only)
            model.module.phase = phase
            model.module.phase_forwards = 0
        iterators = [iter(range(case.microbatches)) for _ in models]
        with torch.no_grad() if forward_only else nullcontext():
            get_forward_backward_func()(
                forward_step_func=forward_step,
                data_iterator=iterators if case.vpp > 1 else iterators[0],
                model=models if case.vpp > 1 else models[0],
                num_microbatches=case.microbatches,
                seq_length=case.tokens,
                micro_batch_size=1,
                forward_only=forward_only,
            )
        if graph_audit is not None:
            graph_audit.check_idle(phase)
        for model in models:
            for layer in model.module.layers:
                for microbatch in range(case.microbatches):
                    prefix = f"{phase}/mb{microbatch}/layer{layer.layer_number}"
                    required = ["Y", "moe_Y", "routing_map", "router_scores"]
                    if not forward_only:
                        required.extend(["dgrad", "moe_dgrad", "router_score_grad"])
                    assert all(f"{prefix}/{name}" in snapshots for name in required), prefix
            model.module.pending_update = None

    def zero_grad():
        optimizer.zero_grad()
        if quantized and case.overlap_param_gather:
            # MXFP8 param AG reuses grad_data: native staging must happen AFTER
            # zeroing, or the next deferred all-gather would gather zeros. The
            # chained optimizer zeros shared chunks once before staging shards.
            optimizer.prepare_model_params_for_param_sync()
        else:
            for model in models:
                model.zero_grad_buffer()

    def record_and_capture(phase):
        zero_grad()
        execution_stream = torch.cuda.current_stream()
        schedule(phase)
        assert (
            torch.cuda.current_stream() == execution_stream
        ), "Native capture changed runtime stream"
        graph_audit.check_capture()
        for key in list(snapshots):
            if key.startswith(phase + "/"):
                del snapshots[key]
        for model in models:
            counts = model.module.owner_completions
            for key in list(counts):
                if key.startswith(phase + "/"):
                    del counts[key]
        param_sync_events[:] = [event for event in param_sync_events if event["phase"] != phase]

    if graph_audit is not None:
        record_and_capture("train0")

    for step in range(case.steps):
        if graph_audit is not None and case.graph_recapture and step == 2:
            graph_audit.release_graphs()
            record_and_capture(f"train{step}")
        zero_grad()
        schedule(f"train{step}")
        _parameter_snapshot(models, snapshots, f"train{step}/wgrad", gradients=True)
        for model in models:
            for layer in model.module.layers:
                for name, owner in layer.mlp.experts.named_parameters():
                    key = f"train{step}/layer{layer.layer_number}/{name}"
                    assert model.module.owner_completions.get(key) == case.microbatches, (
                        key,
                        model.module.owner_completions.get(key),
                        case.microbatches,
                    )
                    assert owner.main_grad.dtype == torch.float32
        success, _, _ = optimizer.step()
        assert success, f"Optimizer skipped step {step}"
        _master_snapshot(optimizer, snapshots, f"train{step}/updated_master")
        for model in models:
            model.module.pending_update = step
        if step == 0:
            if quantized and case.overlap_param_gather:
                optimizer.prepare_model_params_for_param_sync()
            schedule("eval", forward_only=True)
        elif step == case.steps - 1:
            if quantized and case.overlap_param_gather:
                optimizer.prepare_model_params_for_param_sync()
            for model in models:
                model.start_param_sync(force_sync=True)
            _parameter_snapshot(models, snapshots, f"train{step}/updated")
    for value in snapshots.values():
        assert torch.isfinite(value).all(), "Nonfinite numerical observation"
    changed = sum(
        not torch.equal(value, snapshots[key.replace("initial/", "train0/updated/", 1)])
        for key, value in snapshots.items()
        if key.startswith("initial/")
    )
    assert changed > 0, "Optimizer did not change any observable parameter"
    changed_expert_elements = {}
    for key, initial in snapshots.items():
        if key.startswith("initial/") and "/mlp.experts.linear_fc" in key:
            updated = snapshots[key.replace("initial/", "train0/updated/", 1)]
            count = int(torch.count_nonzero(initial != updated))
            assert count > 0, f"Expert parameter did not change after optimizer step: {key}"
            changed_expert_elements[key.removeprefix("initial/")] = count
    assert len(changed_expert_elements) == 2 * case.vpp * case.layers_per_chunk
    changed_routes = 0
    for model in models:
        for layer in model.module.layers:
            prefix = f"layer{layer.layer_number}/routing_map"
            assert not torch.equal(
                snapshots[f"train0/mb0/{prefix}"], snapshots[f"train0/mb1/{prefix}"]
            ), "Router identities did not change between microbatches"
            changed_routes += 1
    changed_masters = []
    for step in range(case.steps):
        previous = "initial_master" if step == 0 else f"train{step - 1}/updated_master"
        current = f"train{step}/updated_master"
        changes = sum(
            not torch.equal(value, snapshots[key.replace(previous, current, 1)])
            for key, value in snapshots.items()
            if key.startswith(previous + "/")
        )
        assert changes, f"FP32 master parameters did not change on optimizer step {step}"
        changed_masters.append(changes)
    if case.overlap_param_gather and case.vpp > 2:
        for step in range(case.steps):
            events = [
                event
                for event in param_sync_events
                if event["phase"] == f"train{step}" and event["chunk"] == 2
            ]
            assert len(events) == 1 and events[0]["forwards_seen"] == 0, events
    metadata = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
        "ep_rank": parallel_state.get_expert_model_parallel_rank(),
        "expert_dp": parallel_state.get_expert_data_parallel_world_size(),
        "dense_dp": parallel_state.get_data_parallel_world_size(),
        "changed_parameters_after_first_step": changed,
        "layers_with_changed_microbatch_routes": changed_routes,
        "changed_master_shards_per_step": changed_masters,
        "changed_expert_elements_after_first_step": changed_expert_elements,
        "owner_completions": {
            key: value for model in models for key, value in model.module.owner_completions.items()
        },
        "snapshots": len(snapshots),
        "param_sync_events": param_sync_events,
        "align_param_gather": ddp_config.align_param_gather,
        "native_mxfp8_param_staging": quantized and case.overlap_param_gather,
    }
    if graph_audit is not None:
        metadata["cuda_graph"] = graph_audit.finish()
        graph_audit.close()
    if case.backend == "dist_moe":
        close_dist_moe_runtime(models)
    for model in models:
        if case.overlap_param_gather:
            model.disable_forward_pre_hook(param_sync=False)
    del optimizer, models
    gc.collect()
    torch.cuda.empty_cache()
    return snapshots, metadata


def compare_snapshots(expected, actual, precision, exact=False):
    """Use predeclared normwise gates, preserving every failure for diagnosis."""
    assert expected.keys() == actual.keys(), "Numerical snapshot coverage differs"
    relative_limit = 0.02 if precision == "bf16" else 0.10
    max_limit = 0.10 if precision == "bf16" else 0.50
    metrics = {}
    for key, reference in expected.items():
        value = actual[key]
        assert reference.shape == value.shape, key
        if exact and torch.equal(reference, value) and torch.isfinite(reference).all():
            metrics[key] = dict(passed=True, relative_l2=0.0, max_absolute=0.0, max_normalized=0.0)
            continue
        reference, value = reference.double(), value.double()
        error = reference - value
        absolute = error.abs().max().item()
        relative = (error.norm() / reference.norm().clamp(min=1e-30)).item()
        normalized_max = absolute / max(reference.abs().max().item(), 1e-30)
        finite = bool(torch.isfinite(reference).all() and torch.isfinite(value).all())
        if exact or key.startswith(("initial/", "initial_master/")) or key.endswith("routing_map"):
            passed = finite and absolute == 0
        elif key.endswith("/loss"):
            passed = finite and relative <= (0.005 if precision == "bf16" else 0.05)
        else:
            passed = (
                finite
                and relative <= relative_limit
                and (absolute <= 1e-8 or normalized_max <= max_limit)
            )
        metrics[key] = dict(
            passed=passed,
            relative_l2=relative,
            max_absolute=absolute,
            max_normalized=normalized_max,
        )
    return metrics


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("vpp", [1, 2, 3], ids=["pp2", "pp2_vpp2", "pp2_vpp3"])
def test_dist_moe_pipeline_numeric(precision, vpp):
    """Check schedules with well-conditioned Adam updates, including native overlap.

    Epsilon=1e-4 prevents near-zero LayerNorm bias gradients from producing
    opposite O(lr) Adam updates solely from BF16/MXFP8 rounding. All original
    numerical gates still apply to losses, activations, gradients, FP32 masters,
    and logical weights. The CLI fixture keeps epsilon=1e-8 by default to retain
    separate diagnostics of ordinary Adam training trajectories.
    """
    from tests.unit_tests.test_utilities import Utils

    pytest.importorskip("dist_moe")
    if Utils.world_size < 8 or Utils.world_size % 8:
        pytest.skip("Requires PP2 * EP4 * expert-DP ranks")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Dist-MoE requires Blackwell")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vpp if vpp > 1 else None,
        expert_model_parallel_size=4,
    )
    try:
        case = PipelineCase(
            precision=precision,
            vpp=vpp,
            backend="alltoall",
            adam_eps=1e-4,
            overlap_grad_reduce=vpp >= 3,
            overlap_param_gather=vpp >= 3,
        )
        expected, _ = run_pipeline_case(case)
        case.backend = "dist_moe"
        actual, _ = run_pipeline_case(case)
        metrics = compare_snapshots(expected, actual, precision)
        failed = {key: value for key, value in metrics.items() if not value["passed"]}
        assert not failed, failed
    finally:
        Utils.destroy_model_parallel()
