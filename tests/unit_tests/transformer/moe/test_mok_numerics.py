# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real MoK entrypoints with MCore DDP: an independent EP1 PyTorch oracle for BF16, and
chunk-wise CUDA graph replay against eager for MXFP8."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


def _assert_numerics(actual, expected, name):
    """Use the pinned MoK BF16 suite's max-absolute and normalized-L1 limits.

    BF16 intermediates and contribution-wise gradient stores are rounded differently
    from PyTorch. Neither cosine similarity alone nor per-element relative error near
    zero describes that difference. The same bounds cover outputs and gradients.
    """
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = (actual - expected).abs()
    normalized_l1 = delta.sum() / expected.abs().sum().clamp_min(1e-12)
    assert (
        delta.max() <= 0.5 and normalized_l1 <= 0.01
    ), f"{name}: max_abs={delta.max().item()}, normalized_l1={normalized_l1.item()}"


def _reference(x, probs, weights):
    """Dense EP1 oracle: two routed SwiGLUs plus an optional gated shared expert.

    Weights are FP32 leaves containing BF16 values: FP32 GEMMs followed by BF16
    activation casts retain FP32 wgrad contributions instead of rounding weight.grad
    to BF16. This reference does not import MoK, its adapters, or its test helpers.
    """

    def linear(value, weight):
        return F.linear(value.float(), weight.float()).to(torch.bfloat16)

    def expert(fc1, fc2):
        gate, up = fc1.chunk(2, dim=0)
        return linear(F.silu(linear(x, gate)) * linear(x, up), fc2)

    shared = expert(
        weights["shared_experts.linear_fc1.weight"], weights["shared_experts.linear_fc2.weight"]
    )
    if "shared_experts.gate_weight" in weights:
        gate = torch.sigmoid(linear(x, weights["shared_experts.gate_weight"]))
        # MoK's gated forward combines in FP32; its backward differentiates the
        # BF16 shared*gate product. Express that precision contract with autograd.
        product = shared * gate
        result = product.float() + (shared.float() * gate.float() - product.float()).detach()
    else:
        result = shared.float()
    for index in range(2):
        routed = expert(
            weights[f"experts.linear_fc1.weight{index}"],
            weights[f"experts.linear_fc2.weight{index}"],
        )
        result = result + routed.float() * probs[:, index : index + 1]
    return result.to(torch.bfloat16)


def _reference_backward(x, probs, weights, upstream):
    x = x.detach().clone().requires_grad_()
    probs = probs.detach().clone().requires_grad_()
    weights = {name: value.detach().float().requires_grad_() for name, value in weights.items()}
    output = _reference(x, probs, weights)
    grads = torch.autograd.grad(output, (x, probs, *weights.values()), upstream)
    return output.detach(), grads[0], grads[1], dict(zip(weights, grads[2:]))


def _check_main_grads(parameters, expected):
    for name, parameter in parameters.items():
        _assert_numerics(parameter.main_grad, expected[name], name)
        assert parameter.grad is None, name
        assert parameter.grad_added_to_main_grad, name


class _MoKExperts(torch.nn.Module):
    """Retain native expert ownership; route probabilities are independent inputs."""

    def __init__(self, layer):
        super().__init__()
        self.experts = layer.experts
        self.shared_experts = layer.shared_experts
        self.megakernel_experts = layer.megakernel_experts

    def forward(self, x, probs):
        return self.megakernel_experts(x, probs, torch.ones_like(probs, dtype=torch.bool))


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
def test_real_mok_ddp_numerics(gated, main_grad_dtype):
    """Two microbatches, all weight buffers, no parameter mutation, and SGD refresh."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MoK requires Blackwell SM100/SM103")
    functional = pytest.importorskip("mok.functional")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.megakernel.mok import runtime
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            gradient_accumulation_fusion=True,
            moe_megakernel_backend="mok",
            moe_shared_expert_gate=gated,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            moe_megakernel_backend_config=dict(
                fwd_num_comm_sms=2,
                bwd_num_comm_sms=2,
                minibatch_size=256,
                macrobatch_size=256,
                schedule_capacity_multiplier=2.0,
            ),
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        assert groups.ep.size() == 1
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=2, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(config, submodules, layer_number=1, pg_collection=groups).cuda()
        model = _MoKExperts(layer)
        # Identical per-rank inputs/weights make DDP's average a no-op numerically.
        generator = torch.Generator(device="cuda").manual_seed(9123)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, device="cuda") / 16
                )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=main_grad_dtype == torch.float32,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
            ),
            model,
            pg_collection=groups,
        )
        parameters = dict(model.named_parameters())
        assert len(parameters) == 6 + int(gated)
        assert len({id(p) for p in parameters.values()}) == len(parameters)
        checkpoint = model.state_dict()
        assert set(parameters) <= set(checkpoint)
        assert all(not name.startswith("megakernel_experts.") for name in checkpoint)
        assert all(not name.startswith("megakernel_experts.") for name in parameters)
        backend = model.megakernel_experts
        if gated:
            assert backend.shared_output_gate_weight is model.shared_experts.gate_weight
        buffers = {name: p.main_grad for name, p in parameters.items()}
        assert all(buf.dtype == main_grad_dtype for buf in buffers.values())
        original = {name: p.detach().clone() for name, p in parameters.items()}
        expected = {name: torch.zeros_like(p.main_grad) for name, p in parameters.items()}
        batches = []
        ddp.zero_grad_buffer()
        for _ in range(2):
            x = torch.randn(512, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
            probs = torch.randn(512, 2, generator=generator, device="cuda").softmax(-1)
            upstream = torch.randn(x.shape, generator=generator, device="cuda", dtype=x.dtype) / 16
            reference, dx, dp, dw = _reference_backward(x, probs, original, upstream)
            x.requires_grad_()
            probs.requires_grad_()
            output = ddp(x, probs)
            _assert_numerics(output, reference, "output")
            output.backward(upstream)
            _assert_numerics(x.grad, dx, "input gradient")
            _assert_numerics(probs.grad, dp, "router probability gradient")
            for name, param in parameters.items():
                # Gate wgrad adds its FP32 contribution before the destination
                # cast; expert kernels round each reduction-store contribution.
                contribution = dw[name]
                if name != "shared_experts.gate_weight":
                    contribution = contribution.to(main_grad_dtype)
                expected[name].add_(contribution)
                assert param.main_grad is buffers[name], name
                torch.testing.assert_close(param, original[name], rtol=0, atol=0)
            _check_main_grads(parameters, expected)
            batches.append((x.detach(), probs.detach(), upstream))
        ddp.finish_grad_sync()
        _check_main_grads(parameters, expected)

        # Use a real optimizer over deduplicated native parameters. This checks
        # payload/cache refresh, not the distributed optimizer or MXFP8 mode.
        reference_weights = {
            name: torch.nn.Parameter(value.clone()) for name, value in original.items()
        }
        optimizer = torch.optim.SGD(parameters.values(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference_weights.values(), lr=0.03)
        for name, param in parameters.items():
            param.grad = param.main_grad.to(param.dtype)
            reference_weights[name].grad = expected[name].to(param.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        backend.is_first_microbatch = True
        x, probs, upstream = batches[0]
        with torch.no_grad():
            updated = ddp(x, probs)
            reference_updated = _reference(x, probs, reference_weights)
            _assert_numerics(updated, reference_updated, "post-update output")
            assert not torch.equal(updated, _reference(x, probs, original))

        if gated and main_grad_dtype == torch.float32:
            # Mutation controls use the real backend. They must fail the same
            # oracle, rather than passing just because a mocked hook was called.
            original_arguments = runtime._gate_up_weight_arguments

            def swapped(*args):
                sg, su, rg, ru = original_arguments(*args)
                return su, sg, rg, ru

            with patch.object(runtime, "_gate_up_weight_arguments", swapped), torch.no_grad():
                wrong = ddp(x, probs)
            with pytest.raises(AssertionError, match="swapped gate/up"):
                _assert_numerics(wrong, reference_updated, "swapped gate/up")
            real_backward = functional.backward
            for missing in ("experts.linear_fc1.weight0", "shared_experts.linear_fc2.weight"):
                ddp.zero_grad_buffer()
                _, _, _, expected_dw = _reference_backward(x, probs, reference_weights, upstream)

                def omit_write(*args, **kwargs):
                    result = real_backward(*args, **kwargs)
                    parameters[missing].main_grad.zero_()
                    return result

                with patch.object(functional, "backward", omit_write):
                    ddp(x.requires_grad_(), probs.requires_grad_()).backward(upstream)
                with pytest.raises(AssertionError, match=missing):
                    _assert_numerics(parameters[missing].main_grad, expected_dw[missing], missing)
    finally:
        functional.clear_workspace_cache()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        Utils.destroy_model_parallel()


def _train_step(ddp, batch, *, eager=False):
    """One training iteration's forward + backward on a chunk-wise model, as the schedules run it.

    ``eager=True`` detaches the captured block graphs for the call, so the same weights run the
    ordinary forward (a block replays only while its graphs are attached). The host state is
    prepared as ``forward_step`` does at the start of every iteration in both modes; only the eager
    forward ever reads ``is_first_microbatch``. Returns the logits and every parameter's
    accumulated ``main_grad``.
    """
    from megatron.core.transformer.cuda_graphs import set_current_microbatch

    decoder = ddp.module.decoder
    graphs = decoder.cuda_graphs
    if eager:
        decoder.cuda_graphs = []
    try:
        ddp.zero_grad_buffer()
        ddp.set_is_first_microbatch()
        set_current_microbatch(ddp, 0)
        logits = ddp(
            input_ids=batch["input_ids"], position_ids=batch["position_ids"], attention_mask=None
        )
        F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten()).backward()
        torch.cuda.synchronize()
    finally:
        decoder.cuda_graphs = graphs
    grads = {
        name: parameter.main_grad.detach().float().clone()
        for name, parameter in ddp.module.named_parameters()
    }
    return logits.detach().float().clone(), grads


def _assert_replay_matches_eager(replay, eager, phase):
    """Replay and eager run the same kernels on the same weights and inputs.

    Measured on GB200 (TE 2.20, mok 0.1.0), both weight layouts: every logit and every main_grad
    is bit-identical between the two, before and after the requantization. A replay that kept the
    capture-time scales instead is off by a relative L2 error of 1.2 on the logits and of 0.8 to
    11 on the gradients, so 1e-3 separates the two cases by almost three orders of magnitude and
    still tolerates a reduction-order change in a future kernel.
    """

    def check(actual, expected, name):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        # A zero reference would make the relative comparison vacuous.
        assert expected.norm() > 0, name
        error = ((actual - expected).norm() / expected.norm()).item()
        assert error <= 1e-3, f"{phase}: {name}: relative L2 error {error:.3e}"

    (logits, grads), (expected_logits, expected_grads) = replay, eager
    check(logits, expected_logits, "logits")
    assert grads.keys() == expected_grads.keys(), phase
    for name, expected in expected_grads.items():
        check(grads[name], expected, f"main_grad of {name}")


@torch.no_grad()
def _requantize_routed_experts(backend, amax_reduction_group):
    """Give the routed experts new MXFP8 scales the way an fp8_param optimizer step does.

    The distributed optimizer keeps an FP32 master of every FP8 parameter and, after each update,
    casts it into the parameter's existing TE storage (``quantize_param_shard`` on the member
    tensors, then ``post_all_gather_processing``); the same two calls are used here on masters
    recovered from the current weights. Every other 32-row band of each expert matrix is scaled
    by 4 and the bands in between by 0.75. An MXFP8 block spans 32 elements, along a row or down
    a column, so every block lies in one band: each block of a x4 band gets a larger E8M0
    exponent while 95 % of its payload bytes stay what they were, and the x0.75 bands change
    every payload byte and lower the exponent of four blocks in ten. Returns the scale bytes
    before and after.
    """
    from megatron.core.fp8_utils import (
        get_grouped_quantized_members,
        is_grouped_tensor_with_quantized_storage,
        post_all_gather_processing,
        quantize_param_shard,
    )

    weights, masters = [], []
    for parameters in (backend.routed_fc1_parameters, backend.routed_fc2_parameters):
        members = []
        for parameter in parameters:
            # One GroupedTensor parameter holds every expert; the optimizer casts its members.
            if is_grouped_tensor_with_quantized_storage(parameter):
                members.extend(get_grouped_quantized_members(parameter))
            else:
                members.append(parameter)
        for expert, weight in enumerate(members):
            master = weight.dequantize().float()
            bands = torch.arange(master.shape[0], device=master.device) // 32
            scaled = (bands + expert) % 2 == 0
            master[scaled] *= 4.0
            master[~scaled] *= 0.75
            weights.append(weight)
            masters.append(master.view(-1))

    def scales():
        return torch.cat(
            [
                scale.flatten()
                for weight in weights
                for scale in (weight._rowwise_scale_inv, weight._columnwise_scale_inv)
            ]
        ).clone()

    before = scales()
    # The optimizer reduces the block maxima over the ranks that share a master's shards. Here
    # every rank holds whole masters (offset 0), so the caller passes a group of this rank only.
    quantize_param_shard(weights, masters, [0] * len(weights), amax_reduction_group)
    post_all_gather_processing(list(backend.routed_fc1_parameters + backend.routed_fc2_parameters))
    return before, scales()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("single_grouped_weight", [False, True], ids=["split", "single-grouped"])
def test_real_mok_mxfp8_chunk_wise_replay_follows_requantized_experts(
    monkeypatch, single_grouped_weight
):
    """A one-layer MoK MXFP8 block graph must replay with the scales of the latest update.

    MoK reads the FP8 payloads straight from TE storage but consumes the E8M0 scales through its
    own tcgen05-layout copies. Eagerly those copies are refreshed when ``is_first_microbatch`` is
    set; a chunk-wise replay runs no Python, so the refresh has to be part of the captured block
    (``MoKMegakernel.quantized_routed_weights``). After capture the experts are requantized to
    new scales and the replayed forward and backward are compared with the eager ones on the same
    weights. The replay runs first: the eager reference refreshes the shared copies on the host.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MoK requires Blackwell SM100/SM103")
    functional = pytest.importorskip("mok.functional")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.num_microbatches_calculator import (
        destroy_num_microbatches_calculator,
        init_num_microbatches_calculator,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import is_te_min_version
    from tests.unit_tests.test_utilities import Utils, restore_process_state, snapshot_process_state

    if not is_te_min_version("2.14.0"):
        pytest.skip("CUDA graphs with the TE RNG tracker and grouped weights require TE >= 2.14.0")
    if single_grouped_weight:
        # Transformer Engine keeps one GroupedTensor parameter per layer only on request.
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")

    # MoK needs at least 512 tokens per rank and sizes in multiples of 256.
    seq_length, vocab_size, hidden_size, num_experts = 512, 128, 256, 4
    # Selects this rank's device, so the device-bound state below is read after it.
    Utils.initialize_model_parallel(expert_model_parallel_size=1)
    process_state = snapshot_process_state()
    cpu_rng_state, cuda_rng_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    previous_stream = torch.cuda.current_stream()
    helper = None
    try:
        # The capture sizes its Transformer Engine warm-up from the run's micro-batch count.
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=Utils.world_size,
            micro_batch_size=1,
            data_parallel_size=Utils.world_size,
        )
        torch.manual_seed(123)
        # Graph-safe RNG tracker, as the training loop enables for CUDA graphs.
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        # Like the training loop, run everything on a side stream: AccumulateGrad nodes keep the
        # stream they were created on, and the default stream must not be pulled into a capture.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(previous_stream)
        torch.cuda.set_stream(side_stream)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=hidden_size,
            moe_shared_expert_intermediate_size=hidden_size,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=single_grouped_weight,
            moe_single_grouped_weight=single_grouped_weight,
            gated_linear_unit=True,
            activation_func=F.silu,
            gradient_accumulation_fusion=True,
            moe_megakernel_backend="mok",
            moe_megakernel_backend_config=dict(
                fwd_num_comm_sms=2,
                bwd_num_comm_sms=2,
                minibatch_size=256,
                macrobatch_size=256,
                schedule_capacity_multiplier=2.0,
            ),
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            cuda_graph_impl="chunk_wise",
            cuda_graph_modules=[],
            cuda_graph_warmup_steps=2,
            # The batch carries no attention mask (causal attention).
            create_attention_mask_in_dataloader=False,
            use_cpu_initialization=True,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        assert groups.ep.size() == 1
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_experts, moe_grouped_gemm=True
            ),
            vocab_size=vocab_size,
            max_sequence_length=seq_length,
            position_embedding_type="rope",
        ).cuda()
        model.train()
        # MoK accumulates its weight gradients into the DDP-owned main_grad buffers.
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
            ),
            model,
            pg_collection=groups,
        )
        backend = model.decoder.layers[0].mlp.megakernel_experts
        assert backend.use_mxfp8_weights
        assert backend.native_single_grouped_weights == single_grouped_weight
        tokens = torch.arange(seq_length, dtype=torch.int64, device="cuda")
        batch = {
            "input_ids": (tokens * 7 % vocab_size).unsqueeze(0),
            "labels": ((tokens + 1) * 7 % vocab_size).unsqueeze(0),
            "position_ids": tokens.unsqueeze(0),
        }

        # cuda_graph_warmup_steps eager iterations: the first one builds MoK's weight views.
        _train_step(ddp, batch)
        eager_before = _train_step(ddp, batch)

        helper = cuda_graphs.TECudaGraphHelper(
            model=[ddp], config=config, seq_length=seq_length, micro_batch_size=1
        )
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert len(model.decoder.cuda_graphs) == 1

        replays = []
        original_replay = torch.cuda.CUDAGraph.replay

        def counting_replay(graph, *args, **kwargs):
            replays.append(graph)
            return original_replay(graph, *args, **kwargs)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting_replay)

        _assert_replay_matches_eager(_train_step(ddp, batch), eager_before, "capture-time weights")
        assert len(replays) == 2, "expected one forward and one backward block replay"

        old_scales, new_scales = _requantize_routed_experts(backend, groups.ep)
        # Every scale of the x4 bands (half of them) and part of the others: 69 % measured.
        changed = (old_scales != new_scales).float().mean().item()
        assert changed >= 0.5, f"the requantization moved only {changed:.0%} of the scales"

        replay_after = _train_step(ddp, batch)
        assert len(replays) == 4, "the step after the requantization did not replay the graphs"
        eager_after = _train_step(ddp, batch, eager=True)
        assert len(replays) == 4, "the eager reference replayed a graph"

        _assert_replay_matches_eager(replay_after, eager_after, "requantized weights")
        # The update is visible: these weights do not reproduce the capture-time results.
        with pytest.raises(AssertionError, match="relative L2 error"):
            _assert_replay_matches_eager(eager_after, eager_before, "requantization")
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_stream(previous_stream)
        if cuda_graphs.is_graph_capturing():
            # A capture that raised leaves Megatron's and Transformer Engine's flags set.
            cuda_graphs._set_capture_end()
            cuda_graphs.te_set_capture_end()
        if helper is not None and helper.graphs_created():
            helper.delete_cuda_graphs()
        functional.clear_workspace_cache()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        restore_process_state(process_state)
