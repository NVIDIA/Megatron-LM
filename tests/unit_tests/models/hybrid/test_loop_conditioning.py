# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.hybrid.hybrid_layer_allocation import get_hybrid_total_layer_count
from megatron.core.models.hybrid.hybrid_layer_specs import (
    adaptive_delta_gated_delta_product_stack_spec,
    adaptive_delta_hybrid_stack_spec,
    loop_conditioned_adaptive_delta_gated_delta_product_stack_spec,
    loop_conditioned_adaptive_delta_hybrid_inference_stack_spec,
    loop_conditioned_adaptive_delta_hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.loop_conditioning import (
    BranchLoopConditioning,
    LayerLoopConditioning,
    LoopConditioner,
    LoopConditioningConfig,
    RegionLoopConditioning,
    initialize_loop_conditioning_state,
    validate_loop_conditioning_checkpoint,
    validate_unconditioned_checkpoint_keys,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("values", [(3, 4, 4), (0, 4, 4), (8, 0, 4), (8, 4, 0), (True, 4, 4)])
def test_conditioning_config_validates_dimensions(values):
    with pytest.raises(ValueError):
        LoopConditioningConfig(*values)


def small_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        normalization="RMSNorm",
    )


@pytest.mark.parametrize("init_on_cpu", [False, True])
def test_conditioning_initialization_matches_across_tp_rng_streams(init_on_cpu):
    config = small_config()
    config.use_cpu_initialization = init_on_cpu
    settings = LoopConditioningConfig(8, 4, 4)
    replicas = []
    device = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device]):
        for tp_rank in (0, 1):
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(
                123,
                tp_rank=tp_rank,
                ep_rank=0,
                etp_rank=0,
                gtp_remat_rank=0,
                egtp_remat_rank=0,
                gtp_remat_world_size=1,
                egtp_remat_world_size=1,
            )
            if tp_rank:
                # A rank-local draw must not affect the replicated CUDA weights.
                torch.rand(32, device="cuda")
            cuda_state = torch.cuda.get_rng_state()
            groups = ProcessGroupCollection()
            replicas.append(
                torch.nn.ModuleList(
                    [
                        LoopConditioner(config, settings, groups, 1),
                        LayerLoopConditioning(config, settings, groups, mixer=True, mlp=False),
                    ]
                )
            )
            assert torch.equal(cuda_state, torch.cuda.get_rng_state())
    for first, second in zip(replicas[0].parameters(), replicas[1].parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert replicas[0][0].encoder_in.weight.count_nonzero()
    assert replicas[0][1].mixer.down.weight.count_nonzero()


def test_conditioning_parameter_count_matches_koi(monkeypatch):
    import megatron.core.models.hybrid.loop_conditioning as implementation

    def meta_linear(config, in_features, out_features, *, zero=False, bias=True):
        return torch.nn.Linear(in_features, out_features, bias=bias, device="meta")

    monkeypatch.setattr(implementation, "_linear", meta_linear)
    config = TransformerConfig(
        num_layers=52, hidden_size=3072, num_attention_heads=24, use_cpu_initialization=True
    )
    settings = LoopConditioningConfig(256, 256, 4)
    controller = LoopConditioner(config, settings, ProcessGroupCollection(), 1)
    branches = [
        LayerLoopConditioning(config, settings, ProcessGroupCollection(), mixer=True, mlp=False)
        for _ in range(24 + 14)
    ]
    count = sum(parameter.numel() for parameter in controller.parameters())
    count += sum(parameter.numel() for branch in branches for parameter in branch.parameters())
    assert count == 155_627_520  # The paper reports 155.6M added conditioning parameters.


def test_progress_and_budget_encoding_and_reference_budget():
    config, settings = small_config(), LoopConditioningConfig(8, 4, 4)
    encoder = LoopConditioner(config, settings, ProcessGroupCollection(), 1)
    assert encoder(0, 4).shape == (8,)
    assert not torch.equal(encoder(0, 4), encoder(1, 4))
    assert not torch.equal(encoder(0, 4), encoder(0, 2))
    phases = 2 * math.pi * torch.tensor([[2 / 4], [4 / 4]]) * encoder.frequencies
    embedding = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten()
    expected = F.linear(
        F.silu(F.linear(embedding, encoder.encoder_in.weight, encoder.encoder_in.bias)),
        encoder.encoder_out.weight,
        encoder.encoder_out.bias,
    )
    torch.testing.assert_close(encoder(1, 4), expected)
    for index in (-1, 4, True, 0.5):
        with pytest.raises(ValueError, match="loop_index"):
            encoder(index, 4)
    # K is a reference scale, not a hard depth cap.
    assert torch.isfinite(encoder(7, 8)).all()
    with pytest.raises(ValueError, match="reference budget"):
        altered = dict(encoder.state_dict())
        altered["reference_max_loops"] = torch.tensor(8)
        encoder.load_state_dict(altered)


def test_branch_modulation_formula_and_gradients():
    config, settings = small_config(), LoopConditioningConfig(8, 4, 4)
    module = BranchLoopConditioning(config, settings)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    x = torch.randn(3, 2, 8, requires_grad=True)
    c = torch.randn(8, requires_grad=True)
    bias = torch.randn(8, requires_grad=True)
    reference_bias = bias.detach().clone().requires_grad_()
    reference_x, reference_c = (
        x.detach().clone().requires_grad_(),
        c.detach().clone().requires_grad_(),
    )
    named = dict(module.named_parameters())
    copies = {name: value.detach().clone().requires_grad_() for name, value in named.items()}

    def linear(name, value):
        return F.linear(value, copies[name + ".weight"], copies.get(name + ".bias"))

    modulated = (1 + linear("input_scale", reference_c)) * reference_x + linear(
        "input_shift", reference_c
    )
    corrected = modulated + linear(
        "up", linear("low_rank_gate", reference_c) * linear("down", modulated)
    )
    expected = (1 + linear("output_scale", reference_c)) * (corrected + reference_bias)
    output, scaled_bias = module.output((module.input(x, c), bias), c)
    actual = output + scaled_bias
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), [x, c, bias, *named.values()])
    expected_grad = torch.autograd.grad(
        expected.square().sum(), [reference_x, reference_c, reference_bias, *copies.values()]
    )
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)


def test_entry_exit_initialization_and_first_pass_zero_gradient_edges():
    config, settings = small_config(), LoopConditioningConfig(8, 4, 4)
    module = RegionLoopConditioning(config, settings)
    u, h, c = torch.randn(3, 1, 8), torch.randn(3, 1, 8), torch.randn(8)
    torch.testing.assert_close(module.entry(u, h, c, 0), u)
    torch.testing.assert_close(
        module.entry(u, h, c, 1), F.rms_norm(u, (8,), eps=config.layernorm_epsilon)
    )
    torch.testing.assert_close(module.exit(c, 0), torch.ones(8))
    torch.testing.assert_close(module.exit(c, 1), torch.zeros(8))
    module.entry(u, h, c, 0).sum().backward()
    for head in (module.entry_scale, module.entry_shift, module.reinjection):
        for parameter in head.parameters():
            assert parameter.grad is not None and not parameter.grad.count_nonzero()


def test_learned_entry_exit_formula_and_gradients():
    config = small_config()
    module = RegionLoopConditioning(config, LoopConditioningConfig(8, 4, 4))
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    u = torch.randn(3, 1, 8, requires_grad=True)
    h = torch.randn_like(u, requires_grad=True)
    c = torch.randn(8, requires_grad=True)
    torch.testing.assert_close(module.entry(u, h, c, 0), u)

    def linear(head):
        return F.linear(c, head.weight, head.bias)

    expected_entry = (
        (1 + linear(module.entry_scale)) * F.rms_norm(u, (8,), eps=module.eps)
        + linear(module.entry_shift)
        + linear(module.reinjection) * h
    )
    expected_exit = linear(module.exit_scale)
    torch.testing.assert_close(module.exit(c, 0), 1 + expected_exit)
    actual = module.entry(u, h, c, 1) + module.exit(c, 1)
    expected = expected_entry + expected_exit
    torch.testing.assert_close(actual, expected)
    variables = [u, h, c, *module.parameters()]
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), variables),
        torch.autograd.grad(expected.square().sum(), variables),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.fixture
def models():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)

    def build(
        pattern="-:*-:-",
        *,
        conditioned=True,
        dtype=torch.float32,
        grad_fusion=False,
        spec=None,
        settings=None,
        distribution="3:1",
        **overrides,
    ):
        config = TransformerConfig(
            num_layers=get_hybrid_total_layer_count(pattern),
            hidden_size=128,
            num_attention_heads=4,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=True,
            hidden_dropout=0,
            attention_dropout=0,
            gradient_accumulation_fusion=grad_fusion,
            normalization="RMSNorm",
            gdn_kernel_backend="torch",
            deterministic_mode="G" in pattern,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            mamba_num_heads=4,
            mamba_head_dim=32,
            mamba_num_groups=4,
            mamba_state_dim=16,
            **overrides,
        )
        spec = spec or (
            loop_conditioned_adaptive_delta_hybrid_stack_spec
            if conditioned
            else adaptive_delta_hybrid_stack_spec
        )
        return HybridModel(
            config,
            spec,
            vocab_size=32,
            max_sequence_length=4,
            hybrid_layer_pattern=pattern,
            n_loops_distribution=distribution,
            loop_conditioning_config=(
                (settings or LoopConditioningConfig(8, 4, 4)) if conditioned else None
            ),
        ).cuda()

    yield build
    Utils.destroy_model_parallel()


def inputs():
    return torch.tensor([[1, 2, 3, 4]], device="cuda"), torch.arange(4, device="cuda").unsqueeze(0)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:G-:-", "-:*:-*:-"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_heads_preserve_gate_only_checkpoint_for_every_depth(models, pattern, dtype):
    source = models(pattern, conditioned=False, dtype=dtype)
    with torch.no_grad():
        for gate in source.decoder.recurrence_gates:
            gate.beta_proj.weight.normal_(std=0.01)
            gate.beta_proj.bias.normal_(std=0.01)
            gate.base_alpha_raw.fill_(0.2)
    target = models(pattern, dtype=dtype)
    state = source.state_dict()
    initialize_loop_conditioning_state(target, state)
    target.load_state_dict(state, strict=True)
    tokens, positions = inputs()
    with torch.no_grad():
        expected = source(tokens, positions, None, n_loops=1)
        for count in (1, 3, 2, 4, 1):
            torch.testing.assert_close(target(tokens, positions, None, n_loops=count), expected)
    for module in target.modules():
        if hasattr(module, "_loop_condition"):
            assert module._loop_condition.get() is None
    assert not hasattr(target.decoder.layers[0], "_loop_condition")
    assert hasattr(target.decoder.layers[-1], "_loop_condition")


@pytest.mark.parametrize("loops", [1, 3])
@pytest.mark.parametrize("grad_fusion", [False, True])
def test_conditioned_ddp_has_gradients_for_all_parameters(models, loops, grad_fusion):
    model = models("-:*:-*:-", grad_fusion=grad_fusion)
    ddp = DistributedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(overlap_grad_reduce=True),
        module=model,
        pg_collection=model.pg_collection,
    )
    ddp.zero_grad_buffer()
    tokens, positions = inputs()
    with ddp.no_sync():
        ddp(tokens, positions, None, n_loops=loops).square().mean().backward()
    ddp(tokens, positions, None, n_loops=loops).square().mean().backward()
    ddp.finish_grad_sync()
    assert all(parameter.grad is None for parameter in model.parameters())  # DDP owns main_grad.
    assert all(torch.isfinite(parameter.main_grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:G-:-", "-:*:-*:-"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_conditioning_matches_independent_per_pass_models(models, pattern, dtype):
    model = models(pattern, dtype=dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "loop_conditioning" in name:
                parameter.normal_(std=0.03)
        for gate in model.decoder.recurrence_gates:
            gate.beta_proj.weight.normal_(std=0.01)
            gate.beta_proj.bias.normal_(std=0.01)
        for region in model.decoder.loop_conditioning.regions:
            region.exit_scale.bias.fill_(0.3)
    tokens, positions = inputs()
    ranges = model.recurrent_layer_ranges
    for count in (1, 3, 2):
        copies = [models(pattern, dtype=dtype) for _ in range(count)]
        for replica in copies:
            replica.load_state_dict(model.state_dict(), strict=True)
        reference = copies[0]
        conditions = [
            replica.decoder.loop_conditioning(i, count) for i, replica in enumerate(copies)
        ]
        model.zero_grad(set_to_none=True)
        actual = model(tokens, positions, None, n_loops=count)
        hidden = reference.embedding(tokens, positions)
        hidden, _ = reference.decoder.layers[0](hidden, attention_mask=None)
        for region_index, (start, end) in enumerate(ranges):
            original = hidden
            for i, replica in enumerate(copies):
                previous = hidden
                combined = previous + original
                modulation = replica.decoder.loop_conditioning.regions[region_index]
                hidden = modulation.entry(combined, original, conditions[i], i).clone()
                for layer_index in range(start, end):
                    hidden, _ = replica.decoder.layers[layer_index](
                        hidden, attention_mask=None, loop_condition=conditions[i]
                    )
                hidden = replica.decoder.recurrence_gates[region_index](
                    previous, hidden, combined, update_scale=modulation.exit(conditions[i], i)
                )
        hidden, _ = reference.decoder.layers[-1](
            hidden, attention_mask=None, loop_condition=conditions[-1]
        )
        expected, _ = reference.output_layer(reference.decoder.final_norm(hidden))
        torch.testing.assert_close(actual, expected.transpose(0, 1).contiguous())
        actual.square().sum().backward()
        expected.square().sum().backward()
        replica_parameters = [dict(replica.named_parameters()) for replica in copies]
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            total = torch.zeros_like(parameter.grad)
            for replica in reversed(replica_parameters):
                if replica[name].grad is not None:
                    total.add_(replica[name].grad)
            torch.testing.assert_close(parameter.grad, total, msg=name)
        assert all(
            module._loop_condition.get() is None
            for replica in [model, *copies]
            for module in replica.modules()
            if hasattr(module, "_loop_condition")
        )


def test_gdp_conditioned_layers_preserve_warmup_and_train(models):
    from megatron.core.ssm.gated_delta_product import check_fla_sequence_packing_support

    supported, reason = check_fla_sequence_packing_support()
    if not supported:
        pytest.skip(reason)
    pattern = "M-:M-:*-"
    source = models(
        pattern,
        conditioned=False,
        dtype=torch.bfloat16,
        spec=adaptive_delta_gated_delta_product_stack_spec,
    )
    target = models(
        pattern,
        dtype=torch.bfloat16,
        spec=loop_conditioned_adaptive_delta_gated_delta_product_stack_spec,
    )
    source = Float16Module(source.config, source).module
    target = Float16Module(target.config, target).module
    state = source.state_dict()
    initialize_loop_conditioning_state(target, state)
    target.load_state_dict(state, strict=True)
    tokens, positions = inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            source(tokens, positions, None, n_loops=1), target(tokens, positions, None, n_loops=3)
        )
    target(tokens, positions, None, n_loops=3).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters())


def test_moe_layer_conditioning_keeps_routing_and_backward(models):
    model = models("-:E-:-", num_moe_experts=2)
    output = model(*inputs(), None, n_loops=3)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_region_modulation_is_independent_and_mtp_unconditioned(models):
    model = models("-:*:-*:-/-")
    regions = model.decoder.loop_conditioning.regions
    assert len(regions) == 2
    assert regions[0].entry_scale.weight.data_ptr() != regions[1].entry_scale.weight.data_ptr()
    assert not any(
        isinstance(module, (LoopConditioner, LayerLoopConditioning))
        for module in model.mtp.modules()
    )


def test_conditioned_optimizer_continuation(models, tmp_path):
    model = models()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens, positions = inputs()
    model(tokens, positions, None, n_loops=3).square().mean().backward()
    optimizer.step()
    path = tmp_path / f"conditioned_optimizer_{Utils.rank}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    restored = models()
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(state["optimizer"])
    for root, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        root(tokens, positions, None, n_loops=2).square().mean().backward()
        opt.step()
    for actual, expected in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_interleaved_forward_graphs_do_not_reuse_conditioning_state(models):
    model = models()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "loop_conditioning" in name:
                parameter.normal_(std=0.03)
    replicas = [models(), models()]
    for replica in replicas:
        replica.load_state_dict(model.state_dict())
    tokens, positions = inputs()
    first = model(tokens, positions, None, n_loops=1)
    second = model(tokens, positions, None, n_loops=3)
    (first.square().mean() + second.square().mean()).backward()
    for count, replica in zip((1, 3), replicas, strict=True):
        replica(tokens, positions, None, n_loops=count).square().mean().backward()
    copy_parameters = [dict(replica.named_parameters()) for replica in replicas]
    for name, parameter in model.named_parameters():
        expected = copy_parameters[0][name].grad + copy_parameters[1][name].grad
        torch.testing.assert_close(parameter.grad, expected, msg=name)


def test_conditioning_import_requires_fresh_optimizer(models):
    from megatron.training.checkpointing import _validate_loop_conditioning_import

    args = SimpleNamespace(init_loop_conditioning=True)
    with pytest.raises(ValueError, match="optimizer parameter layout"):
        _validate_loop_conditioning_import([models()], args, "torch_dist", True)
    with pytest.raises(ValueError, match="conditioned model spec"):
        _validate_loop_conditioning_import([models(conditioned=False)], args, "torch_dist", False)
    args.ckpt_assume_constant_structure = True
    with pytest.raises(ValueError, match="checkpoint structure"):
        _validate_loop_conditioning_import([models()], args, "torch_dist", False)


def test_conditioning_with_bf16_parameters_and_fp32_residuals(models):
    source = models(conditioned=False, dtype=torch.bfloat16, fp32_residual_connection=True)
    target = models(dtype=torch.bfloat16, fp32_residual_connection=True)
    state = source.state_dict()
    initialize_loop_conditioning_state(target, state)
    target.load_state_dict(state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            source(*inputs(), None, n_loops=1), target(*inputs(), None, n_loops=3)
        )
    target(*inputs(), None, n_loops=3).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters())


def test_conditioning_frequencies_stay_fp32(models):
    model = models(dtype=torch.bfloat16)
    wrapped = Float16Module(model.config, model)
    assert wrapped.module.decoder.loop_conditioning.frequencies.dtype == torch.float32


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_distributed_gate_only_import_and_conditioned_resume(models, tmp_path_dist_ckpt, pattern):
    source = models(pattern, conditioned=False)
    path = tmp_path_dist_ckpt / f"conditioning_import_{len(source.decoder.layers)}"
    path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": source.sharded_state_dict()}, path)
    validate_unconditioned_checkpoint_keys(dist_checkpointing.load_tensors_metadata(path))
    target = models(pattern)
    state = target.sharded_state_dict(
        metadata={"dp_cp_group": target.pg_collection.dp_cp, "initialize_loop_conditioning": True}
    )
    loaded = dist_checkpointing.load({"model": state}, path, strict="raise_all")
    target.load_state_dict(loaded["model"], strict=True)
    tokens, positions = inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            source(tokens, positions, None, n_loops=1), target(tokens, positions, None, n_loops=3)
        )
        for module in target.modules():
            if isinstance(module, (LoopConditioner, LayerLoopConditioning)):
                for parameter in module.parameters():
                    parameter.normal_(std=0.03)
    saved = tmp_path_dist_ckpt / f"conditioned_resume_{len(source.decoder.layers)}"
    saved.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": target.sharded_state_dict()}, saved)
    restored = models(pattern)
    restored.load_state_dict(
        dist_checkpointing.load(
            {"model": restored.sharded_state_dict()}, saved, strict="raise_all"
        )["model"],
        strict=True,
    )
    for actual, expected in zip(restored.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    wrong_budget = models(pattern, settings=LoopConditioningConfig(8, 4, 8))
    with pytest.raises(ValueError, match="reference budget"):
        wrong_budget.load_state_dict(
            dist_checkpointing.load(
                {"model": wrong_budget.sharded_state_dict()}, saved, strict="raise_all"
            )["model"],
            strict=True,
        )
    with pytest.raises(ValueError, match="already contains loop conditioning"):
        validate_unconditioned_checkpoint_keys(dist_checkpointing.load_tensors_metadata(saved))
    with pytest.raises(ValueError, match="select a conditioned model spec"):
        validate_loop_conditioning_checkpoint(
            source, dist_checkpointing.load_tensors_metadata(saved)
        )


def test_ordinary_import_preserves_gates_and_backbone_checks(models):
    source = models(conditioned=False)
    with torch.no_grad():
        source.decoder.recurrence_gates[0].base_alpha_raw.fill_(0.3)
    target = models()
    state = source.state_dict()
    initialize_loop_conditioning_state(target, state)
    target.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        target.decoder.recurrence_gates[0].base_alpha_raw,
        source.decoder.recurrence_gates[0].base_alpha_raw,
    )
    with pytest.raises(ValueError, match="already contains loop conditioning"):
        initialize_loop_conditioning_state(target, state)
    del state["output_layer.weight"]
    with pytest.raises(RuntimeError, match="output_layer.weight"):
        target.load_state_dict(state, strict=True)
    with pytest.raises(RuntimeError, match="loop_conditioning"):
        models().load_state_dict(source.state_dict(), strict=True)


def test_conditioned_inference_spec_uncached_evaluation(models):
    source = models(conditioned=False, dtype=torch.bfloat16, add_bias_linear=False)
    target = models(
        dtype=torch.bfloat16,
        add_bias_linear=False,
        transformer_impl="inference_optimized",
        spec=loop_conditioned_adaptive_delta_hybrid_inference_stack_spec,
    )
    state = source.state_dict()
    initialize_loop_conditioning_state(target, state)
    target.load_state_dict(state, strict=True)
    source.eval()
    target.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            source(*inputs(), None, n_loops=1), target(*inputs(), None, n_loops=3)
        )


def test_conditioning_config_requires_a_conditioned_spec(models):
    config = LoopConditioningConfig(8, 4, 4)
    base = models(conditioned=False)
    with pytest.raises(ValueError, match="requires a loop-conditioned stack spec"):
        HybridModel(
            base.config,
            adaptive_delta_hybrid_stack_spec,
            32,
            4,
            hybrid_layer_pattern="-:*-:-",
            loop_conditioning_config=config,
        )


@pytest.mark.parametrize("inference", [False, True])
def test_conditioning_cli_reaches_builder(monkeypatch, inference):
    from megatron.training.argument_utils import hybrid_config_from_args
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.models.hybrid import HybridModelBuilder

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern=:*-:",
            "--spec",
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "loop_conditioned_adaptive_delta_hybrid_stack_spec",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "4",
            "--max-position-embeddings",
            "4",
            "--micro-batch-size",
            "1",
            "--loop-conditioning-dim",
            "8",
            "--loop-conditioning-rank",
            "4",
            "--loop-conditioning-max-loops",
            "8",
        ]
        + (
            [
                "--transformer-impl",
                "inference_optimized",
                "--normalization",
                "RMSNorm",
                "--disable-bias-linear",
            ]
            if inference
            else []
        ),
    )
    args = validate_args(parse_args())
    config = hybrid_config_from_args(args)
    assert config.loop_conditioning_config == LoopConditioningConfig(8, 4, 8)
    assert config.hybrid_stack_spec.submodules.loop_conditioning is not None
    config.vocab_size = 32
    monkeypatch.setattr("megatron.training.models.hybrid.HybridModel", lambda **kwargs: kwargs)
    kwargs = HybridModelBuilder(config).build_model(
        SimpleNamespace(), pre_process=True, post_process=True
    )
    assert kwargs["loop_conditioning_config"] == config.loop_conditioning_config


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dist"])
def test_native_conditioning_import_and_resume(
    models, tmp_path_dist_ckpt, monkeypatch, run_config, ckpt_format
):
    from megatron.training import checkpointing
    from megatron.training.argument_utils import checkpoint_config_from_args
    from megatron.training.arguments import parse_args, validate_args

    path = tmp_path_dist_ckpt / f"native_conditioning_{ckpt_format}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern=-:*-:-",
            "--spec",
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "loop_conditioned_adaptive_delta_hybrid_stack_spec",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "4",
            "--max-position-embeddings",
            "4",
            "--micro-batch-size",
            "1",
            "--normalization",
            "RMSNorm",
            "--position-embedding-type",
            "none",
            "--untie-embeddings-and-output-weights",
            "--loop-conditioning-dim",
            "8",
            "--loop-conditioning-rank",
            "4",
            "--load",
            str(path),
            "--ckpt-format",
            ckpt_format,
            "--finetune",
            "--init-loop-conditioning",
            "--no-load-rng",
        ],
    )
    args = validate_args(parse_args())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "has_nvidia_modelopt", False)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    run_config.tokenizer.padded_vocab_size = 32
    source, target = models(conditioned=False), models()
    saved_args = copy.copy(args)
    saved_args.spec = [
        "megatron.core.models.hybrid.hybrid_layer_specs",
        "adaptive_delta_hybrid_stack_spec",
    ]
    saved_args.loop_conditioning_dim = 256  # Unused defaults in an unconditioned checkpoint.

    def save(model, root, source_args):
        directory = root / "iter_0000007"
        if ckpt_format == "torch_dist":
            directory.mkdir(parents=True, exist_ok=True)
            dist_checkpointing.save(
                {
                    "model": model.sharded_state_dict(),
                    "args": source_args,
                    "iteration": 7,
                    "checkpoint_version": 3.1,
                },
                directory,
            )
        elif Utils.rank == 0:
            (directory / "mp_rank_00").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": source_args,
                    "iteration": 7,
                    "checkpoint_version": 3.1,
                },
                directory / "mp_rank_00" / "model_optim_rng.pt",
            )
        torch.distributed.barrier()
        if Utils.rank == 0:
            (root / "latest_checkpointed_iteration.txt").write_text("7")
        torch.distributed.barrier()

    save(source, path, saved_args)
    checkpointing.load_args_from_checkpoint(args)
    run_config.checkpoint = checkpoint_config_from_args(args)
    assert args.spec[-1] == "loop_conditioned_adaptive_delta_hybrid_stack_spec"
    assert args.loop_conditioning_dim == 8
    groups = dict(
        tp_group=target.pg_collection.tp,
        pp_group=target.pg_collection.pp,
        dp_cp_group=target.pg_collection.dp_cp,
        dp_group=target.pg_collection.dp,
    )
    checkpointing.load_checkpoint([target], None, None, **groups)
    with torch.no_grad():
        torch.testing.assert_close(
            source(*inputs(), None, n_loops=1), target(*inputs(), None, n_loops=3)
        )
        for module in target.modules():
            if isinstance(module, (LoopConditioner, LayerLoopConditioning)):
                for parameter in module.parameters():
                    parameter.normal_(std=0.03)
    path_resume = tmp_path_dist_ckpt / f"native_conditioned_resume_{ckpt_format}"
    args.init_loop_conditioning = False
    save(target, path_resume, copy.copy(args))
    run_config.checkpoint.load = str(path_resume)
    restored = models()
    checkpointing.load_checkpoint([restored], None, None, **groups)
    for actual, expected in zip(restored.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="select a conditioned model spec"):
        checkpointing.load_checkpoint([source], None, None, **groups)
    args.init_loop_conditioning = True
    with pytest.raises(ValueError, match="already contains loop conditioning"):
        checkpointing.load_checkpoint([restored], None, None, **groups)


def test_conditioning_scope_resets_after_exception(models, monkeypatch):
    model = models()
    layer = model.decoder.layers[1]

    def fail(*args, **kwargs):
        raise RuntimeError("intentional failure")

    with monkeypatch.context() as patch:
        patch.setattr(layer.self_attention, "forward", fail)
        with pytest.raises(RuntimeError, match="intentional failure"):
            model(*inputs(), None)
    assert layer._loop_condition.get() is None
    assert torch.isfinite(model(*inputs(), None)).all()
