# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.hybrid.hybrid_layer_allocation import get_hybrid_total_layer_count
from megatron.core.models.hybrid.hybrid_layer_specs import (
    adaptive_delta_gated_delta_product_inference_stack_spec,
    adaptive_delta_hybrid_inference_stack_spec,
    adaptive_delta_hybrid_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.recurrence_gate import (
    AdaptiveDeltaGate,
    initialize_recurrence_gate_state,
    validate_ungated_checkpoint_keys,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.test_utilities import Utils


def test_gate_initialization_and_unbounded_rate():
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
    )
    gate = AdaptiveDeltaGate(config, ProcessGroupCollection())
    assert not gate.beta_proj.weight.count_nonzero()
    assert not gate.beta_proj.bias.count_nonzero()
    assert not gate.base_alpha_raw.count_nonzero()
    assert gate.base_alpha_raw.keep_in_fp32
    assert sum(parameter.numel() for parameter in gate.parameters()) == 8 * 8 + 2 * 8
    state, candidate, entry = torch.zeros(2, 1, 8), torch.ones(2, 1, 8), torch.randn(2, 1, 8)
    torch.testing.assert_close(gate(state, candidate, entry), candidate * (math.log(2) / 2))
    with torch.no_grad():
        gate.base_alpha_raw.fill_(3)
    assert (gate(state, candidate, entry) > 1).all()  # No sigmoid-only clamp or normalization.


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_gate_marks_replicated_sequence_parallel_parameters(sequence_parallel):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        tensor_model_parallel_size=2 if sequence_parallel else 1,
        sequence_parallel=sequence_parallel,
        use_cpu_initialization=True,
    )
    gate = AdaptiveDeltaGate(config, ProcessGroupCollection())
    assert all(p.sequence_parallel == sequence_parallel for p in gate.parameters())


def test_gate_formula_and_gradients():
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=2, use_cpu_initialization=True
    )
    gate = AdaptiveDeltaGate(config, ProcessGroupCollection())
    with torch.no_grad():
        gate.beta_proj.weight.normal_(std=0.2)
        gate.beta_proj.bias.normal_(std=0.1)
        gate.base_alpha_raw.normal_(std=0.2)
    inputs = [torch.randn(3, 2, 8, requires_grad=True) for _ in range(3)]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    parameters = list(gate.parameters())
    copies = [parameter.detach().clone().requires_grad_() for parameter in parameters]
    # named order: the direct base-rate parameter precedes the beta_proj child.
    gamma, weight, bias = copies
    previous, candidate, entry = reference_inputs
    beta = torch.sigmoid(F.linear(entry, weight, bias)) * F.softplus(gamma)
    expected = previous + beta * (candidate - previous)
    actual = gate(*inputs)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs + parameters)
    expected_grads = torch.autograd.grad(expected.square().sum(), reference_inputs + copies)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.fixture
def models():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)

    def build(
        pattern="-:*-:-",
        *,
        spec=adaptive_delta_hybrid_stack_spec,
        dtype=torch.float32,
        distribution="3:1",
        grad_fusion=False,
        **config_overrides,
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
            gdn_kernel_backend="torch",
            deterministic_mode="G" in pattern,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        for name, value in config_overrides.items():
            setattr(config, name, value)
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=32,
            max_sequence_length=4,
            hybrid_layer_pattern=pattern,
            n_loops_distribution=distribution,
        ).cuda()

    yield build
    Utils.destroy_model_parallel()


def inputs():
    return torch.tensor([[1, 2, 3, 4]], device="cuda"), torch.arange(4, device="cuda").unsqueeze(0)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:G-:-", "-:*:-*:-"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_model_matches_independent_unroll(models, pattern, dtype):
    model = models(pattern, dtype=dtype)
    # Nonzero controllers exercise token-dependent rates, not just the constant initializer.
    with torch.no_grad():
        for gate in model.decoder.recurrence_gates:
            gate.beta_proj.weight.normal_(std=0.01)
            gate.beta_proj.bias.normal_(std=0.01)
    tokens, positions = inputs()
    flat = pattern.replace(":", "")
    ranges = model.recurrent_layer_ranges
    for loops in (1, 3, 2, 1):
        order = [0]
        for start, end in ranges:
            for _ in range(loops):
                order.extend(range(start, end))
        order.append(len(flat) - 1)
        reference = models(
            "".join(flat[i] for i in order), spec=hybrid_stack_spec, distribution=None, dtype=dtype
        )
        reference.embedding.load_state_dict(model.embedding.state_dict())
        reference.decoder.final_norm.load_state_dict(model.decoder.final_norm.state_dict())
        reference.output_layer.load_state_dict(model.output_layer.state_dict())
        for layer, original in zip(reference.decoder.layers, order, strict=True):
            layer.load_state_dict(model.decoder.layers[original].state_dict())
        gate_copies = [
            AdaptiveDeltaGate(model.config, model.pg_collection).cuda()
            for _ in range(loops * len(ranges))
        ]
        for index, gate in enumerate(gate_copies):
            gate.load_state_dict(model.decoder.recurrence_gates[index // loops].state_dict())
        model.zero_grad(set_to_none=True)
        actual = model(tokens, positions, None, n_loops=loops)
        hidden = reference.embedding(tokens, positions)
        hidden, _ = reference.decoder.layers[0](hidden, attention_mask=None)
        cursor = 1
        for region, (start, end) in enumerate(ranges):
            region_input = hidden
            for loop in range(loops):
                previous = hidden
                entry = previous + region_input
                hidden = entry.clone()
                for _ in range(end - start):
                    hidden, _ = reference.decoder.layers[cursor](hidden, attention_mask=None)
                    cursor += 1
                hidden = gate_copies[region * loops + loop](previous, hidden, entry)
        hidden, _ = reference.decoder.layers[cursor](hidden, attention_mask=None)
        expected, _ = reference.output_layer(reference.decoder.final_norm(hidden))
        expected = expected.transpose(0, 1).contiguous()
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for index, original in enumerate(model.decoder.layers):
            copies = [
                dict(layer.named_parameters())
                for i, layer in enumerate(reference.decoder.layers)
                if order[i] == index
            ]
            for name, parameter in original.named_parameters():
                assert parameter.grad is not None
                total = torch.zeros_like(parameter.grad)
                for copy_parameters in reversed(copies):
                    total.add_(copy_parameters[name].grad)
                torch.testing.assert_close(parameter.grad, total)
        for region, gate in enumerate(model.decoder.recurrence_gates):
            copies = [
                dict(copy.named_parameters())
                for copy in gate_copies[region * loops : (region + 1) * loops]
            ]
            for name, parameter in gate.named_parameters():
                assert parameter.grad is not None
                total = torch.zeros_like(parameter.grad)
                for copy_parameters in reversed(copies):
                    total.add_(copy_parameters[name].grad)
                torch.testing.assert_close(parameter.grad, total)
        for original, replica in (
            (model.embedding, reference.embedding),
            (model.decoder.final_norm, reference.decoder.final_norm),
            (model.output_layer, reference.output_layer),
        ):
            for parameter, other in zip(original.parameters(), replica.parameters(), strict=True):
                assert parameter.grad is not None and other.grad is not None
                torch.testing.assert_close(parameter.grad, other.grad)


def test_gate_parameters_are_kept_in_fp32_by_bf16_wrapper(models):
    model = models(dtype=torch.bfloat16)
    wrapped = Float16Module(model.config, model)
    assert wrapped.module.decoder.recurrence_gates[0].base_alpha_raw.dtype == torch.float32
    assert wrapped.module.decoder.recurrence_gates[0].beta_proj.weight.dtype == torch.bfloat16


def test_regions_have_independent_parameters_and_mtp_is_ungated(models):
    model = models("-:*:-*:-/-")
    gates = model.decoder.recurrence_gates
    assert len(gates) == 2
    assert gates[0].beta_proj.weight.data_ptr() != gates[1].beta_proj.weight.data_ptr()
    assert not any(isinstance(module, AdaptiveDeltaGate) for module in model.mtp.modules())


@pytest.mark.parametrize("setting", ["wide_residual", "enable_mhc_connections"])
def test_gates_reject_expanded_residual_width(models, setting):
    with pytest.raises(ValueError, match="ordinary residual width"):
        models(**{setting: object() if setting == "wide_residual" else True})


def test_gated_spec_requires_boundaries(models):
    with pytest.raises(ValueError, match="requires ':'"):
        models("-*--", distribution=None)


def test_ordinary_checkpoint_import_is_explicit_and_strict(models, tmp_path):
    source = models("-*--", spec=hybrid_stack_spec, distribution=None)
    path = tmp_path / f"ungated_{Utils.rank}.pt"
    torch.save(source.state_dict(), path)
    target = models(distribution="1:1")
    source_state = torch.load(path, weights_only=False)
    with pytest.raises(RuntimeError, match="recurrence_gates"):
        target.load_state_dict(source_state, strict=True)
    initialize_recurrence_gate_state(target, source_state)
    target.load_state_dict(source_state, strict=True)
    for name, value in source.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, target.state_dict()[name])
    assert not target.decoder.recurrence_gates[0].beta_proj.weight.count_nonzero()
    with pytest.raises(ValueError, match="already contains recurrence gates"):
        initialize_recurrence_gate_state(target, source_state)
    del source_state["output_layer.weight"]
    with pytest.raises(RuntimeError, match="output_layer.weight"):
        target.load_state_dict(source_state, strict=True)
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    tokens, positions = inputs()
    target(tokens, positions, None).square().mean().backward()
    optimizer.step()
    assert any(
        parameter.count_nonzero() for parameter in target.decoder.recurrence_gates[0].parameters()
    )
    gated_path = tmp_path / f"gated_{Utils.rank}.pt"
    torch.save({"model": target.state_dict(), "optimizer": optimizer.state_dict()}, gated_path)
    loaded = torch.load(gated_path, map_location="cpu", weights_only=False)
    resumed = models(distribution="1:1")
    resumed.load_state_dict(loaded["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    for original, restored in zip(target.parameters(), resumed.parameters(), strict=True):
        for name, value in optimizer.state[original].items():
            torch.testing.assert_close(value, resumed_optimizer.state[restored][name])
    for root, opt in ((target, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        root(tokens, positions, None, n_loops=3).square().mean().backward()
        opt.step()
    for original, restored in zip(target.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(original, restored)


@pytest.mark.parametrize("pattern", ["-:*-:-", "-:*:-*:-"])
def test_distributed_ungated_import_and_gated_resume(models, tmp_path_dist_ckpt, pattern):
    source = models(pattern.replace(":", ""), spec=hybrid_stack_spec, distribution=None)
    target = models(pattern, distribution="1:1")
    path = tmp_path_dist_ckpt / f"gate_import_{len(source.decoder.layers)}"
    path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": source.sharded_state_dict()}, path)
    validate_ungated_checkpoint_keys(dist_checkpointing.load_tensors_metadata(path))
    template = target.sharded_state_dict(
        metadata={"dp_cp_group": target.pg_collection.dp_cp, "initialize_recurrence_gates": True}
    )
    loaded = dist_checkpointing.load({"model": template}, path, strict="raise_all")
    target.load_state_dict(loaded["model"], strict=True)
    for name, value in source.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, target.state_dict()[name])
    with torch.no_grad():
        for parameter in target.decoder.recurrence_gates.parameters():
            parameter.fill_(0.2)
    resumed_path = tmp_path_dist_ckpt / f"gated_resume_{len(source.decoder.layers)}"
    resumed_path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save({"model": target.sharded_state_dict()}, resumed_path)
    restored = models(pattern, distribution="1:1")
    with pytest.raises(ValueError, match="already contains recurrence gates"):
        validate_ungated_checkpoint_keys(dist_checkpointing.load_tensors_metadata(resumed_path))
    restored.load_state_dict(
        dist_checkpointing.load(
            {"model": restored.sharded_state_dict()}, resumed_path, strict="raise_all"
        )["model"],
        strict=True,
    )
    for a, b in zip(target.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_import_rejects_loading_an_old_optimizer(models):
    from megatron.training.checkpointing import _validate_recurrence_gate_import

    target = models()
    args = SimpleNamespace(init_recurrence_gates=True)
    with pytest.raises(ValueError, match="optimizer parameter layout"):
        _validate_recurrence_gate_import([target], args, "torch_dist", True)
    with pytest.raises(ValueError, match="gated model spec"):
        source = models("-*--", spec=hybrid_stack_spec, distribution=None)
        _validate_recurrence_gate_import([source], args, "torch_dist", False)
    args.ckpt_assume_constant_structure = True
    with pytest.raises(ValueError, match="checkpoint structures differ"):
        _validate_recurrence_gate_import([target], args, "torch_dist", False)
    args.ckpt_assume_constant_structure = False
    args.load_main_params_from_ckpt = True
    with pytest.raises(ValueError, match="optimizer main parameters"):
        _validate_recurrence_gate_import([target], args, "torch_dist", False)


@pytest.mark.parametrize("grad_fusion", [False, True])
def test_gated_ddp_accumulation(models, grad_fusion):
    model = models("-:*:-*:-", grad_fusion=grad_fusion)
    reference = models("-:*:-*:-")
    reference.load_state_dict(model.state_dict())
    ddp = DistributedDataParallel(
        config=model.config,
        ddp_config=DistributedDataParallelConfig(overlap_grad_reduce=True),
        module=model,
        pg_collection=model.pg_collection,
    )
    tokens, positions = inputs()
    ddp.zero_grad_buffer()
    with ddp.no_sync():
        ddp(tokens, positions, None, n_loops=1).square().mean().backward()
    ddp(tokens, positions, None, n_loops=3).square().mean().backward()
    ddp.finish_grad_sync()
    reference(tokens, positions, None, n_loops=1).square().mean().backward()
    reference(tokens, positions, None, n_loops=3).square().mean().backward()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.main_grad, expected.grad)


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dist"])
def test_training_checkpoint_loader_imports_and_resumes_gates(
    models, tmp_path_dist_ckpt, monkeypatch, run_config, ckpt_format
):
    from megatron.training import checkpointing
    from megatron.training.argument_utils import checkpoint_config_from_args
    from megatron.training.arguments import parse_args, validate_args

    path = tmp_path_dist_ckpt / f"native_gate_{ckpt_format}"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pretrain_hybrid.py",
            "--hybrid-layer-pattern=-:*-:-",
            "--spec",
            "megatron.core.models.hybrid.hybrid_layer_specs",
            "adaptive_delta_hybrid_stack_spec",
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
            "--position-embedding-type",
            "none",
            "--untie-embeddings-and-output-weights",
            "--load",
            str(path),
            "--ckpt-format",
            ckpt_format,
            "--finetune",
            "--init-recurrence-gates",
            "--no-load-rng",
        ],
    )
    args = validate_args(parse_args())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "has_nvidia_modelopt", False)
    # Legacy Megatron payloads deliberately contain a Namespace and enum-valued arguments.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    run_config.tokenizer.padded_vocab_size = 32
    source = models("-*--", spec=hybrid_stack_spec, distribution=None)
    target = models(distribution="1:1")
    source_args = copy.copy(args)
    source_args.hybrid_layer_pattern = "-*--"
    source_args.spec = ["megatron.core.models.hybrid.hybrid_layer_specs", "hybrid_stack_spec"]
    source_args.init_recurrence_gates = False

    def save(model, root, saved_args):
        directory = root / "iter_0000007"
        if ckpt_format == "torch_dist":
            directory.mkdir(parents=True, exist_ok=True)
            dist_checkpointing.save(
                {
                    "model": model.sharded_state_dict(),
                    "args": saved_args,
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
                    "args": saved_args,
                    "iteration": 7,
                    "checkpoint_version": 3.1,
                },
                directory / "mp_rank_00" / "model_optim_rng.pt",
            )
        torch.distributed.barrier()
        if Utils.rank == 0:
            (root / "latest_checkpointed_iteration.txt").write_text("7")
        torch.distributed.barrier()

    save(source, path, source_args)
    checkpointing.load_args_from_checkpoint(args)
    run_config.checkpoint = checkpoint_config_from_args(args)
    assert args.spec[-1] == "adaptive_delta_hybrid_stack_spec"
    groups = dict(
        tp_group=target.pg_collection.tp,
        pp_group=target.pg_collection.pp,
        dp_cp_group=target.pg_collection.dp_cp,
        dp_group=target.pg_collection.dp,
    )
    checkpointing.load_checkpoint([target], None, None, **groups)
    for name, value in source.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, target.state_dict()[name])
    assert not target.decoder.recurrence_gates[0].base_alpha_raw.count_nonzero()
    with torch.no_grad():
        for parameter in target.decoder.recurrence_gates.parameters():
            parameter.fill_(0.2)
    resumed_path = tmp_path_dist_ckpt / f"native_gated_resume_{ckpt_format}"
    gated_args = copy.copy(args)
    gated_args.init_recurrence_gates = False
    save(target, resumed_path, gated_args)
    restored = models(distribution="1:1")
    run_config.checkpoint.load = str(resumed_path)
    args.init_recurrence_gates = False
    checkpointing.load_checkpoint([restored], None, None, **groups)
    for a, b in zip(target.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="select a gated model spec"):
        checkpointing.load_checkpoint([source], None, None, **groups)
    wrong_regions = models("-*:--:", distribution="1:1")
    with pytest.raises(ValueError, match="recurrence boundaries differ"):
        checkpointing.load_checkpoint([wrong_regions], None, None, **groups)
    args.init_recurrence_gates = True
    with pytest.raises(ValueError, match="already contains recurrence gates"):
        checkpointing.load_checkpoint([restored], None, None, **groups)


def test_inference_builders_keep_the_gated_specs():
    from megatron.training.argument_utils import _hybrid_inference_stack_spec

    module = "megatron.core.models.hybrid.hybrid_layer_specs"
    assert (
        _hybrid_inference_stack_spec([module, "adaptive_delta_hybrid_stack_spec"], False)
        is adaptive_delta_hybrid_inference_stack_spec
    )
    assert (
        _hybrid_inference_stack_spec(
            [module, "adaptive_delta_gated_delta_product_stack_spec"], False
        )
        is adaptive_delta_gated_delta_product_inference_stack_spec
    )
