# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small real-model parity checks for the module boundary, including backward."""

import pytest
import torch

from megatron.core.context_parallel import (
    ContextParallelBatch,
    convert_cp_layout,
    get_batches_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import wide_residual_hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.test_utilities import Utils


def _capture_projection(module, values):
    def forward_hook(_module, inputs, output):
        values["x"] = inputs[0].detach().clone()

        def backward_hook(gradient):
            values["dy"] = gradient.detach().clone()

        output[0].register_hook(backward_hook)

    module.register_forward_hook(forward_hook)


def _check_projection_gradient(reference, candidate, values, cp_group):
    """Check wgrad against independent FP64 products, including local rounding error.

    BF16 local partial sums rounded before CP reduction are partition-dependent.
    Check the mathematical gradient separately from that unavoidable rounding.
    """
    for key in ("x", "dy"):
        aligned = convert_cp_layout(values[0][key], "zigzag", "contiguous", cp_group)
        torch.testing.assert_close(aligned, values[1][key], atol=0, rtol=0)
    exact_grads = []
    for module, data in zip((reference, candidate), values):
        x, dy = (data[key].double().flatten(0, -2) for key in ("x", "dy"))
        exact = dy.T @ x
        grad = module.weight.grad
        # FP32 GEMM accumulation followed by the gradient dtype's rounding.
        accumulation_bound = x.shape[0] * torch.finfo(torch.float32).eps * (dy.abs().T @ x.abs())
        rounding_bound = torch.finfo(grad.dtype).eps * exact.abs()
        error = (grad.double() - exact).abs()
        assert torch.all(error <= accumulation_bound + rounding_bound).item()
        torch.distributed.all_reduce(exact, group=cp_group)
        exact_grads.append(exact)
    torch.testing.assert_close(exact_grads[0], exact_grads[1], atol=1e-10, rtol=1e-10)


@pytest.mark.internal
@pytest.mark.parametrize(
    "tp,cp,packed,pattern",
    [
        (1, 1, False, "*-*"),
        (1, 2, False, "*-*"),
        (2, 2, True, "*-*"),
        (2, 2, True, "contiguous-attention"),
        (1, 2, False, "G-*"),
        (1, 2, False, "M-*"),
        (1, 2, False, "gpt"),
        (1, 2, False, "chunkwise"),
        (1, 2, False, "mhc"),
        (1, 2, False, "wide"),
        (1, 2, False, "hash"),
    ],
)
def test_model_residual_layout_forward_backward(tp, cp, packed, pattern):
    if not torch.cuda.is_available() or Utils.world_size < cp * tp:
        pytest.skip("Requires at least TP x CP=2 GPU ranks")
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        # mHC mappings and hash tables also initialize on CPU. CP replicas must
        # start from the same weights before comparing different token ownership.
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config_kwargs = dict(
            num_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            kv_channels=32,
            ffn_hidden_size=256,
            context_parallel_size=cp,
            tensor_model_parallel_size=tp,
            sequence_parallel=tp > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            activation_func=torch.nn.functional.silu,
            mamba_num_groups=4,
        )
        if pattern == "mhc":
            config_kwargs.update(enable_mhc_connections=True, mhc_num_residual_streams=2)
        elif pattern == "hash":
            config_kwargs.update(
                num_moe_experts=4,
                moe_router_topk=1,
                moe_num_hash_layers=1,
                hash_moe_vocab_size=128,
                moe_router_pre_softmax=True,
            )
        elif pattern == "wide":
            config_kwargs.update(wide_residual=WideResidualConfig(num_streams=2))
        if pattern == "contiguous-attention":
            config_kwargs.update(cp_comm_type="all_gather")
        config = TransformerConfig(**config_kwargs, global_cp_layout="zigzag")
        fixed_config = TransformerConfig(
            **config_kwargs,
            global_cp_layout="contiguous",
            attention_cp_layout="contiguous" if pattern == "contiguous-attention" else "auto",
            linear_cp_mode="chunkwise" if pattern == "chunkwise" else "headwise",
            linear_attention_cp_layout="contiguous" if pattern == "M-*" else "auto",
        )

        def build(model_config):
            if pattern == "gpt":
                return GPTModel(
                    model_config,
                    get_gpt_layer_with_transformer_engine_spec(),
                    vocab_size=128,
                    max_sequence_length=128,
                    pg_collection=groups,
                    position_embedding_type="rope",
                ).cuda()
            return HybridModel(
                model_config,
                vocab_size=128,
                max_sequence_length=128,
                hybrid_layer_pattern=(
                    "G-*"
                    if pattern == "chunkwise"
                    else (
                        "*-*"
                        if pattern in ("mhc", "wide", "contiguous-attention")
                        else "*E*" if pattern == "hash" else pattern
                    )
                ),
                pg_collection=groups,
                position_embedding_type="none" if pattern == "contiguous-attention" else "rope",
                **(
                    {"hybrid_stack_spec": wide_residual_hybrid_stack_spec}
                    if pattern == "wide"
                    else {}
                ),
            ).cuda()

        reference, fixed = build(config), build(fixed_config)
        fixed.load_state_dict(reference.state_dict())
        if pattern == "hash":
            assert reference.decoder.layers[1].mlp.router.is_hash_layer
            assert fixed.decoder.layers[1].mlp.router.is_hash_layer
        projection_values = [{}, {}]
        if pattern == "M-*":
            for model, values in zip((reference, fixed), projection_values):
                _capture_projection(model.decoder.layers[0].mixer.out_proj, values)
        # Fixed input values make ownership independent of per-rank RNG state.
        global_hidden = torch.sin(torch.arange(32 * 128, device="cuda").float() / 31).to(
            torch.bfloat16
        )
        batch = {
            "tokens": torch.arange(32, device="cuda").view(1, -1),
            "position_ids": torch.arange(32, device="cuda").view(1, -1),
            "decoder_input": global_hidden.view(1, 32, 128),
        }
        if packed:
            batch.update(
                cu_seqlens=torch.tensor([[0, 0, 7, 17, 32]], dtype=torch.int32, device="cuda"),
                max_seqlen=torch.tensor([15], dtype=torch.int32, device="cuda"),
            )
        views = get_batches_on_this_cp_rank(
            batch,
            boundary_layout="contiguous",
            dynamic_cp=False,
            cp_group=groups.cp,
            additional_layouts=(
                ("zigzag",)
                if pattern == "gpt"
                else reference.cp_required_layouts | fixed.cp_required_layouts
            ),
            sequence_parallel=tp > 1,
            tp_group=groups.tp,
            tp_cp_group=groups.tp_cp,
        ).prepare_module_layouts()
        inputs, outputs = {}, {}
        for mode, model in (("zigzag", reference), ("contiguous", fixed)):
            data = views.get_batch(mode)
            x = data["decoder_input"].transpose(0, 1).contiguous()
            if tp > 1:
                x = x.chunk(tp)[groups.tp.rank()].contiguous()
            x = x.detach().requires_grad_()
            inputs[mode] = x
            model_views = ContextParallelBatch(
                mode, views.batches_by_layout, views.packed_seq_params_by_layout, views.thd_plan
            )
            kwargs = (
                {}
                if pattern == "gpt"
                else dict(cp_batch=model_views, output_processor=lambda **kw: kw["hidden_states"])
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs[mode] = model(
                    input_ids=data["tokens"],
                    position_ids=data["position_ids"],
                    attention_mask=None,
                    decoder_input=x,
                    packed_seq_params=model_views.get_packed_seq_params(),
                    **kwargs,
                )
            if pattern == "gpt":
                outputs[mode] = outputs[mode].transpose(0, 1).contiguous()
        expected = convert_cp_layout(
            outputs["zigzag"],
            "zigzag",
            "contiguous",
            groups.cp,
            tp > 1,
            groups.tp,
            groups.tp_cp,
            views.thd_plan,
        )
        torch.testing.assert_close(outputs["contiguous"], expected, atol=0.03, rtol=0.03)
        # Apply the same nonuniform objective after mapping outputs to one physical
        # layout, then compare input and globally reduced weight gradients.
        weight = 1 + 0.2 * torch.cos(torch.arange(expected.numel(), device="cuda").float()).view(
            expected.shape
        )
        (expected.float().square() * weight).sum().backward()
        (outputs["contiguous"].float().square() * weight).sum().backward()
        expected_grad = convert_cp_layout(
            inputs["zigzag"].grad,
            "zigzag",
            "contiguous",
            groups.cp,
            tp > 1,
            groups.tp,
            groups.tp_cp,
            views.thd_plan,
        )
        torch.testing.assert_close(inputs["contiguous"].grad, expected_grad, atol=0.03, rtol=0.03)
        if pattern == "M-*":
            _check_projection_gradient(
                reference.decoder.layers[0].mixer.out_proj,
                fixed.decoder.layers[0].mixer.out_proj,
                projection_values,
                groups.cp,
            )
        for (name, param), (_, other) in zip(
            reference.named_parameters(), fixed.named_parameters()
        ):
            if pattern == "M-*" and name == "decoder.layers.0.mixer.out_proj.weight":
                # Already checked against the independent reference and local rounding bound.
                continue
            if param.grad is None:
                assert other.grad is None, name
                continue
            # CP owns different tokens in the two policies; compare the same global weight grad.
            grad, other_grad = param.grad.float(), other.grad.float()
            torch.distributed.all_reduce(grad, group=groups.cp)
            torch.distributed.all_reduce(other_grad, group=groups.cp)
            if tp > 1 and getattr(param, "sequence_parallel", False):
                torch.distributed.all_reduce(grad, group=groups.tp)
                torch.distributed.all_reduce(other_grad, group=groups.tp)
            if not torch.allclose(other_grad, grad, atol=0.05, rtol=0.03):
                print(
                    name,
                    "grad_dtype",
                    param.grad.dtype,
                    "max_abs",
                    (other_grad - grad).abs().max().item(),
                    "ref_max",
                    grad.abs().max().item(),
                    flush=True,
                )
            torch.testing.assert_close(
                other_grad, grad, atol=0.05, rtol=0.03, msg=lambda message: name + "\n" + message
            )
    finally:
        Utils.destroy_model_parallel()
