# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical parity of activation recomputation against the no-recompute baseline.

Every recompute path that routes through ``precision_aware_checkpoint`` -- full-layer
recompute, selective MLP / MoE / shared-expert recompute, and MTP recompute -- must
reproduce the baseline loss and every parameter gradient under BF16, MXFP8, and NVFP4.
"""

import gc
import os
import sys
import traceback

import pytest
import torch

from megatron.core import recompute as recompute_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer import transformer_layer as transformer_layer_module
from megatron.core.transformer import utils as transformer_utils
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support, check_nvfp4_support

    _FP8_AVAILABLE, _NO_FP8_REASON = check_fp8_support()
    _NVFP4_AVAILABLE, _NO_NVFP4_REASON = check_nvfp4_support()
except ImportError:
    _FP8_AVAILABLE = False
    _NO_FP8_REASON = "Transformer Engine FP8 support is unavailable"
    _NVFP4_AVAILABLE = False
    _NO_NVFP4_REASON = "Transformer Engine NVFP4 support is unavailable"


_SEED = 1234
_TP = 2
_ATOL = 1e-4
_BLACKWELL_AVAILABLE = torch.cuda.is_available() and get_device_arch_version() >= 10

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]

_FULL = {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1}

# case id -> (model kind, recompute args, modules whose precision_aware_checkpoint must fire)
_RECOMPUTE_CASES = {
    "full": ("dense", _FULL, {"recompute"}),
    "full_distribute_saved_activations": (
        # distribute_saved_activations shards the saved input across TP, which requires the
        # input to be replicated across TP ranks, i.e. no sequence parallelism.
        "dense_no_sp",
        {**_FULL, "distribute_saved_activations": True},
        {"recompute"},
    ),
    "full_with_mtp": ("dense_mtp", _FULL, {"recompute", "multi_token_prediction"}),
    "selective_mlp": (
        "dense",
        {"recompute_granularity": "selective", "recompute_modules": ["mlp"]},
        {"transformer_layer"},
    ),
    "selective_moe": (
        "moe",
        {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
        {"moe_layer"},
    ),
    "selective_shared_experts": (
        "moe_shared_experts",
        {"recompute_granularity": "selective", "recompute_modules": ["shared_experts"]},
        {"moe_layer"},
    ),
}

_CALL_SITE_MODULES = {
    "recompute": recompute_module,
    "transformer_layer": transformer_layer_module,
    "moe_layer": moe_layer_module,
    "multi_token_prediction": mtp_module,
}


def _skip_if_unsupported(precision: str) -> None:
    if Utils.world_size < _TP or Utils.world_size % _TP != 0:
        pytest.skip(f"requires a world size divisible by TP={_TP}, got {Utils.world_size}")
    if precision in ("mxfp8", "nvfp4") and not _BLACKWELL_AVAILABLE:
        pytest.skip(f"{precision} recompute parity requires Blackwell (SM >= 10)")
    if precision == "mxfp8" and not _FP8_AVAILABLE:
        pytest.skip(_NO_FP8_REASON)
    if precision == "nvfp4" and not _NVFP4_AVAILABLE:
        pytest.skip(_NO_NVFP4_REASON)


class TestActivationRecomputeNumerics:
    """Recompute must not change the loss or any gradient, for every checkpointed path."""

    seq_length = 128
    micro_batch_size = 2

    def setup_method(self, method):
        self._old_env = {
            key: os.environ.get(key)
            for key in ("CUDA_DEVICE_MAX_CONNECTIONS", "NVTE_ALLOW_NONDETERMINISTIC_ALGO")
        }
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    def teardown_method(self, method):
        try:
            self._cleanup()
        finally:
            for key, value in self._old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _cleanup(self):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_test_args(self, precision: str, model_kind: str, recompute_args: dict):
        self._cleanup()

        sys.argv = ["test_activation_recompute_numerics.py"]
        args = parse_args()
        args.num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 256
        args.ffn_hidden_size = 512
        args.num_attention_heads = 8
        args.max_position_embeddings = self.seq_length
        args.seq_length = self.seq_length
        args.micro_batch_size = self.micro_batch_size
        args.global_batch_size = self.micro_batch_size * (Utils.world_size // _TP)
        args.create_attention_mask_in_dataloader = True
        args.tensor_model_parallel_size = _TP
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.sequence_parallel = model_kind != "dense_no_sp"
        args.bf16 = True
        args.attention_backend = "unfused"
        args.add_bias_linear = False
        # Dropout on: recompute must replay the exact forward RNG draws (dropout masks),
        # which is what the checkpoint primitives' RNG-state restoration guarantees.
        args.hidden_dropout = 0.1
        args.attention_dropout = 0.1
        args.swiglu = True
        # MTP supports only RoPE / no position embeddings.
        args.position_embedding_type = "rope"
        # No DDP here: let Transformer Engine write .grad instead of a fused main_grad.
        args.gradient_accumulation_fusion = False
        args.save_tokenizer_assets = False

        if model_kind in ("moe", "moe_shared_experts"):
            args.num_experts = 4
            args.moe_layer_freq = 1
            args.moe_ffn_hidden_size = 256
            args.moe_grouped_gemm = True
            args.moe_token_dispatcher_type = "alltoall"
            args.moe_router_topk = 2
            args.moe_router_load_balancing_type = "none"
            args.moe_aux_loss_coeff = 0.0
            args.moe_router_padding_for_quantization = precision != "bf16"
            if model_kind == "moe_shared_experts":
                args.moe_shared_expert_intermediate_size = 256
                args.moe_shared_expert_overlap = False
        if model_kind == "dense_mtp":
            args.mtp_num_layers = 1

        for key, value in recompute_args.items():
            setattr(args, key, value)

        if precision == "mxfp8":
            args.fp8 = "e4m3"
            args.fp8_recipe = "mxfp8"
        elif precision == "nvfp4":
            args.fp4 = "e2m1"
            args.fp4_recipe = "nvfp4"
        elif precision != "bf16":
            raise ValueError(f"Unknown precision test case: {precision}")

        validate_args(args)
        set_global_variables(args, pretrain_cfg_container_from_args(args), build_tokenizer=False)
        return args

    @staticmethod
    def build_model(args):
        config = core_transformer_config_from_args(args)
        model_parallel_cuda_manual_seed(_SEED)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
        )
        mtp_block_spec = None
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config=config, spec=layer_spec, use_transformer_engine=True
            )
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            mtp_block_spec=mtp_block_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
        return model.cuda()

    def get_batch(self):
        data = torch.arange(self.seq_length, dtype=torch.int64, device="cuda")
        input_ids = (data * 7 % 1024).repeat((self.micro_batch_size, 1))
        labels = ((data + 1) * 7 % 1024).repeat((self.micro_batch_size, 1))
        position_ids = data.repeat((self.micro_batch_size, 1))
        attention_mask = torch.ones(
            (self.micro_batch_size, 1, self.seq_length, self.seq_length), dtype=bool, device="cuda"
        )
        loss_mask = torch.ones(
            (self.micro_batch_size, self.seq_length), dtype=torch.float32, device="cuda"
        )
        return input_ids, labels, position_ids, attention_mask, loss_mask

    @staticmethod
    def forward_backward(model, batch):
        model.train()
        model.zero_grad(set_to_none=True)
        # Identical RNG streams for both runs (NVFP4 stochastic rounding draws from them).
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        input_ids, labels, position_ids, attention_mask, loss_mask = batch
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        loss = output.float().mean()
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"{name} received no gradient"
            grads[name] = param.grad.detach().float().clone()
        return loss.detach(), grads

    @staticmethod
    def spy_on_checkpoint(monkeypatch):
        """Record which call sites reach precision_aware_checkpoint, and with what, plus
        which checkpoint primitive each call dispatched to.

        Parity alone cannot tell "recomputed correctly" from "never checkpointed": both
        match the baseline. The primitive record proves every call really checkpointed.
        """
        calls = []
        primitives = []
        real_checkpoint = transformer_utils.precision_aware_checkpoint

        for site, module in _CALL_SITE_MODULES.items():

            def spy(function, config, tp_group, /, *args, _site=site, **kwargs):
                calls.append(
                    {
                        "site": _site,
                        "tp_group": tp_group,
                        "distribute_saved_activations": kwargs.get(
                            "distribute_saved_activations", False
                        ),
                    }
                )
                return real_checkpoint(function, config, tp_group, *args, **kwargs)

            monkeypatch.setattr(module, "precision_aware_checkpoint", spy)

        real_tp_checkpoint = transformer_utils.tensor_parallel.checkpoint

        def tp_checkpoint_spy(*args, **kwargs):
            # args: (function, distribute_saved_activations, *function_args)
            primitives.append(("tensor_parallel.checkpoint", args[1]))
            return real_tp_checkpoint(*args, **kwargs)

        monkeypatch.setattr(transformer_utils.tensor_parallel, "checkpoint", tp_checkpoint_spy)

        from megatron.core.extensions import transformer_engine as te_extensions

        real_te_checkpoint = te_extensions.te_checkpoint

        def te_checkpoint_spy(*args, **kwargs):
            # args: (function, distribute_saved_activations, rng_tracker, tp_group, ...)
            primitives.append(("te_checkpoint", args[1]))
            return real_te_checkpoint(*args, **kwargs)

        # precision_aware_checkpoint imports te_checkpoint lazily, so patching the module
        # attribute intercepts it.
        monkeypatch.setattr(te_extensions, "te_checkpoint", te_checkpoint_spy)
        return calls, primitives

    @staticmethod
    def assert_all_ranks_passed(local_passed: bool, local_error: str) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if not local_passed:
                pytest.fail(local_error)
            return

        pass_flag = torch.tensor(
            [1 if local_passed else 0], dtype=torch.int32, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(pass_flag, op=torch.distributed.ReduceOp.MIN)
        if pass_flag.item() == 1:
            return

        rank = torch.distributed.get_rank()
        if local_passed:
            pytest.fail("At least one distributed rank failed this recompute parity case.")
        pytest.fail(f"Rank {rank} failed this recompute parity case:\n{local_error}")

    def run_parity_case(self, precision: str, case: str, monkeypatch) -> None:
        model_kind, recompute_args, expected_sites = _RECOMPUTE_CASES[case]
        local_passed = True
        local_error = ""
        try:
            baseline_args = self.create_test_args(precision, model_kind, recompute_args={})
            Utils.initialize_model_parallel(tensor_model_parallel_size=_TP)
            batch = self.get_batch()
            calls, primitives = self.spy_on_checkpoint(monkeypatch)

            baseline_model = self.build_model(baseline_args)
            baseline_loss, baseline_grads = self.forward_backward(baseline_model, batch)
            assert (
                not calls and not primitives
            ), f"baseline unexpectedly checkpointed: {calls} {primitives}"
            baseline_state = baseline_model.state_dict()
            del baseline_model

            # Same process groups; only the recompute configuration changes.
            recompute_args_ns = self.create_test_args(precision, model_kind, recompute_args)
            Utils.initialize_model_parallel(tensor_model_parallel_size=_TP)
            recompute_model = self.build_model(recompute_args_ns)
            recompute_model.load_state_dict(baseline_state)
            recompute_loss, recompute_grads = self.forward_backward(recompute_model, batch)

            fired_sites = {call["site"] for call in calls}
            assert (
                fired_sites == expected_sites
            ), f"expected recompute through {sorted(expected_sites)}, got {sorted(fired_sites)}"
            want_distributed = recompute_args.get("distribute_saved_activations", False)
            assert all(
                call["distribute_saved_activations"] == want_distributed for call in calls
            ), f"unexpected distribute_saved_activations: {calls}"
            assert all(call["tp_group"] is not None for call in calls), "missing TP group"
            # FP8 and FP4 recompute must go through TE's checkpoint; everything else
            # through tensor_parallel.checkpoint -- one primitive call per checkpoint.
            expected_primitive = (
                "te_checkpoint" if precision in ("mxfp8", "nvfp4") else "tensor_parallel.checkpoint"
            )
            # ...and must forward distribute_saved_activations unchanged.
            expected_primitives = [(expected_primitive, want_distributed)] * len(calls)
            assert (
                primitives == expected_primitives
            ), f"expected {expected_primitives}, got {primitives}"

            torch.testing.assert_close(recompute_loss, baseline_loss, atol=_ATOL, rtol=0)
            assert recompute_grads.keys() == baseline_grads.keys()
            for name, baseline_grad in baseline_grads.items():
                torch.testing.assert_close(
                    recompute_grads[name],
                    baseline_grad,
                    atol=_ATOL,
                    rtol=0,
                    msg=lambda default, name=name: f"gradient mismatch for {name}: {default}",
                )
        except Exception:
            local_passed = False
            local_error = traceback.format_exc()

        self.assert_all_ranks_passed(local_passed, local_error)

    @pytest.mark.parametrize("case", list(_RECOMPUTE_CASES))
    @pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
    def test_recompute_matches_no_recompute(self, precision, case, monkeypatch):
        """Loss and every gradient with recompute must match the no-recompute baseline."""
        _skip_if_unsupported(precision)
        self.run_parity_case(precision, case, monkeypatch)
