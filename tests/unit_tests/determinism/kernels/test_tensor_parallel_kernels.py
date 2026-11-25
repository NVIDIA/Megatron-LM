# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the local (non-TE) tensor-parallel layers and the apex CUDA extensions.

* ``VocabParallelEmbedding``: the deterministic branch (``weight[ids]``) and the default
  ``F.embedding`` path, with heavily duplicated ids so the backward accumulates thousands of
  rows into the same embedding row; with ``gradient_accumulation_fusion`` the weight gradient
  goes straight into ``main_grad`` (``fused_embedding_wgrad``).
* ``vocab_parallel_cross_entropy``: the unfused path deterministic mode relies on.
* ``ColumnParallelLinear`` / ``RowParallelLinear``: cuBLAS GEMMs under the pinned workspace,
  with and without apex ``fused_weight_gradient_mlp_cuda`` gradient-accumulation fusion.
* ``ColumnParallelLinear`` with BF16 inputs and FP32 output: Transformer Engine
  ``general_gemm`` forward plus the corresponding input- and weight-gradient paths.
* apex ``FusedLayerNorm`` (persistent and non-persistent) and ``FusedScaleMaskSoftmax``
  (causal and padding CUDA kernels).
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.fusions.fused_embedding_wgrad import accumulate_embedding_wgrad
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.ops.loss.cross_entropy import (
    VocabParallelCrossEntropy,
    vocab_parallel_cross_entropy,
)
from megatron.core.ops.norm import fused_layer_norm
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import init_method_normal
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HAVE_WGRAD_ACCUM_EXT = importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None
HAVE_SCALED_SOFTMAX_EXT = importlib.util.find_spec("scaled_masked_softmax_cuda") is not None and (
    importlib.util.find_spec("scaled_upper_triang_masked_softmax_cuda") is not None
)


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestTensorParallelLayers:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap", [False, True])
    def test_packed_gtp_prefetch_and_deferred_gradients(self, monkeypatch, overlap):
        """Three packed owners reuse buffers without stale gathers or premature grad-ready."""
        from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

        from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
        from megatron.core.tensor_parallel.gtp_grouped_weight import attach_packed_gtp_weight

        group = torch.distributed.group.WORLD
        if group.size() != 4:
            pytest.skip("Requires four ranks")
        monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", overlap)
        monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", overlap)
        monkeypatch.setattr(gtp.GTP_CONFIG, "check_param_states", True)
        monkeypatch.setattr(gtp.GTP_CONFIG, "calculate_per_token_loss", False)
        monkeypatch.setattr(gtp.GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", True)
        model = torch.nn.ModuleList()
        notifications = [0, 0, 0]
        async_gathers = []
        original_gather = torch.distributed.all_gather_into_tensor

        def gather(output, shard, **kwargs):
            if kwargs.get("async_op"):
                expected = gtp.get_ag_stream(model[0].weight.chain_id, group)
                assert torch.cuda.current_stream() == expected
                async_gathers.append(True)
            return original_gather(output, shard, **kwargs)

        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
        try:
            for index in range(3):
                module = torch.nn.Module()
                module.weight = torch.nn.Parameter(
                    GroupedTensor.make_grouped_tensor_from_rowwise_data(
                        num_tensors=2,
                        tensor_shape=(2, 8),
                        rowwise_data=torch.zeros(32, device="cuda", dtype=torch.bfloat16),
                        dtype=torch.bfloat16,
                    )
                )
                attach_packed_gtp_weight(module, group, 3, None)
                module.weight.main_grad = torch.ones(
                    module.weight.shape, device="cuda", dtype=torch.bfloat16
                )

                def ready(index=index):
                    notifications[index] += 1

                module.weight.register_grad_accum_hook(None, ready)
                model.append(module)
            gtp.register_prefetch_chain("packed-replay", [m.weight for m in model])
            gtp.classify_gtp_remat_chains(model)
            # A non-default caller also exercises the producer-to-AG stream edge.
            caller = torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            for iteration in range(3):
                expected_weights = []
                observed = []
                with torch.cuda.stream(caller):
                    for index, module in enumerate(model):
                        full = torch.arange(80, device="cuda", dtype=torch.bfloat16).view(2, 5, 8)
                        full.add_(8 * index + 32 * iteration)
                        expected_weights.append(full)
                        padded = F.pad(full, (0, 0, 0, 3))
                        module.weight.rowwise_data.copy_(
                            padded[:, 2 * group.rank() : 2 * (group.rank() + 1)].reshape(-1)
                        )
                        module.weight.main_grad.fill_(1)
                    for _ in range(2):
                        for module in model:
                            observed.append(
                                module.weight.materialize_group_for_forward().rowwise_data.clone()
                            )
                        for index in reversed(range(3)):
                            weight = model[index].weight
                            # Recompute has its own prefetch tickets; it must not overwrite
                            # a backward gather which the next expert has already issued.
                            with monkeypatch.context() as context:
                                context.setattr(gtp, "in_activation_recompute_phase", lambda: True)
                                observed.append(
                                    weight.materialize_group_for_forward().rowwise_data.clone()
                                )
                            observed.append(
                                weight.materialize_group_for_backward().rowwise_data.clone()
                            )
                            gradient = weight.grad_buffer()
                            gradient.fill_((1024, 1, -1024, 1)[group.rank()] * (index + 1))
                            weight.finalize_group_grads(gradient)
                        gtp.wait_for_gtp_grad_reduction_on_current_stream()
                    caller.synchronize()
                torch.cuda.current_stream().wait_stream(caller)
                expected_order = [0, 1, 2, 2, 2, 1, 1, 0, 0] * 2
                for actual, index in zip(observed, expected_order):
                    torch.testing.assert_close(
                        actual, expected_weights[index].reshape(-1), rtol=0, atol=0
                    )
                for index, module in enumerate(model):
                    expected = torch.ones((2, 8, 8), device="cuda", dtype=torch.bfloat16)
                    expected[:, :5].add_(index + 1)
                    shard = expected[:, 2 * group.rank() : 2 * (group.rank() + 1)]
                    torch.testing.assert_close(module.weight.main_grad, shard, rtol=0, atol=0)
                    assert notifications[index] == 2 * (iteration + 1)
            assert bool(async_gathers) == overlap
        finally:
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()

    @pytest.mark.parametrize("per_token", [False, True])
    @pytest.mark.parametrize("fp32_accumulation", [False, True])
    def test_packed_gtp_padding_and_storage_rebind(self, monkeypatch, per_token, fp32_accumulation):
        """Packed padding, storage rebinding and parent gradient completion stay exact."""
        from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

        from megatron.core.fp8_utils import modify_grouped_tensor_rowwise_storage
        from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
        from megatron.core.tensor_parallel.gtp_api import GTP_CONFIG
        from megatron.core.tensor_parallel.gtp_grouped_weight import attach_packed_gtp_weight

        try:
            group = torch.distributed.group.WORLD
            if group.size() != 4:
                pytest.skip("Requires four ranks")
            monkeypatch.setattr(GTP_CONFIG, "calculate_per_token_loss", per_token)
            monkeypatch.setattr(
                GTP_CONFIG, "reduce_scatter_with_fp32_accumulation", fp32_accumulation
            )
            full = torch.arange(120, device="cuda", dtype=torch.bfloat16).view(3, 5, 8)
            # Two rows per rank, three padding rows per expert. The final rank owns only padding.
            padded = torch.nn.functional.pad(full, (0, 0, 0, 3))
            local = padded[:, group.rank() * 2 : (group.rank() + 1) * 2].contiguous()
            module = torch.nn.Module()
            module.weight = torch.nn.Parameter(
                GroupedTensor.make_grouped_tensor_from_rowwise_data(
                    num_tensors=3,
                    tensor_shape=(2, 8),
                    rowwise_data=local.reshape(-1),
                    dtype=torch.bfloat16,
                )
            )
            attach_packed_gtp_weight(module, group, 3, None)
            weight = module.weight
            actual = weight.materialize_group_for_forward().rowwise_data.view(full.shape)
            torch.testing.assert_close(actual, full, rtol=0, atol=0)

            # DDP replaces backing storage. A later gather and checkpoint must read the new owner.
            replacement = torch.empty_like(local)
            modify_grouped_tensor_rowwise_storage(weight, replacement)
            replacement.add_(1)
            actual = weight.materialize_group_for_backward().rowwise_data.view(full.shape)
            torch.testing.assert_close(actual, full + 1, rtol=0, atol=0)
            for index, view in enumerate(weight.checkpoint_expert_views()):
                torch.testing.assert_close(view, replacement[index], rtol=0, atol=0)
                assert view.is_gtp_weight_remat and view.gtp_remat_size == 4

            weight.main_grad = torch.zeros(weight.shape, device="cuda", dtype=torch.float32)
            notifications = []
            weight.register_grad_accum_hook(None, lambda: notifications.append(True))
            for _ in range(2):
                gradient = torch.ones(full.shape, device="cuda", dtype=torch.float32) * (
                    group.rank() + 1
                )
                weight.finalize_group_grads(gradient)
            # Across four distinct-token peers, sum=10 and mean=2.5 per backward.
            expected = torch.zeros_like(padded, dtype=torch.float32)
            expected[:, :5] = 20 if per_token else 5
            expected = expected[:, group.rank() * 2 : (group.rank() + 1) * 2]
            torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)
            assert len(notifications) == 2
            if fp32_accumulation:
                # Cancellation exposes low-precision intermediate sums. Retain FP32
                # through the final main_grad add, matching discrete GTP semantics.
                weight.main_grad = torch.ones(weight.shape, device="cuda", dtype=torch.bfloat16)
                values = (1024, 1, -1024, 1)
                gradient = torch.full(
                    full.shape, values[group.rank()], device="cuda", dtype=torch.bfloat16
                )
                weight.finalize_group_grads(gradient)
                expected = torch.ones_like(padded)
                expected[:, :5] += 2 if per_token else 0.5
                expected = expected[:, group.rank() * 2 : (group.rank() + 1) * 2]
                torch.testing.assert_close(weight.main_grad, expected, rtol=0, atol=0)
        finally:
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()

    @pytest.mark.parametrize("declare_boundary", [False, True])
    def test_gtp_dynamic_accumulation_replays(self, declare_boundary, monkeypatch):
        """Changing accumulation windows replay both widened GEMMs and reduced grads exactly."""
        from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
        from megatron.core.tensor_parallel.layers import (
            linear_with_grad_accumulation_and_async_allreduce,
        )

        group = torch.distributed.group.WORLD
        if not gtp.HAVE_TE or group.size() < 2:
            pytest.skip("Requires GTP and at least two ranks")
        seeded(1234 + group.rank())
        monkeypatch.setattr(gtp.GTP_CONFIG, "async_reduction", False)
        monkeypatch.setattr(gtp.GTP_CONFIG, "weight_prefetch", False)
        model = torch.nn.Module()
        # Keep every shard nonempty under GTP's 16-row alignment.
        model.output_layer = torch.nn.Linear(
            64, 16 * group.size(), bias=False, device="cuda", dtype=torch.bfloat16
        )
        gtp.wrap_module_params_gtp(model.output_layer, ["weight"], group)
        gtp.classify_gtp_remat_chains(model)
        weight = model.output_layer.weight
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        x = torch.randn(16, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run_windows(inputs):
            results = []
            for count in (2, 4, 1):
                weight.main_grad.zero_()
                weight.grad = None
                for index in range(count):
                    output = linear_with_grad_accumulation_and_async_allreduce(
                        inputs * (index + 1),
                        weight,
                        None,
                        False,
                        False,
                        False,
                        tp_group=group,
                        gtp_remat_size=group.size(),
                    )
                    if declare_boundary and index == count - 1:
                        gtp.finalize_gtp_forward_counts(model)
                    output.sum().backward()
                gtp.wait_for_gtp_grad_reduction_on_current_stream()
                results.append(weight.main_grad.detach().clone())
            return results

        try:
            outputs, _ = assert_replays_bit_exact(
                run_windows, (x,), backward=False, replays=3, what="GTP dynamic accumulation"
            )
            assert all(torch.count_nonzero(value) > 0 for value in outputs.values())
        finally:
            gtp.reset_gtp_state()
            gtp._GTP_PARAMS.clear()

    @pytest.mark.parametrize(
        "deterministic_branch", [True, False], ids=["weight_index", "F.embedding"]
    )
    def test_vocab_parallel_embedding_backward_replays(self, deterministic_branch):
        """64 distinct ids over 32k positions: ~500 duplicate rows accumulate per embedding row."""
        seeded()
        config = _config(deterministic_mode=deterministic_branch)
        module = VocabParallelEmbedding(
            8192, 1024, init_method=init_method_normal(0.02), config=config
        ).cuda()
        ids = torch.randint(0, 64, (8, 4096), device="cuda") * 100
        with deterministic_algorithms(deterministic_branch):
            assert_module_replays_bit_exact(
                module,
                (ids,),
                replays=4,
                what=f"VocabParallelEmbedding[det={deterministic_branch}]",
            )

    def test_vocab_parallel_embedding_wgrad_fusion_replays(self):
        """gradient_accumulation_fusion: 64 distinct ids over 4096 positions accumulate into
        main_grad through the sorted partial sums of ``fused_embedding_wgrad``."""
        seeded()
        config = _config(deterministic_mode=False, gradient_accumulation_fusion=True)
        module = VocabParallelEmbedding(
            8192, 1024, init_method=init_method_normal(0.02), config=config
        ).cuda()
        module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
        module.weight.grad_added_to_main_grad = False
        ids = torch.randint(0, 64, (1, 4096), device="cuda") * 100
        assert_module_replays_bit_exact(
            module,
            (ids,),
            replays=4,
            contention=True,
            what="VocabParallelEmbedding[gradient_accumulation_fusion]",
        )
        assert module.weight.grad_added_to_main_grad

    def test_accumulate_embedding_wgrad_replays(self):
        seeded()
        ids = torch.randint(0, 64, (1, 4096), device="cuda") * 100
        grad = torch.randn(1, 4096, 1024, device="cuda", dtype=torch.bfloat16)
        main_grad = torch.randn(8192, 1024, device="cuda")

        def accumulate(ids, grad, main_grad):
            accumulate_embedding_wgrad(ids, grad, main_grad)
            return main_grad

        assert_replays_bit_exact(
            accumulate,
            (ids, grad, main_grad),
            replays=3,
            backward=False,
            contention=True,
            what="accumulate_embedding_wgrad",
        )

    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_vocab_parallel_cross_entropy_replays(self, label_smoothing):
        seeded()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        logits = torch.randn(4096, 32768, device="cuda", dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 32768 * tp_group.size(), (4096,), device="cuda")
        assert_replays_bit_exact(
            lambda l, t: vocab_parallel_cross_entropy(l, t, label_smoothing, tp_group),
            (logits, target),
            replays=3,
            what="vocab_parallel_cross_entropy",
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_vocab_parallel_cross_entropy_staged_helpers_match_composite(self, dtype):
        """The fused path calls the read-only gather, the in-place exp and the in-place gradient
        scaling as separate steps (each compiled on its own); they must produce exactly the
        bytes of the composite ``calculate_predicted_logits`` / ``calculate_gradients``."""
        seeded()
        vocab = 32768
        logits = torch.randn(4096, vocab, device="cuda", dtype=dtype)
        target = torch.randint(0, vocab, (4096,), device="cuda")
        grad_output = torch.rand(4096, device="cuda", dtype=torch.float32)

        composite, logits_max = VocabParallelCrossEntropy.calculate_logits_max(logits.clone())
        staged, staged_max = VocabParallelCrossEntropy.calculate_logits_max(logits.clone())
        assert torch.equal(logits_max, staged_max)

        target_mask, masked_target_1d, predicted, sum_exp, exp_logits = (
            VocabParallelCrossEntropy.calculate_predicted_logits(
                composite, target, logits_max, 0, vocab
            )
        )
        s_mask, s_masked_target_1d, s_predicted = VocabParallelCrossEntropy.gather_predicted_logits(
            staged, target, staged_max, 0, vocab
        )
        assert torch.equal(staged, logits), "gather_predicted_logits must not modify logits"
        VocabParallelCrossEntropy.exp_logits_inplace(staged, staged_max)
        for name, a, b in (
            ("target_mask", target_mask, s_mask),
            ("masked_target_1d", masked_target_1d, s_masked_target_1d),
            ("predicted_logits", predicted, s_predicted),
            ("exp_logits", exp_logits, staged),
            ("sum_exp_logits", sum_exp, staged.sum(dim=-1)),
        ):
            assert torch.equal(a, b), f"{name} differs between staged and composite forward"

        softmax = exp_logits / sum_exp.unsqueeze(dim=-1)
        grad_2d, arange_1d, softmax_update, grad_input = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax.clone(), target_mask
            )
        )
        composite_grad = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
        )
        grad_2d, arange_1d, softmax_update, grad_input = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax.clone(), target_mask
            )
        )
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        VocabParallelCrossEntropy.scale_gradients_inplace(grad_input, grad_output)
        assert torch.equal(composite_grad, grad_input), "staged backward differs from composite"

    @pytest.mark.parametrize(
        "grad_accum_fusion",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not HAVE_WGRAD_ACCUM_EXT, reason="apex fused_weight_gradient_mlp_cuda missing"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("layer", ["column", "row"])
    def test_local_linear_layers_replay(self, layer, grad_accum_fusion):
        seeded()
        config = _config(gradient_accumulation_fusion=grad_accum_fusion)
        if layer == "column":
            module = ColumnParallelLinear(
                4096, 16384, config=config, init_method=init_method_normal(0.02), bias=True
            ).cuda()
            x = torch.randn(4096, 2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        else:
            module = RowParallelLinear(
                16384,
                4096,
                config=config,
                init_method=init_method_normal(0.02),
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
            ).cuda()
            x = torch.randn(4096, 2, 16384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if grad_accum_fusion:
            # The fused wgrad kernel accumulates into a caller-owned fp32 buffer.
            module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
        assert_module_replays_bit_exact(
            module,
            (x,),
            replays=3,
            contention=True,
            what=f"{layer} linear[fusion={grad_accum_fusion}]",
        )

    @pytest.mark.skipif(
        te_general_gemm is None, reason="Transformer Engine general_gemm is not available"
    )
    def test_column_parallel_linear_fp32_output_replays(self):
        """Replay the mixed-output GEMM used by FP32 language-model logits."""
        seeded()
        config = _config(hidden_size=512, num_attention_heads=8)
        module = ColumnParallelLinear(
            512,
            1024,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            output_dtype=torch.float32,
        ).cuda()
        x = torch.randn(2048, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        outputs, _ = assert_module_replays_bit_exact(
            module, (x,), replays=3, contention=True, what="column linear[bf16 input, fp32 output]"
        )
        assert outputs["out[0]"].dtype is torch.float32


# --- apex fused layer norm ------------------------------------------------------------------


@pytest.mark.skipif(
    not (fused_layer_norm.HAVE_PERSIST_LAYER_NORM or fused_layer_norm.HAVE_FUSED_LAYER_NORM),
    reason="apex fused layer norm unavailable",
)
@pytest.mark.parametrize("hidden", [4096, 4000], ids=["persistent", "non-persistent"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fused_layer_norm_replays(hidden, zero_centered_gamma):
    """dgamma/dbeta reduce over 8192 rows -- the cross-row reduction apex has to order."""
    seeded()
    config = _config(
        hidden_size=hidden,
        normalization="LayerNorm",
        layernorm_zero_centered_gamma=zero_centered_gamma,
    )
    module = (
        fused_layer_norm.FusedLayerNorm(
            config,
            hidden,
            eps=1e-5,
            persist_layer_norm=True,
            zero_centered_gamma=zero_centered_gamma,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(8192, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert_module_replays_bit_exact(
        module, (x,), replays=3, contention=True, what=f"FusedLayerNorm[{hidden}]"
    )


# --- apex fused scale-mask softmax -------------------------------------------------------


@pytest.mark.skipif(not HAVE_SCALED_SOFTMAX_EXT, reason="apex scaled softmax extensions missing")
@pytest.mark.parametrize("mask_type", [AttnMaskType.causal, AttnMaskType.padding])
def test_fused_scale_mask_softmax_replays(mask_type):
    seeded()
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=mask_type,
        scaled_masked_softmax_fusion=True,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
    )
    b, np_, sq, sk = 4, 16, 2048, 2048
    scores = torch.randn(b, np_, sq, sk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if mask_type == AttnMaskType.causal:
        mask = None
    else:
        mask = torch.rand(b, 1, sq, sk, device="cuda") < 0.2
    assert module.is_kernel_available(mask, b, np_, sq, sk), "fused softmax kernel not selected"
    assert_replays_bit_exact(
        lambda s: module(s, mask),
        (scores,),
        replays=3,
        what=f"FusedScaleMaskSoftmax[{mask_type.name}]",
    )


@pytest.mark.parametrize("with_caller_mask", [False, True], ids=["swa", "swa+padding"])
def test_sliding_window_softmax_torch_path_replays(with_caller_mask):
    """The torch fallback path behind ``window_size``: the sliding-window mask alone, and
    composed with a caller-provided padding mask (which used to be silently discarded).
    ``window_size`` never routes to the apex kernel, so this covers ``forward_torch_softmax``
    for both fwd and bwd, with and without the composed mask."""
    seeded()
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=False,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=(128, 0),
    )
    b, np_, sq, sk = 2, 8, 512, 512
    scores = torch.randn(b, np_, sq, sk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = (torch.rand(b, 1, sq, sk, device="cuda") < 0.2) if with_caller_mask else None
    assert not module.is_kernel_available(mask, b, np_, sq, sk), "expected the torch path"
    assert_replays_bit_exact(
        lambda s: module(s, mask),
        (scores,),
        replays=3,
        what=f"FusedScaleMaskSoftmax[torch, swa{'+padding' if with_caller_mask else ''}]",
    )


@pytest.mark.skipif(not HAVE_SCALED_SOFTMAX_EXT, reason="apex scaled softmax extensions missing")
@pytest.mark.parametrize("with_caller_mask", [False, True], ids=["swa", "swa+padding"])
def test_windowed_config_with_fusion_uses_torch_path_replays(with_caller_mask):
    """A windowed config must not route to the apex fused kernels (issue #7847).

    The fused kernels have no window parameter, so selecting them for a
    windowed config silently computes full-causal attention. Windowed configs
    must therefore fall through to ``forward_torch_softmax`` even when every
    shape/dtype gate would admit the kernel; this covers that routed path for
    both fwd and bwd, with and without a composed caller mask.
    """
    seeded()
    module = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=True,
        mask_func=attention_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=(128, 0),
    )
    b, np_, sq, sk = 2, 8, 512, 512
    scores = torch.randn(b, np_, sq, sk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = (torch.rand(b, 1, sq, sk, device="cuda") < 0.2) if with_caller_mask else None
    assert not module.is_kernel_available(mask, b, np_, sq, sk), "windowed config took fused path"
    assert_replays_bit_exact(
        lambda s: module(s, mask),
        (scores,),
        replays=3,
        what=f"FusedScaleMaskSoftmax[torch, swa-fusion{'+padding' if with_caller_mask else ''}]",
    )
