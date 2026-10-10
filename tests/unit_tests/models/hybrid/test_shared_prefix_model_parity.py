# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""HybridModel shared-prefix execution against the expanded dense rows.

``HybridModel.forward(shared_prefix_layout=...)`` must compute the completion logits, the
parameter gradients and the MoE expert-bias token counts of running every dense row
``[prompt, completion_g]`` on its own.

Tolerances (whole-tensor relative L2):

* FP32 layer exactness. Local (torch) linears with TF32 disabled make every GEMM IEEE FP32 and
  the router runs in FP64, so shared and dense differ only in summation order (measured: 4e-7
  for parameter gradients). ``LAYER_FP32_TOLERANCE = 1e-5``; TF32 GEMMs alone give 1.6e-4..3.5e-4.
* BF16 models. Shared and dense BF16 arithmetic are two different but equally valid roundings,
  so neither a fixed bound nor a dense rerun (which can be bit-identical) is a sound null.
  Both are compared with an FP32 dense reference of the same BF16-representable weights, and
  shared must be as close to it as dense is: within ``REFERENCE_RATIO = 1.25``. Measured on
  GB200: shared/dense error ratios 0.98..1.00, while an off-by-one RoPE position for the
  completions raises the logit error 17x. MoE top-k choices are replayed from a fixed per-token
  table in every arm, so rounding cannot flip expert choices and token counts must match exactly.
* Distributed (TP2/SP/CP2, TP1/CP4). No FP32 reference exists there (TE context-parallel
  attention is 16-bit only), so the shared-vs-dense gap at the target topology must stay within
  ``TOPOLOGY_RATIO = 1.5`` of the same model's TP1/CP1 gap, measured in the same test (measured
  ratios 1.00..1.08). World-summed expert-bias counts must match exactly.
"""

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout
from tests.unit_tests.models.hybrid.shared_prefix_test_utils import (
    LayerProblemData,
    ReplayedRouting,
    SharedPrefixProblem,
    TokenProblem,
    build_hybrid_model,
    canonical_dense_run,
    canonical_star_run,
    clear_attention_env,
    compare_canonical,
    compare_model_runs,
    copy_params,
    round_params_to,
    run_dense_rows,
    run_layer,
    run_shared,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")
pytest.importorskip("flash_attn")

LAYER_FP32_TOLERANCE = 1e-5
REFERENCE_RATIO = 1.25
TOPOLOGY_RATIO = 1.5
# Additive slack for relative errors that are themselves near zero.
SLACK = 1e-5
PADDING_MULTIPLE = 8  # 2*TP*CP for TP2/CP2 and TP1/CP4; explicit branch padding at TP1/CP1.
PATTERN = "M*EM*E"
MTP_PATTERN = "M*EM*E/*E/*E"

STAR = ((300, (97, 210, 41, 150)),)
FOREST = ((150, (60, 131)), (200, (33, 90, 47, 70)))


@pytest.fixture
def _deterministic_kernels(monkeypatch):
    """Bit-reproducible backward for the test, restored afterwards.

    The Mamba scan, causal_conv1d and TE attention backward kernels accumulate with atomics by
    default, so two identical calls differ in the last bits (Mamba ``A_log`` first). They take
    ordered reductions when torch's deterministic flag (or their environment switch) is set.
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _problem(roots):
    return SharedPrefixProblem(
        roots, padding_multiple=PADDING_MULTIPLE, topology_multiple=PADDING_MULTIPLE
    )


class _OffByOnePositionStar(SharedPrefixLayout):
    """A deliberately wrong layout: completion RoPE positions start at P - 1 instead of P."""

    def padded_position_ids(self, physical_len, device):
        positions = super().padded_position_ids(physical_len, device).clone()
        positions[self.prefix_len : self.total_len] -= 1
        return positions


def _star_with_off_by_one_positions(layout):
    return _OffByOnePositionStar(
        layout.prefix_len,
        layout.completion_lens,
        logical_completion_lens=layout.logical_completion_lens,
        padding_multiple=layout.padding_multiple,
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMoELayerExactness:
    """FP32 MoE TransformerLayer with logical multiplicities against dense rows."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)
        self._tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def teardown_method(self, method):
        torch.backends.cuda.matmul.allow_tf32 = self._tf32
        Utils.destroy_model_parallel()

    def test_moe_layer_matches_dense_rows(self):
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.moe.moe_utils import router_gating_token_blocks
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        from megatron.core.transformer.transformer_layer import (
            MoETransformerLayer,
            TransformerLayerSubmodules,
        )

        hidden = 128
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            num_attention_heads=4,
            normalization="RMSNorm",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp64",
            use_cpu_initialization=True,
        )
        submodules = TransformerLayerSubmodules(
            pre_mlp_layernorm=WrappedTorchNorm,
            mlp=get_moe_module_spec(use_te=False, num_experts=8, moe_grouped_gemm=False),
            mlp_bda=get_bias_dropout_add,
        )
        layer = MoETransformerLayer(config, submodules).cuda().train()
        router = layer.mlp.router
        problem = SharedPrefixProblem(
            ((300, (180, 64, 257, 33)), (37, (5, 90))), padding_multiple=8, extra_padding=8
        )
        layout = problem.layout(forest=True)
        data = LayerProblemData(problem, hidden, seed=3)

        def dense(x):
            return layer(hidden_states=x, attention_mask=None)

        def shared(x):
            # The scoping forward_hybrid_stack_shared_prefix applies around every MoE layer.
            layer.mlp._shared_prefix_token_multiplicities = layout.padded_token_multiplicities(
                x.shape[0], x.device
            )
            try:
                with router_gating_token_blocks():
                    return layer(hidden_states=x, attention_mask=None)
            finally:
                del layer.mlp._shared_prefix_token_multiplicities

        # MoE is token-wise, so the dense rows run as one packed sequence without padding rows
        # (padding rows of a [max_len, rows] batch would be routed and counted).
        packed_input = torch.cat(data.dense_rows(data.dense_input)).unsqueeze(1)
        packed_cotangent = torch.cat(data.dense_rows(data.dense_cotangent)).unsqueeze(1)
        router.local_tokens_per_expert.zero_()
        dense_run = run_layer(layer, dense, packed_input.cuda().float(), packed_cotangent.cuda())
        dense_counts = router.local_tokens_per_expert.clone()
        router.local_tokens_per_expert.zero_()
        shared_run = run_layer(
            layer, shared, data.star_input.cuda().float(), data.star_cotangent.cuda()
        )
        shared_counts = router.local_tokens_per_expert.clone()
        for key in ("output", "input_grad"):
            dense_run[key] = data.unpack_dense_rows(dense_run[key])

        errors = compare_canonical(
            canonical_star_run(data, shared_run), canonical_dense_run(data, dense_run)
        )
        print(f"\n[moe fp32] {errors}")
        for metric in ("output", "input_grad", "params"):
            assert errors[metric] <= LAYER_FP32_TOLERANCE, errors
        torch.testing.assert_close(shared_counts, dense_counts, rtol=0, atol=0)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixHybridModelParity:
    """BF16 HybridModel shared-prefix forward/backward at TP1/CP1 against dense rows."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "roots,forest,position_embedding_type",
        [(STAR, False, "rope"), (FOREST, True, "rope"), (STAR, False, "none")],
        ids=["star-rope", "forest-rope", "star-positionless"],
    )
    def test_model_matches_dense_rows(self, roots, forest, position_embedding_type, monkeypatch):
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(
            PATTERN, torch.bfloat16, position_embedding_type=position_embedding_type
        )
        round_params_to(model, torch.bfloat16)
        reference_model = build_hybrid_model(
            PATTERN, torch.float32, position_embedding_type=position_embedding_type
        )
        copy_params(model, reference_model)
        tokens = TokenProblem(_problem(roots), vocab_size=2048, seed=1)
        layout = tokens.problem.layout(forest)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        reference_routing = ReplayedRouting(reference_model, tokens.num_keys, seed=2)
        try:
            reference = run_dense_rows(reference_model, tokens, reference_routing)
            dense = run_dense_rows(model, tokens, routing)
            shared = run_shared(model, tokens, layout, routing)
            dense_error = compare_model_runs(dense, reference, model)
            shared_error = compare_model_runs(shared, reference, model)
            print(f"\n[{position_embedding_type}] dense={dense_error} shared={shared_error}")
            for metric in ("logits", "grads"):
                assert shared_error[metric] <= REFERENCE_RATIO * dense_error[metric] + SLACK, (
                    f"shared {metric} error {shared_error[metric]:.3e} vs dense "
                    f"{dense_error[metric]:.3e} against the FP32 reference"
                )
            # Replayed routing makes the expert-bias counts exact functions of the layout.
            for shared_count, dense_count in zip(shared.counts, dense.counts):
                torch.testing.assert_close(shared_count, dense_count, rtol=0, atol=0)
            assert len(shared.counts) == PATTERN.count("E")

            if position_embedding_type == "rope" and not forest:
                # Sensitivity guard: the comparison must detect an off-by-one RoPE position.
                wrong = run_shared(model, tokens, _star_with_off_by_one_positions(layout), routing)
                wrong_error = compare_model_runs(wrong, reference, model)
                print(f"  off-by-one RoPE positions: {wrong_error}")
                assert wrong_error["logits"] > 5 * REFERENCE_RATIO * dense_error["logits"]
        finally:
            routing.close()
            reference_routing.close()

    def test_full_recompute_matches_plain_shared_forward(self, monkeypatch):
        """Full uniform recompute replays the same shared forward and counts tokens once."""
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        recompute_model = build_hybrid_model(
            PATTERN,
            torch.bfloat16,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        copy_params(model, recompute_model)
        tokens = TokenProblem(_problem(FOREST), vocab_size=2048, seed=1)
        layout = tokens.problem.layout(forest=True)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        recompute_routing = ReplayedRouting(recompute_model, tokens.num_keys, seed=2)
        try:
            plain = run_shared(model, tokens, layout, routing)
            again = run_shared(model, tokens, layout, routing)
            recomputed = run_shared(recompute_model, tokens, layout, recompute_routing)
        finally:
            routing.close()
            recompute_routing.close()
        for plain_logits, recomputed_logits in zip(plain.logits, recomputed.logits):
            assert torch.equal(plain_logits, recomputed_logits)
        repeat_error = compare_model_runs(again, plain, model)["grads"]
        recompute_error = compare_model_runs(recomputed, plain, model)["grads"]
        print(f"\nrepeat={repeat_error:.3e} recompute={recompute_error:.3e}")
        # BF16 MoE backward accumulates in a nondeterministic order; recompute adds nothing more.
        assert recompute_error <= 2 * repeat_error + SLACK
        for plain_count, recomputed_count in zip(plain.counts, recomputed.counts):
            torch.testing.assert_close(recomputed_count, plain_count, rtol=0, atol=0)

    @pytest.mark.parametrize("exclude", [None, False, True], ids=["default", "counted", "excluded"])
    def test_expert_bias_padding_convention_matches_dense(self, exclude, monkeypatch):
        """The caller's flag selects which dense padding convention the counts reproduce."""
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        tokens = TokenProblem(_problem(FOREST), vocab_size=2048, seed=1)
        layout = tokens.problem.layout(forest=True)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        flag = {}
        if exclude is not None:
            flag["shared_prefix_exclude_sequence_padding_from_expert_bias"] = exclude
        try:
            counted = run_dense_rows(model, tokens, routing)
            masked = run_dense_rows(model, tokens, routing, mask_sequence_padding=True)
            shared = run_shared(model, tokens, layout, routing, **flag)
        finally:
            routing.close()
        # The problem has per-branch padding, so the two dense conventions differ.
        assert sum(map(torch.sum, masked.counts)) < sum(map(torch.sum, counted.counts))
        # Unset on this non-HybridEP dispatcher, padding rows count, as with padding_mask=None.
        assert model.config.moe_token_dispatcher_type != "flex"
        expected = masked if exclude else counted
        assert len(shared.counts) == PATTERN.count("E")
        for shared_count, expected_count in zip(shared.counts, expected.counts):
            torch.testing.assert_close(shared_count, expected_count, rtol=0, atol=0)

    @pytest.mark.parametrize("exclude", [False, True], ids=["counted", "excluded"])
    def test_expert_bias_padding_flag_requires_layout(self, exclude, monkeypatch):
        """Stating the padding convention has no meaning without a shared-prefix layout."""
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        tokens = TokenProblem(_problem(STAR), vocab_size=2048, seed=1)
        with pytest.raises(ValueError, match="requires shared_prefix_layout"):
            model(
                input_ids=tokens.row_ids[0][None].cuda(),
                position_ids=torch.arange(tokens.row_ids[0].numel())[None].cuda(),
                attention_mask=None,
                shared_prefix_exclude_sequence_padding_from_expert_bias=exclude,
            )

    @pytest.mark.parametrize("exclude", [None, False, True], ids=["unset", "counted", "excluded"])
    def test_expert_bias_padding_flag_reaches_the_stack(self, exclude, monkeypatch):
        """Unset, the model leaves the convention to the stack's dispatcher-based inference."""
        from megatron.core.models.hybrid import shared_prefix

        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        tokens = TokenProblem(_problem(STAR), vocab_size=2048, seed=1)
        forward = shared_prefix.forward_hybrid_stack_shared_prefix
        received = []

        def recording_forward(*args, exclude_sequence_padding_from_expert_bias, **kwargs):
            received.append(exclude_sequence_padding_from_expert_bias)
            return forward(
                *args,
                exclude_sequence_padding_from_expert_bias=exclude_sequence_padding_from_expert_bias,
                **kwargs,
            )

        monkeypatch.setattr(shared_prefix, "forward_hybrid_stack_shared_prefix", recording_forward)
        flag = {}
        if exclude is not None:
            flag["shared_prefix_exclude_sequence_padding_from_expert_bias"] = exclude
        run_shared(model, tokens, tokens.problem.layout(forest=False), **flag)
        assert received == [exclude]

    def test_forward_validates_the_stack_once(self, monkeypatch):
        """HybridModel validates before its embedding; the stack forward does not repeat it."""
        from megatron.core.models.hybrid import shared_prefix

        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        tokens = TokenProblem(_problem(STAR), vocab_size=2048, seed=1)
        validate = shared_prefix._validate_hybrid_stack
        physical_lens = []

        def counting_validate(*args, physical_len, **kwargs):
            physical_lens.append(physical_len)
            return validate(*args, physical_len=physical_len, **kwargs)

        monkeypatch.setattr(shared_prefix, "_validate_hybrid_stack", counting_validate)
        run_shared(model, tokens, tokens.problem.layout(forest=False))
        assert physical_lens == [tokens.problem.physical_len]

    @pytest.mark.usefixtures("_deterministic_kernels")
    @pytest.mark.parametrize("pattern", [PATTERN, MTP_PATTERN], ids=["no-mtp", "mtp"])
    def test_explicit_none_layout_is_the_default_path(self, pattern, monkeypatch):
        """``shared_prefix_layout=None`` runs the ordinary forward bit for bit (feature off)."""
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(pattern, torch.bfloat16)
        tokens = TokenProblem(_problem(STAR), vocab_size=2048, seed=1)
        # Natural (not replayed) routing: this is the production default path.
        omitted = run_dense_rows(model, tokens)
        explicit = run_dense_rows(model, tokens, shared_prefix_layout=None)
        for omitted_logits, explicit_logits in zip(omitted.logits, explicit.logits):
            assert torch.equal(omitted_logits, explicit_logits)
        assert omitted.grads.keys() == explicit.grads.keys()
        for name, grad in omitted.grads.items():
            assert torch.equal(grad, explicit.grads[name]), name
        assert len(omitted.counts) == pattern.count("E")
        for omitted_count, explicit_count in zip(omitted.counts, explicit.counts):
            assert torch.equal(omitted_count, explicit_count)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixHybridModelDistributedParity:
    """Shared-prefix TP/SP/CP plumbing: token-multiplicity ownership, zigzag RoPE, gathers."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _gap(self, monkeypatch, roots, forest):
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(PATTERN, torch.bfloat16)
        tokens = TokenProblem(_problem(roots), vocab_size=2048, seed=1)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        try:
            dense = run_dense_rows(model, tokens, routing)
            shared = run_shared(model, tokens, tokens.problem.layout(forest), routing)
        finally:
            routing.close()
        return dense, shared, compare_model_runs(shared, dense, model)

    @pytest.mark.parametrize(
        "tp,cp,mamba_backend",
        [(2, 2, None), (1, 4, None), (2, 2, "state_fork")],
        ids=["tp2-sp-cp2", "tp1-cp4", "tp2-sp-cp2-state-fork"],
    )
    @pytest.mark.parametrize(
        "roots,forest", [(STAR, False), (FOREST, True)], ids=["star", "forest"]
    )
    def test_model_matches_dense_rows_distributed(
        self, tp, cp, mamba_backend, roots, forest, monkeypatch
    ):
        if Utils.world_size < tp * cp or Utils.world_size % (tp * cp):
            pytest.skip(f"requires a world size divisible by {tp * cp}")
        if mamba_backend is None:
            # The default backend, ragged_state_fork.
            monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
        else:
            # The opt-in backend that uses only the public mamba_ssm API.
            monkeypatch.setenv("NRL_SP_MAMBA_IMPL", mamba_backend)
        Utils.initialize_model_parallel(1, 1)
        _, _, baseline = self._gap(monkeypatch, roots, forest)
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp)
        dense, shared, gap = self._gap(monkeypatch, roots, forest)
        print(f"\n[tp{tp} cp{cp}] gap={gap} tp1/cp1 gap={baseline}")
        # World-summed counts: each CP/TP rank must own the multiplicities of its own tokens.
        for shared_count, dense_count in zip(shared.counts, dense.counts):
            torch.testing.assert_close(shared_count, dense_count, rtol=0, atol=0)
        for metric in ("logits", "grads"):
            assert gap[metric] <= TOPOLOGY_RATIO * baseline[metric] + SLACK, (
                f"TP{tp}/CP{cp} shared-vs-dense {metric} gap {gap[metric]:.3e} exceeds "
                f"{TOPOLOGY_RATIO}x the TP1/CP1 gap {baseline[metric]:.3e}"
            )
