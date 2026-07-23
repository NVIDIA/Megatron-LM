# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import itertools
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_num_microbatches,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
    delete_cuda_graphs,
)
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module
from megatron.rl import rl_utils
from megatron.rl.agent.api import TokenRollout
from megatron.rl.inference import ReturnsRaw
from megatron.rl.rollout_granularity import get_rl_parallel_generation_tasks
from megatron.rl.sequence_packing_utils import get_default_packed_seq_params
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from tests.unit_tests.test_utilities import Utils

BATCH = 2
SEQ = 4
VOCAB = 754


class MockModel(LanguageModule):
    def __init__(self, batch=BATCH, seq=SEQ, vocab=VOCAB):
        self.batch = batch
        self.seq = seq
        self.vocab = vocab
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.config = TransformerConfig(
            num_attention_heads=8, num_layers=8, pipeline_dtype=torch.bfloat16
        )
        self.model_type = ModelType.encoder_or_decoder

    def __call__(self, x, position_ids, attention_mask, **kwargs):
        del position_ids
        del attention_mask
        batch, seq = x.shape
        mock_model_outputs = torch.ones((batch, seq, self.vocab), device=x.device)
        return mock_model_outputs

    def load_state_dict(self, params):
        del params

    def train(self, mode=True):
        del mode

    def state_dict(self):
        return {}

    def set_input_tensor(self, input_tensor):
        pass


class MockTokenizer:
    def __init__(self):
        self.pad = 42
        self.eod = 43
        self.vocab_size = VOCAB
        self.bos = None
        self.library = None

    def detokenize(self, tokens):
        return [str(tok) for tok in tokens]


def make_token_rollout(trajectory, logprobs, generation_mask=None, reward=1.0, problem_id="p"):
    """TokenRollout with the per-turn staleness boilerplate derived from the turn count."""
    turns = len(trajectory)
    return TokenRollout(
        trajectory=trajectory,
        reward=reward,
        generation_mask=generation_mask,
        logprobs=logprobs,
        env_id='MEGAENV',
        problem_id=problem_id,
        policy_epoch=[[(0, 0)]] * turns,
        kv_cache_epoch=[[(0, 0)]] * turns,
        num_evictions=[0] * turns,
    )


class DummyLangModule:
    def __init__(self, config):
        self.config = config
        self.rotary_pos_emb = None
        self.eval = MagicMock()
        self.train = MagicMock()

    def modules(self):
        return iter(())


class DummyMoELayer:
    def __init__(self, use_partial_cudagraphs):
        self.use_partial_cudagraphs = use_partial_cudagraphs
        self.transition_calls = []

    def transition_cudagraph_scope(self, mode):
        self.transition_calls.append(mode)
        self.use_partial_cudagraphs = mode == "partial"


@pytest.fixture
def initialize_model_parallel(request, monkeypatch):
    """Fixture to initialize and destroy model parallel.

    Parameters are passed via request.param as a tuple: (tp, pp)
    Skips if world_size < tp * pp.
    """
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("LOG_TO_WANDB", "false")

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    tp, pp = request.param
    world_size = Utils.world_size
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(123)
    dp = world_size // (tp * pp)
    yield world_size, dp, tp, pp
    Utils.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Ensure global state is correctly cleaned up after every test."""
    yield
    destroy_global_vars()
    destroy_num_microbatches_calculator()


class TestRLUtils:
    """Test class for RL utilities."""

    def create_test_args(self, **kwargs):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        args = parse_args(ignore_unknown_args=True)
        args.num_layers = 8
        args.num_attention_heads = 8
        args.vocab_size = VOCAB
        args.hidden_size = 128
        args.max_position_embeddings = 256
        args.seq_length = 256
        args.wandb_project = None

        args.micro_batch_size = 1

        for key, value in kwargs.items():
            setattr(args, key, value)

        args = validate_args(args)
        set_global_variables(args, False)
        return args

    def test_rl_granularity_defaults(self):
        args = self.create_test_args(perform_rl_step=True, grpo_prompts_per_step=8)

        assert args.rl_submission_granularity == "B"
        assert args.rl_consumption_granularity == "B"
        assert args.rl_generation_lag == 0
        assert not hasattr(args, "rl_parallel_generation_tasks")
        assert get_rl_parallel_generation_tasks(args) == 1

    @pytest.mark.parametrize(
        "submission_granularity, generation_lag, expected_parallel_generation_tasks",
        [
            pytest.param("B", 0, 1, id="batch"),
            pytest.param("B", 2, 3, id="batch_with_lag"),
            pytest.param("G", 0, 8, id="group"),
            pytest.param("G", 2, 24, id="group_with_lag"),
            pytest.param("R", 0, 32, id="rollout"),
            pytest.param("R", 2, 96, id="rollout_with_lag"),
        ],
    )
    def test_get_rl_parallel_generation_tasks(
        self, submission_granularity, generation_lag, expected_parallel_generation_tasks
    ):
        args = SimpleNamespace(
            rl_submission_granularity=submission_granularity,
            rl_generation_lag=generation_lag,
            grpo_prompts_per_step=8,
            grpo_group_size=4,
        )

        assert get_rl_parallel_generation_tasks(args) == expected_parallel_generation_tasks

    @pytest.mark.parametrize(
        "rl_partial_rollouts, submission_granularity",
        [
            pytest.param(False, "B", id="non_streaming_batch"),
            pytest.param(True, "B", id="streaming_batch"),
            pytest.param(True, "G", id="streaming_group"),
            pytest.param(True, "R", id="streaming_rollout"),
        ],
    )
    def test_get_rollout_generator_keeps_num_groups_at_trainer_batch_size(
        self, monkeypatch, rl_partial_rollouts, submission_granularity
    ):
        """Regression for the removed ``num_groups=1`` streaming override.

        Previously ``get_rollout_generator`` forced ``num_groups`` to 1 whenever it
        streamed with a non-batch submission granularity. For a multi-environment
        agent that collapses the per-env group distribution so some environments
        receive zero groups (and a degenerate all-zero ``agent_slots``), stalling
        ``get_grouped_rollouts``. ``num_groups`` must stay at the trainer batch size
        (``n_prompts``) regardless of streaming or submission granularity.
        """
        n_prompts = 8
        captured = {}
        rollout_generator = object()

        class Agent:
            def get_grouped_rollouts(self, request):
                captured["request"] = request
                return rollout_generator

        def get_agent(_args, parallel_generation_tasks=None):
            captured["parallel_generation_tasks"] = parallel_generation_tasks
            return Agent()

        monkeypatch.setattr(rl_utils, "_ROLLOUT_GENERATOR", None)
        monkeypatch.setattr(rl_utils, "get_agent", get_agent)

        args = SimpleNamespace(
            rl_partial_rollouts=rl_partial_rollouts,
            rl_submission_granularity=submission_granularity,
            rl_consumption_granularity="B",
            rl_generation_lag=0,
            grpo_prompts_per_step=n_prompts,
            grpo_group_size=4,
            rl_default_temperature=1.0,
            inference_max_seq_length=128,
            rl_default_top_p=1.0,
            rl_default_top_k=0,
            grpo_filter_groups_with_same_reward=False,
        )

        result = rl_utils.get_rollout_generator(
            args, inference_interface=ReturnsRaw(), n_prompts=n_prompts, samples_per_group=4
        )

        assert result is rollout_generator
        assert captured["request"].num_groups == n_prompts
        assert captured["request"].streaming == rl_partial_rollouts
        assert captured["request"].submission_granularity == submission_granularity

    @pytest.mark.parametrize(
        "overrides, match",
        [
            pytest.param(
                {"rl_generation_lag": 1},
                "--rl-generation-lag requires --rl-partial-rollouts",
                id="lag_requires_partial_rollouts",
            ),
            pytest.param(
                {"rl_submission_granularity": "R"},
                "Rollout submission granularity requires streaming grouped rollouts",
                id="rollout_submission_requires_partial_rollouts",
            ),
            pytest.param(
                {"rl_consumption_granularity": "R"},
                "--rl-consumption-granularity R is not currently supported",
                id="rollout_consumption_unsupported",
            ),
            pytest.param(
                {"rl_submission_granularity": "B", "rl_consumption_granularity": "G"},
                "--rl-submission-granularity B with --rl-consumption-granularity G",
                id="batch_submit_group_consume_unsupported",
            ),
        ],
    )
    def test_rl_granularity_validation_rejects_unsupported_modes(self, overrides, match):
        with pytest.raises(AssertionError, match=match):
            self.create_test_args(perform_rl_step=True, **overrides)

    @pytest.mark.parametrize(
        "flag", ["--rl-submission-granularity", "--rl-consumption-granularity"]
    )
    def test_rl_granularity_choices_reject_unknown_value(self, monkeypatch, flag):
        monkeypatch.setattr("sys.argv", ["test", flag, "X"])
        with pytest.raises(SystemExit):
            parse_args(ignore_unknown_args=False)

    def _patch_rl_inference_mode_deps(self, monkeypatch, args):
        interface = MagicMock()
        interface.resume.return_value = object()
        interface.suspend.return_value = object()
        loop = SimpleNamespace(run_until_complete=MagicMock())

        monkeypatch.setattr(rl_utils, "get_args", lambda: args)
        monkeypatch.setattr(rl_utils, "get_asyncio_loop", lambda: loop)
        monkeypatch.setattr(
            rl_utils, "get_nvtx_range", lambda: (lambda *args, **kwargs: nullcontext())
        )
        monkeypatch.setattr(rl_utils, "get_inference_interface", lambda *_args: interface)
        monkeypatch.setattr(
            rl_utils,
            "unwrap_model",
            lambda model: model.module if hasattr(model, "module") else model,
        )
        monkeypatch.setattr(
            rl_utils, "_maybe_prefetch_separate_inference_model_weights", MagicMock()
        )
        monkeypatch.setattr(rl_utils, "set_decode_expert_padding", MagicMock())
        monkeypatch.setattr(rl_utils.dist, "get_rank", lambda: 0)
        return interface, loop

    def _make_toggle_cuda_graphs_mock(self):
        def _toggle(lang_module, set_to):
            assert set_to in {"none", "local"}, f"Invalid CUDA graph implementation: {set_to}"
            lang_module.config.cuda_graph_impl = set_to

        return MagicMock(side_effect=_toggle)

    def test_megatron_rl_inference_mode_restores_training_cuda_graph_state(self, monkeypatch):
        config = SimpleNamespace(
            cuda_graph_impl="none",
            cuda_graph_modules=[CudaGraphModule.attn],
            inference_cuda_graph_scope=InferenceCudaGraphScope.none,
        )
        lang_module = DummyLangModule(config)
        model = [SimpleNamespace(config=config, module=lang_module)]
        args = SimpleNamespace(
            rl_training_cuda_graphs=False,
            num_experts=None,
            curr_iteration=11,
            cuda_graph_impl="local",
            cuda_graph_modules=[CudaGraphModule.attn],
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
        )
        interface, _ = self._patch_rl_inference_mode_deps(monkeypatch, args)
        toggle_cuda_graphs = self._make_toggle_cuda_graphs_mock()
        monkeypatch.setattr(rl_utils, "toggle_cuda_graphs", toggle_cuda_graphs)

        with rl_utils.megatron_rl_inference_mode(model, MagicMock(), "local", False) as result:
            assert result is interface
            assert config.cuda_graph_impl == "local"
            assert config.cuda_graph_modules == []
            assert config.inference_cuda_graph_scope == InferenceCudaGraphScope.block

        assert toggle_cuda_graphs.call_args_list == [
            call(lang_module, "local"),
            call(lang_module, "none"),
        ]
        assert config.cuda_graph_impl == "local"
        assert config.cuda_graph_modules == [CudaGraphModule.attn]
        assert config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        lang_module.eval.assert_called_once()
        lang_module.train.assert_called_once()

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4, 8], [1, 2, 4, 8])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    @pytest.mark.parametrize("use_sequence_packing", [False])
    def test_get_logprobs(self, initialize_model_parallel, use_sequence_packing):
        """Test that getting logprobs at least does not crash."""
        self.create_test_args(rl_use_sequence_packing=use_sequence_packing)

        model = MockModel()
        tokens = torch.ones((BATCH, SEQ), dtype=torch.long)
        logprobs = rl_utils.get_logprobs(
            model, tokens, position_ids=None, sequence_packing=use_sequence_packing
        )
        if is_pp_last_stage(model.pg_collection.pp):
            # We chop off 1 element from the sequence dimension.
            assert logprobs.shape == (BATCH, SEQ - 1)
            # As we return ones as logits, all logprobs should be the same.
            assert torch.all(logprobs == logprobs[0, 0]).item()
        else:
            assert logprobs.shape == (BATCH, SEQ, VOCAB)

    def test_grpo_loss_calculation_all_pi_eq(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ)
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.zeros(BATCH)
        loss, kl_term, ratios, entropy_term, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(kl_term, torch.zeros_like(kl_term))
        torch.testing.assert_close(ratios, torch.ones_like(ratios))
        torch.testing.assert_close(entropy_term, -torch.ones_like(ratios) * torch.e)

    def test_grpo_loss_calculation_2x_ratios(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ) - torch.log(torch.tensor([2.0]))
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.ones(BATCH)
        loss, kl_term, ratios, _, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=2.1,
            clamp_eps_upper=2.1,
            kl_beta=0.0,
            entropy_weight=0.0,
        )
        # Clamping does not affect us, as 2.1 [eps] > 2 [ratio].
        # kl_beta = 0 -> we only have the non-kl term of the loss active.
        torch.testing.assert_close(loss, -torch.ones_like(loss) * 2)
        # pi and pi_{ref} are the same here.
        torch.testing.assert_close(kl_term, torch.zeros_like(kl_term))
        # Current probs are 2x more probable than old pi.
        torch.testing.assert_close(ratios, torch.ones_like(ratios) * 2)

    def test_entropy_calculation(self):
        # All policies are equal: clamping is inactive, ratios are ones.
        current_logprobs = torch.ones(BATCH, SEQ)
        old_logprobs = torch.ones(BATCH, SEQ)
        ref_logprobs = torch.ones(BATCH, SEQ)
        advantages = torch.zeros(BATCH)
        loss, _, ratios, entropy_term, _, _ = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.0,
            entropy_weight=1.0,
        )
        torch.testing.assert_close(loss, torch.ones_like(ratios) * torch.e)
        torch.testing.assert_close(entropy_term, -torch.ones_like(ratios) * torch.e)

    def test_grpo_loss_truncation(self):
        # All ratios are 2
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=torch.ones(BATCH, SEQ),
            old_logprobs=0.5 * torch.ones(BATCH, SEQ),
            ref_logprobs=torch.ones(BATCH, SEQ),
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        assert truncated_from_above.float().mean() == 1
        assert truncated_from_below.float().sum() == 0

        # All ratios are 0.01
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=0.01 * torch.ones(BATCH, SEQ),
            old_logprobs=torch.ones(BATCH, SEQ),
            ref_logprobs=torch.ones(BATCH, SEQ),
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        assert truncated_from_above.float().sum() == 0
        assert truncated_from_below.float().mean() == 1

        # Mixed ratios: [[2., 0.5], [20., 1.]]
        current_logprobs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        old_logprobs = torch.tensor([[0.5, 2.0], [0.05, 1.0]])
        _, _, _, _, truncated_from_above, truncated_from_below = rl_utils.calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=old_logprobs,
            advantages=torch.zeros(BATCH),
            clamp_eps_lower=0.1,
            clamp_eps_upper=0.1,
            kl_beta=0.1,
            entropy_weight=0.0,
        )
        torch.testing.assert_close(
            truncated_from_above, torch.tensor([[True, False], [True, False]])
        )
        torch.testing.assert_close(
            truncated_from_below, torch.tensor([[False, True], [False, False]])
        )

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4, 8], [1, 2, 4, 8])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_prepare_data_for_update(self, initialize_model_parallel):
        """Logprobs path runs; single-turn EOD guard holds; multi-turn rollouts collapse to one
        row each, padded to a DP*microbatch multiple to size the microbatch calculator."""
        world_size, dp, tp, pp = initialize_model_parallel
        # Here I assume that we will be consuming all data in one step.
        group_size = 2
        self.create_test_args(
            micro_batch_size=2,
            seq_length=4,
            curr_iteration=1,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            global_batch_size=dp * 2,
            grpo_prompts_per_step=dp,
            grpo_group_size=group_size,
        )

        model = MockModel()
        tokenizer = MockTokenizer()

        # A single-turn rollout whose only turn is short and lacks eod must be rejected:
        # a single-turn completion has no tool-call boundary to justify stopping early.
        bad = make_token_rollout(
            [[1, 2, 3]], [[0.1, 0.2, 0.3]], [[False, True, True]], reward=3.14, problem_id="2"
        )
        with pytest.raises(AssertionError, match="must end in eod"):
            rl_utils.prepare_data_for_update(
                [model],
                {},
                [[bad] for _ in range(dp)],
                tokenizer,
                sequence_packing=False,
                is_correction=False,
            )

        # Multi-turn rollouts with uneven turn counts: each collapses to ONE combined row
        # (the final turn) regardless of turn count, so a group of 2 rollouts yields 2 rows.
        # token_ids are cumulative and each turn generates the newly-added tokens (no overlap),
        # so the combined generation mask / logprobs are well-formed.
        mt1 = make_token_rollout(
            [[1, 2, 3], [1, 2, 3, 4]],
            [[0.1, 0.2], [0.3]],
            [[False, True, True], [False, False, False, True]],
            problem_id="1",
        )
        mt2 = make_token_rollout(
            [[1, 2], [1, 2, 3], [1, 2, 3, 4]],
            [[0.1], [0.2], [0.3]],
            [[False, True], [False, False, True], [False, False, False, True]],
            reward=0.0,
            problem_id="3",
        )
        rl_utils.prepare_data_for_update(
            [model],
            {},
            [[mt1, mt2] for _ in range(dp)],
            tokenizer,
            sequence_packing=False,
            is_correction=False,
        )
        # 2 rollouts/group * dp groups = 2*dp rows (already a multiple of micro_batch_size*dp);
        # 2*dp / (micro_batch_size 2 * dp) = 1 microbatch.
        assert get_num_microbatches() == 1

        r1 = make_token_rollout(
            torch.tensor([[1, 2, 3, tokenizer.eod]], dtype=torch.float).cuda(),
            torch.tensor([[-0.2, -0.3, -3.2]]).cuda(),
            torch.tensor([[False, True, True, True]], dtype=torch.float).cuda(),
            reward=3.14,
            problem_id="2",
        )
        r2 = make_token_rollout(
            torch.tensor([[1, 2, 234, tokenizer.eod]], dtype=torch.float).cuda(),
            torch.tensor([[-0.2, -0.3, -1.2]]),
            torch.tensor([[False, True, True, True]], dtype=torch.float).cuda(),
            reward=0.14,
            problem_id="2",
        )
        rollouts = [[r1, r2] for _ in range(dp)]
        data_iter, _, _ = rl_utils.prepare_data_for_update(
            [model], {}, rollouts, tokenizer, sequence_packing=False, is_correction=False
        )

        _, _, old_logprobs, _, _, _, _ = next(data_iter)
        # All logits are ones in the MockModel.
        # All probabilities should be uniform.
        torch.testing.assert_close(old_logprobs.exp(), torch.ones_like(old_logprobs) / VOCAB)

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((1, 1), id="tp1-pp1")],
        indirect=["initialize_model_parallel"],
    )
    @pytest.mark.parametrize("scenario", ["fallback_and_pad", "oversampling"])
    def test_prepare_data_for_update_row_accounting(self, initialize_model_parallel, scenario):
        """Rows = one per rollout, except non-prefix rollouts fall back to one row per turn.
        The (padded) row count -- not the rollout count -- sizes the microbatch calculator, and
        oversampling (ratio < 1) consumes a fraction of the rows per step."""
        world_size, dp, tp, pp = initialize_model_parallel
        tokenizer = MockTokenizer()
        model = MockModel()

        def single(problem_id, reward):
            return make_token_rollout(
                [[1, 2, 3, tokenizer.eod]],
                [[0.1, 0.2, 0.3]],
                [[False, True, True, True]],
                reward=reward,
                problem_id=problem_id,
            )

        if scenario == "fallback_and_pad":
            # group = [collapsible single-turn, NON-prefix 2-turn] -> 1 + 2 = 3 rows/group.
            # 3*dp rows padded up to the next multiple of micro_batch_size*dp (= 2*dp) -> 4*dp;
            # 4*dp / (2 * dp) = 2 microbatches.
            self.create_test_args(
                micro_batch_size=2,
                seq_length=4,
                curr_iteration=1,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                global_batch_size=dp * 2,
                grpo_prompts_per_step=dp,
                grpo_group_size=2,
            )
            non_prefix = make_token_rollout(
                [[1, 2, 3, tokenizer.eod], [7, 8, 9, tokenizer.eod]],
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[False, True, True, True], [False, True, True, True]],
                reward=0.0,
                problem_id="x",
            )
            rollouts = [[single("a", 1.0), non_prefix] for _ in range(dp)]
            expected_microbatches = 2
        else:  # oversampling: ratio = global_batch_size/(prompts*group) = 2*dp/(dp*4) = 0.5.
            # 4*dp single-turn rows (already a multiple of 2*dp); ceil(0.5 * 4*dp) = 2*dp;
            # 2*dp / (2 * dp) = 1 microbatch.
            self.create_test_args(
                micro_batch_size=2,
                seq_length=4,
                curr_iteration=1,
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                global_batch_size=dp * 2,
                grpo_prompts_per_step=dp,
                grpo_group_size=4,
            )
            rollouts = [[single(str(i), float(i % 2)) for i in range(4)] for _ in range(dp)]
            expected_microbatches = 1

        rl_utils.prepare_data_for_update(
            [model], {}, rollouts, tokenizer, sequence_packing=False, is_correction=False
        )
        assert get_num_microbatches() == expected_microbatches

    def test_prepare_trajectories(self):
        """Every row kind becomes one padded training row in a single batch: a single-turn
        rollout (one contiguous generated region), a multi-turn rollout collapsed to its final
        sequence (multi-region mask with a gap at the observation, per-turn logprobs
        concatenated), the per-turn fallback rows of a NON-prefix rollout (each trained on its own
        tokens/mask), and an inert PAD_ROW."""
        seq_length = 8
        self.create_test_args(rl_skip_bos_token=False, micro_batch_size=1, seq_length=seq_length)
        tokenizer = MockTokenizer()
        eod, pad = tokenizer.eod, tokenizer.pad

        # Single-turn rollout: one contiguous generated region.
        single = make_token_rollout(
            [[1, 2, 3, eod]],
            [[0.1, 0.2, 0.3]],
            [[False, True, True, True]],
            reward=3.14,
            problem_id="1",
        )
        # Multi-turn prefix chain: turn 0 generates [2, 3, eod]; an observation (token 5) is
        # appended; turn 1 generates [6, eod]. token_ids are cumulative, so this collapses to the
        # final sequence with TWO generated regions (a gap at the observation) and per-turn
        # logprobs concatenated.
        multi = make_token_rollout(
            [[1, 2, 3, eod], [1, 2, 3, eod, 5, 6, eod]],
            [[0.4, 0.5, 0.6], [0.7, 0.75]],
            [[False, True, True, True], [False, False, False, False, False, True, True]],
            reward=0.14,
            problem_id="2",
        )
        # Non-prefix rollout: turn 1 is not a prefix-extension of turn 0, so it is trained as
        # per-turn fallback rows (rollout, turn_idx) -- each on its own tokens/mask.
        non_prefix = make_token_rollout(
            [[1, 2, 3, eod], [7, 8, 9, eod]],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[False, True, True, True], [False, True, True, True]],
            problem_id="3",
        )

        rows = [single, multi, (non_prefix, 0), (non_prefix, 1), rl_utils.PAD_ROW]
        trajs, genmask, inference_logprobs = rl_utils.prepare_trajectories(
            rows, tokenizer, seq_length, skip_bos_token=False
        )

        expected_trajs = torch.tensor(
            [
                [1, 2, 3, eod] + [pad] * 4,
                [1, 2, 3, eod, 5, 6, eod] + [pad] * 1,
                [1, 2, 3, eod] + [pad] * 4,
                [7, 8, 9, eod] + [pad] * 4,
                [pad] * 8,
            ],
            dtype=torch.long,
            device=trajs.device,
        )
        assert torch.equal(trajs, expected_trajs)

        expected_genmask = torch.tensor(
            [
                [False, True, True, True, False, False, False, False],
                # union of both turns' generated spans, with a gap at the observation (pos 4)
                [False, True, True, True, False, True, True, False],
                [False, True, True, True, False, False, False, False],
                [False, True, True, True, False, False, False, False],
                [False] * 8,  # inert PAD_ROW
            ],
            dtype=torch.bool,
            device=genmask.device,
        )
        assert torch.equal(genmask, expected_genmask)

        # Per-row list: unpadded tensor per real row, None for the pad row. (Packing-mode
        # densification happens at the call site via _pad_nonnull_with_zeros, tested separately.)
        expected_logprobs = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7, 0.75],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            None,
        ]
        assert len(inference_logprobs) == len(expected_logprobs)
        for got, exp in zip(inference_logprobs, expected_logprobs):
            if exp is None:
                assert got is None
            else:
                exp_t = torch.tensor(exp, dtype=torch.float32, device=got.device)
                torch.testing.assert_close(got, exp_t, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "trajectory, collapsible",
        [
            pytest.param([[1, 2, 3]], True, id="single_turn"),
            pytest.param([[1, 2], [1, 2, 3, 4]], True, id="prefix_chain"),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 8]], False, id="non_prefix"),
            pytest.param([[1, 2, 3], [1, 2]], False, id="later_turn_shorter"),
        ],
    )
    def test_is_collapsible_rollout(self, trajectory, collapsible):
        """A rollout collapses to one combined row only when each turn's token_ids is an exact
        prefix of the next; otherwise it must fall back to one row per turn."""
        rollout = make_token_rollout(
            trajectory, [[0.0] for _ in trajectory], [[False] * len(t) for t in trajectory]
        )
        assert rl_utils._is_collapsible_rollout(rollout) is collapsible

    @pytest.mark.parametrize(
        "rewards, expected",
        [
            pytest.param(
                [[-1, 1], [4, 4]], [-1.0, 1.0, 0.0, 0.0], id="normalized_and_zero_variance"
            ),
            pytest.param([[2, 2, 2]], [0.0, 0.0, 0.0], id="all_equal"),
        ],
    )
    def test_calculate_grpo_advantages(self, rewards, expected):
        """One group-normalized advantage per rollout (independent of turn counts), flattened
        group-major; a zero-variance group yields zero advantages."""
        advs = rl_utils.calculate_grpo_advantages(rewards)
        torch.testing.assert_close(torch.tensor(advs), torch.tensor(expected), atol=1e-4, rtol=1e-5)

    @pytest.mark.parametrize(
        "scenario, expected_turn_lens, expected_traj_lens, expected_num_turns",
        [
            pytest.param("single_turn_only", [[4, 3]], [[4, 3]], [[1, 1]], id="single_turn_only"),
            pytest.param(
                "multi_and_single", [[4, 3, 4]], [[7, 4]], [[2, 1]], id="multi_and_single"
            ),
        ],
    )
    def test_compute_group_stats(
        self, scenario, expected_turn_lens, expected_traj_lens, expected_num_turns
    ):
        """Length metrics: single-turn rollouts use the plain per-turn length, while a multi-turn
        TokenRollout re-encodes the prior conversation, so its per-turn lengths are reported
        incrementally and its trajectory length is the final conversation length (not the inflated
        overlap sum)."""
        tokenizer = MockTokenizer()
        eod = tokenizer.eod

        def single(traj, reward):
            return make_token_rollout(
                [traj], [[0.0]], [[False] * len(traj)], reward=reward, problem_id="s"
            )

        if scenario == "single_turn_only":
            group = [single([1, 2, 3, eod], 1.0), single([1, 2, eod], 0.0)]
        else:
            # Cumulative per-turn lengths 4 then 7 -> turn 1 adds 3 tokens; trajectory length is
            # the full conversation (7), not 4 + 7 = 11.
            multi = make_token_rollout(
                [[1, 2, 3, eod], [1, 2, 3, eod, 9, 8, eod]],
                [[0.1, 0.2], [0.3, 0.4]],
                [[False, False, True, True], [False, False, False, False, False, True, True]],
                problem_id="m",
            )
            group = [multi, single([1, 2, 3, eod], 0.0)]

        stats = rl_utils.compute_group_stats([group], tokenizer, seq_len=8)
        assert stats.turn_lens == expected_turn_lens
        assert stats.traj_lens == expected_traj_lens
        assert stats.num_turns == expected_num_turns

    @pytest.mark.parametrize(
        "lengths, max_len, expected_shape",
        [
            pytest.param([2, 3, 4], 5, (3, 5), id="normal"),
            pytest.param([5, 5], 5, (2, 5), id="full_size"),
            pytest.param([None, 5], 5, (2, 5), id="some_nones"),
            # All-None (all-PAD rank): still a zero [num_rows, max_len] tensor, so every DP rank
            # produces the same shape and joins the sequence-packing all_gather.
            pytest.param([None, None, None], 42, (3, 42), id="all_nones"),
            pytest.param([5], 4, "larger length", id="too_long_raises"),
        ],
    )
    def test_pad_nonnull_with_zeros(self, lengths, max_len, expected_shape):
        data = [None if l is None else torch.zeros(l) for l in lengths]
        if isinstance(expected_shape, str):
            with pytest.raises(ValueError, match=expected_shape):
                rl_utils._pad_nonnull_with_zeros(data, max_len)
            return
        padded = rl_utils._pad_nonnull_with_zeros(data, max_len)
        assert padded.shape == expected_shape
        assert (padded == 0).all()  # zero inputs and zero-filled padding/None rows

    def test_align_unpacked_inference_logprobs(self):
        """Scatter positions: a generated token at position p lands at index p-1, hopping
        observation gaps; a short logprob list leaves trailing generated slots at old_logprobs
        (train-side eod); all-False (PAD) rows are untouched and never dereference their
        (None) list entry."""
        width = 7
        # Distinct value per cell so any wrong-position write breaks exact equality.
        old = -0.5 - 0.01 * torch.arange(4 * width, dtype=torch.float32).reshape(4, width)
        # Masks are seq_length (width + 1) wide, as in the pipeline: exercises the stats clip.
        masks = torch.tensor(
            [
                [False, True, True, False, True, True, False, False],  # multi-region, gap at 3
                [False, True, True, True, False, False, False, False],  # contiguous, 3 gen slots
                [True, True, False, False, False, False, False, False],  # gen at 0: target -1
                [False] * 8,  # PAD row
            ]
        )
        logprobs = [
            torch.tensor([-0.11, -0.12, -0.21, -0.22]),  # two turns' logprobs, concatenated
            torch.tensor([-0.31, -0.32]),  # one short: the eod slot keeps old_logprobs
            torch.tensor([-0.41, -0.42]),  # first pairs with the dropped target -1
            None,  # PAD: must not be touched
        ]
        expected = old.clone()
        expected[0, [0, 1, 3, 4]] = torch.tensor([-0.11, -0.12, -0.21, -0.22])
        expected[1, [0, 1]] = torch.tensor([-0.31, -0.32])  # index 2 keeps old (ratio 1)
        expected[2, 0] = -0.42

        aligned = rl_utils.align_unpacked_inference_logprobs(
            logprobs, old, masks, SimpleNamespace()
        )
        torch.testing.assert_close(aligned, expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_grad_buffer_offload(self, initialize_model_parallel):
        """Test that grad buffer offload/restore correctly frees and restores GPU memory."""
        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)

        model_parallel_cuda_manual_seed(123)

        # Create a realistic GPTModel as used in RL training
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            bucket_size=None,  # Single bucket for simplicity
        )

        ddp_model = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=gpt_model
        )

        all_buffers = ddp_model.buffers + ddp_model.expert_parallel_buffers

        # Verify initial storage is allocated
        initial_sizes = [buf.grad_data.storage().size() for buf in all_buffers]
        assert all(size > 0 for size in initial_sizes), "Expected non-zero initial storage"

        # Offload grad buffers to CPU
        ddp_model.offload_grad_buffers()

        # Verify storage is released
        for buf in all_buffers:
            assert buf.grad_data.storage().size() == 0, "Expected zero storage after offload"

        # Restore grad buffers to GPU
        ddp_model.restore_grad_buffers()

        # Verify storage is restored
        restored_sizes = [buf.grad_data.storage().size() for buf in all_buffers]
        assert (
            initial_sizes == restored_sizes
        ), f"Expected restored sizes {restored_sizes} to match initial {initial_sizes}"

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_optimizer_offload(self, initialize_model_parallel):
        """Test that optimizer offload_to_cpu/restore_from_cpu correctly moves state to/from CPU."""
        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
        model_parallel_cuda_manual_seed(123)

        # Create a realistic GPTModel as used in RL training
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            bucket_size=None,  # Single bucket for simplicity
        )

        ddp_model = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=gpt_model
        )

        # Create optimizer
        optimizer_config = OptimizerConfig(
            optimizer='adam', bf16=True, use_distributed_optimizer=True
        )
        optimizer = get_megatron_optimizer(optimizer_config, [ddp_model])

        # Manually initialize optimizer state (simulating what happens after first step)
        # This avoids needing to run a full forward/backward/step cycle
        for opt in optimizer.chained_optimizers:
            if hasattr(opt, 'optimizer') and opt.optimizer is not None:
                for group in opt.optimizer.param_groups:
                    for p in group['params']:
                        if len(opt.optimizer.state[p]) == 0:
                            # Initialize Adam state (exp_avg and exp_avg_sq) on GPU
                            opt.optimizer.state[p]['exp_avg'] = torch.rand_like(p.data)
                            opt.optimizer.state[p]['exp_avg_sq'] = torch.rand_like(p.data)
                            opt.optimizer.state[p]['step'] = torch.tensor(1)

        # Helper to check if optimizer state tensors are on GPU or CPU
        def get_optimizer_state_devices():
            devices = set()
            for opt in optimizer.chained_optimizers:
                if hasattr(opt, 'optimizer') and opt.optimizer is not None:
                    for state_dict in opt.optimizer.state.values():
                        for v in state_dict.values():
                            if isinstance(v, torch.Tensor):
                                devices.add(str(v.device))
            return devices

        # Verify optimizer state is initially on GPU
        initial_devices = get_optimizer_state_devices()
        assert any(
            'cuda' in d for d in initial_devices
        ), f"Expected optimizer state on GPU initially, got devices: {initial_devices}"

        # Record GPU memory before offload
        torch.cuda.synchronize()
        memory_before_offload = torch.cuda.memory_allocated()

        # Offload optimizer state to CPU
        optimizer.offload_to_cpu()

        # Verify GPU memory decreased (optimizer state should be freed)
        torch.cuda.synchronize()
        memory_after_offload = torch.cuda.memory_allocated()
        assert memory_after_offload < memory_before_offload, (
            f"Expected GPU memory to decrease after offload. "
            f"Before: {memory_before_offload}, After: {memory_after_offload}"
        )

        # Verify optimizer state is now on CPU
        offloaded_devices = get_optimizer_state_devices()
        assert all(
            'cpu' in d for d in offloaded_devices
        ), f"Expected all optimizer state on CPU after offload, got devices: {offloaded_devices}"

        # Restore optimizer state to GPU
        optimizer.restore_from_cpu()

        # Verify optimizer state is back on GPU
        restored_devices = get_optimizer_state_devices()
        assert any(
            'cuda' in d for d in restored_devices
        ), f"Expected optimizer state on GPU after restore, got devices: {restored_devices}"

        # Verify GPU memory increased after restore (optimizer state reallocated)
        torch.cuda.synchronize()
        memory_after_restore = torch.cuda.memory_allocated()
        assert memory_after_restore > memory_after_offload, (
            f"Expected GPU memory to increase after restore. "
            f"After offload: {memory_after_offload}, After restore: {memory_after_restore}"
        )

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [
            pytest.param((tp, pp), id=f"tp{tp}-pp{pp}")
            for tp, pp in itertools.product([1, 2, 4], [1, 2])
            if tp * pp <= Utils.world_size
        ],
        indirect=["initialize_model_parallel"],
    )
    def test_gpt_logprobs(self, initialize_model_parallel):
        """Test get logprobs on an actual model, not on a mocked one.

        This can be useful for quick benchmarking/analyzing regressions too.
        """

        world_size, dp, tp, pp = initialize_model_parallel
        micro_batch_size = 1
        self.create_test_args(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            bf16=True,
            micro_batch_size=micro_batch_size,
        )
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=10,
            hidden_size=128,
            num_attention_heads=16,
            use_cpu_initialization=True,
            embedding_init_method_std=1.0,
            bf16=True,
            pipeline_dtype=torch.bfloat16,  # Without this, pp!=1 runs will fail.
        )
        vocab_size = 10_000
        pp_group = ProcessGroupCollection.use_mpu_process_groups().pp
        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=vocab_size,
            max_sequence_length=4192,
            pre_process=is_pp_first_stage(pp_group),
            post_process=is_pp_last_stage(pp_group),
        ).cuda()
        sequence_length = gpt_model.max_sequence_length

        gpt_model = Float16Module(gpt_model.config, gpt_model)

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()

        with torch.no_grad():
            logprobs = rl_utils.compute_logprobs_batch(
                model=gpt_model,
                data_loader=[(input_ids, position_ids)],
                forward_backward_func=get_forward_backward_func(pp_size=pp),
                packing_context=None,
                trajs_batch_size=micro_batch_size,
                seq_length=sequence_length,
                logprobs_batch_size=micro_batch_size,
                decoder_seq_length=sequence_length,
                dtype=torch.bfloat16,
                pp_group=gpt_model.pg_collection.pp,
                is_correction=False,
                collect_non_loss_data=True,
            )
        if is_pp_last_stage(pp_group):
            assert logprobs.shape == (micro_batch_size, sequence_length - 1)

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((1, 1), id="tp1-pp1")],
        indirect=["initialize_model_parallel"],
    )
    def test_get_logprobs_cuda_graphs(self, initialize_model_parallel):
        """Test that get_logprobs reuses CUDA graphs created during training forward pass.

        This test verifies that rl_utils.get_logprobs can reuse CUDA graphs by:
        1. Running a training-style forward pass on some model to record CUDA graph runners.
        2. Creating the CUDA graphs.
        3. Running `get_logprobs` to verify it reuses the same forward graph from training.
        """

        num_layers = 2

        world_size, dp, tp, pp = initialize_model_parallel
        self.create_test_args(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            rl_training_cuda_graphs=True,
            cuda_graph_impl="local",
            bf16=True,
            rl_sequence_packing_max_sequences_per_bin=4,
        )

        # Create a model with training CUDA graphs enabled
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            bf16=True,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=256,
            max_sequence_length=32,
        ).cuda()

        # Wrap in Float16Module so it accepts fp32_output argument from get_logprobs
        wrapped_model = Float16Module(transformer_config, model)
        # Cudagraph backward capture assumes the model has DDP so create main_grads for params
        for param in wrapped_model.parameters():
            param.main_grad = torch.zeros_like(param)

        # Create test inputs (batch_size=1 required for thd format with sequence packing)
        batch_size = 1
        seq_length = 16
        tokens = torch.randint(0, 256, (batch_size, seq_length), dtype=torch.long).cuda()
        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()

        # Create packed_seq_params for dummy data
        packed_seq_params = get_default_packed_seq_params(
            seq_length=seq_length, max_sequences_per_bin=4, device=tokens.device
        )

        # Run a single training forward pass to record cudagraphs
        output = wrapped_model(
            tokens,
            position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=True,
            fp32_output=False,
        )

        # Run backward to reset runner status from BWD_READY back to FWD_READY
        # This is needed because get_logprobs runs in no_grad mode and expects FWD_READY
        loss = output.sum()
        loss.backward()

        # Collect all CudaGraphManager instances and their runners
        cudagraph_managers = []
        for module in wrapped_model.modules():
            if hasattr(module, 'cudagraph_manager') and module.cudagraph_manager is not None:
                cudagraph_managers.append(module.cudagraph_manager)

        # Record runner count before creating graphs
        runners_before = {id(mgr): len(mgr.cudagraph_runners) for mgr in cudagraph_managers}

        create_cudagraphs()

        # Verify that each runner has a fwd_graph created
        for mgr in cudagraph_managers:
            for runner in mgr.cudagraph_runners:
                assert runner.fwd_graph is not None, (
                    f"Expected runner to have fwd_graph created after create_cudagraphs(), "
                    f"but fwd_graph is None"
                )

        # Now test `get_logprobs`; this should reuse the existing CUDA graphs
        # We do not pass packed_seq_params; it should be created within `get_logprobs`
        logprobs = rl_utils.get_logprobs(
            wrapped_model, tokens, position_ids=position_ids, sequence_packing=True
        )

        # Verify that no new runners were created and graph was reused
        runners_after = {id(mgr): len(mgr.cudagraph_runners) for mgr in cudagraph_managers}
        for mgr_id, count_before in runners_before.items():
            count_after = runners_after[mgr_id]
            assert count_after == count_before, (
                f"Expected runner count to remain {count_before} after `get_logprobs`, "
                f"but got {count_after}. `get_logprobs` should not create new runners."
            )

        # Verify outputs are valid
        assert output is not None, "Training forward pass should return valid output"
        assert logprobs is not None, "get_logprobs should return valid output"

        # Destroy all captured graphs deterministically
        for l in model.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

        # Ensure all pending work is complete and graph destruction runs now
        torch.cuda.synchronize()

        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None
        CudaGraphManager.fwd_mempools = None
        CudaGraphManager.bwd_mempools = None

    @pytest.mark.parametrize(
        "initialize_model_parallel",
        [pytest.param((1, 1), id="tp1-pp1")],
        indirect=["initialize_model_parallel"],
    )
    def test_prep_wandb_metrics(self, initialize_model_parallel):
        # This tests the computation and makes us fail noisily if
        # inputs assumptions are changed, e.g. we expect rewards to come in groups (list[list[int]]).
        traj_lens = [[3, 3], [1, 2]]
        turn_lens = [[1, 2, 1, 1, 1], [1, 2]]
        rewards = [[1, 1], [-1, 2]]
        num_turns = [[42, 2], [10, 8]]
        advantages = [0, 1]
        # Per-token epoch stamps, grouped by group then rollout
        policy_epoch = [[[4, 5], [2, 3]], [[5], [0, 1]]]
        kv_cache_epoch = [[[4, 5], [3, 4]], [[5], [1, 2]]]
        # Per-turn max epoch stamps (when each turn completed)
        completed_epochs = [[5, 3], [5, 1]]
        num_evictions = [[0, 1], [0, 0]]
        current_iteration = 6
        metrics = rl_utils.prep_wandb_metrics(
            MagicMock(),
            traj_lens,
            turn_lens,
            rewards,
            num_turns,
            advantages,
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
            completed_epochs=completed_epochs,
            num_evictions=num_evictions,
            current_iteration=current_iteration,
        )
        assert metrics["mean_reward"] == 0.75
        assert metrics["mean_advantage"] == 0.5
        assert metrics["nonzero_groups_ratio"] == 0.5
        assert metrics["max_traj_length"] == 3
        assert metrics["min_traj_length"] == 1
        assert metrics["mean_traj_length"] == 2.25
        assert metrics["mean_traj_length_std"] == 0.25
        assert metrics["max_turn_length"] == 2
        assert metrics["min_turn_length"] == 1
        assert metrics["mean_turn_length"] == 1.35
        assert np.isclose(metrics["mean_turn_length_std"], 0.45)
        assert metrics["mean_num_turns"] == 15.5
        assert metrics["max_num_turns"] == 42
        assert metrics["min_num_turns"] == 2
        # true_policy_staleness = [6-4, 6-2, 6-5, 6-0] = [2, 4, 1, 6] -> mean=3.25, max=6, min=1
        assert metrics["mean_policy_staleness"] == np.mean([2, 4, 1, 6])
        assert metrics["max_policy_staleness"] == 6
        assert metrics["min_policy_staleness"] == 1
        # true_kv_staleness = [6-4, 6-3, 6-5, 6-1] = [2, 3, 1, 5] -> mean=2.75, max=5, min=1
        assert metrics["mean_kv_cache_staleness"] == np.mean([2, 3, 1, 5])
        assert metrics["max_kv_cache_staleness"] == 5
        assert metrics["min_kv_cache_staleness"] == 1
        # last_token (max epoch per rollout): policy=[5, 3, 5, 1] -> staleness=[1, 3, 1, 5]
        assert metrics["mean_policy_last_token_staleness"] == np.mean([1, 3, 1, 5])
        assert metrics["max_policy_last_token_staleness"] == 5
        assert metrics["min_policy_last_token_staleness"] == 1
        # last_token (max epoch per rollout): kv=[5, 4, 5, 2] -> staleness=[1, 2, 1, 4]
        assert metrics["mean_kv_cache_last_token_staleness"] == np.mean([1, 2, 1, 4])
        assert metrics["max_kv_cache_last_token_staleness"] == 4
        assert metrics["min_kv_cache_last_token_staleness"] == 1
        assert metrics["total_eviction_count"] == 1
        assert metrics["max_num_evictions"] == 1
        # mean_completion_gap = mean([6-5, 6-3, 6-5, 6-1]) = mean([1, 3, 1, 5]) = 2.5
        assert metrics["mean_completion_gap"] == 2.5
