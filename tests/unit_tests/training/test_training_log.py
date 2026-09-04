from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.training.training import training_log


def _make_args(log_interval: int):
    return SimpleNamespace(
        log_interval=log_interval,
        timing_log_level=0,
        perform_rl_step=False,
        rl_use_sequence_packing=False,
        rl_profile=False,
        num_experts=None,
        mtp_num_layers=None,
        dsa_indexer_loss_coeff=None,
        log_throughput=False,
        log_energy=False,
        log_timers_to_tensorboard=False,
        record_memory_history=False,
        log_memory_interval=None,
        tensorboard_log_interval=1,
        train_iters=100,
        consumed_train_samples=0,
        skipped_train_samples=0,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=1024,
        freeze_all_layers=False,
        gtp_weight_remat_size=1,
        profile_ranks=[],
        hybrid_layer_pattern=None,
        group_query_attention=False,
        num_attention_heads=1,
        num_query_groups=None,
        hidden_size=1,
        num_layers=1,
        ffn_hidden_size=4,
        kv_channels=1,
        moe_ffn_hidden_size=None,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=None,
        swiglu=False,
        padded_vocab_size=1,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="learned_absolute",
        moe_router_topk=1,
        expert_model_parallel_size=1,
        multi_latent_attention=False,
        attention_output_gate=False,
        experimental_attention_variant=None,
    )


@pytest.mark.parametrize(
    ("log_interval", "is_first_iteration", "expected_reset"),
    [(1, True, True), (1, False, True), (10, True, False), (10, False, True)],
)
def test_training_log_resets_completed_logging_window(
    monkeypatch, log_interval, is_first_iteration, expected_reset
):
    args = _make_args(log_interval)

    timer = MagicMock()
    timer.elapsed.return_value = 1.0

    timers = MagicMock()
    timers.return_value = timer

    monkeypatch.setattr("megatron.training.training.get_args", lambda: args)
    monkeypatch.setattr("megatron.training.training.get_timers", lambda: timers)
    monkeypatch.setattr("megatron.training.training.get_tensorboard_writer", lambda: None)
    monkeypatch.setattr("megatron.training.training.get_wandb_writer", lambda: None)
    monkeypatch.setattr("megatron.training.training.get_one_logger", lambda: None)
    monkeypatch.setattr("megatron.training.training.get_energy_monitor", lambda: MagicMock())
    monkeypatch.setattr(
        "megatron.training.training.reduce_max_stat_across_model_parallel_group",
        lambda value, group=None: value,
    )
    monkeypatch.setattr("megatron.training.training.get_num_microbatches", lambda: 1)
    monkeypatch.setattr("megatron.training.training.print_rank_last", lambda _: None)
    monkeypatch.setattr("megatron.training.training.one_logger_utils", MagicMock())

    total_loss_dict = {}

    training_log(
        loss_dict={},
        total_loss_dict=total_loss_dict,
        learning_rate=0.001,
        iteration=1 if is_first_iteration else log_interval,
        loss_scale=1.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=None,
        params_norm=None,
        num_zeros_in_grad=None,
        max_attention_logit=None,
        is_first_iteration=is_first_iteration,
    )

    timer.elapsed.assert_called_once_with(barrier=True, reset=expected_reset)

    expected_count = 0 if expected_reset else 1
    assert total_loss_dict["advanced iterations"] == expected_count
