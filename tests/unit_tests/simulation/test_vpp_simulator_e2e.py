# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-node E2E smoke tests for VPP simulation."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.simulation.model_executor import get_pg_collection_for_simulation
from megatron.training.simulation.vpp_simulate import VppSimulator
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def cleanup_parallel_state():
    yield
    destroy_num_microbatches_calculator()
    Utils.destroy_model_parallel()


def _build_args(tmp_path):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    return SimpleNamespace(
        rank=rank,
        world_size=world_size,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=PipelineParallelLayerLayout(
            layout="Et|tL",
            pipeline_model_parallel_size=2,
        ),
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        ffn_hidden_size=64,
        seq_length=8,
        vocab_size=32,
        micro_batch_size=1,
        global_batch_size=world_size,
        data_parallel_size=world_size,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        execute_mode="router_balanced",
        skip_execute=False,
        simulate_result_dir=str(tmp_path / "simulation_results"),
        simulation_warmup_times=1,
        simulation_measure_times=1,
        simulation_measure_skip_times=0,
        microbatch_group_size_per_vp_stage=None,
        moe_layer_freq=[0, 0],
    )


def _build_gpt_model(args, pp_rank):
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        pipeline_dtype=torch.float32,
        context_parallel_size=1,
        pipeline_model_parallel_layout=args.pipeline_model_parallel_layout,
        sequence_parallel=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.local,
        use_cpu_initialization=True,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_length,
        pre_process=pp_rank == 0,
        post_process=pp_rank == args.pipeline_model_parallel_size - 1,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="learned_absolute",
        pg_collection=get_pg_collection_for_simulation(
            pp_rank=pp_rank,
            pp_size=args.pipeline_model_parallel_size,
        ),
    ).cuda()


def _build_gpt_batch(args):
    device = torch.cuda.current_device()
    tokens = torch.arange(args.seq_length, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = tokens.repeat(args.micro_batch_size, 1) % args.vocab_size
    labels = (input_ids + 1) % args.vocab_size
    position_ids = tokens.repeat(args.micro_batch_size, 1)
    attention_mask = torch.ones(
        args.micro_batch_size,
        1,
        args.seq_length,
        args.seq_length,
        dtype=torch.bool,
        device=device,
    )
    return input_ids, labels, position_ids, attention_mask


def test_single_node_simulation_e2e_writes_results_and_reports_virtual_world_size(
    tmp_path, monkeypatch, capsys
):
    if not torch.cuda.is_available():
        pytest.skip("VPP simulation E2E test requires CUDA")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    args = _build_args(tmp_path)
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    set_args(args)
    init_num_microbatches_calculator(
        rank=args.rank,
        rampup_batch_size=None,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=args.data_parallel_size,
    )

    model_parallel_cuda_manual_seed(123)

    def fake_create_model(pp_rank, model_provider):
        parallel_state.set_pipeline_model_parallel_rank(pp_rank)
        parallel_state.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
        return [_build_gpt_model(args, pp_rank)]

    def gpt_forward_step(data_iterator, model):
        input_ids, labels, position_ids, attention_mask = _build_gpt_batch(args)
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        def loss_func(output_tensor, non_loss_data=False):
            loss = output_tensor.float().mean()
            if non_loss_data:
                return {"loss": loss.detach()}
            return loss, {"loss": loss.detach()}

        return output, loss_func

    monkeypatch.setattr(
        "megatron.training.simulation.vpp_simulate.create_model",
        fake_create_model,
    )
    monkeypatch.setattr(
        "megatron.training.training.num_floating_point_operations",
        lambda parsed_args, batch_size: 1000000000000,
    )

    simulator = VppSimulator(
        train_data_iterator=iter(()),
        model_provider=lambda **_: _build_gpt_model(args, 0),
        forward_step_func=gpt_forward_step,
    )
    simulator.run_global_step()

    if args.rank != 0:
        return

    result_dir = Path(args.simulate_result_dir)
    finished_tasks = json.loads((result_dir / "finished_tasks.json").read_text())
    task_durations = json.loads((result_dir / "task_durations.json").read_text())

    assert set(finished_tasks) == {"0", "1"}
    num_microbatches = args.global_batch_size // (
        args.micro_batch_size * args.data_parallel_size
    )
    max_task_count = args.pipeline_model_parallel_size * num_microbatches * 2
    assert 0 < len(task_durations) <= max_task_count
    assert all(task["duration_ms"] > 0 for task in task_durations.values())
    assert {task["task_type"] for task in task_durations.values()} == {
        "forward",
        "backward",
    }
    assert {task["pp_rank"] for task in task_durations.values()} == {
        0,
        1,
    }

    output = capsys.readouterr().out
    assert "GBS Execution Statistics" in output
    assert f"Executor World Size: {args.world_size}" in output
    assert (
        f"VTrainer World Size: {args.world_size * args.pipeline_model_parallel_size}"
        in output
    )
    assert "Pipeline Parallel Size: 2" in output

    throughput_match = re.search(r"Throughput: ([0-9.]+) TFLOPS per GPU", output)
    executor_throughput_match = re.search(
        r"Executor-normalized sampled throughput: ([0-9.]+) TFLOPS per executor GPU",
        output,
    )
    assert throughput_match is not None
    assert executor_throughput_match is not None
    throughput = float(throughput_match.group(1))
    executor_throughput = float(executor_throughput_match.group(1))
    assert executor_throughput == pytest.approx(
        throughput * args.pipeline_model_parallel_size,
        rel=1e-3,
    )
