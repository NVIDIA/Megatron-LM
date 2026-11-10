# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from tools.pipeline_layout_planner import layout as layouts
from tools.pipeline_layout_planner.__main__ import main
from tools.pipeline_layout_planner.cost_model import CostModel, ProfilingRun, fit
from tools.pipeline_layout_planner.model import ModelShape, shape_from_megatron_args
from tools.pipeline_layout_planner.planner import plan, profiling_layouts, score
from tools.pipeline_layout_planner.schedule import simulate_1f1b
from tools.pipeline_layout_planner.timings import parse_log, summarize

# 7B-A1B MoE recipe with one MTP layer on four pipeline stages, one micro-batch of 4096 tokens.
SHAPE = ModelShape(
    num_layers=16,
    hidden_size=2048,
    ffn_hidden_size=8192,
    num_experts=32,
    moe_ffn_hidden_size=1024,
    moe_router_topk=8,
    num_attention_heads=32,
    num_query_groups=4,
    kv_channels=64,
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=32,
    mtp_num_layers=1,
    full_recompute=True,
)

# Known unit costs (ms per micro-batch) used to synthesize timings.
TRUTH = CostModel(
    forward_ms={"E": 1.5, "t": 4.0, "m": 19.0, "L": 15.0, "c0": 0.8},
    backward_ms={"E": 0.5, "t": 11.0, "m": 41.0, "L": 30.0, "c0": 0.4},
    overhead_ms=500.0,
    p2p_ms=0.2,
)


def _true_step_ms(stages):
    forward, backward = zip(*(TRUTH.stage_ms(stage) for stage in stages))
    result = simulate_1f1b(forward, backward, SHAPE.num_microbatches, TRUTH.p2p_ms)
    return result.makespan_ms + TRUTH.overhead_ms, result.makespan_ms


def _profiling_run(layout_str):
    stages = layouts.parse(layout_str)
    step_ms, forward_backward_ms = _true_step_ms(stages)
    per_stage = [TRUTH.stage_ms(stage) for stage in stages]
    mb = SHAPE.num_microbatches
    return ProfilingRun(
        stages=stages,
        num_microbatches=mb,
        forward_ms=[f * mb for f, _ in per_stage],
        backward_ms=[b * mb for _, b in per_stage],
        step_ms=step_ms,
        forward_backward_ms=forward_backward_ms,
    )


class TestPipelineLayoutPlanner:
    def test_baseline_layouts(self):
        assert layouts.to_string(layouts.megatron_default(16, 4, 1)) == "Et*4|t*4|t*4|t*4mL"
        assert layouts.to_string(layouts.count_based(16, 4, 1)) == "Et*4|t*5|t*5|t*2mL"
        # Six decoder layers on four stages: decoder counts [1, 2, 2, 1].
        assert layouts.to_string(layouts.count_based(6, 4, 0)) == "Et|t*2|t*2|tL"
        stages = layouts.parse("Et*3|(tt|)*29,m|L")
        assert len(stages) == 32 and sum(s.count("t") for s in stages) == 61

    def test_enumerated_layouts_pass_megatron_validation(self):
        found = [layouts.to_string(s) for s in layouts.enumerate_layouts(16, 4, 1)]
        # 19 units with MTP kept whole, split into 4 non-empty stages: C(18, 3) layouts.
        assert len(found) == 816
        assert "Et*4|t*4|t*4|t*4mL" in found and "Et*7|t*5|t*4m|L" in found
        assert not layouts.is_valid(layouts.parse("Et*4|t*4|t*4|t*4Lm"), 4, 16, 1)

    @pytest.mark.parametrize("pp_size, num_microbatches", [(1, 8), (4, 32), (8, 16)])
    def test_1f1b_matches_closed_form_for_equal_stages(self, pp_size, num_microbatches):
        result = simulate_1f1b([1.0] * pp_size, [2.0] * pp_size, num_microbatches)
        assert result.makespan_ms == pytest.approx((num_microbatches + pp_size - 1) * 3.0)

    def test_fit_recovers_unit_costs_and_predicts_unseen_layout(self):
        runs = [_profiling_run(s) for s in profiling_layouts(16, 4, 1)]
        model = fit(SHAPE, runs, prior_weight=0.01)
        for direction in ("forward_ms", "backward_ms"):
            for unit in ("t", "m", "L"):
                expected = getattr(TRUTH, direction)[unit]
                assert getattr(model, direction)[unit] == pytest.approx(expected, rel=0.03)
        unseen = layouts.parse("Et*5|t*6|t*5|mL")
        expected_step, _ = _true_step_ms(unseen)
        assert score(SHAPE, model, unseen).step_ms == pytest.approx(expected_step, rel=0.02)

    def test_plan_matches_exhaustive_search_and_beats_baselines(self):
        result = plan(SHAPE, TRUTH, pp_size=4, hbm_gb=80)
        best = min(
            score(SHAPE, TRUTH, s).step_ms
            for s in layouts.enumerate_layouts(16, 4, 1)
            if not any("m" in stage and "L" not in stage for stage in s)
        )
        assert result.top[0].step_ms == pytest.approx(best)
        assert result.baselines["megatron_default"].step_ms > 1.2 * result.top[0].step_ms
        assert result.baselines["count_based"].step_ms > result.top[0].step_ms
        assert all(max(c.stage_memory_gb) <= 80 for c in result.top)

    def test_parse_log_reads_steps_timers_and_memory(self):
        log = parse_log(
            [
                " [2026-11-09 10:00:00] iteration        6/   54164 | consumed samples: 192 |"
                " elapsed time per iteration (ms): 4321.0 | lm loss: 7.5E+00 |",
                "[Rank 3] (after 6 iterations) memory (MB) | allocated: 1.0 |"
                " max allocated: 37000.0 | reserved: 1.0 | max reserved: 2.0",
                "times across ranks (ms):",
                "  forward-compute:",
                "     rank  0: 1000.00",
                "     rank  3: 2500.00",
                "  backward-compute:",
                "     rank  0: 2000.00",
                " [2026-11-09 10:00:05] iteration        7/   54164 |"
                " elapsed time per iteration (ms): 4300.0 |",
            ]
        )
        assert log.step_ms == {6: 4321.0, 7: 4300.0}
        assert log.timers[6]["forward-compute"] == {0: 1000.0, 3: 2500.0}
        assert log.max_allocated_mb == {3: 37000.0}
        summary = summarize(log, warmup=5)
        assert summary.num_iterations == 2 and summary.timers["backward-compute"] == {0: 2000.0}

    def test_shape_matches_handbook_parameter_counts(self):
        shape = shape_from_megatron_args(
            "--num-layers 16 --hidden-size 2048 --ffn-hidden-size 8192 --num-experts 64 "
            "--moe-ffn-hidden-size 1024 --moe-router-topk 8 --num-attention-heads 32 "
            "--num-query-groups 4 --kv-channels 64 --seq-length 4096 "
            "--recompute-granularity full"
        )
        assert shape.full_recompute and shape.num_query_groups == 4
        # The MoE handbook's 7B-A1B model: about 6.86B total and 1.22B active parameters.
        assert shape.total_params() / 1e9 == pytest.approx(6.86, abs=0.01)
        assert shape.active_params() / 1e9 == pytest.approx(1.221, abs=0.002)

    def test_cli_plan_writes_json(self, tmp_path, capsys):
        shape_path = tmp_path / "shape.json"
        shape_path.write_text(SHAPE.to_json())
        main(["plan", "--shape", str(shape_path), "--pp-size", "4", "--tflops", "150"])
        out = json.loads(capsys.readouterr().out)
        assert out["top"][0]["layout"].endswith("|mL")
        assert set(out["baselines"]) == {"megatron_default", "count_based"}
