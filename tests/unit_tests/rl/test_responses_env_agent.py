# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest

from megatron.rl import GenericGenerationArgs
from megatron.rl.agent.api import TokenRollout
from megatron.rl.agent.registry import get_agent_class
from megatron.rl.agent.responses_env_agent import (
    EnvConnectExhausted,
    EnvRunRequest,
    ResponsesEnvAgent,
    RunResult,
    generation_args_to_run_payload,
    next_curriculum_index,
    resolve_curriculum_training_state,
    run_result_to_evaluation_response,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask


def _write_dataset(tmp_path, text=None):
    rows = (
        {"responses_create_params": {"input": f"q {i}"}, "problem_id": f"p{i}"} for i in range(4)
    )
    path = tmp_path / "dataset.jsonl"
    path.write_text((text if text is not None else "\n".join(map(json.dumps, rows))) + "\n")
    return str(path)


def _agent(tmp_path, **kwargs):
    kwargs.setdefault("agent_name", "test_env")
    if "dataset_file" not in kwargs:
        kwargs["dataset_file"] = _write_dataset(tmp_path)
    kwargs.setdefault("env_server_host_port", "envhost:11000")
    return ResponsesEnvAgent(**kwargs)


def _turn(prompt_ids, generation_ids, **extra):
    ids = {"prompt_token_ids": prompt_ids, "generation_token_ids": generation_ids}
    return {"type": "message", "generation_log_probs": [-0.1] * len(generation_ids), **ids, **extra}


def _run_result(output, reward=1.0, **extra):
    return RunResult.model_validate({"reward": reward, "response": {"output": output}, **extra})


def _install_transport(agent, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    object.__setattr__(agent, "_client", client)


def _request(**generation_args):
    return SimpleNamespace(
        validation=True, generation_args=GenericGenerationArgs(**generation_args)
    )


_GOOD_BODY = {"reward": 0.5, "response": {"output": [_turn([1], [2])]}}
_UNTOKENIZED = {"type": "message", "content": "no token metadata"}
_CAP100 = {"max_output_tokens_per_step": 100}


def _mo(value):
    return {"max_output_tokens": value}


_EVAL_OUTPUT = [
    {"type": "message", "role": "assistant", "content": "part one"},
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "part two"}],
    },
    {"type": "message", "role": "user", "content": "ignored"},
]


class TestResponsesEnvAgent:

    # (output items, result extras, agent kwargs, expected rollout attributes)
    @pytest.mark.parametrize(
        "output, result_extra, agent_kwargs, expected",
        [
            (
                [_turn([1, 2], [3]), {"type": "function_call_output"}, _turn([1, 2, 3, 4], [5, 6])],
                {},
                {},
                dict(
                    trajectory=[[1, 2, 3], [1, 2, 3, 4, 5, 6]],
                    generation_mask=[[False] * 2 + [True], [False] * 4 + [True] * 2],
                    logprobs=[[-0.1], [-0.1] * 2],
                    reward=1.0,
                    env_id="test_env",
                    problem_id="p0",
                    rollout_status="ok",
                    completion_ids=[],
                ),
            ),
            (
                [_turn([1], [2], completion_id="c-0"), _turn([1, 2], [3], completion_id="c-1")],
                {},
                {},
                dict(completion_ids=["c-0", "c-1"]),
            ),
            (
                [_turn([1], [2], completion_id="c-0"), _turn([1, 2], [3])],
                {},
                {},
                dict(completion_ids=[]),
            ),
            (
                [_UNTOKENIZED],
                {},
                {},
                dict(
                    trajectory=[],
                    reward=0.0,
                    problem_id="placeholder",
                    rollout_status="placeholder",
                    failure_reason="no_trainable_output",
                ),
            ),
            (
                [_turn([1], [2])],
                {"mask_sample": True},
                {},
                dict(trajectory=[], rollout_status="masked", failure_reason="env_masked"),
            ),
            (
                [_turn([1], [2])],
                {"mask_sample": True, "failure_reason": "agent_timeout"},
                {},
                dict(rollout_status="masked", failure_reason="agent_timeout"),
            ),
            (
                [_turn([1], [2])],
                {"instance_config": {"mask_sample": True}},
                {},
                dict(rollout_status="masked", failure_reason="env_masked"),
            ),
            (
                [_turn([1], [2])],
                {"final_eval_time": 12.5},
                {"max_output_tokens_per_step": 128},
                dict(rollout_status="graded", generation_cap=128),
            ),
        ],
        ids=[
            "multi-turn-rows-skip-untokenized",
            "completion-ids-stamped-when-complete",
            "partial-completion-ids-dropped",
            "no-trainable-output-pads-placeholder",
            "masked-direct-field",
            "masked-keeps-server-reason",
            "masked-via-instance-config",
            "graded-with-generation-cap",
        ],
    )
    def test_build_token_rollout(self, tmp_path, output, result_extra, agent_kwargs, expected):
        agent = _agent(tmp_path, **agent_kwargs)
        rollout = agent.build_token_rollout(_run_result(output, **result_extra), "p0")
        for attr, value in expected.items():
            assert getattr(rollout, attr) == value, attr

    def test_token_rollout_placeholder(self):
        rollout = TokenRollout.placeholder("e", failure_reason="http_500", generation_cap=7)
        assert (rollout.trajectory, rollout.generation_mask, rollout.reward) == ([], [], 0.0)
        assert (rollout.problem_id, rollout.rollout_status) == ("placeholder", "placeholder")
        assert (rollout.failure_reason, rollout.generation_cap) == ("http_500", 7)

    # (generation args, agent kwargs, pre-existing params, expected params subset, warns);
    # _CAP100 caps the agent at 100 output tokens per step, _mo(v) = {"max_output_tokens": v}.
    @pytest.mark.parametrize(
        "ga, agent_kwargs, existing, expected, warns",
        [
            ({"temperature": 0.7, "top_p": 0.9}, {}, {}, {"temperature": 0.7, "top_p": 0.9}, False),
            ({"top_k": 5}, {}, {}, {}, True),
            ({}, _CAP100, {}, _mo(100), False),
            ({}, _CAP100, _mo(50), _mo(50), False),
            ({}, _CAP100, _mo(300), _mo(100), False),
            ({}, {"defer_output_cap_to_engine": True}, _mo(300), _mo(None), False),
            ({}, {}, _mo(300), _mo(300), False),
        ],
        ids=[
            "generation-args-merged",
            "top-k-warns",
            "cap-fills-empty",
            "cap-keeps-smaller",
            "cap-wins-over-larger",
            "defer-nulls-the-cap",
            "no-cap-passthrough",
        ],
    )
    def test_run_payload_and_output_caps(
        self, tmp_path, ga, agent_kwargs, existing, expected, warns
    ):
        agent = _agent(tmp_path, **agent_kwargs)
        row = {"responses_create_params": dict(existing), "problem_id": "p0"}
        with pytest.warns(UserWarning, match="top_k") if warns else nullcontext():
            payload = agent._stamp_output_caps(
                generation_args_to_run_payload(GenericGenerationArgs(**ga), row)
            )
        params = payload["responses_create_params"]
        for key, value in expected.items():
            assert key in params and params[key] == value, key
        # The dataset row itself is never mutated.
        assert row["responses_create_params"] == existing

    @pytest.mark.parametrize(
        "cursor, share, expected",
        [(None, 1.0, (48, 49)), (50, 1.0, (50, 51)), (None, 0.25, (12, 13)), (17, 0.25, (17, 18))],
        ids=[
            "restart-seed",
            "monotonic-never-rewinds",
            "seed-scales-with-share",
            "monotonic-with-share",
        ],
    )
    def test_next_curriculum_index(self, cursor, share, expected):
        assert (
            next_curriculum_index(
                cursor, collections=4, prompts_per_collection=12, prompt_share=share
            )
            == expected
        )

    # (args attributes, expected (collections, prompts_per_collection); None = raises)
    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"curr_iteration": 3, "grpo_prompts_per_step": 8}, (3, 8)),
            ({"iteration": 5}, (5, 64)),
            (
                {
                    "curr_iteration": 8,
                    "grpo_prompts_per_step": 4,
                    "grpo_group_size": 16,
                    "global_batch_size": 32,
                },
                (4, 4),
            ),
            (
                {
                    "curr_iteration": 8,
                    "grpo_prompts_per_step": 4,
                    "grpo_group_size": 16,
                    "global_batch_size": 32,
                    "grpo_iterations": 2,
                },
                (2, 4),
            ),
            ({"curr_iteration": 1, "grpo_prompts_per_step": 0}, None),
        ],
        ids=[
            "cadence-one-passthrough",
            "iteration-fallback-and-default-prompts",
            "batches-per-collection-cadence",
            "grpo-iterations-cadence",
            "invalid-prompts-raises",
        ],
    )
    def test_resolve_curriculum_training_state(self, attrs, expected):
        args = SimpleNamespace(**attrs)
        if expected is None:
            with pytest.raises(ValueError, match="must be positive"):
                resolve_curriculum_training_state(args)
        else:
            assert resolve_curriculum_training_state(args) == expected

    @pytest.mark.asyncio
    async def test_curriculum_get_prompt(self, tmp_path, monkeypatch):
        agent = _agent(tmp_path, curriculum_sampling=True)
        state = SimpleNamespace(curr_iteration=0, grpo_prompts_per_step=2)
        monkeypatch.setattr("megatron.training.global_vars.get_args", lambda: state)
        rows = [await agent.get_prompt(validation=False) for _ in range(3)]
        assert [row["problem_id"] for row in rows] == ["p0", "p1", "p2"]

        # A restart at a later iteration re-seeds the cursor from training progress.
        agent2 = _agent(tmp_path, curriculum_sampling=True)
        state.curr_iteration = 1
        assert (await agent2.get_prompt(validation=False))["problem_id"] == "p2"

        # Validation and missing training state fall back to uniform, cursor untouched.
        assert await agent2.get_prompt(validation=True) in agent2.dataset

        def _no_training_state():
            raise RuntimeError("no args")

        monkeypatch.setattr("megatron.training.global_vars.get_args", _no_training_state)
        assert await agent2.get_prompt(validation=False) in agent2.dataset
        assert agent2._curriculum_cursor == 3

    def test_weighted_multi_task_stamps_prompt_share(self, tmp_path):
        agent_cls = get_agent_class("ResponsesEnvAgent")
        assert agent_cls is ResponsesEnvAgent

        def config(name, weight, **kwargs):
            args = {
                "agent_name": name,
                "dataset_file": _write_dataset(tmp_path),
                "env_server_host_port": "h:1",
            }
            return AgentConfig(agent_type=agent_cls, agent_args=args, weight=weight, **kwargs)

        multi = WeightedMultiTask(
            [config("a", 3.0), config("b", 1.0), config("c", 1.0, evaluation_only=True)]
        )
        shares = {a.agent_name: getattr(a, "_prompt_share", None) for a in multi.agents}
        assert shares == {"a": pytest.approx(0.75), "b": pytest.approx(0.25), "c": None}

    @pytest.mark.parametrize(
        "kind, error",
        [
            ("env-var-expanded", None),
            ("env-var-unset", "unexpanded environment variable"),
            ("bad-json", "Error decoding dataset JSON"),
        ],
        ids=["env-var-expanded", "env-var-unset", "bad-json"],
    )
    def test_dataset_loading(self, tmp_path, monkeypatch, kind, error):
        _write_dataset(tmp_path)
        if kind == "env-var-expanded":
            monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path))
            dataset_file = "${TEST_DATA_ROOT}/dataset.jsonl"
        elif kind == "env-var-unset":
            monkeypatch.delenv("SURELY_UNSET_VAR", raising=False)
            dataset_file = "${SURELY_UNSET_VAR}/dataset.jsonl"
        else:
            dataset_file = _write_dataset(tmp_path, text='{"ok": 1}\nnot json')
        with pytest.raises(ValueError, match=error) if error else nullcontext():
            assert len(_agent(tmp_path, dataset_file=dataset_file).dataset) == 4

    # (canned responses, agent kwargs, expected failure_reason/detail, expected backoff sleeps)
    @pytest.mark.parametrize(
        "responses, agent_kwargs, failure, detail, sleeps",
        [
            ([(200, _GOOD_BODY)], {}, None, None, []),
            ([(503, "busy"), (503, "busy"), (200, _GOOD_BODY)], {}, None, None, [5, 10]),
            ([(500, "boom"), (500, "boom")], {"http_retries": 2}, "http_500", "boom", [5]),
            ([(404, "gone")], {}, "http_404", "gone", []),
            ([(200, "not json")], {}, "ValidationError", None, []),
            (
                [("hang", None)],
                {"rollout_timeout_s": 0.01, "http_retries": 1},
                "rollout_timeout",
                None,
                [],
            ),
        ],
        ids=[
            "ok-first-try",
            "5xx-retried-then-ok",
            "5xx-exhaustion-fails-as-value",
            "4xx-fails-without-retry",
            "malformed-body-fails-as-value",
            "rollout-timeout-fails-as-value",
        ],
    )
    @pytest.mark.asyncio
    async def test_http_handling(
        self, tmp_path, monkeypatch, responses, agent_kwargs, failure, detail, sleeps
    ):
        agent = _agent(tmp_path, **agent_kwargs)
        remaining, seen_sleeps = list(responses), []

        async def handler(request):
            status, body = remaining.pop(0)
            if status == "hang":  # outlast the wall-clock budget (immune to the sleep patch)
                await asyncio.Event().wait()
            return httpx.Response(
                status, **({"json": body} if isinstance(body, dict) else {"text": body})
            )

        _install_transport(agent, handler)

        async def fake_sleep(seconds):
            seen_sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))

        assert seen_sleeps == sleeps
        if failure is None:
            assert not remaining and response.result.reward == 0.5
        else:
            assert (response.result, response.failure_reason) == (None, failure)
            assert detail is None or response.failure_detail == detail

    @pytest.mark.asyncio
    async def test_connect_failures_refresh_routes_then_hit_the_cap(self, tmp_path):
        refreshes = []

        class RefreshCounting(ResponsesEnvAgent):
            async def refresh_routes(self):
                refreshes.append(1)

        agent = RefreshCounting(
            agent_name="t",
            dataset_file=_write_dataset(tmp_path),
            env_server_host_port="h:1",
            connect_failure_cap=5,
            connect_refresh_every=2,
            connect_retry_wait_s=0,
        )

        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        _install_transport(agent, handler)
        with pytest.raises(EnvConnectExhausted):
            await agent._post_run({})
        assert len(refreshes) == 2  # at failures 2 and 4; the cap hits at 5

        # Through the pipeline path the exhaustion travels as a failure value.
        response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))
        assert (response.result, response.failure_reason) == (None, "EnvConnectExhausted")

    @pytest.mark.parametrize("kind", ["ok", "http-failure"])
    @pytest.mark.asyncio
    async def test_episode_end_to_end(self, tmp_path, kind):
        agent = _agent(tmp_path, max_output_tokens_per_step=64)
        seen, ok_body = [], {
            "reward": 1.0,
            "response": {"output": [_turn([1, 2], [3], completion_id="c-0")]},
        }

        def handler(request):
            seen.append(json.loads(request.content))
            return (
                httpx.Response(200, json=ok_body)
                if kind == "ok"
                else httpx.Response(404, text="gone")
            )

        _install_transport(agent, handler)
        params = await agent.prepare_group_rollout(_request(temperature=0.7, top_p=0.9))
        rollout = await params.build_rollout(await params.run_episode())

        assert seen[0]["responses_create_params"]["temperature"] == 0.7
        assert seen[0]["responses_create_params"]["max_output_tokens"] == 64
        if kind == "ok":
            assert (rollout.trajectory, rollout.completion_ids) == ([[1, 2, 3]], ["c-0"])
            assert rollout.generation_cap == 64
            # The non-grouped path composes the same episode primitives.
            request = _request()
            request.num_rollouts = 2
            assert [r.trajectory for r in await agent.get_reward_rollouts(request)] == [
                [[1, 2, 3]]
            ] * 2
        else:
            assert (rollout.rollout_status, rollout.failure_reason) == ("placeholder", "http_404")
            assert rollout.env_id == "test_env"

    @pytest.mark.parametrize(
        "input_value, expected_prompt",
        [
            ("2+2?", "2+2?"),
            (
                [
                    {"type": "message", "role": "system", "content": "be brief"},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "2+2?"}],
                    },
                    {"type": "function_call", "name": "ignored"},
                ],
                [("system", "be brief"), ("user", "2+2?")],
            ),
        ],
        ids=["string-input-passes-through", "message-items-converted"],
    )
    def test_evaluation_response_building(self, input_value, expected_prompt):
        result = _run_result(
            _EVAL_OUTPUT, reward=1.0, responses_create_params={"input": input_value}
        )
        response = run_result_to_evaluation_response(result, "p0", "test_env")

        assert response.env_id == "test_env"
        [entry] = response.results
        assert (entry.reward, entry.problem_id) == (1.0, "p0")
        assert entry.response.content == "part one\npart two"
        prompt = (
            entry.prompt
            if isinstance(expected_prompt, str)
            else [(m.role, m.content) for m in entry.prompt]
        )
        assert prompt == expected_prompt
