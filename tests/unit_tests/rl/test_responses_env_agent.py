# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import json
import re
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest

from megatron.rl import GenericGenerationArgs
from megatron.rl.agent.api import EvaluationRequest
from megatron.rl.agent.registry import get_agent_class
from megatron.rl.agent.responses_env_agent import (
    CAPTURE_CONTROL_PREFIX,
    ROLLOUT_ID_KEY,
    EnvConnectExhausted,
    EnvRunRequest,
    EpisodeKeys,
    ResponsesEnvAgent,
    RunResult,
    curriculum_collections,
    episode_keys_from_manifest,
    generation_args_to_run_payload,
    new_rollout_id,
    run_result_to_evaluation_response,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask

TOKEN_ENV = "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN"
# Gym's capture-key charset: a key is a path segment and a file name.
ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    kwargs.setdefault("capture_control_host_port", "policyhost:12000")
    return ResponsesEnvAgent(**kwargs)


def _row(call_id, parent=None, prev_len=0, cum_len=4, response_id=None, **extra):
    """One manifest row; the staging key is the engine uid the ledger join pops."""
    return {
        "model_call_id": call_id,
        "parent_call_id": parent,
        "prev_len": prev_len,
        "cum_len": cum_len,
        "staging_key": f"uid-{call_id}",
        "response_id": response_id or f"resp-{call_id}",
        **extra,
    }


def _manifest(*rows, failures=(), **extra):
    return {"capture_key": "r", "records": list(rows), "failures": list(failures), **extra}


# A three-turn conversation (c1 -> c2 -> c3) plus a retried second turn (c2b, a dead branch).
_CHAIN = _manifest(
    _row("c1", cum_len=4), _row("c2", "c1", 4, 9), _row("c2b", "c1", 4, 8), _row("c3", "c2", 9, 15)
)


def _run_result(reward=1.0, response_id="resp-c3", **extra):
    return RunResult.model_validate(
        {"reward": reward, "response": {"id": response_id, "output": []}, **extra}
    )


def _install_transport(agent, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    object.__setattr__(agent, "_client", client)


def _request(**generation_args):
    return SimpleNamespace(
        validation=True, generation_args=GenericGenerationArgs(**generation_args)
    )


_GOOD_BODY = {"reward": 0.5, "response": {"id": "resp-c1", "output": []}}
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


class _ControlServer:
    """Scripted env server + policy control routes behind one MockTransport."""

    def __init__(self, run_responses, manifest=None, manifest_status=200, retire_status=200):
        self.run_responses = list(run_responses)
        self.manifest = manifest
        self.manifest_status = manifest_status
        self.retire_status = retire_status
        self.runs, self.manifests, self.retired, self.auth = [], [], [], []

    def __call__(self, request):
        path = request.url.path
        if path == "/run":
            self.runs.append(json.loads(request.content))
            status, body = self.run_responses.pop(0)
            return httpx.Response(
                status, **({"json": body} if isinstance(body, dict) else {"text": body})
            )
        self.auth.append(request.headers.get("authorization"))
        assert path.startswith(CAPTURE_CONTROL_PREFIX + "/rollouts/")
        if path.endswith("/manifest"):
            self.manifests.append(path.split("/")[-2])
            if self.manifest_status != 200:
                return httpx.Response(self.manifest_status, text="gone")
            return httpx.Response(200, json=self.manifest)
        assert path.endswith("/retire")
        self.retired.extend(json.loads(request.content)["rollout_ids"])
        return httpx.Response(self.retire_status, json={"removed": [], "absent": []})


class TestResponsesEnvAgent:

    # (manifest, scored response id, declared response id, expected EpisodeKeys attributes)
    @pytest.mark.parametrize(
        "manifest, scored, declared, expected",
        [
            (
                _CHAIN,
                "resp-c3",
                None,
                dict(
                    turns=["uid-c1", "uid-c2", "uid-c3"],
                    others=["uid-c2b"],
                    failure_reason=None,
                    terminal_selection="response_id",
                ),
            ),
            (
                _CHAIN,
                "resp-c2b",
                None,
                dict(turns=["uid-c1", "uid-c2b"], others=["uid-c2", "uid-c3"]),
            ),
            (
                _CHAIN,
                "resp-c3",
                "resp-c2",
                dict(turns=["uid-c1", "uid-c2"], terminal_selection="declared"),
            ),
            (
                _CHAIN,
                "resp-c3",
                "resp-nope",
                dict(
                    turns=[],
                    others=["uid-c1", "uid-c2", "uid-c2b", "uid-c3"],
                    failure_reason="declared_terminal_not_captured",
                    terminal_selection="declared",
                ),
            ),
            # No witness: the parent-link heuristic walks the extended branch to its leaf.
            (
                _manifest(
                    _row("c1"),
                    _row("c2", "c1", 4, 9),
                    _row("c2b", "c1", 4, 8),
                    _row("c3", "c2", 9, 15),
                ),
                None,
                None,
                dict(turns=["uid-c1", "uid-c2", "uid-c3"], terminal_selection="heuristic"),
            ),
            (
                _manifest(_row("c1"), _row("c2", "c1", 4, 9), _row("c2b", "c1", 4, 8)),
                "resp-unknown",
                None,
                dict(turns=[], failure_reason="ambiguous_terminal", terminal_selection="heuristic"),
            ),
            # A scored response id that two calls share selects neither: the heuristic decides.
            (
                _manifest(
                    _row("c1"),
                    _row("c2", "c1", 4, 9, response_id="resp-x"),
                    _row("c2a", "c1", 4, 9, response_id="resp-x"),
                ),
                "resp-x",
                None,
                dict(turns=[], failure_reason="ambiguous_terminal", terminal_selection="heuristic"),
            ),
            (
                _manifest(
                    _row("c1"),
                    failures=[{"model_call_id": "c2", "reason": "worker_capture_failed"}],
                ),
                "resp-c1",
                None,
                dict(turns=[], others=["uid-c1"], failure_reason="worker_capture_failed"),
            ),
            # A call that never returned a completion is off-chain and does not poison.
            (
                _manifest(
                    _row("c1"),
                    failures=[
                        {
                            "model_call_id": "c2",
                            "reason": "request_finished_without_staged_coordinates",
                        }
                    ],
                ),
                "resp-c1",
                None,
                dict(turns=["uid-c1"], failure_reason=None),
            ),
            (
                _manifest(_row("c1"), {"model_call_id": "bad"}),
                "resp-c1",
                None,
                dict(turns=[], others=["uid-c1"], failure_reason="invalid_manifest_row"),
            ),
            (
                _manifest(),
                None,
                None,
                dict(turns=[], others=[], failure_reason="no_captured_calls"),
            ),
            (
                _manifest(_row("c1"), _row("c2", "c1", 5, 9)),
                "resp-c2",
                None,
                dict(
                    turns=[], others=["uid-c1", "uid-c2"], failure_reason="parent_length_mismatch"
                ),
            ),
            (
                _manifest(_row("c2", "c0", 4, 9)),
                "resp-c2",
                None,
                dict(turns=[], others=["uid-c2"], failure_reason="missing_parent"),
            ),
        ],
        ids=[
            "scored-id-selects-the-chain",
            "scored-id-on-the-retry-makes-the-other-branch-dead",
            "declared-id-is-authoritative",
            "declared-id-unknown-never-falls-back",
            "heuristic-prefers-the-extended-branch",
            "heuristic-ambiguity-fails-closed",
            "shared-scored-id-falls-back",
            "failure-row-poisons",
            "uncommitted-call-is-off-chain",
            "invalid-row-poisons-keeps-keys",
            "empty-manifest",
            "non-contiguous-chain",
            "missing-parent",
        ],
    )
    def test_episode_keys_from_manifest(self, manifest, scored, declared, expected):
        keys = episode_keys_from_manifest(
            manifest, scored_response_id=scored, declared_response_id=declared
        )
        for attr, value in expected.items():
            assert getattr(keys, attr) == value, attr
        assert sorted(keys.all_keys) == sorted(
            row["staging_key"] for row in manifest["records"] if "staging_key" in row
        )

    def test_new_rollout_id(self):
        rollout_id = new_rollout_id("my env/2")
        assert rollout_id.startswith("my-env-2-") and ROLLOUT_ID_PATTERN.match(rollout_id)
        # The last segment starts with a letter Gym's attempt suffix, -a<N>, cannot start with.
        assert rollout_id.rsplit("-", 1)[1].startswith("r")
        assert new_rollout_id("").startswith("episode-")
        assert len({new_rollout_id("e") for _ in range(50)}) == 50

    # (result extras, episode keys, agent kwargs, expected rollout attributes)
    @pytest.mark.parametrize(
        "result_extra, keys, agent_kwargs, expected",
        [
            (
                {},
                EpisodeKeys(turns=["u1", "u2"], others=["u3"]),
                {},
                dict(
                    trajectory=None,
                    generation_mask=None,
                    logprobs=None,
                    reward=1.0,
                    env_id="test_env",
                    problem_id="p0",
                    rollout_status="ok",
                    completion_ids=["u1", "u2"],
                    retired_completion_ids=["u3"],
                ),
            ),
            (
                {},
                EpisodeKeys(others=["u1", "u2"], failure_reason="worker_capture_failed"),
                {},
                dict(
                    trajectory=[],
                    reward=0.0,
                    rollout_status="placeholder",
                    failure_reason="worker_capture_failed",
                    completion_ids=["u1", "u2"],
                    retired_completion_ids=[],
                ),
            ),
            (
                {},
                None,
                {},
                dict(
                    rollout_status="placeholder",
                    failure_reason="manifest_unavailable",
                    completion_ids=[],
                ),
            ),
            (
                {"mask_sample": True},
                EpisodeKeys(turns=["u1"], others=["u2"]),
                {},
                dict(
                    trajectory=[],
                    rollout_status="masked",
                    failure_reason="env_masked",
                    completion_ids=["u1", "u2"],
                    retired_completion_ids=[],
                ),
            ),
            (
                {"mask_sample": True, "failure_reason": "agent_timeout"},
                EpisodeKeys(turns=["u1"]),
                {},
                dict(rollout_status="masked", failure_reason="agent_timeout"),
            ),
            (
                {"instance_config": {"mask_sample": True}},
                EpisodeKeys(turns=["u1"]),
                {},
                dict(rollout_status="masked", failure_reason="env_masked"),
            ),
            (
                {"final_eval_time": 12.5},
                EpisodeKeys(turns=["u1"]),
                {"max_output_tokens_per_step": 128},
                dict(rollout_status="graded", generation_cap=128),
            ),
        ],
        ids=[
            "chain-becomes-completion-ids",
            "poisoned-capture-pads-placeholder-draining-all",
            "no-manifest-pads-placeholder",
            "masked-direct-field-drains-all",
            "masked-keeps-server-reason",
            "masked-via-instance-config",
            "graded-with-generation-cap",
        ],
    )
    def test_build_token_rollout(self, tmp_path, result_extra, keys, agent_kwargs, expected):
        agent = _agent(tmp_path, **agent_kwargs)
        rollout = agent.build_token_rollout(
            _run_result(**result_extra), "p0", keys, generation_cap=agent.max_output_tokens_per_step
        )
        for attr, value in expected.items():
            assert getattr(rollout, attr) == value, attr
        assert rollout.is_placeholder == (rollout.rollout_status in ("placeholder", "masked"))

    def test_placeholder(self, tmp_path):
        rollout = _agent(tmp_path)._placeholder("http_500", "p0", ["u1"], rollout_status="masked")
        assert (rollout.trajectory, rollout.generation_mask, rollout.logprobs) == ([], [], [])
        assert (rollout.reward, rollout.env_id, rollout.problem_id) == (0.0, "test_env", "p0")
        assert (rollout.rollout_status, rollout.failure_reason) == ("masked", "http_500")
        assert rollout.is_placeholder and rollout.completion_ids == ["u1"]

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

    # (iteration, prompts per collection, group size, grpo iterations, global batch size, expected)
    @pytest.mark.parametrize(
        "iteration, prompts, group_size, grpo_iterations, global_batch_size, expected",
        [
            (3, 8, 1, 1, None, 3),
            (8, 4, 16, 1, 32, 4),
            (8, 4, 16, 2, 32, 2),
            (12, 4, 2, 3, 4, 2),
            (1, 0, 1, 1, None, ValueError),
        ],
        ids=[
            "one-batch-per-collection",
            "batches-per-collection-cadence",
            "grpo-iterations-cadence",
            "both-cadences",
            "invalid-prompts-raises",
        ],
    )
    def test_curriculum_collections(
        self, iteration, prompts, group_size, grpo_iterations, global_batch_size, expected
    ):
        args = (iteration, prompts, group_size, grpo_iterations, global_batch_size)
        if expected is ValueError:
            with pytest.raises(ValueError, match="must be positive"):
                curriculum_collections(*args)
        else:
            assert curriculum_collections(*args) == expected

    @pytest.mark.asyncio
    async def test_curriculum_get_prompt(self, tmp_path, monkeypatch):
        agent = _agent(tmp_path, curriculum_sampling=True)
        train_state = SimpleNamespace(iteration=0)
        rl = SimpleNamespace(grpo_prompts_per_step=2, grpo_group_size=1, grpo_iterations=1)
        run_config = SimpleNamespace(rl=rl)
        monkeypatch.setattr("megatron.training.global_vars.get_train_state", lambda: train_state)
        monkeypatch.setattr("megatron.training.global_vars.get_run_config", lambda: run_config)
        args = SimpleNamespace(global_batch_size=None)
        monkeypatch.setattr("megatron.training.global_vars.get_args", lambda: args)
        rows = [await agent.get_prompt(validation=False) for _ in range(3)]
        assert [row["problem_id"] for row in rows] == ["p0", "p1", "p2"]
        # The cursor never rewinds: one collection later its seed (2) is behind the cursor (3).
        train_state.iteration = 1
        assert (await agent.get_prompt(validation=False))["problem_id"] == "p3"

        # A restart at that iteration seeds the cursor from training progress, scaled by the
        # agent's share of the prompt groups.
        agent2 = _agent(tmp_path, curriculum_sampling=True)
        assert (await agent2.get_prompt(validation=False))["problem_id"] == "p2"
        half_share = _agent(tmp_path, curriculum_sampling=True)
        object.__setattr__(half_share, "_prompt_share", 0.5)
        assert (await half_share.get_prompt(validation=False))["problem_id"] == "p1"

        # Validation and missing training state fall back to uniform, cursor untouched.
        assert await agent2.get_prompt(validation=True) in agent2.dataset
        monkeypatch.setattr("megatron.training.global_vars.get_train_state", lambda: None)
        assert await agent2.get_prompt(validation=False) in agent2.dataset

        def _no_run_config():
            # What get_run_config really raises when Megatron global state is uninitialized.
            raise AssertionError("run config is not initialized.")

        monkeypatch.setattr("megatron.training.global_vars.get_run_config", _no_run_config)
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
                "capture_control_host_port": "h:2",
            }
            return AgentConfig(agent_type=agent_cls, agent_args=args, weight=weight, **kwargs)

        multi = WeightedMultiTask(
            [config("a", 3.0), config("b", 1.0), config("c", 1.0, evaluation_only=True)]
        )
        # The evaluation-only agent takes no rollout prompts and keeps the default share.
        shares = {a.agent_name: a._prompt_share for a in multi.agents}
        assert shares == {"a": pytest.approx(0.75), "b": pytest.approx(0.25), "c": 1.0}

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
            valid = '{"responses_create_params": {"input": "q"}, "problem_id": "p"}'
            dataset_file = _write_dataset(tmp_path, text=valid + "\nnot json")
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
    async def test_connect_failures_hit_the_cap(self, tmp_path):
        agent = _agent(tmp_path, connect_failure_cap=5, connect_retry_wait_s=0)
        attempts = []

        def handler(request):
            attempts.append(request.url.path)
            raise httpx.ConnectError("connection refused", request=request)

        _install_transport(agent, handler)
        with pytest.raises(EnvConnectExhausted):
            await agent._post_run({})
        assert attempts == ["/run"] * 5

        # Through the pipeline path the exhaustion travels as a failure value.
        response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))
        assert (response.result, response.failure_reason) == (None, "EnvConnectExhausted")

    @pytest.mark.parametrize(
        "kind", ["ok", "row-cap-binds", "http-failure", "manifest-gone", "retire-fails"]
    )
    @pytest.mark.asyncio
    async def test_episode_end_to_end(self, tmp_path, monkeypatch, kind):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        if kind == "row-cap-binds":
            # A row-level max_output_tokens below the agent cap must win the stamp AND be
            # what the rollout self-reports as generation_cap (termination-check input).
            row = {"responses_create_params": {"input": "q 0", "max_output_tokens": 32}}
            capped = tmp_path / "capped.jsonl"
            capped.write_text(json.dumps({**row, "problem_id": "p0"}) + "\n")
            agent = _agent(tmp_path, max_output_tokens_per_step=64, dataset_file=str(capped))
        else:
            agent = _agent(tmp_path, max_output_tokens_per_step=64)
        expected_cap = 32 if kind == "row-cap-binds" else 64
        run = (
            (404, "gone")
            if kind == "http-failure"
            else (200, {**_GOOD_BODY, "reward": 1.0, "response": {"id": "resp-c3", "output": []}})
        )
        server = _ControlServer(
            [run] * 5,
            manifest=_CHAIN,
            manifest_status=410 if kind == "manifest-gone" else 200,
            retire_status=500 if kind == "retire-fails" else 200,
        )
        _install_transport(agent, server)

        params = await agent.prepare_group_rollout(_request(temperature=0.7, top_p=0.9))
        rollout = await params.build_rollout(await params.run_episode())

        body = server.runs[0]
        assert body["responses_create_params"]["temperature"] == 0.7
        assert body["responses_create_params"]["max_output_tokens"] == expected_cap
        rollout_id = body[ROLLOUT_ID_KEY]
        assert rollout_id.startswith("test_env-") and ROLLOUT_ID_PATTERN.match(rollout_id)
        # The manifest is read (and the ledger retired) for every episode, failed ones included.
        assert server.manifests == [rollout_id]
        assert set(server.auth) == {"Bearer s3cret"}
        assert server.retired == ([] if kind == "manifest-gone" else [rollout_id])
        if kind == "http-failure":
            assert (rollout.rollout_status, rollout.failure_reason) == ("placeholder", "http_404")
            # Whatever Gym recorded for the failed episode is still drained from the ledger.
            assert sorted(rollout.completion_ids) == ["uid-c1", "uid-c2", "uid-c2b", "uid-c3"]
            assert rollout.env_id == "test_env"
            # Failure placeholders keep the real problem_id of the drawn row.
            assert rollout.problem_id in {"p0", "p1", "p2", "p3"}
        elif kind == "manifest-gone":
            assert (rollout.failure_reason, rollout.completion_ids) == ("manifest_unavailable", [])
        else:
            assert rollout.trajectory is None
            assert rollout.completion_ids == ["uid-c1", "uid-c2", "uid-c3"]
            assert rollout.retired_completion_ids == ["uid-c2b"]
            assert rollout.generation_cap == expected_cap
            assert not rollout.is_placeholder
        if kind == "ok":
            # The pipeline runs every rollout of a group from the same params: each episode gets
            # its own capture key and reads its own manifest.
            episodes = [await params.run_episode() for _ in range(2)]
            built = [await params.build_rollout(episode) for episode in episodes]
            keys = [run[ROLLOUT_ID_KEY] for run in server.runs]
            assert len(set(keys)) == 3 and server.manifests == keys and server.retired == keys
            assert [r.completion_ids for r in built] == [["uid-c1", "uid-c2", "uid-c3"]] * 2
            # The non-grouped path composes the same episode primitives.
            request = _request()
            request.num_rollouts = 2
            rollouts = await agent.get_reward_rollouts(request)
            assert [r.completion_ids for r in rollouts] == [["uid-c1", "uid-c2", "uid-c3"]] * 2
            assert len({run[ROLLOUT_ID_KEY] for run in server.runs}) == 5

    @pytest.mark.asyncio
    async def test_missing_control_token_fails_the_episode_as_a_value(self, tmp_path, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        agent = _agent(tmp_path)
        _install_transport(agent, _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN))
        params = await agent.prepare_group_rollout(_request())
        rollout = await params.build_rollout(await params.run_episode())
        assert (rollout.rollout_status, rollout.failure_reason) == (
            "placeholder",
            "manifest_unavailable",
        )

    @pytest.mark.asyncio
    async def test_evaluation_reports_every_recorded_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        agent = _agent(tmp_path)
        server = _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN)
        _install_transport(agent, server)
        response = await agent._evaluation(agent.dataset[0], agent.dataset[0], _request())
        [entry] = response.results
        assert (entry.reward, entry.problem_id) == (0.5, "p0")
        assert sorted(entry.completion_ids) == ["uid-c1", "uid-c2", "uid-c2b", "uid-c3"]
        assert server.retired == [server.runs[0][ROLLOUT_ID_KEY]]

    @pytest.mark.asyncio
    async def test_run_evaluation_reports_a_dropped_prompts_keys(self, tmp_path, monkeypatch):
        """A prompt with a failed episode is dropped, and every call its episodes recorded is
        still reported, the sample that was running when its sibling failed included."""
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        agent = _agent(tmp_path, max_k=2)
        runs = []

        async def handler(request):
            path = request.url.path
            if path == "/run":
                runs.append(json.loads(request.content))
                p1_runs = [run for run in runs if run["problem_id"] == "p1"]
                if runs[-1]["problem_id"] == "p1" and len(p1_runs) == 1:
                    return httpx.Response(400, text="bad request")  # not retried
                if runs[-1]["problem_id"] == "p1":
                    await asyncio.sleep(0.05)  # still running when its sibling fails
                return httpx.Response(200, json=_GOOD_BODY)
            if path.endswith("/manifest"):  # one recorded call per episode, keyed by its rollout
                return httpx.Response(200, json=_manifest(_row(path.split("/")[-2])))
            return httpx.Response(200, json={"removed": [], "absent": []})

        _install_transport(agent, handler)
        request = EvaluationRequest.model_construct(inference_interface=None, num_prompts=2)
        with pytest.warns(UserWarning, match="top_k"):  # the kept prompt's greedy episode
            response = await agent.run_evaluation(request)

        [kept] = response.results
        assert kept.problem_id == "p0"
        # p0 ran two samples and a greedy episode; p1 one failed sample and one slow sample.
        assert [run["problem_id"] for run in runs].count("p1") == 2
        assert sorted(kept.completion_ids) == sorted(f"uid-{run[ROLLOUT_ID_KEY]}" for run in runs)
        assert len(kept.completion_ids) == 5
        assert agent._evaluation_completion_ids == []

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
        result = RunResult.model_validate(
            {
                "reward": 1.0,
                "response": {"id": "resp-1", "output": _EVAL_OUTPUT},
                "responses_create_params": {"input": input_value},
            }
        )
        response = run_result_to_evaluation_response(result, "p0", "test_env", ["u1"])

        assert response.env_id == "test_env"
        [entry] = response.results
        assert (entry.reward, entry.problem_id, entry.completion_ids) == (1.0, "p0", ["u1"])
        assert entry.response.content == "part one\npart two"
        prompt = (
            entry.prompt
            if isinstance(expected_prompt, str)
            else [(m.role, m.content) for m in entry.prompt]
        )
        assert prompt == expected_prompt
