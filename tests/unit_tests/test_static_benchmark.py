# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import sys
from unittest import mock

import pytest

from tests.performance_tests.client import static_benchmark


@pytest.mark.parametrize(
    ("batch_size", "data_parallel_size", "expected"),
    [(1, 8, 8), (8, 8, 8), (32, 8, 32), (1, 4, 4), (128, 4, 128), (1, 1, 1)],
)
def test_get_warmup_batch_size(batch_size, data_parallel_size, expected):
    assert static_benchmark._get_warmup_batch_size(batch_size, data_parallel_size) == expected


def test_parse_args_preserves_single_worker_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["static_benchmark.py"])

    assert static_benchmark.parse_args().data_parallel_size == 1


@pytest.mark.asyncio
async def test_run_batch_request_count_override(monkeypatch):
    single_request = mock.AsyncMock(return_value=(512, 128, 0.1))
    monkeypatch.setattr(static_benchmark, "_single_request", single_request)
    args = argparse.Namespace(
        batch_size=1, model="gpt_583m", num_output_tokens=128, temperature=0.0
    )

    inputs, outputs, latencies, _ = await static_benchmark._run_batch(
        mock.sentinel.session,
        args,
        "http://localhost:5000/v1/completions",
        ["prompt 0", "prompt 1"],
        iter_start_index=1,
        request_count=8,
    )

    assert single_request.await_count == 8
    assert [call.args[3] for call in single_request.await_args_list] == [
        "prompt 1",
        "prompt 0",
        "prompt 1",
        "prompt 0",
        "prompt 1",
        "prompt 0",
        "prompt 1",
        "prompt 0",
    ]
    assert inputs == [512] * 8
    assert outputs == [128] * 8
    assert latencies == [0.1] * 8

    single_request.reset_mock()
    await static_benchmark._run_batch(
        mock.sentinel.session,
        args,
        "http://localhost:5000/v1/completions",
        ["prompt"],
        iter_start_index=0,
    )
    single_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_widens_only_warmup_batches_and_preserves_timed_prompts(monkeypatch):
    calls = []

    async def fake_run_batch(session, args, url, prompts, iter_start_index, request_count=None):
        count = args.batch_size if request_count is None else request_count
        calls.append((iter_start_index, count))
        return [512] * count, [128] * count, [0.1] * count, 1.0

    class FakeClientSession:
        async def __aenter__(self):
            return mock.sentinel.session

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(static_benchmark, "_run_batch", fake_run_batch)
    monkeypatch.setattr(static_benchmark.aiohttp, "TCPConnector", mock.Mock())
    monkeypatch.setattr(
        static_benchmark.aiohttp, "ClientSession", mock.Mock(return_value=FakeClientSession())
    )
    args = argparse.Namespace(
        server_url="http://localhost:5000/v1",
        model="gpt_583m",
        batch_size=1,
        data_parallel_size=8,
        dataset="synthetic",
        num_input_tokens=512,
        num_output_tokens=128,
        temperature=0.0,
        num_warmup_iters=2,
        num_iters=2,
    )

    summary = await static_benchmark.main(args)

    assert calls == [(0, 8), (8, 8), (2, 1), (3, 1)]
    assert summary["batch_size"] == 1
