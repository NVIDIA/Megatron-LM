# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sampling oracles separate deterministic forward results from shared RNG draws."""

import asyncio
from unittest import mock

import flashinfer
import pytest
import torch

from megatron.core.inference.inference_request import deserialize_tensor
from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]


@pytest.mark.parametrize("offset_by_dp", [False, True], ids=["shared-seed", "rank-seed"])
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
@pytest.mark.parametrize(
    "backend,filters",
    [
        pytest.param("torch", (4, 0.0), id="torch-top-k"),
        pytest.param("torch", (0, 0.7), id="torch-top-p"),
        pytest.param("flashinfer", (4, 0.0), id="flashinfer-top-k"),
        pytest.param("flashinfer", (0, 0.7), id="flashinfer-top-p"),
        pytest.param("flashinfer", (4, 0.7), id="flashinfer-joint"),
    ],
)
async def test_routed_stochastic_first_step_distribution(
    monkeypatch, backend, logprobs_mode, filters, offset_by_dp
):
    """Same prompt/history gives the same distribution, not necessarily the same draw."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=1,
        temperature=0.7,
        top_k=filters[0],
        top_p=filters[1],
        return_log_probs=True,
        skip_prompt_log_probs=True,
    )
    async with routed_model(
        monkeypatch,
        sampling_backend=backend,
        logprobs_mode=logprobs_mode,
        offset_sampling_seed_by_dp_rank=offset_by_dp,
    ) as h:
        samples = []
        sampler = h.engine.controller._sampling
        sample = sampler.sample_kernel
        rng = h.engine.controller.sampling_rng
        owner_rank = torch.distributed.get_rank(group=h.engine.controller.dp_group)
        assert sampler._rng is rng
        assert rng.initial_seed() == h.config.random_seed + (owner_rank if offset_by_dp else 0)
        joint = mock.Mock(wraps=flashinfer.sampling.top_k_top_p_sampling_from_logits)
        if backend == "flashinfer" and all(filters):
            monkeypatch.setattr(flashinfer.sampling, "top_k_top_p_sampling_from_logits", joint)

        def observe(logits, n, context, **kwargs):
            indices = kwargs.get("gather_indices")
            rows = logits[:n] if indices is None else logits[indices[:n]]
            raw = rows.detach().clone()
            distribution = sampler.log_probs_kernel(raw.float(), context).detach().clone()
            rng_before = rng.get_state()
            output = sample(logits, n, context, **kwargs)
            assert not torch.equal(
                rng_before, rng.get_state()
            ), "Stochastic sampling must consume RNG"
            ids = context.request_ids[
                context.paused_request_count : context.total_request_count
            ].tolist()
            samples.append((ids, raw.cpu(), distribution.cpu(), output[:n].clone().cpu()))
            return output

        monkeypatch.setattr(sampler, "sample_kernel", observe)
        await h.direct(prompt, params)
        assert samples and len(samples[0][0]) == 1
        _, direct_logits, direct_distribution, _ = samples[0]
        samples.clear()
        await h.start()
        # Direct/reference and startup work cannot witness routed joint filtering.
        joint.reset_mock()
        results, owners, local_ids = await h.run_paused_batch(
            [(client, prompt, params) for client in range(h.dp_size)]
        )
        assert local_ids == [(client, 0) for client in range(h.dp_size)]
        assert len(set(owners.values())) == h.dp_size
        assert all(
            len(r["generated_tokens"]) == len(r["generated_log_probs"]) == 1 for r in results
        )
        assert len(samples) == 1, "One-token requests sample exactly one distribution"
        ids, logits, distribution, tokens = samples[0]
        assert len(ids) == 1 and ids[0] != 10001
        assert any(ids[0] in step["ids"] for step in h.witnesses)
        torch.testing.assert_close(logits, direct_logits, rtol=0, atol=0)
        torch.testing.assert_close(distribution, direct_distribution, rtol=0, atol=0)
        result = next(r for r in results if r["request_id"] == ids[0])
        token = int(tokens[0])
        assert result["generated_tokens"] == [token]
        raw_distribution = torch.log_softmax(logits.float(), dim=-1)
        target = raw_distribution if logprobs_mode == "raw_logprobs" else distribution
        atol = 8 * torch.finfo(target.dtype).eps
        assert result["generated_log_probs"][0] == pytest.approx(
            target[0, token].item(), rel=0, abs=atol
        )
        assert abs(raw_distribution[0, token] - distribution[0, token]) > atol
        assert torch.isfinite(distribution[0, token])
        assert 1 < torch.isfinite(distribution[0]).sum() < h.config.vocab_size
        evidence = [None] * h.dp_size
        torch.distributed.all_gather_object(evidence, (ids[0], rng.initial_seed(), token, True))
        assert [(owner, seed) for owner, seed, _, _ in sorted(evidence)] == [
            (rank, h.config.random_seed + (rank if offset_by_dp else 0))
            for rank in range(h.dp_size)
        ]
        assert joint.call_count == int(backend == "flashinfer" and all(filters))
        print(f"sampling owner evidence: {sorted(evidence)}")
        h.assert_retired()


@pytest.mark.parametrize("interval", [1, 3])
async def test_routed_stream_holdback_keeps_top_n_scores_aligned(monkeypatch, interval):
    """Withheld stop-prefix tokens must not leak their top-N scores in a partial."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=12,
        return_log_probs=True,
        skip_prompt_log_probs=True,
        top_n_logprobs=3,
        streaming_interval=interval,
    )
    async with routed_model(monkeypatch) as h:
        direct = await h.direct(prompt, params)
        # Choose a genuine three-token stop pattern absent from the model's output.
        # Its two-token holdback still executes at every streaming boundary.
        generated = direct["generated_tokens"]
        stop = next(
            [token, token, token]
            for token in range(h.config.vocab_size)
            if not any(generated[i : i + 3] == [token] * 3 for i in range(len(generated) - 2))
        )
        params.stop_words = [" ".join(map(str, stop))]
        await h.start()
        replies = None
        if h.rank == 0:
            stream = h.clients[0].add_request_streaming(prompt, params)

            async def collect():
                return [item async for item in stream]

            replies = await asyncio.wait_for(collect(), timeout=60)
        box = [replies]
        torch.distributed.broadcast_object_list(box, src=0)
        replies = box[0]
        final = replies[-1]["final"]
        assert final["generated_tokens"] == generated
        partials = [r["partial"] for r in replies if "partial" in r]
        assert partials, "The stop holdback must coexist with real incremental delivery"
        sent = 0
        for part in partials:
            n = len(part["new_tokens"])
            assert len(part["new_top_n_logprobs"]) == n
            assert len(part["new_log_probs"]) == n
            assert part["new_tokens"] == generated[sent : sent + n]
            assert part["new_top_n_logprobs"] == final["generated_top_n_logprobs"][sent : sent + n]
            sent += n
        assert sent <= len(generated) - 2, "Two stop-prefix tokens remain withheld until final"
        await h.barrier()
        h.assert_retired()


@pytest.mark.parametrize(
    "echo,detokenize,eos",
    [(False, False, False), (True, True, False), (True, False, False), (False, False, True)],
)
async def test_routed_output_opt_ins(monkeypatch, echo, detokenize, eos):
    """Wire prompt/text opt-ins survive routing while total length stays exact."""
    prompt = list(range(4, 20))
    params = greedy_params(
        num_tokens_to_generate=None,
        num_tokens_total=len(prompt) + 5,
        return_prompt_tokens=echo,
        detokenize_generations=detokenize,
    )
    async with routed_model(monkeypatch) as h:
        if eos:
            direct = await h.direct(prompt, params)
            params.termination_id = direct["generated_tokens"][0]
        await h.start()
        if h.rank == 0:
            result = await asyncio.wait_for(h.clients[0].add_request(prompt, params), timeout=60)
            assert result["prompt_length"] == len(prompt)
            assert (
                result["generated_tokens"] == [params.termination_id]
                if eos
                else len(result["generated_tokens"]) == 5
            )
            assert (result["prompt_tokens"] is not None) == echo
            if detokenize:
                assert result["generated_text"] == h.tokenizer.detokenize(
                    result["generated_tokens"]
                )
            else:
                assert result["generated_text"] is None
            assert any(0 in step["ids"] for step in h.witnesses)
        await h.barrier()
        h.assert_retired()


@pytest.mark.parametrize("keep,skip", [(False, False), (False, True), (True, False), (True, True)])
async def test_routed_bos_stop_after_metadata_compaction(monkeypatch, keep, skip):
    params = greedy_params(
        num_tokens_to_generate=4,
        return_prompt_tokens=True,
        return_log_probs=True,
        skip_prompt_log_probs=skip,
        top_n_logprobs=3,
        detokenize_stop_sequence=keep,
    )
    async with routed_model(
        monkeypatch, hidden_size=128, materialize_only_last_token_logits=False
    ) as h:
        h.tokenizer.bos = 3
        direct = await h.direct([3, 4, 5], params)
        params.stop_words = [h.tokenizer.detokenize(direct["generated_tokens"][:2])]
        params.add_BOS = True
        sampler, samples = h.engine.controller._sampling, []
        sample = sampler.sample_kernel

        def observe(logits, n, context, **kwargs):
            ids = context.request_ids[: context.total_request_count].tolist()
            samples.append((ids, context.active_request_metadata["top_k"][: len(ids)].tolist()))
            return sample(logits, n, context, **kwargs)

        monkeypatch.setattr(sampler, "sample_kernel", observe)
        await h.start()
        samples.clear()
        short = greedy_params(num_tokens_to_generate=1, top_k=4, temperature=0.7)
        (short, filler, target), owners, _ = await h.run_paused_batch(
            [(0, "4 5", choice) for choice in (short, short, params)]
        )
        first, other, last = (result["request_id"] for result in (short, filler, target))
        assert owners[first] == owners[last] != owners[other]
        assert target["status"] == "COMPLETED"
        assert deserialize_tensor(target["prompt_tokens"][1]).tolist() == [3, 4, 5]
        assert target["generated_tokens"] == direct["generated_tokens"][:2] * keep
        for key in ("generated_log_probs", "generated_top_n_logprobs"):
            assert len(target[key]) == 2 * keep
        assert target["generated_log_probs"] == [
            scores[str(token)]
            for token, scores in zip(target["generated_tokens"], target["generated_top_n_logprobs"])
        ]
        for key in ("prompt_log_probs", "prompt_top_n_logprobs"):
            assert target[key] == direct[key]
        assert await h.sync.all_reduce_max(
            ([first, last], [4, 1]) in samples, ([last], [1]) in samples
        ) == (1, 1)
        h.assert_retired()
