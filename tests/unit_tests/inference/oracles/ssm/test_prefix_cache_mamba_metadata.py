# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the Mamba prefix-cache metadata that #5119 rewrote as `PrefixCachedMambaMetadata`
(megatron/core/inference/contexts/attention_context/mamba_metadata.py:772) and the extraction kernels
it drives, `_scatter_intermediate_ssm_kernel` / `_scatter_intermediate_conv_kernel`
(megatron/core/ssm/ops/common/intermediate_extraction.py), through the commit into the durable cache
(`MambaSlotAllocator.commit_intermediate_states`) and the restore of a later request
(`restore_to_live`), driven by the dynamic engine on a real hybrid model.

This is a direct value oracle of the metadata computation, step by step: every step's metadata is
compared with an independent reference, and every state the cache stores or restores is checked
bit for bit against the state it must be. It is not a stateful search over request histories (the
step 5 stateful machines cover Mamba state across prefix hits); its scenarios are directed at the
metadata's edges.

Contract (from the docstrings of `compute_and_store_offsets`, `_update_intermediate_metadata`,
`collect_commit_records`, `commit_intermediate_states`, `restore_to_live`, the scan's documented
chunk layout (`MambaMetadata.compute_cpu_metadata`: each prefill request of a step is cut every
`mamba_chunk_size` tokens from its own first computed token; the scan returns, per chunk, the state
after its last token) and `_compute_prefix_match`, written before reading the bodies). Notation: a
prefill request j of a step computes the absolute prompt positions [s, e) (s = the tokens already in
its KV cache: earlier chunks plus a prefix skip), n = e - s tokens; bs = block size; c = Mamba chunk
size; q = the SSM chunk quantum (`ssm_chunk_alignment`, a multiple of c; bs is a multiple of q).

* **Staging.** A block boundary B is staged for request j only if s < B < e, B % bs == 0 and
  (B - s) % q == 0 (a chunk exit of this step's scan), at most three per request, ascending. At
  least the boundaries a later request needs are staged when they satisfy that: the prompt's last
  complete block boundary floor(L / bs) * bs (L = prompt length), the last block boundary before e
  (ceil(e / bs) - 1) * bs, and, for a request's first chunk, the end of its KV-matched prefix.
  A staged B is recorded against the request's own KV block B / bs - 1.
* **Extraction metadata.** Staged boundaries are numbered request by request (prefill order), then
  ascending. Slot i of boundary B of request j: `intermediate_chunk_indices[i]` = R_j + (B - s) / c - 1,
  where R_j = sum over earlier prefill requests k of max(1, ceil(n_k / c)) (the scan's chunk rows);
  `intermediate_abs_positions[i]` = T_j + (B - s), where T_j = sum of n_k over earlier prefill
  requests (B's offset in the step's prefill tokens); `intermediate_real_count` = the number of
  slots. Entries past it in the per-step view hold the safe defaults chunk 0 and position d_conv.
* **Extraction kernels.** Scratch row i (per Mamba layer, the layer whose live state the mixer
  updates) = the scan's raw state at row chunk_indices[i], converted to the cache dtype; the conv
  scratch row i = the pre-conv inputs at prefill positions [pos_i - d_conv, pos_i), transposed.
  Rows past the real count are not written.
* **End of chunk.** When e % bs == 0 the live state after the step is the state for KV block
  e / bs - 1 and is committed from the live buffer.
* **Commit.** After the step, each committed block's durable slot holds exactly its scratch row (or
  the live state), in every layer, and the registry maps the prompt-prefix hash of that block
  (`precomputed_block_hashes[B / bs - 1]`) to it. Distinct registered blocks own distinct slots.
  When the durable cache is full, a commit may be dropped (not registered), never misfiled.
* **Resume after a prefix hit.** A request's first chunk of length l skips p tokens: with m the
  number of leading prompt blocks whose prefix hash is KV-cached (within the blocks the chunk
  touches) and k the farthest k <= m whose block has cached Mamba state, p = k * bs when k * bs < l,
  else the farthest cached j < k gives p = j * bs (a chunk must compute at least one token); p = 0
  for continuation chunks. Its live state is then restored from the block registered under prefix
  hash p / bs - 1, and equals, bit for bit, the state committed for that prefix.

Tolerance class: exact everywhere (integer metadata; copies, gathers and transposes of states).
The scan's numerics are the chunk-scan oracle's (test_chunk_scan_varlen.py); here the raw states are
whatever the scan returned, and the oracle checks which of them land where.

Scenarios (structure-derived: bs = 256, c = q = 128): prompts of bs - 1, bs, bs + 1, 2.5 bs and 3 bs
tokens packed in one step (several requests, so R_j and T_j are nontrivial); followers admitted
while those decode (mixed steps, decode tokens ahead of the prefill tokens) with full-block hits, a
one-token resume chunk, a whole-prompt hit that backs off, partial-block hits (KV match ends inside
the shared prefix) and a restore from a divergence boundary; chunked prefill with chunk starts
aligned to c (384) and unaligned (320), where continuation chunks stage boundaries and commit chunk
ends; and a durable cache of three slots, so commits are dropped and slots are evicted and reused
before later requests resume. The `graphs` case replays prefill steps from CUDA graphs (the
extraction kernels' values are then checked only through the commit and resume chain, and the
metadata's padded entries are checked).

Not covered: the Gated Delta Product mixer's own chunk indices (`gdp_intermediate_chunk_indices`,
64-token chunks with Householder expansion); batch-invariant mode's boundary caching; MTP.
"""

import collections
import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, PrefixCachingEvictionPolicy
from megatron.core.inference.engine_factory import build_dynamic_inference_engine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm import mamba_mixer as mamba_mixer_module
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    delete_cuda_graphs,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.test_utilities import Utils

BLOCK_SIZE = 256
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048
PATTERN = "M*M-"
"""Two Mamba layers, so the scratch, durable and live buffers are indexed by layer."""
MAX_STEPS = 200


def _requirements():
    from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
    from megatron.core.utils import is_fa_min_version

    available, reason = _check_mamba_sequence_packing_support(for_inference_not_training=True)
    if not available:
        return reason
    if not is_fa_min_version("2.7.3"):
        return "dynamic batching needs flash-attn >= 2.7.3"
    return None


def _unavailable(reason):
    """Fail when the run targets this hardware (a skip there is not evidence), else skip."""
    if ou.session_target_sm() == ou.HOPPER:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def model_parallel():
    reason = _requirements()
    if reason is not None:
        _unavailable(reason)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield
    delete_cuda_graphs()
    Utils.destroy_model_parallel()


def _build_model(graphs):
    model_parallel_cuda_manual_seed(
        123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    torch.manual_seed(123)
    config = TransformerConfig(
        flash_attention_version=2,
        params_dtype=torch.bfloat16,
        num_layers=len(PATTERN),
        hidden_size=256,
        mamba_num_heads=16,
        num_attention_heads=16,
        use_cpu_initialization=True,
        cuda_graph_impl="local" if graphs else "none",
        inference_rng_tracker=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=True,
        is_hybrid_model=True,
    )
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        parallel_output=True,
        hybrid_layer_pattern=PATTERN,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    ).cuda()
    for param in model.parameters():
        param.data = param.data.to(config.params_dtype)
    model.eval()
    return model


def _reset_cuda_graph_state(model):
    delete_cuda_graphs()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []
    CudaGraphManager.global_mempool = None
    for module in model.modules():
        if isinstance(module, CudaGraphManager):
            module.cudagraph_runners.clear()
            module.custom_cudagraphs_lookup_table.clear()


def _mamba_gb_for_slots(mamba_config, max_tokens, max_requests, durable_slots):
    """A `prefix_caching_mamba_gb` that leaves `durable_slots` durable slots after the scratch."""
    layers = sum(1 for symbol in mamba_config.layer_type_list if symbol == "M")
    per_slot = layers * (
        math.prod(mamba_config.conv_states_shape) * mamba_config.conv_states_dtype.itemsize
        + math.prod(mamba_config.ssm_states_shape) * mamba_config.ssm_states_dtype.itemsize
    )
    scratch = min(math.ceil(max_tokens / BLOCK_SIZE), 3 * max_requests)
    return (scratch + durable_slots + 0.5) * per_slot / 1024**3


def _build_engine(model, scenario):
    from megatron.core.inference.config import MambaInferenceStateConfig

    _reset_cuda_graph_state(model)
    # FP32 recurrent state, as the prefix-caching engine tests use; the cache dtype is not what
    # this oracle measures.
    mamba_config = MambaInferenceStateConfig.from_model(model, ssm_states_dtype=torch.float32)
    max_requests = 16
    if scenario.durable_slots is None:
        mamba_gb = 2.0
    else:
        mamba_gb = _mamba_gb_for_slots(
            mamba_config, scenario.max_tokens, max_requests, scenario.durable_slots
        )
    config = InferenceConfig(
        max_sequence_length=MAX_SEQ_LEN,
        buffer_size_gb=0.5,
        block_size_tokens=BLOCK_SIZE,
        mamba_inference_state_config=mamba_config,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
        prefix_caching_mamba_gb=mamba_gb,
        enable_chunked_prefill=scenario.chunked,
        max_tokens=scenario.max_tokens,
        max_requests=max_requests,
        num_cuda_graphs=4 if scenario.graphs else None,
        use_cuda_graphs_for_non_decode_steps=True,
        unified_memory_level=0,
        sampling_backend="torch",
        async_sched_mode=scenario.async_sched_mode,
    )
    tokenizer = SimpleNamespace(vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "")
    return build_dynamic_inference_engine(model=model, tokenizer=tokenizer, inference_config=config)


class _Oracle:
    """Wraps the metadata, the extraction kernels, the commit and the restore of one engine."""

    def __init__(self, engine, monkeypatch, *, check_kernels):
        self.ctx = engine.context
        self.md = self.ctx.mamba_metadata
        self.msa = self.ctx.mamba_slot_allocator
        assert self.msa is not None, "the engine built no Mamba prefix cache"
        self.registry = self.ctx.prefix_cache_registry
        self.bs = self.ctx.block_size_tokens
        self.c = self.md.mamba_chunk_size
        self.q = self.ctx.ssm_chunk_alignment
        self.d_conv = self.md.intermediate_conv_out.shape[-1]
        self.check_kernels = check_kernels
        self.requests = {}
        self.admissions = {}  # request id -> dict(m=, skip=, restore_hash=)
        self.cached = {}  # prefix hash -> (ssm [layers, ...], conv [layers, ...]) as committed
        self.step_entries = None
        self.current_live_layer = None
        self.counts = collections.Counter()
        self._wrap(monkeypatch)

    # -- reference ---------------------------------------------------------------------------

    def _prefill_requests(self):
        ctx = self.ctx
        dims = ctx.batch_dimensions
        start = ctx.paused_request_count + dims.decode_req_count
        rows = []
        row_base = tok_base = 0
        for j in range(dims.prefill_req_count):
            idx = start + j
            s = int(ctx.request_kv_length_offsets[idx])
            n = int(ctx.request_query_lengths[idx])
            rid = int(ctx.request_ids[idx])
            rows.append(dict(j=j, idx=idx, rid=rid, s=s, n=n, e=s + n, R=row_base, T=tok_base))
            row_base += max(1, -(-n // self.c))
            tok_base += n
        return rows

    def _required(self, r):
        """Boundaries the contract requires staged for prefill row r (when legal)."""
        req = self.requests[r["rid"]]
        L = len(req.prompt_tokens)
        wanted = {(L // self.bs) * self.bs, (-(-r["e"] // self.bs) - 1) * self.bs}
        admission = self.admissions[r["rid"]]  # this step's chunk of the request
        if admission["first"]:
            wanted.add(admission["m"] * self.bs)
        return {B for B in wanted if self._legal(r, B)}

    def _legal(self, r, B):
        return r["s"] < B < r["e"] and B % self.bs == 0 and (B - r["s"]) % self.q == 0

    def _expected_skip(self, req, chunk_length):
        hashes = list(req.precomputed_block_hashes or [])
        touched = -(-chunk_length // self.bs)
        hashes = hashes[: min(touched, len(hashes))]
        m = 0
        while m < len(hashes) and hashes[m] in self.registry.kv_hash_to_block_id:
            m += 1
        cached = [i + 1 for i in range(m) if hashes[i] in self.registry.mamba_hash_to_block_id]
        k = max(cached, default=0)
        if k * self.bs >= chunk_length:
            k = max([i for i in cached if i < k], default=0)
        return m, k * self.bs, (hashes[k - 1] if k > 0 else None)

    # -- wrappers ----------------------------------------------------------------------------

    def _wrap(self, monkeypatch):
        oracle = self
        ctx, md, msa = self.ctx, self.md, self.msa

        add_request = ctx.add_request

        def checked_add_request(req, prefill_chunk_length=None):
            chunk = (
                req.remaining_prompt_length
                if prefill_chunk_length is None
                else prefill_chunk_length
            )
            first = req.finished_chunk_token_count == 0
            if first:
                m, skip, restore_hash = oracle._expected_skip(req, chunk)
            else:
                m, skip, restore_hash = 0, 0, None
            before = req.num_cached_tokens
            add_request(req, prefill_chunk_length)
            oracle.requests[req.request_id] = req
            got = req.num_cached_tokens - before
            assert got == skip, (
                f"request {req.request_id}: skipped {got} prompt tokens, the reference skips {skip}"
                f" (chunk {chunk}, KV-matched blocks {m})"
            )
            oracle.admissions[req.request_id] = dict(
                m=m, skip=skip, restore_hash=restore_hash, first=first
            )
            if not first:
                oracle.counts["continuation chunks"] += 1
            if skip:
                oracle.counts["prefix hits"] += 1
            if first and m * oracle.bs >= chunk and m > 0:
                oracle.counts["whole-chunk KV hits"] += 1

        monkeypatch.setattr(ctx, "add_request", checked_add_request)

        update = md._update_intermediate_metadata

        def checked_update(offsets, counts, real_prefill_count, padded_prefill_count, *a, **k):
            update(offsets, counts, real_prefill_count, padded_prefill_count, *a, **k)
            oracle._check_metadata(offsets, counts, real_prefill_count, padded_prefill_count)

        monkeypatch.setattr(md, "_update_intermediate_metadata", checked_update)

        ssm_prefill = MambaMixer.ssm_prefill

        def tracked_ssm_prefill(mixer, zxBCdt, conv_state, ssm_state, context, *a, **k):
            live = [
                i
                for i, t in enumerate(context.mamba_ssm_states)
                if t.data_ptr() == ssm_state.data_ptr()
            ]
            oracle.current_live_layer = live[0] if len(live) == 1 else None
            return ssm_prefill(mixer, zxBCdt, conv_state, ssm_state, context, *a, **k)

        monkeypatch.setattr(MambaMixer, "ssm_prefill", tracked_ssm_prefill)

        scatter_ssm = mamba_mixer_module.scatter_intermediate_ssm
        scatter_conv = mamba_mixer_module.scatter_intermediate_conv

        def checked_scatter_ssm(states, chunk_indices, real_count, out):
            if not oracle.check_kernels or torch.cuda.is_current_stream_capturing():
                return scatter_ssm(states, chunk_indices, real_count, out)
            before = out.clone()
            scatter_ssm(states, chunk_indices, real_count, out)
            oracle._check_scatter(out, before, kind="ssm", source=states)

        def checked_scatter_conv(src, abs_positions, real_count, out, d_conv):
            if not oracle.check_kernels or torch.cuda.is_current_stream_capturing():
                return scatter_conv(src, abs_positions, real_count, out, d_conv)
            before = out.clone()
            scatter_conv(src, abs_positions, real_count, out, d_conv)
            oracle._check_scatter(out, before, kind="conv", source=src)

        monkeypatch.setattr(mamba_mixer_module, "scatter_intermediate_ssm", checked_scatter_ssm)
        monkeypatch.setattr(mamba_mixer_module, "scatter_intermediate_conv", checked_scatter_conv)

        commit = msa.commit_intermediate_states

        def checked_commit():
            expected = oracle._expected_commits()
            commit()
            oracle._check_commits(expected)

        monkeypatch.setattr(msa, "commit_intermediate_states", checked_commit)

        restore = msa.restore_to_live

        def checked_restore(request_idx, block_id):
            restored = restore(request_idx, block_id)
            oracle._check_restore(request_idx, block_id, restored)
            return restored

        monkeypatch.setattr(msa, "restore_to_live", checked_restore)

        on_evicted = msa._on_mamba_evicted

        def counted_evict(block_ids):
            oracle.counts["evicted Mamba blocks"] += len(block_ids)
            return on_evicted(block_ids)

        monkeypatch.setattr(msa, "_on_mamba_evicted", counted_evict)
        self.registry.set_mamba_evict_callback(counted_evict)

    # -- checks ------------------------------------------------------------------------------

    def _check_metadata(self, offsets, counts, real_prefill_count, padded_prefill_count):
        md = self.md
        max_count = min(padded_prefill_count * 3, md.max_intermediate_count)
        entries = []
        if offsets is not None and real_prefill_count > 0:
            offsets = offsets.cpu()
            counts = counts.cpu()
            rows = self._prefill_requests()
            assert (
                len(rows) == real_prefill_count
            ), f"{len(rows)} prefill requests in the context, {real_prefill_count} in the metadata"
            for r in rows:
                staged = [r["s"] + int(o) for o in offsets[r["j"], : int(counts[r["j"]])]]
                assert (
                    staged == sorted(set(staged)) and len(staged) <= 3
                ), f"request {r['rid']}: staged boundaries {staged} are not ascending and unique"
                bad = [B for B in staged if not self._legal(r, B)]
                assert not bad, (
                    f"request {r['rid']} computes [{r['s']}, {r['e']}): staged boundaries {bad} are"
                    f" not interior block boundaries on the {self.q}-token SSM quantum"
                )
                missing = self._required(r) - set(staged)
                assert not missing, (
                    f"request {r['rid']} computes [{r['s']}, {r['e']}) of a"
                    f" {len(self.requests[r['rid']].prompt_tokens)}-token prompt: boundaries"
                    f" {sorted(missing)} are not staged (staged: {staged})"
                )
                req = self.requests[r["rid"]]
                for i, B in enumerate(staged):
                    block = int(self.ctx.request_to_kv_block_ids[r["idx"]][B // self.bs - 1])
                    recorded = int(md._intermediate_block_ids_cpu[r["idx"], i])
                    assert recorded == block, (
                        f"request {r['rid']}: boundary {B} recorded against KV block {recorded},"
                        f" its block {B // self.bs - 1} is {block}"
                    )
                    entries.append(
                        dict(
                            r,
                            B=B,
                            row=r["R"] + (B - r["s"]) // self.c - 1,
                            pos=r["T"] + (B - r["s"]),
                            block=block,
                            hash=req.precomputed_block_hashes[B // self.bs - 1],
                        )
                    )
                    self.counts["staged boundaries"] += 1
                    if r["j"] > 0:
                        self.counts["staged in a later prefill request"] += 1
                    if self.ctx.batch_dimensions.decode_req_count > 0:
                        self.counts["staged in a mixed step"] += 1
                    if not self.admissions[r["rid"]]["first"]:
                        self.counts["staged in a continuation chunk"] += 1
        real = len(entries)
        assert (
            int(md.intermediate_real_count.item()) == real
        ), f"real count {int(md.intermediate_real_count.item())}, the reference stages {real}"
        chunk_indices = md.intermediate_chunk_indices.cpu()
        abs_positions = md.intermediate_abs_positions.cpu()
        assert chunk_indices.numel() == max_count and abs_positions.numel() == max_count
        ou.assert_exact(
            chunk_indices[:real],
            torch.tensor([e["row"] for e in entries], dtype=torch.int64),
            what="intermediate chunk indices",
        )
        ou.assert_exact(
            abs_positions[:real],
            torch.tensor([e["pos"] for e in entries], dtype=torch.int32),
            what="intermediate absolute positions",
        )
        ou.assert_exact(
            chunk_indices[real:], torch.zeros(max_count - real, dtype=torch.int64), what="padding"
        )
        ou.assert_exact(
            abs_positions[real:],
            torch.full((max_count - real,), self.d_conv, dtype=torch.int32),
            what="padding positions",
        )
        extracting = offsets is not None and real_prefill_count > 0
        assert md.per_request_intermediate_counts == (
            [int(x) for x in counts.tolist()] if extracting else []
        )
        self.step_entries = entries

    def _scratch_layer(self, out, kind):
        scratch = self.md.intermediate_ssm_out if kind == "ssm" else self.md.intermediate_conv_out
        layers = [k for k in range(scratch.shape[0]) if scratch[k].data_ptr() == out.data_ptr()]
        assert len(layers) == 1, f"the {kind} extraction writes outside the scratch layers"
        return layers[0]

    def _check_scatter(self, out, before, *, kind, source):
        torch.cuda.synchronize()
        layer = self._scratch_layer(out, kind)
        assert layer == self.current_live_layer, (
            f"the mixer updating live layer {self.current_live_layer} wrote {kind} scratch layer"
            f" {layer}: commit and restore copy layers by position"
        )
        entries = self.step_entries or []
        for i, e in enumerate(entries):
            if kind == "ssm":
                expected = source[e["row"]].to(out.dtype)
            else:
                expected = source[0, e["pos"] - self.d_conv : e["pos"], :].transpose(0, 1)
                expected = expected.to(out.dtype)
            ou.assert_exact(
                out[i], expected, what=f"{kind} scratch row {i} (request {e['rid']}, B={e['B']})"
            )
        ou.assert_exact(out[len(entries) :], before[len(entries) :], what=f"{kind} rows past real")
        if entries:
            self.counts[f"{kind} extractions checked"] += len(entries)

    def _expected_commits(self):
        """Blocks this step's commit must fill, with the states they must hold."""
        ctx, md = self.ctx, self.md
        expected = []
        for i, e in enumerate(self.step_entries or []):
            expected.append(
                dict(
                    block=e["block"],
                    hash=e["hash"],
                    ssm=md.intermediate_ssm_out[:, i].clone(),
                    conv=md.intermediate_conv_out[:, i].clone(),
                    kind="interior",
                )
            )
        for r in self._prefill_requests():
            if r["e"] % self.bs == 0 and r["e"] > 0:
                req = self.requests[r["rid"]]
                b = r["e"] // self.bs - 1
                if b >= len(req.precomputed_block_hashes):
                    continue
                mamba_idx = int(md.request_to_mamba_state_idx[r["idx"]])
                expected.append(
                    dict(
                        block=int(ctx.request_to_kv_block_ids[r["idx"]][b]),
                        hash=req.precomputed_block_hashes[b],
                        ssm=ctx.mamba_ssm_states[:, mamba_idx].clone(),
                        conv=ctx.mamba_conv_states[:, mamba_idx].clone(),
                        kind="end of chunk",
                    )
                )
        self.step_entries = None
        return expected

    def _check_commits(self, expected):
        msa = self.msa
        for x in expected:
            if self.registry.mamba_hash_to_block_id.get(x["hash"]) != x["block"]:
                assert msa.free_count == 0 or x["hash"] in self.registry.mamba_hash_to_block_id, (
                    f"{x['kind']} state of block {x['block']} was neither committed nor dropped for"
                    " lack of capacity"
                )
                self.counts["commits dropped"] += 1
                continue
            slot = int(msa.block_to_slot[x["block"]])
            assert slot >= 0, f"registered block {x['block']} has no durable slot"
            ou.assert_exact(msa.ssm_states[:, slot], x["ssm"], what=f"{x['kind']} ssm commit")
            ou.assert_exact(msa.conv_states[:, slot], x["conv"], what=f"{x['kind']} conv commit")
            self.cached[x["hash"]] = (x["ssm"], x["conv"])
            self.counts[f"{x['kind']} commits checked"] += 1
        blocks = list(self.registry.mamba_hash_to_block_id.values())
        slots = [int(msa.block_to_slot[b]) for b in blocks]
        assert min(slots, default=0) >= 0, "a registered Mamba block has no durable slot"
        assert len(set(slots)) == len(slots), "two registered Mamba blocks share a durable slot"

    def _check_restore(self, request_idx, block_id, restored):
        ctx, msa = self.ctx, self.msa
        rid = int(ctx.request_ids[request_idx])
        admission = self.admissions[rid]
        assert admission["skip"] > 0, f"request {rid} restored a state without a prefix skip"
        assert (
            restored
        ), f"request {rid} skipped {admission['skip']} tokens but block {block_id} had no state"
        h = admission["restore_hash"]
        assert self.registry.mamba_hash_to_block_id.get(h) == block_id, (
            f"request {rid} restored block {block_id}, the prefix of {admission['skip']} tokens is"
            f" registered at {self.registry.mamba_hash_to_block_id.get(h)}"
        )
        slot = int(msa.block_to_slot[block_id])
        mamba_idx = int(self.md.request_to_mamba_state_idx[request_idx])
        ssm, conv = self.cached[h]
        ou.assert_exact(ctx.mamba_ssm_states[:, mamba_idx], ssm, what=f"request {rid} ssm resume")
        ou.assert_exact(
            ctx.mamba_conv_states[:, mamba_idx], conv, what=f"request {rid} conv resume"
        )
        ou.assert_exact(msa.ssm_states[:, slot], ssm, what=f"durable slot {slot} ssm")
        self.counts["restores checked"] += 1


def _tokens(seed, n):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (n,), generator=generator, dtype=torch.int64)


def _cat(*parts):
    return torch.cat(parts)


def _edges_waves():
    bs = BLOCK_SIZE
    T = {k: _tokens(100 + k, 4 * bs) for k in range(1, 6)}
    U = {k: _tokens(200 + k, 2 * bs) for k in range(1, 6)}
    P = {
        1: T[1][: bs - 1],
        2: T[2][:bs],
        3: T[3][: bs + 1],
        4: T[4][: 5 * bs // 2],
        5: T[5][: 3 * bs],
    }
    q4p = _cat(P[4][:300], U[5][:400])
    return [
        (0, [P[1], P[2], P[3], P[4], P[5]]),
        (
            3,
            [
                _cat(P[3], U[3][:343]),  # full-block hit at bs, then a staged last-aligned 512
                _cat(P[4][: 2 * bs], U[4][:1]),  # hit at 2 bs, one-token resume chunk
                P[5].clone(),  # whole prompt cached at 3 bs: backs off to 2 bs
                _cat(P[2][:200], U[2][:100]),  # partial-block share: no KV block, no skip
                q4p,  # one KV block matched (no Mamba state there): stages its divergence
            ],
        ),
        (6, [_cat(q4p[: 2 * bs], U[1][:88])]),  # resumes from q4p's divergence-chunk state
    ]


def _chunked_waves():
    W = {k: _tokens(300 + k, 4 * BLOCK_SIZE) for k in range(1, 4)}
    return [(0, [W[1][:1000], W[2][:600]]), (8, [_cat(W[1][:1000], W[3][:40]), W[2][:600].clone()])]


def _evict_waves():
    bs = BLOCK_SIZE
    X = {k: _tokens(400 + k, 3 * bs) for k in range(1, 9)}
    first = [X[1][:600], X[2][: 2 * bs], X[3][:700]]
    second = [X[4][:600], X[5][:600], X[6][: 2 * bs]]
    return [
        (0, first),
        (4, [_cat(first[0][: 2 * bs], X[7][:50])]),
        (8, second),
        (12, [_cat(p[: 2 * bs], X[8][:30]) for p in first + second]),
    ]


SCENARIOS = {
    "edges": SimpleNamespace(
        waves=_edges_waves,
        chunked=False,
        max_tokens=4096,
        durable_slots=None,
        graphs=False,
        async_sched_mode="async",
    ),
    "edges-legacy": SimpleNamespace(
        waves=_edges_waves,
        chunked=False,
        max_tokens=4096,
        durable_slots=None,
        graphs=False,
        async_sched_mode="legacy",
    ),
    "edges-graphs": SimpleNamespace(
        waves=_edges_waves,
        chunked=False,
        max_tokens=4096,
        durable_slots=None,
        graphs=True,
        async_sched_mode="async",
    ),
    "chunked": SimpleNamespace(
        waves=_chunked_waves,
        chunked=True,
        max_tokens=384,
        durable_slots=None,
        graphs=False,
        async_sched_mode="async",
    ),
    "chunked-unaligned": SimpleNamespace(
        waves=_chunked_waves,
        chunked=True,
        max_tokens=320,
        durable_slots=None,
        graphs=False,
        async_sched_mode="async",
    ),
    "evict": SimpleNamespace(
        waves=_evict_waves,
        chunked=False,
        max_tokens=4096,
        durable_slots=3,
        graphs=False,
        async_sched_mode="async",
    ),
}

COVERAGE = {
    "edges": [
        "staged boundaries",
        "staged in a later prefill request",
        "staged in a mixed step",
        "prefix hits",
        "whole-chunk KV hits",
        "restores checked",
        "interior commits checked",
        "end of chunk commits checked",
        "ssm extractions checked",
        "conv extractions checked",
    ],
    "chunked": [
        "continuation chunks",
        "staged in a continuation chunk",
        "end of chunk commits checked",
        "restores checked",
        "ssm extractions checked",
    ],
    "evict": ["restores checked", "evicted Mamba blocks", "interior commits checked"],
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_prefix_cache_metadata_matches_reference(model_parallel, monkeypatch, name):
    """Every step's staging, extraction metadata, extracted states, commits and resumes match the
    reference, over block-size edges, partial-block and whole-prompt hits, chunked-prefill
    continuations and eviction."""
    scenario = SCENARIOS[name]
    # The unit-test conftest pins the TE attention backend through NVTE_* variables; the model's
    # config sets them itself (flash-attn 2), and monkeypatch restores them after the test.
    for variable in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(variable, raising=False)
    model = _build_model(scenario.graphs)
    engine = _build_engine(model, scenario)
    if scenario.durable_slots is not None:
        slots = engine.context.mamba_slot_allocator.max_slots
        assert 1 <= slots <= scenario.durable_slots + 1, f"{slots} durable Mamba slots"
    oracle = _Oracle(engine, monkeypatch, check_kernels=not scenario.graphs)
    waves = scenario.waves()
    next_id, step = 0, 0
    pending = sorted(waves, key=lambda w: w[0])
    while pending or engine.has_unfinished_requests():
        while pending and pending[0][0] <= step:
            for prompt in pending.pop(0)[1]:
                engine._add_request(
                    DynamicInferenceRequest(
                        request_id=next_id,
                        prompt_tokens=prompt.cuda(),
                        sampling_params=SamplingParams(
                            num_tokens_to_generate=4, termination_id=-1, top_k=1
                        ),
                        block_size_tokens=BLOCK_SIZE,
                        enable_prefix_caching=True,
                    )
                )
                next_id += 1
        if engine.has_unfinished_requests():
            engine.step_modern()
        step += 1
        assert step < MAX_STEPS, "the requests did not finish"
    torch.cuda.synchronize()
    del engine
    _reset_cuda_graph_state(model)

    required = COVERAGE.get(name.split("-")[0] if name != "chunked-unaligned" else "chunked", [])
    if name == "chunked-unaligned":
        required = ["continuation chunks", "end of chunk commits checked", "restores checked"]
    if scenario.graphs:
        required = [c for c in required if "extractions" not in c]
    missing = [c for c in required if oracle.counts[c] == 0]
    assert not missing, f"the scenario never reached {missing}; counts: {dict(oracle.counts)}"


# ---------------------------------------------------------------------------------------------
# Direct tests of the two extraction kernels (the engine cases cannot select their tile edges)
# ---------------------------------------------------------------------------------------------

SSM_BLOCK_N = 1024  # columns per program, intermediate_extraction.py:scatter_intermediate_ssm
CONV_BLOCK_C = 128  # channels per program, intermediate_extraction.py:scatter_intermediate_conv
NEMOTRON_H_8B_SSM_STATE = (
    128,
    64,
    128,
)  # heads, head dim, state dim of a Nemotron-H 8B Mamba layer
NEMOTRON_H_8B_CONV_DIM = 10240  # d_inner 8192 + 2 x 8 groups x 128 state
TEST_MODEL_CONV_DIM = 3072  # the engine cases' layer: d_inner 1024 + 2 x 8 groups x 128 state
CHUNK_ROWS = [7, 0, 8, 3]  # scratch slot -> source chunk row: out of order, first and last rows


def _ssm_case(state_shape, real, source_dtype, generator):
    from megatron.core.ssm.ops.common.intermediate_extraction import scatter_intermediate_ssm

    states = torch.randn((9,) + state_shape, generator=generator, device="cuda").to(source_dtype)
    rows = torch.tensor(CHUNK_ROWS, dtype=torch.int64, device="cuda")
    out = ou.guarded_empty((len(CHUNK_ROWS),) + state_shape, torch.float32)
    real_count = torch.tensor([real], dtype=torch.int32, device="cuda")
    before = states.clone()
    scatter_intermediate_ssm(states, rows, real_count, out.tensor)
    ou.check_guards(out=out)
    ou.assert_exact(out.tensor[:real], states[rows[:real]].to(torch.float32), what="gathered rows")
    ou.assert_poisoned(out.tensor[real:], what="slots past the real count")
    ou.assert_exact(states, before, what="states are read-only")


def _conv_case(conv_dim, d_conv, real, strided, generator):
    from megatron.core.ssm.ops.common.intermediate_extraction import scatter_intermediate_conv

    seq_len = 300
    width = conv_dim + 40 if strided else conv_dim
    packed = torch.randn((1, seq_len, width), generator=generator, device="cuda").bfloat16()
    src = packed[..., 17 : 17 + conv_dim] if strided else packed  # the mixer's xBC is a split view
    ends = torch.tensor([d_conv, 129, seq_len, 200], dtype=torch.int32, device="cuda")
    out = ou.guarded_empty((len(ends), conv_dim, d_conv), torch.bfloat16)
    real_count = torch.tensor([real], dtype=torch.int32, device="cuda")
    scatter_intermediate_conv(src, ends, real_count, out.tensor, d_conv=d_conv)
    ou.check_guards(out=out)
    for i, end in enumerate(ends[:real].tolist()):
        window = src[0, end - d_conv : end, :].transpose(0, 1)
        ou.assert_exact(out.tensor[i], window, what=f"conv window ending at {end}")
    ou.assert_poisoned(out.tensor[real:], what="slots past the real count")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "state_shape",
    [
        (1, 1, SSM_BLOCK_N - 1),
        (1, 1, SSM_BLOCK_N),
        (1, 1, SSM_BLOCK_N + 1),
        (16, 64, 128),
        NEMOTRON_H_8B_SSM_STATE,
    ],
    ids=["1023", "1024", "1025", "test-model", "nemotron-h-8b"],
)
@pytest.mark.parametrize("real", [0, 1, 4])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_scatter_intermediate_ssm_gathers_rows(state_shape, real, source_dtype):
    """Slot i < real holds row chunk_indices[i] of the states converted to the cache dtype
    (exact); slots past real are not written; nothing outside the scratch is written. Widths
    around the 1024-column program; the engine cases' layer (16 heads x 64 x 128) and a
    Nemotron-H 8B layer (128 x 64 x 128, a million elements per state)."""
    _ssm_case(state_shape, real, source_dtype, ou.make_generator(device="cuda"))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "conv_dim",
    [CONV_BLOCK_C - 1, CONV_BLOCK_C, CONV_BLOCK_C + 1, TEST_MODEL_CONV_DIM, NEMOTRON_H_8B_CONV_DIM],
    ids=str,
)
@pytest.mark.parametrize("d_conv", [2, 4])
@pytest.mark.parametrize("real", [0, 1, 4])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_scatter_intermediate_conv_gathers_windows(conv_dim, d_conv, real, strided):
    """Slot i < real holds the inputs at positions [end_i - d_conv, end_i), transposed to
    (conv_dim, d_conv) (exact), including windows ending at d_conv and at the last token; slots
    past real are not written. Channels around the 128-channel program, the engine cases' layer
    (3072) and a Nemotron-H 8B layer (10240); the strided source is the mixer's split view of the
    projection."""
    _conv_case(conv_dim, d_conv, real, strided, ou.make_generator(device="cuda"))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """The compute-sanitizer case: both kernels at one tile edge past the block width."""
    generator = ou.make_generator(device="cuda")
    _ssm_case((1, 1, SSM_BLOCK_N + 1), 4, torch.bfloat16, generator)
    _conv_case(CONV_BLOCK_C + 1, 4, 4, True, generator)
