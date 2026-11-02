# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay dev-only kernel families when integrating main's whole-tree registry.

These are real kernel entry points, not numerical stand-ins. Architecture/dependency
requirements remain explicit; skipped Blackwell kernels are not qualified on Hopper.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_nonuniform_tp_mapping_replays(monkeypatch):
    """Replay the actual CPU integer bincount used to construct NTP shard metadata."""
    from megatron.core.distributed import nonuniform_tp

    module = torch.nn.Linear(8, 8, bias=False)
    module.weight.tensor_model_parallel = True
    module.weight.partition_dim = 0
    config = SimpleNamespace(tp_base=4, tp_spares=1)
    monkeypatch.setattr(nonuniform_tp.dist, "get_rank", lambda: 0)
    for name in (
        "get_data_parallel_rank",
        "get_context_parallel_rank",
        "get_pipeline_model_parallel_rank",
    ):
        monkeypatch.setattr(nonuniform_tp.parallel_state, name, lambda: 0)
    monkeypatch.setattr(nonuniform_tp, "_ntp_get_non_active_ranks", lambda *args: None)

    def map_shards():
        nonuniform_tp.ntp_map(module, config, num_shards=16)
        return torch.tensor(module.weight.send_splits), torch.tensor(module.weight.recv_splits)

    assert_replays_bit_exact(map_shards, (), backward=False, what="NTP integer shard mapping")


def test_linear_cross_entropy_replays():
    """Exercise the Blackwell fused GEMM/loss and both input and weight backward kernels."""
    from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused linear cross entropy targets Blackwell")
    pytest.importorskip("cutlass")
    seeded()
    hidden = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 1024, (256,), device="cuda")
    assert_replays_bit_exact(
        lambda h, w: linear_cross_entropy(h, w, labels, reduction="none"),
        (hidden, weight),
        contention=True,
        what="fused linear cross entropy",
    )


def test_fused_mrope_replays():
    """Run the fused SBHD multimodal rotary forward and backward, with no native fallback."""
    from megatron.core.fusions.fused_mrope import fused_apply_mrope

    pytest.importorskip("triton")
    seeded()
    value = torch.randn(128, 2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    freqs = torch.randn(3, 2, 128, 32, device="cuda")
    assert_replays_bit_exact(
        lambda x: fused_apply_mrope(x, freqs, [16, 8, 8]),
        (value,),
        contention=True,
        what="fused mRoPE",
    )


def test_mtp_packed_cp1_roll_replays():
    """Replay physical THD padding with fewer logical than padded boundaries."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.multi_token_prediction import roll_tensor

    logical = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    physical = torch.tensor([0, 3, 5, 7, 7], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        qkv_format="thd",
    )
    value = torch.arange(1, 8, dtype=torch.float32, device="cuda", requires_grad=True)

    def roll(x):
        return roll_tensor(x, packed_seq_params=packed, return_sum=False)[0]

    assert torch.equal(roll(value), value.new_tensor([2, 3, 0, 5, 0, 0, 0]))
    assert_replays_bit_exact(roll, (value,), contention=True, what="MTP packed CP1 roll")


def test_mtp_packed_cp1_roll_replays_in_a_cuda_graph_across_packings():
    """Chunk CUDA graphs capture the packed roll with static ``cu_seqlens`` buffers whose values
    change per replay (repeated final boundaries pad the entry count). The roll plan therefore
    keeps empty slots instead of compacting them with a data-dependent boolean index, and the
    captured graph must reproduce the eager roll of every packing that fits the static slots."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.multi_token_prediction import roll_tensor

    capacity, slots = 8, 4

    def cu_seqlens(seqlens):
        boundaries = [0]
        for length in seqlens:
            boundaries.append(boundaries[-1] + length)
        # A dummy tail sequence pads the tokens to the capacity; the final boundary repeats so
        # the metadata always has ``slots + 1`` entries (static THD padding).
        if boundaries[-1] < capacity:
            boundaries.append(capacity)
        boundaries += [capacity] * (slots + 1 - len(boundaries))
        return torch.tensor(boundaries, dtype=torch.int32, device="cuda")

    def roll(x, cu):
        packed = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            qkv_format="thd",
        )
        return roll_tensor(x, packed_seq_params=packed, return_sum=False)[0]

    packings = ([3, 2], [4, 3], [8], [1, 1, 1])
    value = torch.arange(1, capacity + 1, dtype=torch.float32, device="cuda")
    references = [roll(value, cu_seqlens(seqlens)) for seqlens in packings]
    assert torch.equal(references[0], value.new_tensor([2, 3, 0, 5, 0, 7, 8, 0]))

    static_x = value.clone()
    static_cu = cu_seqlens(packings[0]).clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            roll(static_x, static_cu)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = roll(static_x, static_cu)
    for seqlens, reference in zip(packings, references):
        static_cu.copy_(cu_seqlens(seqlens))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_out, reference), seqlens


def test_mtp_prefix_replays():
    """Include an exact-zero acceptance in the fused prefix-product backward."""
    from megatron.core.fusions.fused_mtp_prefix import (
        fused_mtp_prefix_unavailable_reason,
        mtp_e2e_prefix_objective,
    )

    seeded()
    acceptance = torch.rand(7, 257, 2, device="cuda")
    acceptance[3, 0] = 0
    acceptance.requires_grad_()
    assert fused_mtp_prefix_unavailable_reason(acceptance) is None
    assert_replays_bit_exact(
        mtp_e2e_prefix_objective, (acceptance,), contention=True, what="MTP prefix objective"
    )


def test_mtp_tv_replays():
    """Replay the fused full-vocabulary TV reduction and draft-logit gradient."""
    from megatron.core.fusions.fused_mtp_tv import (
        fused_mtp_tv_unavailable_reason,
        vocab_parallel_tv_distance,
    )

    seeded()
    draft = torch.randn(17, 1025, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn_like(draft)
    assert fused_mtp_tv_unavailable_reason(draft, target) is None
    assert_replays_bit_exact(
        lambda x: vocab_parallel_tv_distance(x, target, logits_are_vocab_sharded=False),
        (draft,),
        contention=True,
        what="MTP TV distance",
    )


@pytest.mark.parametrize("implementation", ["core", "mot"])
def test_bagel_compiled_flex_attention_replays(implementation):
    """Replay the actual compiled callable exported by each BAGEL attention module."""
    from torch.nn.attention.flex_attention import create_block_mask

    from megatron.core.models.bagel import flex_attention, transformer_mot_layer

    seeded()
    fn = (
        flex_attention.compiled_flex_attention
        if implementation == "core"
        else transformer_mot_layer.flex_attention
    )
    mask = create_block_mask(lambda b, h, q, k: q >= k, 1, 2, 128, 128, device="cuda")
    inputs = tuple(
        torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    assert_replays_bit_exact(
        lambda q, k, v: fn(q, k, v, block_mask=mask),
        inputs,
        contention=True,
        what=f"BAGEL {implementation} compiled flex attention",
    )


def test_indexer_scale_prefix_replays():
    """Replay the torch.compile THD scale-prefix refresh on integer GPU metadata."""
    from megatron.core.quantization.indexer_quantization import (
        refresh_indexer_mxfp8_scale_cu_seqlens,
    )

    boundaries = torch.tensor([0, 17, 17, 130, 257], dtype=torch.int32, device="cuda")

    def refresh(cu):
        destination = torch.empty_like(cu)
        refresh_indexer_mxfp8_scale_cu_seqlens(destination, cu, num_heads=4)
        return destination

    assert_replays_bit_exact(
        refresh, (boundaries,), backward=False, contention=True, what="indexer scale prefixes"
    )


def test_indexer_mxfp8_quantize_replays():
    """Exercise TE quantization and the Triton physical-scale packer on Blackwell."""
    from megatron.core.quantization import indexer_quantization

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("MXFP8 indexer quantization requires Blackwell")
    if not indexer_quantization.HAVE_TE_MXFP8:
        pytest.skip("TE MXFP8 quantizer is unavailable")
    seeded()
    inputs = torch.randn(2, 129, 4, 128, dtype=torch.bfloat16, device="cuda")
    assert_replays_bit_exact(
        indexer_quantization.quantize_indexer_mxfp8,
        (inputs,),
        backward=False,
        contention=True,
        what="indexer MXFP8 quantization and scale packing",
    )


def test_kda_recurrence_replays():
    """Replay KDA's raw gate preparation and the selected FLA recurrence at CP1."""
    from megatron.core.ssm.gated_delta_net import kda

    if kda.chunk_kda is None:
        pytest.skip("FLA KDA kernel is unavailable")
    seeded()
    shape = (1, 128, 4, 64)
    q, k, v = (
        torch.randn(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for _ in range(3)
    )
    raw_g = torch.randn(1, 128, 256, device="cuda", requires_grad=True)
    raw_beta = torch.randn(1, 128, 4, device="cuda", requires_grad=True)
    a_log = torch.zeros(4, device="cuda", requires_grad=True)
    dt_bias = torch.zeros(256, device="cuda", requires_grad=True)
    layer = SimpleNamespace(key_head_dim=64)

    def run(q, k, v, g, beta, a_log, dt_bias):
        gate, extra = kda.KimiDeltaAttention._compute_gates(layer, a_log, dt_bias, 1, 128, g, beta)
        output, _ = kda.chunk_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=extra["beta"],
            A_log=a_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=None,
            output_final_state=False,
            safe_gate=False,
        )
        return output

    assert_replays_bit_exact(
        run, (q, k, v, raw_g, raw_beta, a_log, dt_bias), contention=True, what="KDA recurrence"
    )


def test_mok_route_adapter_replays():
    """Replay the fixed-width route gather and its probability-gradient scatter."""
    from megatron.core.transformer.moe.megakernel.mok.route_adapter import routing_map_to_mok_inputs

    seeded()
    probs = torch.rand(1025, 16, device="cuda", requires_grad=True)
    routing = torch.zeros_like(probs, dtype=torch.bool)
    routing[:, 0] = True
    routing[1::2, 7] = True
    assert_replays_bit_exact(
        lambda p: routing_map_to_mok_inputs(p, routing, topk=2),
        (probs,),
        contention=True,
        what="MOK route adapter",
    )


@pytest.mark.parametrize("rank", range(4))
def test_balanced_route_metadata_replays(rank, monkeypatch):
    """Rebuild integer bincount/sort routing metadata, without executing the deferred indexer."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer

    group = SimpleNamespace(size=lambda: 4, rank=lambda: rank)
    for name in ("_LAST_PLAN", "_SEEN_CU", "_ZZ_PACK_OK"):
        monkeypatch.setattr(cp_balanced_indexer, name, {})
    boundaries = torch.tensor([0, 8192, 8192, 12288, 16384], device="cuda", dtype=torch.int32)

    def build(cu):
        cp_balanced_indexer._LAST_PLAN.clear()
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
        cp_balanced_indexer.prebuild_balanced_layouts(packed, cp_group=group)
        plan = packed._dsa_cp_balance_layout_cache[("zigzag", rank)]
        assert sum(plan["disp_in_splits"]) == 4096
        assert sum(plan["disp_out_splits"]) == 4096
        return {
            **{
                key: plan[key]
                for key in (
                    "gather_idx",
                    "inv_idx",
                    "disp_send_rows",
                    "disp_recv_rows",
                    "cmb_send_rows",
                    "cmb_recv_rows",
                )
            },
            "counts": torch.tensor(plan["disp_out_splits"], device=cu.device),
        }

    assert_replays_bit_exact(
        build, (boundaries,), backward=False, contention=True, what="balanced route metadata"
    )
