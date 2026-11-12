# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay dev-only kernel families when integrating main's whole-tree registry.

These are real kernel entry points, not numerical stand-ins. Architecture/dependency
requirements remain explicit; skipped Blackwell kernels are not qualified on Hopper.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    deterministic_algorithms,
    seeded,
)

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


@pytest.mark.parametrize("impl", ["eager", "fla"])
def test_mtp_attn_res_final_head_replays(impl):
    """Replay the AttnRes output head an MTP depth applies over the trunk depth-source tuple.

    ``MultiTokenPredictionLayer._postprocess`` aggregates ``[*attn_res_sources, hidden]`` with
    the depth's ``final_attn_res`` before the final layernorm; the tuple reaches it through
    ``MultiTokenPredictionInputs.attn_res_sources``.
    """
    from megatron.core.transformer.attention_residual import (
        AttentionResidual,
        _get_fla_fused_attnres,
    )
    from megatron.core.transformer.multi_token_prediction import (
        MultiTokenPredictionInputs,
        MultiTokenPredictionLayer,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    if impl == "fla" and _get_fla_fused_attnres() is None:
        pytest.skip("flash-linear-attention with fused AttnRes is unavailable")
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        enable_attention_residuals=True,
        attn_res_block_layers=2,
        attn_res_impl=impl,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
    )

    class FinalHead(torch.nn.Module):
        """The attributes ``_postprocess`` reads, built as MultiTokenPredictionLayer builds them."""

        def __init__(self):
            super().__init__()
            self.config = config
            self.attn_res_enabled = config.enable_attention_residuals
            self.mhc_enabled = False
            self.final_attn_res = AttentionResidual(config)
            self.final_layernorm = torch.nn.RMSNorm(
                config.hidden_size, eps=config.layernorm_epsilon, dtype=torch.bfloat16
            )

        def forward(self, hidden_states, attn_res_sources):
            inputs = MultiTokenPredictionInputs(
                input_ids=None,
                position_ids=None,
                hidden_states=hidden_states,
                decoder_input=None,
                mhc_multistream=None,
                labels=None,
                loss_mask=None,
                mtp_input_mask=None,
                packed_seq_params=None,
                attn_res_sources=attn_res_sources,
            )
            assert isinstance(inputs.attn_res_sources, tuple)
            return MultiTokenPredictionLayer._postprocess(
                self, inputs.hidden_states, attn_res_sources=inputs.attn_res_sources
            )

    seeded()
    head = FinalHead().cuda()
    with torch.no_grad():
        head.final_attn_res.pseudo_query.normal_(std=0.05)
        head.final_attn_res.key_norm_weight.uniform_(0.8, 1.2)
    sources = tuple(
        torch.randn(512, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    hidden = torch.randn(512, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs, grads = assert_module_replays_bit_exact(
        head,
        {"hidden_states": hidden, "attn_res_sources": sources},
        contention=True,
        what=f"MTP AttnRes final head ({impl})",
    )
    assert len(outputs) == 1
    # pseudo_query, key_norm_weight, final_layernorm.weight, the hidden state and 3 sources.
    assert len(grads) == 7


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


def test_kda_low_rank_projection_replays():
    """Replay KimiDeltaAttention through its bias-free low-rank F-decay / output-gate path.

    ``kda_f_lora_rank`` / ``kda_gate_lora_rank`` take the layer off the legacy fused
    Q/K/V/F/G projection: the input projection emits Q/K/V only, the raw decay comes from
    ``f_a_proj -> f_b_proj``, the output gate from ``g_a_proj -> g_b_proj`` and beta from a
    TP-replicated projection scattered back to the rank, all feeding the FLA chunk_kda
    recurrence. Under ``--deterministic-mode`` the convolution and the L2 norm take their
    torch paths; forward and backward must replay bit-exactly, parameter gradients included.
    """
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import HAVE_TE
    from megatron.core.ssm.gated_delta_net import kda

    if kda.chunk_kda is None:
        pytest.skip("FLA KDA kernel is unavailable")
    if not HAVE_TE:
        pytest.skip("needs Transformer Engine")

    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    f_rank, gate_rank = 16, 32
    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        seeded()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            kda_f_lora_rank=f_rank,
            kda_gate_lora_rank=gate_rank,
            normalization="RMSNorm",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            experimental_attention_variant="kda",
            is_hybrid_model=True,
            transformer_impl="transformer_engine",
            deterministic_mode=True,
        )
        spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
        module = kda.KimiDeltaAttention(
            config,
            submodules=spec.submodules,
            layer_number=1,
            conv_init=1.0,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        ).cuda()
        assert not module.use_legacy_fused_projections
        assert module.in_proj_split_names == ["query", "key", "value"]
        assert module.in_proj_dim == 2 * module.qk_dim + module.v_dim
        assert module.in_proj.weight.shape == (module.in_proj_dim, config.hidden_size)
        assert module.f_a_proj.weight.shape == (f_rank, config.hidden_size)
        assert module.f_b_proj.weight.shape == (module.qk_dim, f_rank)
        assert module.g_a_proj.weight.shape == (gate_rank, config.hidden_size)
        assert module.g_b_proj.weight.shape == (module.v_dim, gate_rank)
        assert module.beta_proj.weight.shape == (module.num_key_heads, config.hidden_size)
        assert module.gated_delta_rule is kda.chunk_kda
        # The streamed pre-GDR fusion (gdn_pre_gated_delta_rule_fusion) is opt-in and rejected
        # under deterministic_mode; the deterministic replay stays on pre_gated_delta_rule.
        assert not module.gdn_pre_gated_delta_rule_fusion
        # So is kda_kernel_backend='cudnn' (test_ssm_kernels.py): this replay covers the FLA
        # backend, the only one deterministic_mode accepts.
        assert config.kda_kernel_backend == "fla"

        hidden_states = torch.randn(
            128, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with (
            deterministic_algorithms(True),
            patch.object(
                kda.KimiDeltaAttention, "_fused_streamed_pre_kda", autospec=True
            ) as fused_pre_kda,
        ):
            outputs, grads = assert_module_replays_bit_exact(
                module,
                (hidden_states, None),
                replays=3,
                contention=True,
                what="KDA low-rank F/G projections",
            )
        assert fused_pre_kda.call_count == 0
        assert outputs["out[0]"].shape == hidden_states.shape
        assert "in[0]" in grads
        for name in ("in_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "beta_proj"):
            assert f"chunk0.{name}.weight" in grads, name
        for name in ("chunk0.conv1d.weight", "chunk0.A_log", "chunk0.dt_bias"):
            assert name in grads, name
    finally:
        Utils.destroy_model_parallel()


def test_kda_recurrence_with_fused_gated_output_norm_replays():
    """Replay the KDA recurrence feeding its fused sigmoid output norm (PR #7609).

    With ``gdn_gated_output_norm_fusion`` set, ``KimiDeltaAttention._apply_gated_norm`` routes
    the ``chunk_kda`` output through ``fused_gated_norm(gate_activation="sigmoid")`` instead of
    the jit-fused norm and gate. The chain gates -> chunk_kda -> fused RMSNorm/sigmoid is
    replayed forward and backward with a projection-backed strided gate, comparing every
    input gradient and the RMSNorm weight gradient.
    """
    from megatron.core.fusions import fused_gated_norm
    from megatron.core.ssm.gated_delta_net import kda

    if kda.chunk_kda is None:
        pytest.skip("FLA KDA kernel is unavailable")
    te = pytest.importorskip("transformer_engine.pytorch")
    seeded()
    batch, length, heads, dim = 1, 128, 4, 64
    shape = (batch, length, heads, dim)
    q, k, v = (
        torch.randn(shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for _ in range(3)
    )
    raw_g = torch.randn(batch, length, heads * dim, device="cuda", requires_grad=True)
    raw_beta = torch.randn(batch, length, heads, device="cuda", requires_grad=True)
    a_log = torch.zeros(heads, device="cuda", requires_grad=True)
    dt_bias = torch.zeros(heads * dim, device="cuda", requires_grad=True)
    # The output gate is a strided slice of the in_proj output, as in pre_gated_delta_rule.
    projection = torch.randn(
        length, batch, heads * dim * 2 + 32, device="cuda", dtype=torch.bfloat16
    )
    gate = projection[..., 17 : 17 + heads * dim].view(length, batch, heads, dim)
    gate = gate.permute(1, 0, 2, 3).detach().requires_grad_()
    gate_layer = SimpleNamespace(key_head_dim=dim)

    class KDAOutput(torch.nn.Module):
        """The KDA attributes the fused dispatcher and ``validate_gated_norm`` read."""

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                deterministic_mode=False, gdn_gated_output_norm_fusion=True
            )
            self.activation = "silu"
            self.out_norm = te.RMSNorm(dim, eps=1e-6, params_dtype=torch.bfloat16, device="cuda")
            self.value_head_dim = dim

        def forward(self, q, k, v, g, beta, a_log, dt_bias, gate):
            decay, extra = kda.KimiDeltaAttention._compute_gates(
                gate_layer, a_log, dt_bias, batch, length, g, beta
            )
            core_attn_out, _ = kda.chunk_kda(
                q=q,
                k=k,
                v=v,
                g=decay,
                beta=extra["beta"],
                A_log=a_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                initial_state=None,
                output_final_state=False,
                safe_gate=False,
            )
            out = kda.KimiDeltaAttention._apply_gated_norm(self, core_attn_out, gate)
            assert out.shape == (batch * length * heads, dim)
            return out

    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "g": raw_g,
        "beta": raw_beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "gate": gate,
    }
    with patch.object(
        fused_gated_norm, "fused_gated_norm", wraps=fused_gated_norm.fused_gated_norm
    ) as fused:
        outputs, grads = assert_module_replays_bit_exact(
            KDAOutput(),
            inputs,
            replays=3,
            contention=True,
            what="KDA recurrence with fused sigmoid output norm",
        )
        assert fused.call_count == 3
        assert all(call.kwargs.get("gate_activation") == "sigmoid" for call in fused.call_args_list)
    assert len(outputs) == 1
    # A_log / dt_bias gradients are compared too whenever chunk_kda produces them.
    compared = {"in.q", "in.k", "in.v", "in.g", "in.beta", "in.gate", "chunk0.out_norm.weight"}
    assert compared <= set(grads)


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
