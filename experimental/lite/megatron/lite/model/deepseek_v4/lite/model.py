# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek V4 (ds4flash) lite native model.

A clone of the Kimi-K2 (deepseek_v3) lite model -- same approach as GLM-5 --
inheriting Kimi's Megatron plumbing (SBHD ``[S, B, H]`` layout, VocabParallel
embed/head, ``set_input_tensor``, ``build_pipeline_chunk_layout`` boundaries,
MTP via ``layout.has_mtp``, dist-opt/distckpt via the protocol).  Three
model-wide DS4 deviations (documented inline at each site):

1. Attention = CSA, not MLA.  The CSA primitive is batch-first ``[B, S, H]`` and
   needs explicit ``position_ids``; ``DeepseekV4CSAAttention`` wraps it with an
   SBHD<->BSHD shim (like GLM-5's DSA shim).  Per-layer behaviour is driven by
   ``config.compress_ratios[layer_idx]`` inside CSA.

2. mHC (multi-head hyper-connection): the hidden carries ``hc_mult`` parallel
   residual streams (4-D ``[S, B, hc_mult, H]`` in SBHD), expanded after embed
   and contracted before the head, persisting across layers and PP stages; each
   layer wraps attn/FFN in a ``HyperConnection``.  At PP boundaries the 4-D
   hidden folds to 3-D ``[S, B, hc_mult*H]`` to match the P2P buffer
   (``_infer_pipeline_tensor_shape`` scales hidden by ``hc_mult``); fold/unfold
   live in ``primitive/parallel/mhc.py``.

3. MoE = hash-routed DeepSeek: first ``num_hash_layers`` use a token-id hash
   route, the rest the shared sigmoid-topk router (``DeepseekV4MoE`` over the
   shared Experts/Router/Dispatcher).

CSA is not TP-capable, so DS4 is a documented TP=1 case (protocol gate raises
for TP>1/ETP>1); VPP/PP/EP/CP work, inherited from the Kimi skeleton.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.modules.attention.hca import HyperConnection
from megatron.lite.primitive.modules.attention.mhc import MultiHeadHyperConnectionHead
from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ParallelState,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.mhc import (
    contract_mhc_hidden_for_pipeline,
    expand_mhc_hidden_for_pipeline,
    fold_mhc_hidden_for_pipeline,
    unfold_mhc_hidden_from_pipeline,
)
from megatron.lite.primitive.parallel.thd import _roll_packed_thd_left_local
from megatron.lite.primitive.utils import build_fp8_recipe


def _roll_mtp_left(
    tensor: torch.Tensor,
    *,
    packed_seq_params=None,
    dims: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift labels/ids one position left (next-token target for MTP depth d).

    THD-packed inputs concatenate multiple sequences, so a plain ``torch.roll``
    leaks the next sequence's first token across the boundary. Route through the
    per-sequence THD-aware roll (contiguous cu_seqlens at CP==1, where DS4 MTP
    runs) when packed_seq_params is available; fall back to plain roll otherwise.
    """
    cu_seqlens = None if packed_seq_params is None else getattr(packed_seq_params, "cu_seqlens_q", None)
    if cu_seqlens is not None:
        # DS4 uses a CONTIGUOUS THD/CP layout (not TE/Megatron zigzag), so roll
        # per-sequence via cu_seqlens directly -- never the zigzag reconstruction
        # in roll_packed_thd_left (which GLM/Kimi use for their zigzag layout).
        dim = dims if dims >= 0 else tensor.dim() + dims
        return _roll_packed_thd_left_local(tensor, cu_seqlens_padded=cu_seqlens, dims=dim)
    dim = dims if dims >= 0 else tensor.dim() + dims
    rolled = torch.roll(tensor, shifts=-1, dims=dim)
    rolled.select(dim, -1).zero_()
    return rolled, rolled.sum()


# -- DS4 ONLY: CSA attention wrapper.  Holds ``CompressedSparseAttention`` and
# adapts it to the skeleton's SBHD-in / SBHD-out attention contract, mirroring
# GLM-5's ``Glm5DSAAttention`` DSA shim.  Per-layer behaviour (window / compress
# / indexer) is selected inside CSA via ``config.compress_ratios[layer_idx]``.
class DeepseekV4CSAAttention(nn.Module):
    """SBHD ``[S, B, H]`` shim around the batch-first CSA primitive.

    The skeleton hands attention a 3-D SBHD tensor ``[S, B, H]`` (the
    hyper-connection has already collapsed the ``hc_mult`` streams to a single
    pre-mix stream).  CSA is hard-wired ``[B, S, H]`` and needs ``position_ids``.
    This wrapper transposes ``[S, B, H] -> [B, S, H]``, runs CSA, and transposes
    the ``[B, S, H]`` output back to ``[S, B, H]``.  The skeleton therefore never
    observes the batch-first interior.
    """

    def __init__(self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState):
        super().__init__()
        self.ps = ps
        self.self_attn = CompressedSparseAttention(config, layer_idx=layer_idx, ps=ps)

    def forward(
        self, x: torch.Tensor, *, position_ids: torch.Tensor, packed_seq_params: Any = None
    ) -> torch.Tensor:
        # Skeleton feeds SBHD [S, B, H]; CSA needs batch-first [B, S, H].
        # packed_seq_params (cu_seqlens etc.) passes through unchanged: the THD
        # token axis lives in S, so the [S, B] <-> [B, S] transpose leaves it alone.
        x_bsh = x.transpose(0, 1).contiguous()
        out_bsh = self.self_attn(
            x_bsh,
            position_ids=position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        # Back to SBHD [S, B, H] for the skeleton.
        return out_bsh.transpose(0, 1).contiguous()


class DeepseekV4Layer(nn.Module):
    """One decoder layer.

    Structurally the Kimi layer, but the plain ``x = x + sublayer(norm(x))``
    residual is replaced by DS4's per-layer ``HyperConnection`` (mHC), and
    attention is CSA via the SBHD shim above.  The hidden is the 4-D SBHD mHC
    tensor ``[S, B, hc_mult, H]`` end-to-end.  Attribute names (``self_attn`` /
    ``input_layernorm`` / ``post_attention_layernorm`` / ``mlp`` / ``attn_hc`` /
    ``ffn_hc``) are preserved from the previous DS4 model so the HF checkpoint
    weight names are unchanged.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ps = ps
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # DS4 ONLY: CSA attention behind the SBHD shim (Kimi builds MLA here).
        self.self_attn = DeepseekV4CSAAttention(config, layer_idx=layer_idx, ps=ps)
        # DS4 ONLY: hash-routed MoE family (shared Experts/Router/dispatcher).
        self.mlp = DeepseekV4MoE(config, ps, layer_idx=layer_idx, use_deepep=use_deepep)
        # DS4 ONLY: per-layer multi-head hyper-connections wrapping attn + ffn.
        self.attn_hc = HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )
        self.ffn_hc = HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        packed_seq_params: Any = None,
    ) -> torch.Tensor:
        # x is SBHD mHC [S, B, hc_mult, H].  HyperConnection collapses the
        # streams to a 3-D [S, B, H] pre-mix, the sub-block runs SBHD, and
        # HyperConnection.post recombines into the 4-D residual streams.
        residual = x
        attn_in, post, comb = self.attn_hc(x)
        attn_out = self.self_attn(
            self.input_layernorm(attn_in),
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        x = HyperConnection.post(attn_out, residual, post, comb)

        residual = x
        ffn_in, post, comb = self.ffn_hc(x)
        # DS4 hash-routed MoE indexes tid2eid[input_ids.reshape(-1)] and must
        # align with the flattened hidden.  The skeleton is SBHD, so the FFN
        # input flattens in (S, B) order; transpose input_ids [B, S] -> [S, B]
        # so its flatten matches.  (No-op semantics for non-hash layers.)
        mlp_input_ids = None if input_ids is None else input_ids.transpose(0, 1).contiguous()
        ffn_out = self.mlp(self.post_attention_layernorm(ffn_in), input_ids=mlp_input_ids)
        return HyperConnection.post(ffn_out, residual, post, comb)


class DeepseekV4MTPLayer(DeepseekV4Layer):
    """MTP depth layer: Kimi's MTP combiner, adapted to DS4's mHC hidden.

    Like Kimi's ``KimiK2MTPLayer`` it owns ``enorm`` / ``hnorm`` and a projection
    to fuse the rolled-token embedding with the running hidden, then runs one
    transformer layer.  DS4 differences: the projection uses DS4's
    ``e_proj`` / ``h_proj`` names (preserved for HF export), the embedding /
    hidden are lifted into the ``hc_mult`` streams, and ``contract`` collapses
    the streams via ``hc_head`` + ``norm`` before the (shared) head.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        embedding: VocabParallelEmbedding,
        use_deepep: bool,
        detach_encoder: bool,
    ):
        super().__init__(config, ps, layer_idx, use_deepep=use_deepep)
        self.config = config
        object.__setattr__(self, "embedding", embedding)
        self.detach_encoder = detach_encoder
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = MultiHeadHyperConnectionHead(config.hidden_size, config.hc_mult, config.hc_eps)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states is the per-stream mHC source [S, B, hc_mult, H].
        embedded = self.embedding(input_ids)
        embedded = scatter_to_sequence_parallel(embedded, self.ps)
        if self.detach_encoder:
            embedded = embedded.detach()
            hidden_states = hidden_states.detach()
        embedded = self.enorm(embedded)
        # e_proj on the [S, B, H] embedding, broadcast across the hc_mult streams;
        # h_proj on the normed mHC hidden keeps the per-stream state.
        projected = self.e_proj(embedded).unsqueeze(2) + self.h_proj(self.hnorm(hidden_states))
        return super().forward(projected, position_ids=position_ids, input_ids=input_ids)

    def contract(self, x: torch.Tensor) -> torch.Tensor:
        # Collapse the hc_mult streams [S, B, hc_mult, H] -> [S, B, H].
        return self.norm(self.hc_head(x))


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("DeepseekV4Model supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


def _apply_attention_backend_override(backend: str | None) -> None:
    if backend in (None, "flash"):
        backend = "fused"
    env = {
        "auto": ("1", "1", "1"),
        "flash": ("1", "0", "0"),
        "fused": ("0", "1", "0"),
        "unfused": ("0", "0", "1"),
        "local": ("0", "0", "1"),
    }.get(backend)
    if env is None:
        raise ValueError(
            "attention_backend_override must be one of "
            "{'auto', 'flash', 'fused', 'unfused', 'local'}"
        )
    (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) = env


class DeepseekV4Model(nn.Module):
    """DS4 model.  Mirrors ``KimiK2Model``: ``build_pipeline_chunk_layout`` for
    embed/head/layer placement, ``set_input_tensor`` for the PP recv buffer, a
    shared embed/head, MTP gated on ``layout.has_mtp``, SBHD scatter at embed,
    and a single forward producing the loss / logits / MTP outputs.

    DS4 differences (all documented inline): the hidden is the 4-D SBHD mHC
    tensor; it is expanded after embed and contracted before the head; pipeline
    stage boundaries fold/unfold the ``hc_mult`` streams to match the 3-D P2P
    buffer; attention is CSA; the MoE is hash-routed and so layers receive
    ``input_ids``.  Attribute names (``embed_tokens`` / ``norm`` / ``lm_head`` /
    ``hc_head`` / ``layers`` ModuleDict) are preserved from the previous DS4 so
    HF checkpoint names are unchanged.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        use_deepep: bool = False,
    ):
        super().__init__()
        del hf_path, use_thd  # DS4 CSA derives its own masking from position_ids.
        _apply_attention_backend_override(attention_backend_override)
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self.hc_mult = int(config.hc_mult)
        self._input_tensor: torch.Tensor | None = None
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor

        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers,
            ps,
            train_config.vpp,
            vpp_chunk_id,
            num_mtp_layers=config.num_nextn_predict_layers if mtp_enable else 0,
        )
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        # DS4 does not tie embeddings; the attribute is preserved for the
        # dist-opt / distckpt interface (matches the previous DS4 model).
        self.share_embeddings_and_output_weights = False
        self.vision_model: nn.Module | None = None

        self.embed_tokens: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        # Key by LOCAL pipeline-stage position (0..len-1), not the global layer id,
        # so parameter names ("layers.{local}.…") follow the same convention as the
        # other lite models (glm5 / kimi_k2 use nn.ModuleList → local names). The
        # shared HF weight map (build via enumerate(layer_indices)) is keyed by local
        # position; keying this dict by the global id instead double-maps under an
        # uneven PP split (a stage owning [1,2] would remap layers.1→layers.2 and
        # drop layer 1 on export/load). The global id is still passed to the layer
        # for its dense-vs-MoE / per-layer logic.
        self.layers = nn.ModuleDict(
            {
                str(local): DeepseekV4Layer(config, ps, global_idx, use_deepep=use_deepep)
                for local, global_idx in enumerate(self.layer_indices)
            }
        )

        self.norm: nn.Module | None = None
        self.hc_head: MultiHeadHyperConnectionHead | None = None
        self.lm_head: VocabParallelOutput | None = None
        if layout.has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hc_head = MultiHeadHyperConnectionHead(
                config.hidden_size, config.hc_mult, config.hc_eps
            )
            self.lm_head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: nn.ModuleList = nn.ModuleList()
        if mtp_enable and config.num_nextn_predict_layers > 0 and layout.has_mtp:
            mtp_embedding = self.embed_tokens
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
                self.mtp_embed = mtp_embedding
            self.mtp = nn.ModuleList(
                [
                    DeepseekV4MTPLayer(
                        config,
                        ps,
                        config.num_hidden_layers + idx,
                        embedding=mtp_embedding,
                        use_deepep=use_deepep,
                        detach_encoder=mtp_detach_encoder,
                    )
                    for idx in range(config.num_nextn_predict_layers)
                ]
            )

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("DeepseekV4Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def _embed_or_recv(
        self, input_ids: torch.Tensor | None, hidden_states: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Return (mHC hidden [S, B, hc_mult, H], input_ids, seq_len) for this stage."""
        if self.embed_tokens is not None:
            assert input_ids is not None
            # Embed -> SBHD [S, B, H] -> SP scatter (no-op at TP=1) -> expand to
            # the hc_mult parallel residual streams.
            h = self.embed_tokens(input_ids)
            h = scatter_to_sequence_parallel(h, self.ps)
            seq_len = h.size(0)
            h = expand_mhc_hidden_for_pipeline(h, hc_mult=self.hc_mult)
            return h, input_ids, seq_len
        if hidden_states is None:
            hidden_states = self._input_tensor
        assert hidden_states is not None
        # Non-first PP stage: the previous stage folded the streams into the
        # hidden dim ([S, B, hc_mult * H]); unfold back to [S, B, hc_mult, H]
        # instead of re-expanding (which would discard the cross-stream state).
        h = unfold_mhc_hidden_from_pipeline(hidden_states, hc_mult=self.hc_mult)
        return h, input_ids, h.size(0)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
        enable_mtp: bool = True,
        packed_seq_params: Any = None,
    ) -> dict:
        # THD-packed inputs may arrive 1-D ([total_tokens]); VocabParallelEmbedding
        # and the dense skeleton expect [B, S], so add the batch row (matches the
        # previous DS4 and the [1, S] single-sequence convention).
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if loss_mask is not None and loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(0)
        h, input_ids, _seq_len = self._embed_or_recv(input_ids, hidden_states)

        # CSA is batch-first and needs [B, S] position ids; build them from the
        # SBHD hidden when not supplied.  position_ids is the only forward arg
        # that crosses into the batch-first CSA interior.
        if position_ids is None:
            seq_len, batch = h.size(0), h.size(1)
            position_ids = (
                torch.arange(seq_len, device=h.device).unsqueeze(0).expand(batch, -1)
            )

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe(self.train_config))
            if self.train_config.fp8
            else nullcontext()
        )
        with fp8_ctx:
            for layer in self.layers.values():
                h = layer(
                    h,
                    position_ids=position_ids,
                    input_ids=input_ids,
                    packed_seq_params=packed_seq_params,
                )

        output: dict = {"hidden_states": fold_mhc_hidden_for_pipeline(h)}
        if self.lm_head is None or self.norm is None or self.hc_head is None:
            # Non-last PP stage: fold the hc_mult streams into the hidden dim so
            # the pipeline P2P buffer ([S, B, hc_mult * H]) carries the full
            # hyper-connection state.
            return output

        # Last stage: contract the mHC streams, then run head / MTP / loss
        # exactly as the Kimi skeleton does.
        mtp_source = h
        hidden_for_head = contract_mhc_hidden_for_pipeline(h, norm=self.norm, head=self.hc_head)

        # MTP runs on the head stage (where self.mtp is built with a valid bound
        # embedding -- self.embed_tokens when present, else the self.mtp_embed
        # fallback, mirroring Kimi).  Disabled at CP>1 (the rolled MTP targets are
        # not CP-sliced) and when no input_ids are available.
        run_mtp = (
            enable_mtp
            and input_ids is not None
            and len(self.mtp) > 0
            and self.ps.cp_size == 1
        )
        mtp_hidden_states = self._apply_mtp(
            mtp_source, input_ids=input_ids, position_ids=position_ids, run_mtp=run_mtp,
            packed_seq_params=packed_seq_params,
        )
        if mtp_hidden_states is not None:
            output["mtp_hidden_states"] = mtp_hidden_states

        if labels is not None:
            temperature_value = _temperature_to_float(temperature)
            mtp_result = self._apply_mtp_loss(
                hidden_for_head,
                mtp_hidden_states=mtp_hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                temperature=temperature_value,
                use_fused_kernels=use_fused_kernels,
                packed_seq_params=packed_seq_params,
            )
            if mtp_result is not None:
                hidden_for_head, mtp_loss = mtp_result
                output["mtp_loss"] = mtp_loss
            labels_sb = labels.transpose(0, 1).contiguous()
            if use_fused_kernels:
                hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
                log_probs, entropy = linear_cross_entropy(
                    hidden_full,
                    self._head_weight_for_fused_ce(hidden_full),
                    labels_sb,
                    temperature_value,
                    self.ps.tp_group,
                )
                output["loss"] = (-log_probs).mean()
                output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                if calculate_entropy:
                    output["entropy"] = entropy.transpose(0, 1).contiguous()
            else:
                logits = self.lm_head(hidden_for_head)
                if temperature_value != 1.0:
                    logits = logits / temperature_value
                loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                output["loss"] = loss.mean()
                output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                if calculate_entropy:
                    entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                    output["entropy"] = entropy.transpose(0, 1).contiguous()
        else:
            logits = self.lm_head(hidden_for_head)
            output["logits"] = self.lm_head.gather(logits).transpose(0, 1).contiguous()
            if mtp_hidden_states is not None:
                output["mtp_logits"] = [
                    self.lm_head.gather(self.lm_head(mtp_hidden)).transpose(0, 1).contiguous()
                    for mtp_hidden in mtp_hidden_states
                ]
        return output

    def _apply_mtp(
        self,
        mtp_source: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        run_mtp: bool,
        packed_seq_params=None,
    ) -> list[torch.Tensor] | None:
        if not run_mtp:
            return None
        assert input_ids is not None
        # DS4 MTP rolls input_ids per depth and runs each MTP layer on the
        # running per-stream source, contracting each depth's output to [S, B, H]
        # for the head.  Mirrors Kimi's KimiK2MTPBlock loop.
        mtp_input_ids = input_ids
        source = mtp_source
        outputs: list[torch.Tensor] = []
        for mtp_layer in self.mtp:
            mtp_input_ids, _ = _roll_mtp_left(mtp_input_ids, packed_seq_params=packed_seq_params, dims=-1)
            source = mtp_layer(
                input_ids=mtp_input_ids,
                hidden_states=source,
                position_ids=position_ids,
            )
            outputs.append(mtp_layer.contract(source))
        return outputs

    def _apply_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        mtp_hidden_states: list[torch.Tensor] | None,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        temperature: float,
        use_fused_kernels: bool,
        packed_seq_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if mtp_hidden_states is None or not self.mtp_enable_train:
            return None
        if loss_mask is None:
            mtp_loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            mtp_loss_mask = loss_mask.to(dtype=torch.float32).clone()
        mtp_labels = labels.clone()

        mtp_loss_values = []
        for mtp_hidden in mtp_hidden_states:
            mtp_labels, _ = _roll_mtp_left(mtp_labels, packed_seq_params=packed_seq_params, dims=-1)
            mtp_loss_mask, num_tokens = _roll_mtp_left(mtp_loss_mask, packed_seq_params=packed_seq_params, dims=-1)
            labels_sb = mtp_labels.transpose(0, 1).contiguous()
            mask_sb = mtp_loss_mask.transpose(0, 1).contiguous()

            if use_fused_kernels:
                mtp_hidden_full = gather_from_sequence_parallel(mtp_hidden, self.ps)
                log_probs, _entropy = linear_cross_entropy(
                    mtp_hidden_full,
                    self._head_weight_for_fused_ce(mtp_hidden_full),
                    labels_sb,
                    temperature,
                    self.ps.tp_group,
                )
                token_loss = -log_probs
            else:
                logits = self.lm_head(mtp_hidden)
                if temperature != 1.0:
                    logits = logits / temperature
                token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)

            token_loss = token_loss * mask_sb.to(dtype=token_loss.dtype)
            num_tokens = num_tokens.to(dtype=token_loss.dtype).clamp_min(1.0)
            mtp_loss_values.append(token_loss.sum() / num_tokens)

            mtp_loss_scale = self.mtp_loss_scaling_factor / max(len(mtp_hidden_states), 1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states,
                mtp_loss_scale * token_loss / num_tokens,
            )

        if not mtp_loss_values:
            return None
        return (
            hidden_states,
            torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean(),
        )

    def _head_weight_for_fused_ce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        weight = self.lm_head.col.linear.weight
        return (
            weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)
        )


__all__ = [
    "CompressedSparseAttention",
    "DeepseekV4CSAAttention",
    "DeepseekV4Layer",
    "DeepseekV4Model",
    "DeepseekV4MTPLayer",
    "HyperConnection",
    "MTPLossAutoScaler",
    "MultiHeadHyperConnectionHead",
]
