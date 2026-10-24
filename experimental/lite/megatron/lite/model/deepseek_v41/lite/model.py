# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from torch import nn

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttentionSubmodules,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import ensure_params_ready
from megatron.lite.model.deepseek_v41.lite.components.assembly import (
    DS41AttentionSources,
    SourceContext,
)
from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
from megatron.lite.model.deepseek_v41.lite.components.block import DS41Block
from megatron.lite.model.deepseek_v41.lite.components.carrier import pack_sources, unpack_sources
from megatron.lite.model.deepseek_v41.lite.components.config import (
    core_config,
    core_groups,
    validate_runtime_cp,
)
from megatron.lite.model.deepseek_v41.lite.components.engram import (
    DS41Engram,
    hash_multipliers,
    prime_buckets,
)
from megatron.lite.model.deepseek_v41.lite.moe import bind_sorted_selection
from megatron.lite.model.deepseek_v41.topology import validate_build
from megatron.lite.primitive.modules.engram import NgramHash
from megatron.lite.primitive.modules.projection import ModelNorm
from megatron.lite.primitive.parallel import build_pipeline_chunk_layout
from megatron.lite.primitive.parallel.source_state import KVSourceState


class CoreAttentionSources(DS41AttentionSources):
    def __init__(self, *, compress_ratio=0, rotary_pos_emb=None, **kwargs):
        super().__init__(compress_ratio=compress_ratio, rotary_pos_emb=rotary_pos_emb, **kwargs)


class DeepseekV41Model(MegatronModule):
    def __init__(self, release, impl, ps, *, token_map, load_reader=None):
        config = core_config(
            release,
            impl.parallel,
            initialize=impl.initialize_weights,
            attention_backend=impl.attention_backend_override or "none",
            deterministic_mode=impl.deterministic_mode,
        )
        config.persist_layer_norm = False
        super().__init__(config=config)
        self.release, self.impl, self.ps = release, impl, ps
        self._load_reader = load_reader
        self.parallel_state = ps
        self.tp_group = ps.tp_group
        self.source_state = KVSourceState()
        self.context = SourceContext(
            release,
            config,
            core_groups(ps, release),
            self.source_state,
            validate_build(release, context_parallel_size=ps.cp_size),
        )
        self.layout = build_pipeline_chunk_layout(release.num_hidden_layers, ps)
        indices = self.layout.layer_indices
        self.local_layer_range = (indices[0], indices[-1] + 1) if indices else (0, 0)
        self.hc_mult, self._input_tensor = release.hc_mult, None
        self.layers = nn.ModuleDict()
        self.engrams = nn.ModuleDict()
        if self.layout.has_embed:
            self.embed = VocabParallelEmbedding(
                release.vocab_size,
                release.hidden_size,
                init_method=config.init_method,
                config=config,
                tp_group=ps.tp_group,
            )
            self.embed.weight.is_embedding_or_output_parameter = True
        for index in indices:
            attention = self._build_attention(config, index)
            mlp = self._build_mlp(config, release, index)
            self.layers[str(index)] = self._build_block(config, index, attention, mlp)
            if index in release.engram_layer_ids:
                if token_map is None:
                    raise ValueError("DS4.1 Engram requires the release compressed token_map")
                if impl.trainable_engram and impl.shard_engram:
                    raise NotImplementedError("DS4.1 trainable Engram requires shard_engram=False")
                table_group = ps.dp_cp_group if impl.shard_engram else ps.tp_group
                self.engrams[str(index)] = DS41Engram(
                    release, config, index, table_group, frozen=not impl.trainable_engram
                )
                if load_reader is not None:
                    from megatron.lite.model.deepseek_v41.lite.checkpoint import prefetch_engram

                    prefetch_engram(load_reader, self.engrams[str(index)].embed, index)
        if self.engrams:
            self.hash = NgramHash(
                token_map,
                release.engram_pad_token_id,
                hash_multipliers(
                    release.engram_layer_ids,
                    release.engram_max_ngram_size,
                    release.engram_compressed_vocab_size,
                ),
                prime_buckets(
                    release.engram_layer_ids,
                    release.engram_max_ngram_size,
                    release.engram_n_heads,
                    release.engram_vocab_size,
                ),
            )
        if self.layout.has_head:
            self.norm, self.head = self._build_head(config, release)
        for module in self.modules():
            if isinstance(module, MegatronModule) and getattr(module, "tp_group", None) is None:
                groups = getattr(module, "pg_collection", None)
                module.tp_group = getattr(groups, "tp", None) or ps.tp_group
        self.to(device=impl.device)
        self.ds41_frozen_parameters = tuple(
            name for name, parameter in self.named_parameters() if not parameter.requires_grad
        )

    def _build_attention(self, config, index):
        attention = DS41SelfAttention(
            config,
            DSv4HybridSelfAttentionSubmodules(
                q_layernorm=ModelNorm,
                kv_layernorm=ModelNorm,
                linear_q_down_proj=TELinear,
                linear_q_up_proj=TEColumnParallelLinear,
                linear_kv_proj=TEColumnParallelLinear,
                core_attention=ModuleSpec(CoreAttentionSources, params={"context": self.context}),
                linear_proj=TERowParallelLinear,
            ),
            index + 1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=self.context.groups,
            compress_ratio=config.csa_compress_ratios[index],
        )
        return attention

    def _build_mlp(self, config, release, index):
        mlp = get_mlp_module_spec(
            use_te=True, num_experts=release.n_routed_experts, moe_grouped_gemm=True
        )(
            config=config,
            layer_number=index + 1,
            pg_collection=self.context.groups,
            hash_moe_layer_threshold=release.num_hash_layers,
        )
        bind_sorted_selection(mlp.router)
        return mlp

    def _build_block(self, config, index, attention, mlp):
        return DS41Block(config, index + 1, attention, mlp)

    def _build_head(self, config, release):
        norm = ModelNorm(config, release.hidden_size, release.rms_norm_eps)
        head = ColumnParallelLinear(
            release.hidden_size,
            release.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            tp_group=self.ps.tp_group,
        )
        head.weight.is_embedding_or_output_parameter = True
        return norm, head

    def set_input_tensor(self, value):
        self._input_tensor = value[0] if isinstance(value, (tuple, list)) else value

    def forward(
        self,
        input_ids,
        *,
        packed_seq_params,
        labels=None,
        loss_mask=None,
        full_input_ids=None,
        token_mask=None,
        temperature=1.0,
        calculate_entropy=False,
        hidden_states=None,
    ):
        ensure_params_ready(self.parameters())
        params = packed_seq_params
        validate_runtime_cp(self.release, self.ps.cp_group, params)
        ids = input_ids.reshape(-1)
        rows = ids.shape[0]
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        if self.layout.has_embed:
            embedding = self.embed(ids).unsqueeze(1)
            hidden = embedding.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
            pre = hidden.new_zeros(rows, 1, self.hc_mult, dtype=torch.float32)
            pre[..., 0] = 1
        else:
            incoming = self._input_tensor if hidden_states is None else hidden_states
            hidden, pre, imported = unpack_sources(incoming, prefix, self.ps.cp_group)
        hashes = None
        if self.engrams:
            if full_input_ids is None:
                if self.ps.cp_size > 1:
                    raise ValueError("DS4.1 Engram CP requires full packed IDs from the protocol")
                full_input_ids = ids
            hashes = self.hash(full_input_ids.reshape(1, -1), cu_seqlens=prefix)
            hashes = hashes[:, self.ps.cp_rank * rows : (self.ps.cp_rank + 1) * rows].transpose(
                0, 1
            )
        with self.source_state.microbatch():
            if not self.layout.has_embed:
                self.source_state.values.update(imported)
            for key, block in self.layers.items():
                if key in self.engrams:
                    module = self.engrams[key]
                    mask = None if token_mask is None else token_mask.reshape(rows, 1)
                    hidden = module(hidden, hashes[:, :, module.layer_hash_index], mask)
                hidden, pre = block(
                    hidden,
                    pre,
                    packed_seq_params=params,
                    input_ids=ids,
                    source_state=self.source_state,
                    recompute=self.impl.recompute,
                )
            if not self.layout.has_head:
                return {
                    "hidden_states": pack_sources(
                        hidden, pre, self.source_state, prefix, self.ps.cp_group
                    )
                }
            last = self.layers[str(self.local_layer_range[1] - 1)]
            collapsed = last.ffn_mixes.aggregate(hidden.flatten(2).float(), pre.float()).to(
                hidden.dtype
            )
            collapsed = self.norm(collapsed)
            logits = self.head(collapsed)[0]
            if float(temperature) != 1:
                logits = logits / float(temperature)
            output = {"rows": torch.arange(rows, device=ids.device) + self.ps.cp_rank * rows}
            if labels is None:
                output["logits"] = logits.squeeze(1)
                return output
            token_loss = vocab_parallel_cross_entropy(
                logits.float(), labels.reshape(rows, 1), tp_group=self.ps.tp_group
            ).squeeze(1)
            mask = (
                torch.ones_like(token_loss)
                if loss_mask is None
                else loss_mask.reshape(-1).to(token_loss.dtype)
            )
            output.update(
                log_probs=-token_loss,
                loss=(token_loss * mask).sum() / mask.sum().clamp_min(1),
                valid_mask=mask.bool(),
            )
            if calculate_entropy:
                logp = logits.float().log_softmax(-1)
                output["entropy"] = -(logp.exp() * logp).sum(-1).squeeze(1)
            return output
