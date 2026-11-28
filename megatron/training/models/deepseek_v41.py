# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 configuration and builder using main's existing training lifecycle."""

import json
from dataclasses import dataclass
from typing import ClassVar

from megatron.core.models.deepseek_v41.engram import released_engram_config
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from megatron.training.vocab_utils import calculate_padded_vocab_size


def configure_v41_args(args, kwargs):
    """Validate the V/E schedule and read explicit conditional-memory configuration."""
    pattern = (args.hybrid_layer_pattern or '').replace('|', '')
    if pattern != 'VE' * (args.num_layers // 2):
        raise ValueError('V4.1 requires a VE pair per logical block, with optional PP separators')
    scheduled_varlen = args.use_varlen_dataset and args.sequence_packing_scheduler == 'dp_balanced'
    if (
        getattr(args, 'dynamic_context_parallel', False)
        or getattr(args, 'hybrid_context_parallel', False)
        or (args.sequence_packing_scheduler is not None and not scheduled_varlen)
    ):
        raise NotImplementedError(
            'V4.1 uses static CP with the packed dataloader or dp_balanced varlen scheduling'
        )
    if args.context_parallel_size > 1 and not (
        args.sft or args.dataloader_inter_document_masking or scheduled_varlen
    ):
        raise ValueError('V4.1 CP requires SFT packing or --dataloader-inter-document-masking')
    if args.create_attention_mask_in_dataloader:
        raise ValueError('V4.1 requires --no-create-attention-mask-in-dataloader')
    if args.experimental_attention_variant not in (None, 'dsv4_hybrid'):
        raise ValueError('V4.1 requires experimental_attention_variant=dsv4_hybrid')
    if args.spec is not None:
        raise ValueError('V4.1 uses its explicit static model spec; omit --spec')
    kwargs['experimental_attention_variant'] = 'dsv4_hybrid'
    if getattr(args, "engram_config", None) is not None:
        if getattr(args, "engram_vocab_sizes", None) is not None:
            raise ValueError("Choose the released V4.1 Engram config or the common Engram flags")
        with open(args.engram_config, encoding="utf-8") as source:
            options = json.load(source)
        path = getattr(args, "engram_tokenizer_map", None) or getattr(
            args, "engram_token_map", None
        )
        kwargs["engram_config"] = released_engram_config(options, path)


@dataclass(kw_only=True, init=False)
class DeepSeekV41ModelConfig(HybridModelConfig):
    """Select the V4.1 builder; the transformer holds all architecture options."""

    builder: ClassVar[str] = 'megatron.training.models.deepseek_v41.DeepSeekV41ModelBuilder'

    def __init__(self, *, transformer, **kwargs):
        # HybridModelConfig proxies overlapping writes to the transformer. Keep one
        # object for the memory constructor, optimizer and distributed table lifecycle.
        shared = kwargs.get("engram_config") or transformer.engram_config
        transformer.engram_config = shared
        transformer.engram_enabled = shared is not None
        super().__init__(transformer=transformer, **{**kwargs, "engram_config": shared})

    @property
    def engram_config(self):
        """Return the same memory configuration used to construct the transformer."""
        return self.transformer.engram_config


class DeepSeekV41ModelBuilder(HybridModelBuilder):
    """Reuse Hybrid's distributed wrapping, optimizer and checkpoint integration."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Create one physical or virtual pipeline stage."""
        c = self._model_config
        if c.share_embeddings_and_output_weights:
            raise ValueError('V4.1 requires --untie-embeddings-and-output-weights')
        vocab = (
            calculate_padded_vocab_size(
                c.vocab_size,
                c.make_vocab_size_divisible_by,
                c.transformer.tensor_model_parallel_size,
            )
            if c.should_pad_vocab
            else c.vocab_size
        )
        return DeepSeekV41Model(
            config=c.transformer,
            vocab_size=vocab,
            max_sequence_length=c.seq_length,
            hybrid_layer_pattern=c.hybrid_layer_pattern or c.transformer.hybrid_pattern,
            pre_process=is_pp_first_stage(pg_collection.pp) if pre_process is None else pre_process,
            post_process=(
                is_pp_last_stage(pg_collection.pp) if post_process is None else post_process
            ),
            parallel_output=c.parallel_output,
            fp16_lm_cross_entropy=c.fp16_lm_cross_entropy,
            logit_dtype=c.logit_dtype,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )


def estimate_v41_training_flops(config, vocab_size, total_tokens, squared_lengths):
    """Matmul FLOP estimate using actual owners; excludes routing, norms and recompute.

    Sparse support uses the configured window/Top-K upper bound. Causal full
    indexing uses sum(L^2)/2; Reindex is capped by its candidate pool. This is a
    reporting estimate, not executed-kernel work or a throughput measurement.
    """
    t, h, d, heads = total_tokens, config.hidden_size, config.v_head_dim, config.num_attention_heads
    qrank, groups, orank = (
        config.q_lora_rank,
        config.output_projection_groups,
        config.output_projection_lora_rank,
    )
    n = config.mhc_num_residual_streams
    forward = 0
    index_work = 0
    for layer in range(0, config.num_layers, 2):
        ratio = config.csa_compress_ratios[layer]
        forward += (
            2 * t * (h * qrank + qrank * heads * d + h * d + heads * d * orank + groups * orank * h)
        )
        forward += (
            4 * t * heads * d * (config.csa_window_size + (config.dsa_indexer_topk if ratio else 0))
        )
        if layer in config.csa2_kv_source_layers:
            forward += 2 * t * h * d * (2 if ratio == 2 else 1)
            index_work += (
                2
                * t
                / ratio
                * d
                * config.dsa_indexer_head_dim
                * (3 if config.dsa_indexer_loss_coeff else 1)
            )
        if layer in config.csa2_index_source_layers:
            index_work += (
                2
                * t
                * config.dsa_indexer_n_heads
                * (qrank * config.dsa_indexer_head_dim + h)
                * (3 if config.dsa_indexer_loss_coeff else 1)
            )
            pairs = squared_lengths / (2 * ratio)
            source = config.csa2_candidate_source_layer
            if source is not None and layer > source:
                pairs = min(
                    pairs, t * config.csa2_candidate_topk_blocks * config.csa2_candidate_block_size
                )
            index_work += 2 * pairs * config.dsa_indexer_n_heads * config.dsa_indexer_head_dim
            if config.dsa_indexer_loss_coeff:
                supervised = (
                    min(pairs, t * config.dsa_indexer_topk)
                    if config.dsa_indexer_use_sparse_loss
                    else pairs
                )
                index_work += (
                    6 * supervised * config.dsa_indexer_n_heads * config.dsa_indexer_head_dim
                )
                index_work += 2 * supervised * heads * d
    experts = config.moe_ffn_hidden_size * config.moe_router_topk + (
        config.moe_shared_expert_intermediate_size or 0
    )
    forward += (config.num_layers // 2) * 6 * t * h * experts
    forward += config.num_layers * 2 * t * (n * h * (n * n + 2 * n) + n * n * h + n * h)
    if config.engram_config is not None:
        memory = config.engram_config
        forward += len(memory.layer_ids) * 2 * t * memory.total_memory_dim * (n + 1) * h
    forward += 2 * t * h * vocab_size
    return int(3 * forward + index_work)
