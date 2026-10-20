# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 configuration and builder using main's existing training lifecycle."""

import json
from dataclasses import dataclass
from typing import ClassVar

import torch

from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.pipeline_parallel.typed_p2p_communication import create_pipeline_control_group
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from megatron.training.vocab_utils import calculate_padded_vocab_size


def configure_v41_args(args, kwargs):
    """Validate the V/E schedule and read explicit conditional-memory configuration."""
    pattern = (args.hybrid_layer_pattern or '').replace('|', '')
    if pattern != 'VE' * (args.num_layers // 2):
        raise ValueError('V4.1 requires a VE pair per logical block, with optional PP separators')
    if args.hybrid_context_parallel or args.sequence_packing_scheduler is not None:
        raise NotImplementedError(
            'V4.1 entrypoint currently uses static CP and the packed dataloader'
        )
    if args.context_parallel_size > 1 and not (args.sft or args.dataloader_inter_document_masking):
        raise ValueError('V4.1 CP requires SFT packing or --dataloader-inter-document-masking')
    if args.create_attention_mask_in_dataloader:
        raise ValueError('V4.1 requires --no-create-attention-mask-in-dataloader')
    if args.experimental_attention_variant not in (None, 'dsv4_hybrid'):
        raise ValueError('V4.1 requires experimental_attention_variant=dsv4_hybrid')
    if args.spec is not None:
        raise ValueError('V4.1 uses its explicit static model spec; omit --spec')
    kwargs['experimental_attention_variant'] = 'dsv4_hybrid'
    if args.engram_config is not None:
        with open(args.engram_config, encoding='utf-8') as source:
            kwargs['engram_config'] = EngramConfig(**json.load(source))
        if args.engram_token_map is None:
            raise ValueError('--engram-config requires --engram-token-map')


@dataclass(kw_only=True)
class DeepSeekV41ModelConfig(HybridModelConfig):
    """Select the V4.1 builder; the transformer holds all architecture options."""

    builder: ClassVar[str] = 'megatron.training.models.deepseek_v41.DeepSeekV41ModelBuilder'
    engram_token_map: str | None = None


class DeepSeekV41ModelBuilder(HybridModelBuilder):
    """Reuse Hybrid's distributed wrapping, optimizer and checkpoint integration."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Create one stage and reuse one typed-control group across its VPP chunks."""
        c = self._model_config
        if c.share_embeddings_and_output_weights:
            raise ValueError('V4.1 requires --untie-embeddings-and-output-weights')
        if not hasattr(self, '_pipeline_control_group'):
            self._pipeline_control_group = (
                create_pipeline_control_group(
                    pg_collection.pp, world_group=torch.distributed.group.WORLD
                )
                if pg_collection.pp.size() > 1
                else None
            )
        vocab = (
            calculate_padded_vocab_size(
                c.vocab_size,
                c.make_vocab_size_divisible_by,
                c.transformer.tensor_model_parallel_size,
            )
            if c.should_pad_vocab
            else c.vocab_size
        )
        token_map = (
            torch.load(c.engram_token_map, map_location='cpu', weights_only=True)
            if c.engram_token_map is not None
            else None
        )
        return DeepSeekV41Model(
            config=c.transformer,
            vocab_size=vocab,
            max_sequence_length=c.seq_length,
            hybrid_layer_pattern=c.hybrid_layer_pattern or c.transformer.hybrid_pattern,
            token_map=token_map,
            pre_process=is_pp_first_stage(pg_collection.pp) if pre_process is None else pre_process,
            post_process=(
                is_pp_last_stage(pg_collection.pp) if post_process is None else post_process
            ),
            parallel_output=c.parallel_output,
            fp16_lm_cross_entropy=c.fp16_lm_cross_entropy,
            logit_dtype=c.logit_dtype,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            pipeline_control_group=self._pipeline_control_group,
        )


def estimate_v41_training_flops(args, total_tokens, squared_lengths):
    """Matmul FLOP estimate using actual owners; excludes routing, norms and recompute.

    Sparse support uses the configured window/Top-K upper bound. Causal full
    indexing uses sum(L^2)/2; Reindex is capped by its candidate pool. This is a
    reporting estimate, not executed-kernel work or a throughput measurement.
    """
    t, h, d, heads = total_tokens, args.hidden_size, args.v_head_dim, args.num_attention_heads
    qrank, groups, orank = (
        args.q_lora_rank,
        args.output_projection_groups,
        args.output_projection_lora_rank,
    )
    n = args.mhc_num_residual_streams
    forward = 0
    index_work = 0
    for layer in range(0, args.num_layers, 2):
        ratio = args.csa_compress_ratios[layer]
        forward += (
            2 * t * (h * qrank + qrank * heads * d + h * d + heads * d * orank + groups * orank * h)
        )
        forward += (
            4 * t * heads * d * (args.csa_window_size + (args.dsa_indexer_topk if ratio else 0))
        )
        if layer in args.csa2_kv_source_layers:
            forward += 2 * t * h * d * (2 if ratio == 2 else 1)
            index_work += (
                2
                * t
                / ratio
                * d
                * args.dsa_indexer_head_dim
                * (3 if args.dsa_indexer_loss_coeff else 1)
            )
        if layer in args.csa2_index_source_layers:
            index_work += (
                2
                * t
                * args.dsa_indexer_n_heads
                * (qrank * args.dsa_indexer_head_dim + h)
                * (3 if args.dsa_indexer_loss_coeff else 1)
            )
            pairs = squared_lengths / (2 * ratio)
            source = args.csa2_candidate_source_layer
            if source is not None and layer > source:
                pairs = min(
                    pairs, t * args.csa2_candidate_topk_blocks * args.csa2_candidate_block_size
                )
            index_work += 2 * pairs * args.dsa_indexer_n_heads * args.dsa_indexer_head_dim
            if args.dsa_indexer_loss_coeff:
                supervised = (
                    min(pairs, t * args.dsa_indexer_topk)
                    if args.dsa_indexer_use_sparse_loss
                    else pairs
                )
                index_work += 6 * supervised * args.dsa_indexer_n_heads * args.dsa_indexer_head_dim
                index_work += 2 * supervised * heads * d
    experts = args.moe_ffn_hidden_size * args.moe_router_topk + (
        args.moe_shared_expert_intermediate_size or 0
    )
    forward += (args.num_layers // 2) * 6 * t * h * experts
    forward += args.num_layers * 2 * t * (n * h * (n * n + 2 * n) + n * n * h + n * h)
    # The args namespace keeps a path; the embedded transformer carries the dataclass.
    options = args.engram_config
    if isinstance(options, str):
        with open(options, encoding='utf-8') as source:
            options = EngramConfig(**json.load(source))
    if options is not None:
        width = (options.max_ngram_size - 1) * options.n_heads * options.head_dim
        forward += len(options.layer_ids) * 2 * t * width * (n + 1) * h
    forward += 2 * t * h * args.padded_vocab_size
    return int(3 * forward + index_work)
