# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-tower Mamba-hybrid model for block-wise diffusion language modelling.

Architecture overview:

    **Context tower** — processes the *clean* token sequence and caches
    per-layer Attention KV pairs and Mamba (conv + SSM) states at every
    block boundary.

    **Denoiser tower** — processes the *noisy* (diffusion-corrupted) token
    sequence.  Each denoiser block *N* attends to context blocks
    ``0 .. N-1`` (strict past) and itself (bidirectional within block)
    via a block-causal attention mask, and is initialised with the
    Mamba states from context block ``N-1``.

Training forward pass:
    1. Corrupt input via the active :class:`DiffusionProcess`.
    2. Embed clean tokens (context) and noisy tokens (denoiser).
    3. Context tower forward → KV + Mamba caches.
    4. Build block-causal attention mask.
    5. Denoiser tower forward with cached context states.
    6. Output projection → logits → per-token diffusion loss.

Key ``--tt-diffusion-*`` flags (see ``arguments.py``):
    ``--tt-diffusion-tied-towers``
        Share weights between context and denoiser towers.
    ``--tt-diffusion-no-freeze-context``
        Allow gradients to flow into the context tower (default: frozen).
    ``--tt-diffusion-mask-token-id``
        Token ID used as the absorbing mask state in mask diffusion.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.diffusion.two_tower.diffusion_process import MaskDiffusionProcess
from megatron.diffusion.two_tower.layer_utils import (
    forward_mamba_layer_batched_with_initial_states,
    forward_mamba_layer_parallel_with_all_states,
    forward_mamba_layer_varlen_with_states,
    forward_mamba_layer_with_states,
)
from megatron.diffusion.two_tower.time_conditioning import (
    TimestepEmbedder,
    get_modulation_params,
    modulate,
)

if TYPE_CHECKING:
    from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizer

#: Per-layer list of ``(key, value)`` tensors, each ``[S, B, H, D]``.
LayerKVCache = List[Tuple[Tensor, Tensor]]
#: Per-layer list of ``(conv_states, ssm_states)`` tensors.
LayerMambaCache = List[Tuple[Tensor, Tensor]]
#: Full context cache: ``(kv_cache, mamba_cache, final_hidden_states)``.
ContextCache = Tuple[LayerKVCache, LayerMambaCache, Tensor]


def create_block_causal_mask(
    num_blocks: int, block_size: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Build an additive block-causal attention mask for the denoiser tower.

    The KV sequence is the concatenation of full context tokens followed by
    full denoiser tokens, so KV length is ``2 * seq_len``.  The mask enforces:

    * Denoiser block *N* may attend to **context** blocks ``0 .. N-1``
      (strictly past context).
    * Denoiser block *N* may attend to **denoiser** block *N* only
      (bidirectional within the current block; no cross-block denoiser
      attention).
    * All other positions are masked with ``-inf``.

    Args:
        num_blocks (int): Number of diffusion blocks in the sequence.
        block_size (int): Tokens per block.
        device (torch.device): Target device.
        dtype (torch.dtype): Floating-point dtype for the mask values.

    Returns:
        Tensor: Additive mask ``[1, 1, seq_len, 2 * seq_len]`` where attended
            positions are ``0.0`` and masked positions are ``-inf``.
    """
    seq_len = num_blocks * block_size

    q_idx = torch.arange(seq_len, device=device)[:, None]
    kv_idx = torch.arange(2 * seq_len, device=device)[None, :]

    is_denoiser_kv = kv_idx >= seq_len
    block_q = q_idx // block_size
    block_kv = torch.where(is_denoiser_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size)

    past_context = (block_q > block_kv) & ~is_denoiser_kv
    current_denoiser = (block_q == block_kv) & is_denoiser_kv
    attend_mask = past_context | current_denoiser

    min_val = torch.finfo(dtype).min
    additive_mask = torch.zeros(seq_len, 2 * seq_len, dtype=dtype, device=device)
    additive_mask.masked_fill_(~attend_mask, min_val)

    return additive_mask.unsqueeze(0).unsqueeze(0)


class TwoTowerMambaModel(LanguageModule):
    """Two-tower Mamba-hybrid model for block-wise diffusion language modelling.

    Extends :class:`LanguageModule` with a duplicated encoder stack (context
    tower + denoiser tower), per-tower embeddings and output heads, a
    pluggable :class:`DiffusionProcess`, and block-causal cross-attention
    between the denoiser and cached context states.

    Args:
        config (TransformerConfig): Transformer / Mamba hybrid configuration.
        mamba_stack_spec (ModuleSpec): Layer specification for
            :class:`MambaStack` (shared between towers).
        vocab_size (int): Vocabulary size (including any mask token).
        max_sequence_length (int): Maximum input sequence length.
        hybrid_layer_pattern (Optional[str]): Unified layer pattern string
            (e.g. ``"M*-"`` for Mamba + Attention + MLP).  Parsed via
            ``megatron.core.ssm.mamba_hybrid_layer_allocation``.
        pre_process (bool): ``True`` if this pipeline rank holds embeddings.
        post_process (bool): ``True`` if this pipeline rank holds the output
            layer and computes the loss.
        fp16_lm_cross_entropy (bool): Compute cross-entropy in FP16.
        parallel_output (bool): Keep logits split across TP ranks.
        share_embeddings_and_output_weights (bool): Tie embedding and output
            projection weights.
        position_embedding_type (str): ``"learned_absolute"``, ``"rope"``,
            or ``"none"``.
        rotary_percent (float): Fraction of hidden dim for RoPE.
        rotary_base (int): RoPE base frequency.
        scatter_embedding_sequence_parallel (bool): Scatter embeddings for SP.
        seq_len_interpolation_factor (Optional[float]): RoPE interpolation
            factor for length extrapolation.
        freeze_context (bool): Freeze the context tower, context embedding, and
            context output layer (no gradients). Default ``True``.
        tied_towers (bool): Share weights between towers (denoiser parameters
            point to context parameters). Default ``False``.
        use_time_conditioning (bool): Enable PixArt-alpha adaLN-single time
            conditioning on the denoiser tower.  Adds a
            :class:`TimestepEmbedder`, a global modulation MLP (``t_block``),
            and per-layer ``scale_shift_tables``.  Default ``False``.
        bidirectional_mamba (bool): Run each Mamba layer in the denoiser
            tower bidirectionally (forward + reversed-sequence backward pass,
            outputs averaged).  The context tower is always unidirectional.
            Default ``False``.
        context_ar_loss (bool): Add a standard next-token-prediction loss
            from the context tower to the diffusion loss.  Requires
            ``tied_towers=True`` and ``freeze_context=False`` so that the
            context output head and shared tower body both receive gradients.
            Default ``False``.
        mask_token_id (int): Token ID for the absorbing mask state.
        pg_collection (Optional[ProcessGroupCollection]): Communication groups.
        vp_stage (Optional[int]): Virtual pipeline stage index.
    """

    def __init__(
        self,
        config: TransformerConfig,
        mamba_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        hybrid_layer_pattern: Optional[str] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        freeze_context: bool = True,
        tied_towers: bool = False,
        use_time_conditioning: bool = False,
        bidirectional_mamba: bool = False,
        context_ar_loss: bool = False,
        mask_token_id: int = 0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.freeze_context = freeze_context
        self.tied_towers = tied_towers
        self.use_time_conditioning = use_time_conditioning
        self.bidirectional_mamba = bidirectional_mamba
        self.context_ar_loss = context_ar_loss
        self.mask_token_id = mask_token_id
        self._single_tower_mode = False

        if context_ar_loss:
            if not tied_towers:
                raise ValueError(
                    "--tt-diffusion-context-ar-loss requires --tt-diffusion-tied-towers "
                    "so the context tower body receives gradients via the shared denoiser."
                )
            if freeze_context:
                raise ValueError(
                    "--tt-diffusion-context-ar-loss requires --tt-diffusion-no-freeze-context "
                    "so the context output head can be trained."
                )
        self.vp_stage = vp_stage

        self.diffusion = MaskDiffusionProcess(mask_token_id=mask_token_id, vocab_size=vocab_size)
        self.diffusion.set_cross_entropy_fn(self._tp_cross_entropy)

        self.model_type = ModelType.encoder_or_decoder

        # Parse hybrid layer pattern via main's utilities
        from megatron.core.ssm.mamba_hybrid_layer_allocation import (
            parse_hybrid_pattern,
            select_pipeline_segment,
        )

        parsed = parse_hybrid_pattern(hybrid_layer_pattern)
        layer_type_list, layer_offset = select_pipeline_segment(
            parsed.main_pattern or '',
            self.pg_collection.pp,
            vp_stage,
            first_stage_layers=self.config.num_layers_in_first_pipeline_stage,
            last_stage_layers=self.config.num_layers_in_last_pipeline_stage,
        )

        # Separate embeddings per tower
        if self.pre_process:
            self.context_embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )
            self.denoiser_embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        # Context tower
        self.context_tower = build_module(
            mamba_stack_spec,
            self.config,
            pre_process=self.pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=layer_offset,
            post_process=self.post_process,
            dtype=config.params_dtype,
            pg_collection=self.pg_collection,
        )

        # Denoiser tower
        self.denoiser_tower = build_module(
            mamba_stack_spec,
            self.config,
            pre_process=self.pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=layer_offset,
            post_process=self.post_process,
            dtype=config.params_dtype,
            pg_collection=self.pg_collection,
        )

        # Denoiser output layer
        if post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=(
                    self.pre_process and self.share_embeddings_and_output_weights
                ),
                tp_group=self.pg_collection.tp,
            )

        # Context output layer (separate LM head for context tower)
        if post_process:
            self.context_output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=False,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self._setup_two_tower_embeddings_and_output_layer()

        self.num_attention_layers = sum(
            1 for lt in self.context_tower.layer_type_list if lt == LayerSymbols.ATTENTION
        )

        if use_time_conditioning:
            hidden_size = self.config.hidden_size
            num_denoiser_layers = len(self.denoiser_tower.layers)
            self.t_embedder = TimestepEmbedder(hidden_size)
            self.t_block = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
            self.scale_shift_tables = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(3, hidden_size) / (hidden_size**0.5))
                    for _ in range(num_denoiser_layers)
                ]
            )

        if tied_towers:
            self._tie_tower_weights()
            if not context_ar_loss:
                self._freeze_context_output_layer()
        elif freeze_context:
            self._freeze_context_tower()

    def _setup_two_tower_embeddings_and_output_layer(self) -> None:
        """Tag embedding and output-layer weights with ``is_embedding_or_output_parameter``.

        Megatron's optimiser uses this flag to apply separate learning-rate
        scaling to embedding / output parameters.  Called once during ``__init__``
        when ``pre_process`` or ``post_process`` is ``True``.
        """
        if self.pre_process:
            self.context_embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
            self.denoiser_embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process:
            if self.output_layer.weight is not None:
                self.output_layer.weight.is_embedding_or_output_parameter = True
            if self.context_output_layer.weight is not None:
                self.context_output_layer.weight.is_embedding_or_output_parameter = True

    def _freeze_context_tower(self) -> None:
        """Freeze the entire context side: embedding, tower layers, and output head.

        Sets ``requires_grad = False`` on all parameters and switches modules
        to ``eval()`` mode so that dropout and batch-norm behave deterministically.
        """
        if self.pre_process:
            for param in self.context_embedding.parameters():
                param.requires_grad = False
            self.context_embedding.eval()
        for param in self.context_tower.parameters():
            param.requires_grad = False
        self.context_tower.eval()
        self._freeze_context_output_layer()

    def _freeze_context_output_layer(self) -> None:
        """Freeze the context output head (LM head) only.

        Called separately when towers are tied — the tower body shares
        denoiser weights (which must remain trainable), but the context
        output head is always frozen because it is not used during training.
        """
        if self.post_process:
            for param in self.context_output_layer.parameters():
                param.requires_grad = False
            self.context_output_layer.eval()

    def _tie_tower_weights(self) -> None:
        """Make the denoiser tower share the context tower's parameter tensors.

        Recursively replaces each denoiser-side ``nn.Parameter`` with the
        corresponding context-side tensor so that DDP treats them as a single
        set of parameters.  The context tower owns the pretrained weights and
        the denoiser borrows them.
        """
        tied_count = 0

        def _replace_params(source: nn.Module, target: nn.Module):
            nonlocal tied_count
            src_params = dict(source.named_parameters(recurse=False))
            for name in list(target._parameters.keys()):
                if name in src_params:
                    target._parameters[name] = src_params[name]
                    tied_count += 1
            src_children = dict(source.named_children())
            for child_name, target_child in target.named_children():
                if child_name in src_children:
                    _replace_params(src_children[child_name], target_child)

        if self.pre_process:
            _replace_params(self.context_embedding, self.denoiser_embedding)
        _replace_params(self.context_tower, self.denoiser_tower)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Accept activation tensors from the previous pipeline stage.

        Args:
            input_tensor (Tensor): Hidden states from the preceding PP rank.
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.context_tower.set_input_tensor(input_tensor[0])
        self.denoiser_tower.set_input_tensor(input_tensor[0])

    def _tp_cross_entropy(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy over a vocabulary sharded across TP ranks.

        Transposes between Megatron's ``(S, B, ...)`` and the diffusion
        process's ``(B, S, ...)`` convention.

        Args:
            logits (Tensor): Predictions ``(B, S, V_local)`` (TP-local vocab).
            labels (Tensor): Target token IDs ``(B, S)`` (global vocab IDs).

        Returns:
            Tensor: Per-token losses ``(B, S)``.
        """
        logits_sbv = logits.transpose(0, 1).contiguous()
        labels_sb = labels.transpose(0, 1).contiguous()
        loss_sb = tensor_parallel.vocab_parallel_cross_entropy(logits_sbv, labels_sb)
        return loss_sb.transpose(0, 1).contiguous()

    def _context_hidden_to_logits(self, hidden: Tensor) -> Tensor:
        """Project context tower hidden states to logits via the context output head.

        Applies ``context_tower.final_norm`` (if the tower owns ``post_process``)
        and then the ``context_output_layer`` linear projection.

        Args:
            hidden (Tensor): Context tower output ``(S, B, D)``.

        Returns:
            Tensor: Logits ``(B, S, V)`` (full vocabulary, gathered across TP).
        """
        if self.context_tower.post_process and self.context_tower.post_layer_norm:
            hidden = self.context_tower.final_norm(hidden)
        logits, _ = self.context_output_layer(hidden, weight=None, runtime_gather_output=True)
        return logits.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        block_size: Optional[int] = None,
        noisy_input_ids: Optional[Tensor] = None,
        decoder_input: Tensor = None,
        runtime_gather_output: Optional[bool] = None,
        inference_context: BaseInferenceContext = None,
    ) -> Tensor:
        """Execute the full two-tower forward pass (training or inference).

        When *labels* are provided the method returns per-token losses
        ``(B, S)``; otherwise it returns logits ``(B, S, V)``.

        Steps:
            1. Corrupt *input_ids* via the active :class:`DiffusionProcess`
               (skipped if *noisy_input_ids* is supplied directly).
            2. Embed clean tokens (context tower) and noisy tokens (denoiser).
            3. Run the context tower, caching Attention KV pairs and Mamba
               states at every block boundary.
            4. Build a block-causal additive mask for the denoiser.
            5. Run the denoiser tower with cached context.
            6. Project to logits and, if *labels* are present, compute the
               diffusion training loss.

        Args:
            input_ids (Tensor): Clean token IDs ``(B, S)``.
            position_ids (Tensor): Position indices ``(B, S)``.
            attention_mask (Tensor): Causal mask ``(B, 1, S, S)`` for the
                context tower.
            labels (Optional[Tensor]): Target token IDs ``(B, S)``.  When
                provided, the return value is per-token loss instead of logits.
                For ``mask_diffusion`` mode, labels are overridden to
                ``input_ids.clone()`` (same-position targets).
            block_size (Optional[int]): Tokens per diffusion block.  Defaults
                to ``self.block_size``.
            noisy_input_ids (Optional[Tensor]): Pre-corrupted tokens ``(B, S)``.
                If ``None``, corruption is applied inside this method.
            decoder_input (Optional[Tensor]): Hidden states from a prior
                pipeline stage (pipeline parallelism).
            runtime_gather_output (Optional[bool]): Override ``parallel_output``
                for this call.
            inference_context (Optional[BaseInferenceContext]): Unused.

        Returns:
            Tensor: Per-token losses ``(B, S)`` when *labels* is not ``None``,
                or logits ``(B, S, V)`` otherwise.
        """
        if block_size is None:
            block_size = getattr(self, 'block_size', 1)

        batch_size, seq_len = input_ids.shape
        num_blocks = seq_len // block_size
        assert seq_len % block_size == 0

        aux: Dict[str, Tensor] = {}
        if noisy_input_ids is None:
            noisy_input_ids, aux = self.diffusion.corrupt_suffix(input_ids)

        ar_labels = labels
        if labels is not None:
            labels = input_ids.clone()

        if self.pre_process:
            clean_embeds = self.context_embedding(input_ids=input_ids, position_ids=position_ids)
            noisy_embeds = self.denoiser_embedding(
                input_ids=noisy_input_ids, position_ids=position_ids
            )
        else:
            clean_embeds = decoder_input
            noisy_embeds = decoder_input

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_pos_emb = self.rotary_pos_emb(seq_len)

        ctx_manager = torch.no_grad() if self.freeze_context else nullcontext()
        with ctx_manager:
            context_kv, context_mamba, context_hidden = self._forward_context_tower(
                hidden_states=clean_embeds,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                block_size=block_size,
            )

        block_mask = create_block_causal_mask(
            num_blocks=num_blocks,
            block_size=block_size,
            device=input_ids.device,
            dtype=clean_embeds.dtype,
        )

        t_emb = None
        if self.use_time_conditioning and 't' in aux:
            t = aux['t'].to(device=input_ids.device, dtype=clean_embeds.dtype)
            t_emb = self.t_block(self.t_embedder(t))

        hidden_states = self._forward_denoiser_tower(
            hidden_states=noisy_embeds,
            context_kv=context_kv,
            context_mamba=context_mamba,
            rotary_pos_emb=rotary_pos_emb,
            block_mask=block_mask,
            block_size=block_size,
            t_emb=t_emb,
        )

        if not self.post_process:
            return hidden_states.transpose(0, 1)

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        if labels is not None:
            logits_bsv = logits.transpose(0, 1).contiguous()
            loss = self.diffusion.training_loss(logits_bsv, labels, aux)

            if self.context_ar_loss and ar_labels is not None:
                context_logits = self._context_hidden_to_logits(context_hidden)
                ar_loss = F.cross_entropy(
                    context_logits.reshape(-1, context_logits.size(-1)),
                    ar_labels.reshape(-1),
                    reduction='mean',
                )
                loss = loss + ar_loss

            return loss

        return logits.transpose(0, 1).contiguous()

    def _forward_context_tower(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
        block_size: int,
    ) -> ContextCache:
        """Run the context tower and extract per-layer caches.

        Iterates through the context tower's layers.  For each:

        * **Attention layer** — extracts ``(key, value)`` with RoPE applied,
          appending to *kv_cache*.
        * **Mamba layer** — runs a specialised forward that exposes SSM and
          conv states at every block boundary, appending to *mamba_cache*.
          Requires ``block_size == mixer.chunk_size``.
        * **Other layers** (e.g. MLP) — standard forward.

        Args:
            hidden_states (Tensor): Embedded clean tokens ``(S, B, D)``.
            attention_mask (Tensor): Causal mask ``(B, 1, S, S)``.
            rotary_pos_emb (Optional[Tensor]): Rotary position embeddings.
            block_size (int): Tokens per diffusion block.

        Returns:
            ContextCache: ``(kv_cache, mamba_cache, final_hidden_states)``.
        """
        kv_cache: LayerKVCache = []
        mamba_cache: LayerMambaCache = []

        for layer_type, layer in zip(self.context_tower.layer_type_list, self.context_tower.layers):
            if layer_type == LayerSymbols.ATTENTION:
                hidden_states, kv = self._forward_attention_layer_with_kv_extract(
                    layer, hidden_states, attention_mask, rotary_pos_emb
                )
                kv_cache.append(kv)
            elif layer_type == LayerSymbols.MAMBA:
                if block_size != layer.mixer.chunk_size:
                    raise RuntimeError(
                        f"Context Mamba requires block_size ({block_size}) == "
                        f"chunk_size ({layer.mixer.chunk_size})."
                    )
                hidden_states, conv_states, ssm_states = (
                    forward_mamba_layer_parallel_with_all_states(layer, hidden_states, block_size)
                )
                mamba_cache.append((conv_states.detach(), ssm_states.detach()))
            else:
                if isinstance(layer, TransformerLayer):
                    hidden_states, _ = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                    )
                else:
                    hidden_states = layer(
                        hidden_states=hidden_states, attention_mask=attention_mask
                    )

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        return kv_cache, mamba_cache, hidden_states

    def _forward_attention_layer_with_kv_extract(
        self,
        layer: TransformerLayer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_pos_emb: Optional[Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Run one attention layer and capture its Key / Value projections.

        The captured KV pair (with RoPE already applied to Key) is later
        concatenated with the denoiser's own KV in
        :meth:`_forward_block_attention`.

        Args:
            layer (TransformerLayer): A single attention layer from the context
                tower.
            hidden_states (Tensor): Input ``(S, B, D)``.
            attention_mask (Tensor): Causal mask ``(B, 1, S, S)``.
            rotary_pos_emb (Optional[Tensor]): Rotary embeddings.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: ``(output [S, B, D],
                (key [S, B, H, D_head], value [S, B, H, D_head]))``.
        """
        self_attn = layer.self_attention
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        query, key, value = self_attn.get_query_key_value_tensors(
            hidden_states, key_value_states=None, split_qkv=True
        )

        if rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )

        captured_kv = (key.detach(), value.detach())

        attn_output = self_attn.core_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            attn_mask_type=self_attn.attn_mask_type,
        )

        attention_output_with_bias = self_attn.linear_proj(attn_output)

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, layer.hidden_dropout
            )

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            residual = hidden_states
            pre_mlp_output = layer.pre_mlp_layernorm(hidden_states)
            mlp_output_with_bias = layer.mlp(pre_mlp_output)
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, layer.hidden_dropout
                )

        return hidden_states, captured_kv

    def _forward_denoiser_tower(
        self,
        hidden_states: Tensor,
        context_kv: LayerKVCache,
        context_mamba: LayerMambaCache,
        rotary_pos_emb: Optional[Tensor],
        block_mask: Tensor,
        block_size: int,
        t_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the denoiser tower using cached context states.

        For each layer:

        * **Attention** — concatenates context KV with denoiser KV and applies
          the block-causal mask via :meth:`_forward_block_attention`.
        * **Mamba** — initialises each block's SSM / conv state from the
          corresponding context cache entry (block *N* gets context state from
          block *N-1*; block 0 gets zeros).  All blocks are processed in one
          batched kernel call.
        * **Other** — standard forward (no context conditioning).

        When ``use_time_conditioning`` is enabled and *t_emb* is not ``None``,
        per-layer modulation parameters ``(shift, scale, gate)`` are derived
        from *t_emb* and the learned ``scale_shift_tables`` and applied to
        every denoiser sub-layer.

        Args:
            hidden_states (Tensor): Denoiser embeddings ``(S, B, D)``.
            context_kv (LayerKVCache): Attention KV pairs from the context tower.
            context_mamba (LayerMambaCache): Mamba conv + SSM states from the
                context tower.
            rotary_pos_emb (Optional[Tensor]): Rotary embeddings.
            block_mask (Tensor): Additive block-causal mask
                ``(1, 1, S, 2*S)`` from :func:`create_block_causal_mask`.
            block_size (int): Tokens per diffusion block.
            t_emb (Optional[Tensor]): Global time embedding ``(B, 3*D)``
                from ``t_block``.  ``None`` when time conditioning is off or
                when the diffusion process did not provide a timestep.

        Returns:
            Tensor: Output hidden states ``(S, B, D)``.
        """
        seq_len = hidden_states.shape[0]
        num_blocks = seq_len // block_size

        attn_idx = 0
        mamba_idx = 0

        for layer_idx, (layer_type, layer) in enumerate(
            zip(self.denoiser_tower.layer_type_list, self.denoiser_tower.layers)
        ):
            mod_params = None
            if self.use_time_conditioning and t_emb is not None:
                mod_params = get_modulation_params(t_emb, self.scale_shift_tables[layer_idx])

            if layer_type == LayerSymbols.ATTENTION:
                k_ctx, v_ctx = context_kv[attn_idx]
                hidden_states = self._forward_block_attention(
                    layer=layer,
                    hidden_states=hidden_states,
                    context_kv=(k_ctx, v_ctx),
                    rotary_pos_emb=rotary_pos_emb,
                    block_mask=block_mask,
                    mod_params=mod_params,
                )
                attn_idx += 1

            elif layer_type == LayerSymbols.MAMBA:
                conv_states_ctx, ssm_states_ctx = context_mamba[mamba_idx]

                if block_size != layer.mixer.chunk_size:
                    raise RuntimeError(
                        f"Denoiser Mamba requires block_size ({block_size}) == "
                        f"chunk_size ({layer.mixer.chunk_size})."
                    )

                batch_size = hidden_states.shape[1]
                mixer = layer.mixer
                d_inner = mixer.cp.d_inner_local_tpcp
                ngroups = mixer.cp.ngroups_local_tpcp
                nheads = mixer.cp.nheads_local_tpcp
                d_state = mixer.d_state
                headdim = mixer.headdim
                conv_dim = d_inner + 2 * ngroups * d_state
                conv_state_width = mixer.d_conv - 1

                zero_conv = torch.zeros(
                    batch_size,
                    1,
                    conv_dim,
                    conv_state_width,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                zero_ssm = torch.zeros(
                    batch_size,
                    1,
                    nheads,
                    headdim,
                    d_state,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

                if num_blocks > 1:
                    initial_conv_states = torch.cat(
                        [zero_conv, conv_states_ctx[:, : num_blocks - 1]], dim=1
                    )
                    initial_ssm_states = torch.cat(
                        [zero_ssm, ssm_states_ctx[:, : num_blocks - 1]], dim=1
                    )
                else:
                    initial_conv_states = zero_conv
                    initial_ssm_states = zero_ssm

                hidden_states = forward_mamba_layer_batched_with_initial_states(
                    layer,
                    hidden_states,
                    block_size,
                    initial_conv_states,
                    initial_ssm_states,
                    bidirectional=self.bidirectional_mamba,
                    mod_params=mod_params,
                )
                mamba_idx += 1

            else:
                if isinstance(layer, TransformerLayer):
                    if mod_params is not None:
                        shift, scale, gate = mod_params
                        residual = hidden_states
                        hidden_states = layer.pre_mlp_layernorm(hidden_states)
                        hidden_states = modulate(hidden_states, shift, scale)
                        mlp_output = layer.mlp(hidden_states)
                        if isinstance(mlp_output, tuple):
                            mlp_output = mlp_output[0]
                        mlp_output = gate.unsqueeze(0) * mlp_output
                        hidden_states = residual + mlp_output
                    else:
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                        )
                else:
                    hidden_states = layer(hidden_states=hidden_states, attention_mask=None)

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        if self.denoiser_tower.post_process and self.denoiser_tower.post_layer_norm:
            hidden_states = self.denoiser_tower.final_norm(hidden_states)

        return hidden_states

    def _forward_block_attention(
        self,
        layer: TransformerLayer,
        hidden_states: Tensor,
        context_kv: Tuple[Tensor, Tensor],
        rotary_pos_emb: Optional[Tensor],
        block_mask: Tensor,
        mod_params: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tensor:
        """Denoiser attention with block-causal masking over context + self KV.

        Concatenates context KV (from :meth:`_forward_attention_layer_with_kv_extract`)
        with the denoiser's own QKV projections, applies GQA expansion if
        ``num_query_groups < num_attention_heads``, then runs
        ``scaled_dot_product_attention`` with the additive block-causal mask.

        When *mod_params* is provided, the post-layernorm hidden states are
        modulated with ``(shift, scale)`` before the QKV projection, and the
        attention output is multiplied by ``gate`` before the residual add.

        The head counts for GQA expansion are derived from actual tensor shapes
        rather than config attributes to remain correct when Megatron's
        ``SelfAttention`` adjusts heads for tensor parallelism.

        Args:
            layer (TransformerLayer): Denoiser attention layer.
            hidden_states (Tensor): Input ``(S, B, D)``.
            context_kv (Tuple[Tensor, Tensor]): ``(key, value)`` from the
                corresponding context attention layer.
            rotary_pos_emb (Optional[Tensor]): Rotary embeddings.
            block_mask (Tensor): Additive mask ``(1, 1, S, 2*S)``.
            mod_params (Optional[Tuple[Tensor, Tensor, Tensor]]): Per-layer
                ``(shift, scale, gate)`` from :func:`get_modulation_params`.

        Returns:
            Tensor: Output hidden states ``(S, B, D)``.
        """
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        if mod_params is not None:
            shift, scale, gate = mod_params
            hidden_states = modulate(hidden_states, shift, scale)

        self_attn = layer.self_attention

        query, key, value = self_attn.get_query_key_value_tensors(
            hidden_states, key_value_states=None, split_qkv=True
        )

        if rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )

        k_ctx, v_ctx = context_kv
        key_full = torch.cat([k_ctx, key], dim=0)
        value_full = torch.cat([v_ctx, value], dim=0)

        num_q_heads = query.shape[2]
        num_kv_heads = key.shape[2]
        if num_kv_heads != num_q_heads:
            repeats = num_q_heads // num_kv_heads
            key_full = key_full.repeat_interleave(repeats, dim=2)
            value_full = value_full.repeat_interleave(repeats, dim=2)

        query = query.permute(1, 2, 0, 3).contiguous()
        key_full = key_full.permute(1, 2, 0, 3).contiguous()
        value_full = value_full.permute(1, 2, 0, 3).contiguous()

        dropout_p = self_attn.config.attention_dropout if self.training else 0.0
        # TODO: This bypasses Megatron's --attention-backend (flash, TE fused,
        # etc.) because the block-causal cross-attention pattern (context KV
        # concatenated with denoiser KV + additive mask) may not be supported by
        # those backends.  This affects determinism properties and memory
        # efficiency at long sequence lengths.
        attn_output = F.scaled_dot_product_attention(
            query, key_full, value_full, attn_mask=block_mask, dropout_p=dropout_p, is_causal=False
        )

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        s_full, b, h, d = attn_output.shape
        attn_output = attn_output.reshape(s_full, b, h * d)

        attention_output_with_bias = self_attn.linear_proj(attn_output)

        if mod_params is not None:
            attn_out = (
                attention_output_with_bias[0]
                if isinstance(attention_output_with_bias, tuple)
                else attention_output_with_bias
            )
            attn_out = gate.unsqueeze(0) * attn_out
            attention_output_with_bias = (
                (attn_out, attention_output_with_bias[1])
                if isinstance(attention_output_with_bias, tuple)
                else attn_out
            )

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, layer.hidden_dropout
            )

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            residual = hidden_states
            pre_mlp_output = layer.pre_mlp_layernorm(hidden_states)
            mlp_output_with_bias = layer.mlp(pre_mlp_output)
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, layer.hidden_dropout
                )

        return hidden_states

    def load_from_single_tower(
        self, state_dict: dict, strict: bool = False, prefix: str = ''
    ) -> Tuple[List[str], List[str]]:
        """Load weights from a single-tower MambaModel checkpoint.

        Maps single-tower keys into the two-tower layout so that both towers
        start from the same pretrained weights:

        * ``decoder.*``     → ``context_tower.*``  **and** ``denoiser_tower.*``
        * ``embedding.*``   → ``context_embedding.*`` **and** ``denoiser_embedding.*``
        * ``output_layer.*`` → ``output_layer.*``  **and** ``context_output_layer.*``

        After loading, ``self._single_tower_mode`` is set to ``True`` so that
        the engine delegates to :meth:`generate_ar` instead of
        :meth:`generate_diffusion`.

        Args:
            state_dict (dict): Single-tower state dict (CPU or GPU).
            strict (bool): Passed to :meth:`nn.Module.load_state_dict`.
            prefix (str): Key prefix to strip before mapping.

        Returns:
            Tuple[List[str], List[str]]: ``(missing_keys, unexpected_keys)``
                from :meth:`load_state_dict`.
        """
        if prefix:
            state_dict = {
                k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()
            }

        full_state_dict = {}
        for key, value in state_dict.items():
            if value is None:
                continue
            if key.startswith('decoder.'):
                full_state_dict[key.replace('decoder.', 'context_tower.')] = value
                full_state_dict[key.replace('decoder.', 'denoiser_tower.')] = value
            elif key.startswith('embedding.'):
                full_state_dict[key.replace('embedding.', 'context_embedding.')] = value
                full_state_dict[key.replace('embedding.', 'denoiser_embedding.')] = value
            elif key.startswith('output_layer.'):
                full_state_dict[key] = value
                full_state_dict[key.replace('output_layer.', 'context_output_layer.')] = value
            else:
                full_state_dict[key] = value

        missing, unexpected = self.load_state_dict(full_state_dict, strict=strict)

        if self.tied_towers:
            if not self.context_ar_loss:
                self._freeze_context_output_layer()
        elif self.freeze_context:
            self._freeze_context_tower()
        elif not self.context_ar_loss:
            self._freeze_context_output_layer()

        self._single_tower_mode = True
        return missing, unexpected

    def _forward_context_with_cache(self, input_ids: Tensor, return_logits: bool = False) -> dict:
        """Build the initial context cache for inference.

        Processes *input_ids* through the context tower layer-by-layer,
        collecting Attention KV pairs and Mamba final states for each layer.

        Args:
            input_ids (Tensor): ``(B, S)`` input token IDs.
            return_logits (bool): If ``True``, project the final hidden states
                through ``context_output_layer`` and include ``"logits"`` in
                the returned dict.

        Returns:
            dict: ``{"kv": List[(K,V)], "mamba": List[(conv, ssm)],
            "len": int}`` and optionally ``"logits": (B, S, V)``.
        """
        B, S = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        hidden = self.context_embedding(input_ids=input_ids, position_ids=position_ids)
        rotary = self.rotary_pos_emb(S) if self.position_embedding_type == 'rope' else None

        kv_cache = []
        mamba_cache = []

        for layer_type, layer in zip(self.context_tower.layer_type_list, self.context_tower.layers):
            if layer_type == LayerSymbols.ATTENTION:
                hidden, kv = self._forward_attention_layer_with_kv_extract(
                    layer, hidden, None, rotary
                )
                kv_cache.append(kv)
            elif layer_type == LayerSymbols.MAMBA:
                hidden, (conv_final, ssm_final) = forward_mamba_layer_with_states(
                    layer, hidden, None, None, bidirectional=False
                )
                mamba_cache.append((conv_final.detach(), ssm_final.detach()))
            else:
                hidden, _ = layer(hidden_states=hidden, attention_mask=None)

        result = {"kv": kv_cache, "mamba": mamba_cache, "len": S}
        if return_logits:
            result["logits"] = self._context_hidden_to_logits(hidden)
        return result

    @staticmethod
    def _split_by_token_budget(
        prompt_ids_list: List[Tensor], max_tokens: int
    ) -> List[List[Tensor]]:
        """Greedily partition prompts so each group fits within a token budget.

        Args:
            prompt_ids_list (List[Tensor]): 1-D token ID tensors to partition.
            max_tokens (int): Maximum total tokens per group.

        Returns:
            List[List[Tensor]]: Groups of prompts, each with total tokens
            ``<= max_tokens``.
        """
        batches: List[List[Tensor]] = []
        current_batch: List[Tensor] = []
        current_tokens = 0
        for prompt in prompt_ids_list:
            prompt_len = prompt.shape[0]
            if current_batch and current_tokens + prompt_len > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(prompt)
            current_tokens += prompt_len
        if current_batch:
            batches.append(current_batch)
        return batches

    @torch.no_grad()
    def _prefill(
        self,
        prompt_ids_list: List[Tensor],
        return_last_logits: bool = False,
        max_tokens: int = DynamicInferenceContext.DEFAULT_MAX_TOKENS,
    ) -> List[dict]:
        """Token-packed context tower prefill for variable-length prompts.

        Packs prompts into a single forward pass bounded by
        ``max_tokens`` (matching the dynamic engine's budget) and runs
        the context tower using varlen attention and Mamba kernels.  No
        padding token noise enters the caches.  If the total token count
        exceeds the budget, prompts are split across the fewest passes
        necessary.

        Args:
            prompt_ids_list (List[Tensor]): List of B 1-D token ID tensors.
            return_last_logits (bool): If ``True``, attach the final-token
                logit per request as ``cache["last_logits"]`` (shape
                ``(1, V)``).
            max_tokens (int): Maximum total tokens per forward pass.
                Defaults to ``DynamicInferenceContext.DEFAULT_MAX_TOKENS``
                to match the dynamic engine's token budget and keep SSM
                intermediate memory bounded.

        Returns:
            List[dict]: Per-request cache dicts with keys
            ``"kv"``, ``"mamba"``, ``"len"``.
        """
        all_caches: List[dict] = []
        for batch in self._split_by_token_budget(prompt_ids_list, max_tokens):
            all_caches.extend(self._prefill_packed(batch, return_last_logits))
        return all_caches

    @torch.no_grad()
    def _run_context_tower_packed(
        self, prompt_ids_list: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[int], List[Tuple[Tensor, Tensor]], List[Tuple[Tensor, Tensor]]]:
        """Pack prompts and run the context tower, returning raw outputs.

        Args:
            prompt_ids_list (List[Tensor]): 1-D token ID tensors to pack.

        Returns:
            Tuple of ``(hidden, cu_seqlens, lengths, kv_cache, mamba_cache)``
            where *hidden* is ``(total_tokens, 1, D)``, *kv_cache* is a
            list of ``(K, V)`` pairs per attention layer, and *mamba_cache*
            is a list of ``(conv_states, ssm_states)`` per Mamba layer.
        """
        B = len(prompt_ids_list)
        device = prompt_ids_list[0].device
        lengths = [p.shape[0] for p in prompt_ids_list]
        max_seqlen = max(lengths)

        packed_ids = torch.cat(prompt_ids_list)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
        )

        positions = torch.cat([torch.arange(seq_len, device=device) for seq_len in lengths])

        hidden = self.context_embedding(
            input_ids=packed_ids.unsqueeze(0), position_ids=positions.unsqueeze(0)
        )

        rotary_packed = None
        if self.position_embedding_type == 'rope':
            full_rotary = self.rotary_pos_emb(max_seqlen)
            rotary_packed = full_rotary[positions]

        kv_cache: List[Tuple[Tensor, Tensor]] = []
        mamba_cache: List[Tuple[Tensor, Tensor]] = []

        for layer_type, layer in zip(self.context_tower.layer_type_list, self.context_tower.layers):
            if layer_type == LayerSymbols.ATTENTION:
                hidden, kv = self._forward_attention_layer_varlen(
                    layer, hidden, rotary_packed, cu_seqlens, max_seqlen
                )
                kv_cache.append(kv)
            elif layer_type == LayerSymbols.MAMBA:
                hidden, states = forward_mamba_layer_varlen_with_states(layer, hidden, cu_seqlens)
                mamba_cache.append(states)
            else:
                hidden, _ = layer(hidden_states=hidden, attention_mask=None)

        return hidden, cu_seqlens, lengths, kv_cache, mamba_cache

    @torch.no_grad()
    def _prefill_packed(
        self, prompt_ids_list: List[Tensor], return_last_logits: bool = False
    ) -> List[dict]:
        """Run a packed batch through the context tower and extract caches.

        Args:
            prompt_ids_list (List[Tensor]): 1-D token ID tensors to pack
                into a single varlen forward pass.
            return_last_logits (bool): If ``True``, attach the final-token
                logit per request.

        Returns:
            List[dict]: Per-request cache dicts.
        """
        hidden, cu_seqlens, lengths, kv_cache, mamba_cache = self._run_context_tower_packed(
            prompt_ids_list
        )
        B = len(prompt_ids_list)

        packed_logits = None
        if return_last_logits:
            packed_logits = self._context_hidden_to_logits(hidden).squeeze(0)

        per_request_caches: List[dict] = []
        for i in range(B):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            req_kv = []
            for k_packed, v_packed in kv_cache:
                req_kv.append((k_packed[start:end].clone(), v_packed[start:end].clone()))

            req_mamba = []
            for conv_states, ssm_states in mamba_cache:
                req_mamba.append((conv_states[i : i + 1].clone(), ssm_states[i : i + 1].clone()))

            cache: dict = {"kv": req_kv, "mamba": req_mamba, "len": lengths[i]}
            if packed_logits is not None:
                cache["last_logits"] = packed_logits[end - 1 : end]
            per_request_caches.append(cache)

        return per_request_caches

    def _forward_attention_layer_varlen(
        self,
        layer: TransformerLayer,
        hidden_states: Tensor,
        rotary_packed: Optional[Tensor],
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Attention layer forward using ``flash_attn_varlen_func``.

        Operates on packed ``(total_tokens, 1, D)`` hidden states.
        Returns the output and captured ``(K, V)`` with RoPE applied,
        both in ``(total_tokens, 1, H, D_head)`` layout.

        Args:
            layer (TransformerLayer): Context tower attention layer.
            hidden_states (Tensor): ``(total_tokens, 1, D)`` packed input.
            rotary_packed (Optional[Tensor]): ``(total_tokens, 1, 1, D_rot)``
                per-token RoPE or ``None``.
            cu_seqlens (Tensor): ``(B+1,)`` int32 cumulative sequence lengths.
            max_seqlen (int): Maximum sequence length in the batch.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: ``(output, (key, value))``
            where tensors are ``(total_tokens, 1, H, D_head)``.
        """
        from flash_attn import flash_attn_varlen_func

        self_attn = layer.self_attention
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        query, key, value = self_attn.get_query_key_value_tensors(
            hidden_states, key_value_states=None, split_qkv=True
        )
        # query/key/value: (total_tokens, 1, H, D_head)

        if rotary_packed is not None:
            rpe = (rotary_packed, rotary_packed)
            q_emb, k_emb = rpe
            query = apply_rotary_pos_emb(
                query, q_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
            )
            key = apply_rotary_pos_emb(
                key, k_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
            )

        captured_kv = (key.detach(), value.detach())

        # flash_attn_varlen_func expects (total_tokens, H, D_head)
        q_fa = query.squeeze(1)
        k_fa = key.squeeze(1)
        v_fa = value.squeeze(1)

        attn_out = flash_attn_varlen_func(
            q_fa, k_fa, v_fa, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )
        # attn_out: (total_tokens, H, D_head)

        attn_out = attn_out.unsqueeze(1)  # (total_tokens, 1, H, D_head)
        t, b, h, d = attn_out.shape
        attn_out = attn_out.reshape(t, b, h * d)

        attention_output_with_bias = self_attn.linear_proj(attn_out)

        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, layer.hidden_dropout
            )

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            residual = hidden_states
            pre_mlp_output = layer.pre_mlp_layernorm(hidden_states)
            mlp_output_with_bias = layer.mlp(pre_mlp_output)
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, layer.hidden_dropout
                )

        return hidden_states, captured_kv

    def _extend_context_cache(
        self, new_tokens: Tensor, cache: dict, return_logits: bool = False
    ) -> dict:
        """Extend the context cache after generating a block.

        Processes *new_tokens* through the context tower, attending to past
        context via :meth:`_forward_context_attn_with_past_kv` for attention
        layers and continuing the Mamba recurrence from the last cached state.

        Args:
            new_tokens (Tensor): ``(B, block_len)`` newly generated tokens.
            cache (dict): Existing cache from :meth:`_forward_context_with_cache`
                or a previous call to this method.
            return_logits (bool): If ``True``, include ``"logits"`` for
                *new_tokens* in the updated cache.

        Returns:
            dict: Updated cache with extended KV, refreshed Mamba states,
            and incremented ``"len"``.
        """
        B = new_tokens.shape[0]
        seq_pos = cache["len"]
        new_len = new_tokens.shape[1]

        position_ids = (
            torch.arange(seq_pos, seq_pos + new_len, device=new_tokens.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        hidden = self.context_embedding(input_ids=new_tokens, position_ids=position_ids)

        rotary = None
        if self.position_embedding_type == 'rope':
            rotary = self.rotary_pos_emb(seq_pos + new_len)[seq_pos:]

        kv_idx, mamba_idx = 0, 0
        for layer_type, layer in zip(self.context_tower.layer_type_list, self.context_tower.layers):
            if layer_type == LayerSymbols.ATTENTION:
                past_k, past_v = cache["kv"][kv_idx]
                hidden, new_kv = self._forward_context_attn_with_past_kv(
                    layer, hidden, past_k, past_v, rotary
                )
                cache["kv"][kv_idx] = (
                    torch.cat([past_k, new_kv[0]], dim=0),
                    torch.cat([past_v, new_kv[1]], dim=0),
                )
                kv_idx += 1
            elif layer_type == LayerSymbols.MAMBA:
                last_conv, last_ssm = cache["mamba"][mamba_idx]
                init_conv = last_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
                init_ssm = last_ssm.contiguous()
                hidden, (new_conv, new_ssm) = forward_mamba_layer_with_states(
                    layer, hidden, init_conv, init_ssm, bidirectional=False
                )
                cache["mamba"][mamba_idx] = (new_conv.detach(), new_ssm.detach())
                mamba_idx += 1
            else:
                hidden, _ = layer(hidden_states=hidden, attention_mask=None)

        cache["len"] = seq_pos + new_len
        if return_logits:
            cache["logits"] = self._context_hidden_to_logits(hidden)
        return cache

    def _forward_attn_with_past(
        self,
        layer: TransformerLayer,
        hidden: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        rotary: Optional[Tensor],
        mod_params: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Denoiser attention against cached context KV.

        Queries come from *hidden* (denoiser tokens), while keys and values
        are the concatenation of *past_k* / *past_v* (context cache) and the
        denoiser's own KV projections.  Attention is **non-causal** by default
        (the denoiser block sees all past context and its own tokens
        bidirectionally).  An optional additive *attn_mask* can restrict
        visibility (e.g. to exclude left-padding in batched caches).

        Handles GQA by repeating KV heads to match query heads.

        Args:
            layer (TransformerLayer): Denoiser attention layer.
            hidden (Tensor): ``(S, B, D)`` denoiser hidden states.
            past_k (Tensor): ``(past_S, B, H, D_head)`` cached context keys.
            past_v (Tensor): ``(past_S, B, H, D_head)`` cached context values.
            rotary (Optional[Tensor]): Rotary embeddings for denoiser positions.
            mod_params (Optional[Tuple]): ``(shift, scale, gate)`` for time
                conditioning.
            attn_mask (Optional[Tensor]): Additive attention mask broadcastable
                to ``(B, H, S, past_S + S)``.  ``None`` means fully visible.

        Returns:
            Tensor: Output ``(S, B, D)``.
        """
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        if mod_params is not None:
            shift, scale, gate = mod_params
            hidden = modulate(hidden, shift, scale)

        q, k, v = layer.self_attention.get_query_key_value_tensors(hidden, None)

        if rotary is not None:
            q = layer.self_attention.rotary_pos_emb(q, rotary)
            k = layer.self_attention.rotary_pos_emb(k, rotary)

        k_full = torch.cat([past_k, k], dim=0)
        v_full = torch.cat([past_v, v], dim=0)

        q_p = q.permute(1, 2, 0, 3)
        k_p = k_full.permute(1, 2, 0, 3)
        v_p = v_full.permute(1, 2, 0, 3)

        num_q_heads = q_p.shape[1]
        num_kv_heads = k_p.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k_p = k_p.repeat_interleave(repeat_factor, dim=1)
            v_p = v_p.repeat_interleave(repeat_factor, dim=1)

        out = F.scaled_dot_product_attention(q_p, k_p, v_p, attn_mask=attn_mask, is_causal=False)
        out = out.permute(2, 0, 1, 3).contiguous()
        out = out.reshape(out.shape[0], out.shape[1], -1)

        attn_out, _ = layer.self_attention.linear_proj(out)

        if mod_params is not None:
            attn_out = gate.unsqueeze(0) * attn_out

        with layer.bias_dropout_add_exec_handler():
            hidden = layer.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                (attn_out, None), residual, layer.hidden_dropout
            )

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            residual = hidden
            mlp_out = layer.mlp(layer.pre_mlp_layernorm(hidden))
            with layer.bias_dropout_add_exec_handler():
                hidden = layer.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_out, residual, layer.hidden_dropout
                )

        return hidden

    def _forward_context_attn_with_past_kv(
        self,
        layer: TransformerLayer,
        hidden_states: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        rotary_pos_emb: Optional[Tensor],
        ext_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Context attention for cache extension, returning new KV.

        Queries come from *hidden_states* (new tokens being appended),
        keys / values are ``[past | new]``.  Causal masking is applied
        within the new-token block so that position *i* only attends to
        past context and new tokens ``0..i``.

        Args:
            layer (TransformerLayer): Context attention layer.
            hidden_states (Tensor): ``(S_new, B, D)`` new-token hidden states.
            past_k (Tensor): ``(S_past, B, H, D_head)`` cached context keys.
            past_v (Tensor): ``(S_past, B, H, D_head)`` cached context values.
            rotary_pos_emb (Optional[Tensor]): Rotary embeddings for new
                positions only.
            ext_attn_mask (Optional[Tensor]): Additive attention mask
                broadcastable to ``(B, H, S_new, S_past + S_new)``.
                ``None`` applies default causal masking.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: ``(output, (new_key, new_value))``
            where ``new_key`` / ``new_value`` are the KV projections of the
            new tokens (for cache concatenation).
        """
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        self_attn = layer.self_attention

        query, key, value = self_attn.get_query_key_value_tensors(
            hidden_states, key_value_states=None, split_qkv=True
        )

        if rotary_pos_emb is not None:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query, q_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key, k_pos_emb, config=self_attn.config, cp_group=self_attn.pg_collection.cp
                )

        new_kv = (key.detach(), value.detach())

        key_full = torch.cat([past_k, key], dim=0)
        value_full = torch.cat([past_v, value], dim=0)

        num_q_heads = query.shape[2]
        num_kv_heads = key.shape[2]
        if num_kv_heads != num_q_heads:
            repeats = num_q_heads // num_kv_heads
            key_full = key_full.repeat_interleave(repeats, dim=2)
            value_full = value_full.repeat_interleave(repeats, dim=2)

        query = query.permute(1, 2, 0, 3).contiguous()
        key_full = key_full.permute(1, 2, 0, 3).contiguous()
        value_full = value_full.permute(1, 2, 0, 3).contiguous()

        dropout_p = self_attn.config.attention_dropout if self.training else 0.0
        q_len = query.shape[2]
        k_len = key_full.shape[2]

        if ext_attn_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                query, key_full, value_full, attn_mask=ext_attn_mask, dropout_p=dropout_p
            )
        elif q_len == 1:
            attn_output = F.scaled_dot_product_attention(
                query, key_full, value_full, dropout_p=dropout_p, is_causal=False
            )
        else:
            past_len = k_len - q_len
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device)
            mask = torch.tril(mask, diagonal=past_len)
            attn_mask = torch.zeros(q_len, k_len, dtype=query.dtype, device=query.device)
            attn_mask.masked_fill_(~mask, float('-inf'))
            attn_output = F.scaled_dot_product_attention(
                query, key_full, value_full, attn_mask=attn_mask, dropout_p=dropout_p
            )

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        s_out, b, h, d = attn_output.shape
        attn_output = attn_output.reshape(s_out, b, h * d)

        attention_output_with_bias = self_attn.linear_proj(attn_output)
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, layer.hidden_dropout
            )

        if hasattr(layer, 'mlp') and layer.mlp is not None:
            residual = hidden_states
            pre_mlp_output = layer.pre_mlp_layernorm(hidden_states)
            mlp_output_with_bias = layer.mlp(pre_mlp_output)
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, layer.hidden_dropout
                )

        return hidden_states, new_kv

    @torch.no_grad()
    def forward_for_likelihood(
        self,
        prompt_ids_list: List[Tensor],
        max_tokens: int = DynamicInferenceContext.DEFAULT_MAX_TOKENS,
    ) -> List[Tensor]:
        """Token-packed context tower forward for likelihood evaluation.

        Runs the context tower via :meth:`_run_context_tower_packed` and
        returns all-position logits per request.  No padding — uses varlen
        packing with the same token budget as :meth:`_prefill`.

        Args:
            prompt_ids_list (List[Tensor]): List of 1-D token ID tensors.
            max_tokens (int): Maximum total tokens per forward pass.

        Returns:
            List[Tensor]: Per-request logits, each ``(seq_len, V)``.
        """
        all_logits: List[Tensor] = []
        for batch in self._split_by_token_budget(prompt_ids_list, max_tokens):
            hidden, cu_seqlens, lengths, _, _ = self._run_context_tower_packed(batch)
            packed_logits = self._context_hidden_to_logits(hidden).squeeze(0)
            B = len(batch)
            for i in range(B):
                all_logits.append(packed_logits[cu_seqlens[i] : cu_seqlens[i + 1]])
        return all_logits

    @torch.no_grad()
    def generate_ar(
        self,
        prompt_ids_list: List[Tensor],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[Tensor], int]:
        """Single-tower AR generation using only the context tower.

        Uses :meth:`_prefill` for token-packed prefill, then generates
        one token at a time via greedy or sampling-based decoding through
        the batched cache extension path.

        Args:
            prompt_ids_list (List[Tensor]): List of B 1-D token ID tensors.
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Sampling temperature (0 = greedy).
            top_k (Optional[int]): Top-k filtering (``None`` = disabled).
            top_p (Optional[float]): Nucleus sampling threshold.
            eos_token_id (Optional[int]): EOS token ID (reserved for future
                early-stopping support).

        Returns:
            Tuple[List[Tensor], int]: ``(output_list, nfe)`` — per-request
            output tensors (prompt + generated) and forward-pass count.
        """
        B = len(prompt_ids_list)

        per_request_caches = self._prefill(prompt_ids_list, return_last_logits=True)
        nfe = 1

        # Extract last-token logits and merge caches for batched decode
        last_logits_list = [c.pop("last_logits") for c in per_request_caches]
        next_logits = torch.stack(last_logits_list).squeeze(1)  # (B, V)

        batched_cache = self._merge_context_caches(per_request_caches)

        generated = []
        for _ in range(max_new_tokens):
            if temperature <= 0 or temperature is None:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / max(temperature, 1e-8), dim=-1)
                if top_k is not None and top_k > 0:
                    kth = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1).values[..., -1:]
                    probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumsum = torch.cumsum(sorted_p, dim=-1)
                    mask = cumsum > top_p
                    mask_shift = torch.cat(
                        [torch.zeros_like(mask[..., :1]), mask[..., :-1]], dim=-1
                    )
                    sorted_p = torch.where(mask_shift, torch.zeros_like(sorted_p), sorted_p)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_p)
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            batched_cache = self._extend_context_cache_batched(
                next_token, batched_cache, return_logits=True
            )
            nfe += 1

            next_logits = batched_cache["logits"][:, -1, :]

        if generated:
            all_generated = torch.cat(generated, dim=1)
        else:
            all_generated = torch.empty(B, 0, dtype=torch.long, device=prompt_ids_list[0].device)

        results: List[Tensor] = []
        for i in range(B):
            results.append(torch.cat([prompt_ids_list[i], all_generated[i]]))

        return results, nfe

    @torch.no_grad()
    def generate_diffusion(
        self,
        prompt_ids_list: List[Tensor],
        max_new_tokens: int,
        block_length: int = 1,
        steps_per_block: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        step_callback: Optional[Callable] = None,
        adaptive_unmasking: bool = False,
        sampling_strategy: str = "predict_and_noise",
        posterior_float64: bool = False,
        noise_schedule: str = "linear",
        confidence_threshold: float = 1e6,
        debug_tokenizer: Optional['MegatronTokenizer'] = None,
    ) -> Tuple[List[Tensor], int]:
        """Block-wise mask-diffusion generation for variable-length prompts.

        Uses :meth:`_prefill` for token-packed context tower processing,
        then runs the batched denoiser loop across all requests.

        Args:
            prompt_ids_list (List[Tensor]): List of B 1-D token ID tensors.
            max_new_tokens (int): Tokens to generate (divisible by
                *block_length*).
            block_length (int): Tokens per diffusion block.
            steps_per_block (int): Denoising iterations per block.
            temperature (float): Sampling temperature.
            top_k (Optional[int]): Top-k filtering (``None`` = disabled).
            top_p (Optional[float]): Nucleus sampling threshold.
            eos_token_id (Optional[int]): EOS token ID (reserved for future
                early-stopping support).
            step_callback (Optional[Callable]): Called after each denoising
                step with the current block tensor.
            adaptive_unmasking (bool): Enable adaptive unmasking during
                denoising.
            sampling_strategy (str): ``"predict_and_noise"``, ``"posterior"``,
                or ``"confidence_unmasking"``.
            posterior_float64 (bool): Use float64 for posterior computation.
            noise_schedule (str): Noise schedule type (``"linear"``, etc.).
            confidence_threshold (float): Threshold for confidence unmasking.
            debug_tokenizer (Optional['MegatronTokenizer']): Tokenizer for
                debug logging.

        Returns:
            Tuple[List[Tensor], int]: ``(output_list, nfe)`` — per-request
            output tensors (prompt + generated) and total denoiser
            forward-pass count.
        """
        B = len(prompt_ids_list)
        device = prompt_ids_list[0].device
        num_blocks = max_new_tokens // block_length
        assert max_new_tokens % block_length == 0

        per_request_caches = self._prefill(prompt_ids_list)
        batched_cache = self._merge_context_caches(per_request_caches)

        nfe = 0
        generated_blocks: List[Tensor] = []

        for block_idx in range(num_blocks):
            block_ids = torch.full(
                (B, block_length), self.mask_token_id, dtype=torch.long, device=device
            )

            def run_denoiser(x_t: Tensor, t: Optional[Tensor] = None) -> Tensor:
                nonlocal nfe
                nfe += 1
                return self._run_denoiser_step_batched(x_t, batched_cache, t=t)

            denoised = self.diffusion.sample_block(
                run_denoiser,
                init_ids=block_ids,
                num_steps=steps_per_block,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
                top_k=top_k if top_k is not None else 0,
                step_callback=step_callback,
                adaptive_unmasking=adaptive_unmasking,
                posterior_float64=posterior_float64,
                noise_schedule=noise_schedule,
                confidence_threshold=confidence_threshold,
            )

            generated_blocks.append(denoised)
            batched_cache = self._extend_context_cache_batched(denoised, batched_cache)

        if generated_blocks:
            all_generated = torch.cat(generated_blocks, dim=1)
        else:
            all_generated = torch.empty(B, 0, dtype=torch.long, device=device)

        results: List[Tensor] = []
        for i in range(B):
            results.append(torch.cat([prompt_ids_list[i], all_generated[i]]))

        return results, nfe

    def _merge_context_caches(self, caches: List[dict]) -> dict:
        """Merge per-request caches into a batched cache with left-padded KV.

        Each request's KV cache is left-padded with zeros so that all have
        the same sequence-length dimension.  Mamba states are stacked along
        the batch dimension.  The ``"pad_amounts"`` list records the
        per-request left-padding size (constant throughout generation).

        Args:
            caches (List[dict]): Per-request cache dicts from :meth:`_prefill`.

        Returns:
            dict: Batched cache with keys ``"kv"``, ``"mamba"``, ``"len"``,
            and ``"pad_amounts"``.
        """
        lengths = [c["len"] for c in caches]
        max_len = max(lengths)
        pad_amounts = [max_len - l for l in lengths]

        batched_kv = []
        for layer_idx in range(len(caches[0]["kv"])):
            ks, vs = [], []
            for i, c in enumerate(caches):
                k, v = c["kv"][layer_idx]
                pa = pad_amounts[i]
                if pa > 0:
                    k = F.pad(k, (0, 0, 0, 0, 0, 0, pa, 0))
                    v = F.pad(v, (0, 0, 0, 0, 0, 0, pa, 0))
                ks.append(k)
                vs.append(v)
            batched_kv.append((torch.cat(ks, dim=1), torch.cat(vs, dim=1)))

        batched_mamba = []
        for layer_idx in range(len(caches[0]["mamba"])):
            convs, ssms = [], []
            for c in caches:
                conv, ssm = c["mamba"][layer_idx]
                convs.append(conv)
                ssms.append(ssm)
            batched_mamba.append((torch.cat(convs, dim=0), torch.cat(ssms, dim=0)))

        return {
            "kv": batched_kv,
            "mamba": batched_mamba,
            "len": max_len,
            "pad_amounts": pad_amounts,
        }

    def _run_denoiser_step_batched(
        self, x_t: Tensor, cache: dict, t: Optional[Tensor] = None
    ) -> Tensor:
        """Batched denoiser forward for variable-length prompt contexts.

        Runs the denoiser tower using per-request position IDs and an
        attention mask that excludes left-padding in the KV cache.

        Args:
            x_t (Tensor): ``(B, L)`` noisy block token IDs.
            cache (dict): Batched cache from :meth:`_merge_context_caches`.
            t (Optional[Tensor]): Diffusion timestep embedding.

        Returns:
            Tensor: ``(B, L, V)`` logits over the vocabulary.
        """
        B, L = x_t.shape
        pad_amounts = cache["pad_amounts"]
        padded_len = cache["len"]
        actual_lengths = [padded_len - pa for pa in pad_amounts]
        device = x_t.device

        position_ids = torch.stack(
            [torch.arange(al, al + L, device=device) for al in actual_lengths]
        )

        hidden = self.denoiser_embedding(input_ids=x_t, position_ids=position_ids)

        rotary = None
        if self.position_embedding_type == 'rope':
            max_pos = max(actual_lengths) + L
            full_rotary = self.rotary_pos_emb(max_pos)
            rotary = torch.cat([full_rotary[al : al + L] for al in actual_lengths], dim=1)

        t_emb = None
        if self.use_time_conditioning and t is not None:
            t = t.to(device=device, dtype=hidden.dtype)
            t_repr = self.t_embedder(t)
            t_emb = self.t_block(t_repr)

        attn_mask = self._build_batched_kv_mask(pad_amounts, padded_len, L, device, hidden.dtype)

        kv_idx, mamba_idx = 0, 0
        for layer_idx, (layer_type, layer) in enumerate(
            zip(self.denoiser_tower.layer_type_list, self.denoiser_tower.layers)
        ):
            mod_params = None
            if self.use_time_conditioning and t_emb is not None:
                mod_params = get_modulation_params(t_emb, self.scale_shift_tables[layer_idx])

            if layer_type == LayerSymbols.ATTENTION:
                ctx_k, ctx_v = cache["kv"][kv_idx]
                hidden = self._forward_attn_with_past(
                    layer, hidden, ctx_k, ctx_v, rotary, mod_params=mod_params, attn_mask=attn_mask
                )
                kv_idx += 1
            elif layer_type == LayerSymbols.MAMBA:
                last_conv, last_ssm = cache["mamba"][mamba_idx]
                init_conv = last_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
                init_ssm = last_ssm.contiguous()
                hidden, _ = forward_mamba_layer_with_states(
                    layer,
                    hidden,
                    init_conv,
                    init_ssm,
                    bidirectional=self.bidirectional_mamba,
                    mod_params=mod_params,
                )
                mamba_idx += 1
            else:
                if mod_params is not None:
                    shift, scale, gate = mod_params
                    residual = hidden
                    hidden = layer.pre_mlp_layernorm(hidden)
                    hidden = modulate(hidden, shift, scale)
                    mlp_output = layer.mlp(hidden)
                    if isinstance(mlp_output, tuple):
                        mlp_output = mlp_output[0]
                    mlp_output = gate.unsqueeze(0) * mlp_output
                    hidden = residual + mlp_output
                else:
                    hidden, _ = layer(hidden_states=hidden, attention_mask=None)

        hidden = self.denoiser_tower.final_norm(hidden)
        weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else None
        )
        logits, _ = self.output_layer(hidden, weight=weight, runtime_gather_output=True)
        return logits.transpose(0, 1).contiguous()

    def _extend_context_cache_batched(
        self, new_tokens: Tensor, cache: dict, return_logits: bool = False
    ) -> dict:
        """Extend batched context cache after generating a block.

        Like :meth:`_extend_context_cache` but handles left-padded KV
        caches and per-request position offsets.  Builds a causal attention
        mask that excludes padding while preserving causal ordering within
        the new tokens.

        Args:
            new_tokens (Tensor): ``(B, S_new)`` accepted block token IDs.
            cache (dict): Batched cache to extend in-place.
            return_logits (bool): If ``True``, store next-token logits in
                ``cache["logits"]``.

        Returns:
            dict: The updated *cache*.
        """
        B = new_tokens.shape[0]
        new_len = new_tokens.shape[1]
        padded_len = cache["len"]
        pad_amounts = cache["pad_amounts"]
        actual_lengths = [padded_len - pa for pa in pad_amounts]
        device = new_tokens.device

        position_ids = torch.stack(
            [torch.arange(al, al + new_len, device=device) for al in actual_lengths]
        )
        hidden = self.context_embedding(input_ids=new_tokens, position_ids=position_ids)

        rotary = None
        if self.position_embedding_type == 'rope':
            max_pos = max(actual_lengths) + new_len
            full_rotary = self.rotary_pos_emb(max_pos)
            rotary = torch.cat([full_rotary[al : al + new_len] for al in actual_lengths], dim=1)

        ext_attn_mask = self._build_batched_causal_kv_mask(
            pad_amounts, padded_len, new_len, device, hidden.dtype
        )

        kv_idx, mamba_idx = 0, 0
        for layer_type, layer in zip(self.context_tower.layer_type_list, self.context_tower.layers):
            if layer_type == LayerSymbols.ATTENTION:
                past_k, past_v = cache["kv"][kv_idx]
                hidden, new_kv = self._forward_context_attn_with_past_kv(
                    layer, hidden, past_k, past_v, rotary, ext_attn_mask=ext_attn_mask
                )
                cache["kv"][kv_idx] = (
                    torch.cat([past_k, new_kv[0]], dim=0),
                    torch.cat([past_v, new_kv[1]], dim=0),
                )
                kv_idx += 1
            elif layer_type == LayerSymbols.MAMBA:
                last_conv, last_ssm = cache["mamba"][mamba_idx]
                init_conv = last_conv.transpose(-1, -2).contiguous().transpose(-1, -2)
                init_ssm = last_ssm.contiguous()
                hidden, (new_conv, new_ssm) = forward_mamba_layer_with_states(
                    layer, hidden, init_conv, init_ssm, bidirectional=False
                )
                cache["mamba"][mamba_idx] = (new_conv.detach(), new_ssm.detach())
                mamba_idx += 1
            else:
                hidden, _ = layer(hidden_states=hidden, attention_mask=None)

        cache["len"] = padded_len + new_len
        if return_logits:
            cache["logits"] = self._context_hidden_to_logits(hidden)
        return cache

    def _build_batched_kv_mask(
        self,
        pad_amounts: List[int],
        past_kv_len: int,
        q_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Build attention mask for left-padded KV caches (denoiser attention).

        Returns ``None`` when no padding exists.  For the denoiser, all
        query positions have the same mask (bidirectional within the block),
        so the query dimension is a broadcast singleton.

        Args:
            pad_amounts (List[int]): Per-request left-padding sizes.
            past_kv_len (int): Sequence length of the cached KV.
            q_len (int): Number of query (denoiser block) tokens.
            device (torch.device): Target device.
            dtype (torch.dtype): Mask dtype (should match hidden states).

        Returns:
            Optional[Tensor]: ``(B, 1, 1, past_kv_len + q_len)`` additive
            float mask, or ``None`` when no padding exists.
        """
        if all(pa == 0 for pa in pad_amounts):
            return None

        total_kv = past_kv_len + q_len
        kv_pos = torch.arange(total_kv, device=device)
        pad_t = torch.tensor(pad_amounts, device=device)
        valid = kv_pos.unsqueeze(0) >= pad_t.unsqueeze(1)

        mask = torch.zeros(len(pad_amounts), 1, 1, total_kv, device=device, dtype=dtype)
        mask.masked_fill_(~valid.unsqueeze(1).unsqueeze(1), float('-inf'))
        return mask

    def _build_batched_causal_kv_mask(
        self,
        pad_amounts: List[int],
        past_kv_len: int,
        q_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Build causal attention mask for context cache extension with padding.

        Returns ``None`` when no padding exists.  New-token queries attend
        causally within themselves and fully to all non-padding past context.

        Args:
            pad_amounts (List[int]): Per-request left-padding sizes.
            past_kv_len (int): Sequence length of the cached KV.
            q_len (int): Number of new-token queries.
            device (torch.device): Target device.
            dtype (torch.dtype): Mask dtype (should match hidden states).

        Returns:
            Optional[Tensor]: ``(B, 1, q_len, past_kv_len + q_len)`` additive
            float mask, or ``None`` when no padding exists.
        """
        if all(pa == 0 for pa in pad_amounts):
            return None

        B = len(pad_amounts)
        total_kv = past_kv_len + q_len
        kv_pos = torch.arange(total_kv, device=device)
        pad_t = torch.tensor(pad_amounts, device=device)
        q_idx = torch.arange(q_len, device=device)

        past_valid = (kv_pos >= pad_t.unsqueeze(1)) & (kv_pos < past_kv_len)
        past_valid = past_valid.unsqueeze(1).expand(-1, q_len, -1)

        new_valid = (kv_pos >= past_kv_len) & ((kv_pos - past_kv_len) <= q_idx.unsqueeze(1))
        new_valid = new_valid.unsqueeze(0).expand(B, -1, -1)

        valid = past_valid | new_valid
        mask = torch.zeros(B, 1, q_len, total_kv, device=device, dtype=dtype)
        mask.masked_fill_(~valid.unsqueeze(1), float('-inf'))
        return mask
