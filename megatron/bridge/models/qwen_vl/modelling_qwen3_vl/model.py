# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py
# Source license: Apache-2.0.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as Qwen3VLConfigHF

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
    get_vision_model_config,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    PatchMergerSubmodules,
    collapse_thw,
    ensure_requires_grad_for_cp_collective,
    get_dist_train_vision_dp_data,
    get_vision_cp_data,
    pack_dist_train_vision_module_output,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    split_deepstack_embs,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.utils.packed_seq_utils import (
    get_packed_seq_cp_partition_indices,
    get_packed_seq_q_cu_seqlens,
    preprocess_packed_seqs,
)
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec


def _is_mrope_position_ids(position_ids: torch.Tensor | None) -> bool:
    """Return whether ``position_ids`` is explicit Qwen MRoPE metadata."""
    return (
        isinstance(position_ids, torch.Tensor)
        and position_ids.dim() == 3
        and position_ids.size(0) == 3
    )


def _select_sequence(
    val: torch.Tensor | None, index: torch.Tensor, *, seq_dim: int
) -> torch.Tensor | None:
    """Index-select one sequence dimension and keep the result contiguous."""
    if val is None:
        return None
    return val.index_select(seq_dim, index).contiguous()


def _get_packed_seq_padding_mask(
    packed_seq_params: PackedSeqParams | None, *, total_tokens: int, device: torch.device
) -> torch.Tensor | None:
    """Build the MoE padding mask from logical and physical THD boundaries."""
    if packed_seq_params is None or packed_seq_params.cu_seqlens_q_padded is None:
        return None

    cu_seqlens, physical_cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if (
        not isinstance(cu_seqlens, torch.Tensor)
        or not isinstance(physical_cu_seqlens, torch.Tensor)
        or cu_seqlens.dim() != 1
        or physical_cu_seqlens.shape != cu_seqlens.shape
        or cu_seqlens.numel() < 2
    ):
        raise ValueError("Packed MoE padding requires matching 1D query cu_seqlens metadata.")

    cu_seqlens = cu_seqlens.to(device=device)
    physical_cu_seqlens = physical_cu_seqlens.to(device=device)
    logical_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    physical_lengths = physical_cu_seqlens[1:] - physical_cu_seqlens[:-1]
    if bool(
        torch.any(logical_lengths <= 0)
        or torch.any(physical_lengths < logical_lengths)
        or cu_seqlens[0] != 0
        or physical_cu_seqlens[0] != 0
        or physical_cu_seqlens[-1] > total_tokens
    ):
        raise ValueError(
            "Packed MoE padding metadata has invalid logical or physical query boundaries."
        )
    if (
        torch.equal(logical_lengths, physical_lengths)
        and int(physical_cu_seqlens[-1].item()) == total_tokens
    ):
        return None

    positions = torch.arange(total_tokens, device=device)
    segment_indices = torch.bucketize(positions, physical_cu_seqlens[1:], right=True)
    in_physical_segment = segment_indices < logical_lengths.numel()
    safe_segment_indices = segment_indices.clamp(max=logical_lengths.numel() - 1)
    offsets = positions - physical_cu_seqlens[:-1].index_select(0, safe_segment_indices)
    valid_tokens = in_physical_segment & (
        offsets < logical_lengths.index_select(0, safe_segment_indices)
    )
    return (~valid_tokens).unsqueeze(0)


def _split_if_full_sequence(
    val: torch.Tensor | None,
    *,
    cp_size: int,
    seq_dim: int,
    cp_rank: int,
    full_sequence_length: int | None,
) -> tuple[torch.Tensor | None, bool]:
    """CP-split ``val`` only when it still carries the full sequence length."""
    if val is None or full_sequence_length is None or val.shape[seq_dim] != full_sequence_length:
        return val, False
    return split_data_cp_rank(val, cp_size, seq_dim, cp_rank), True


def _is_packed_input_pre_sharded(
    input_ids: torch.Tensor | None, packed_seq_params: PackedSeqParams | None, *, cp_size: int
) -> bool:
    """Return whether a THD input already uses MCore's zigzag CP layout."""
    if (
        input_ids is None
        or packed_seq_params is None
        or cp_size <= 1
        or input_ids.dim() != 2
        or input_ids.size(0) != 1
        or packed_seq_params.qkv_format != "thd"
    ):
        return False

    _, physical_cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if (
        not isinstance(physical_cu_seqlens, torch.Tensor)
        or physical_cu_seqlens.dim() != 1
        or physical_cu_seqlens.numel() < 2
    ):
        return False

    full_token_count = int(physical_cu_seqlens[-1].item())
    if full_token_count != cp_size * input_ids.numel():
        return False
    if int(physical_cu_seqlens[0].item()) != 0:
        raise ValueError("Pre-sharded packed CP metadata must start at token offset 0.")

    chunk_count = 2 * cp_size
    segment_lengths = physical_cu_seqlens[1:] - physical_cu_seqlens[:-1]
    if bool(torch.any(segment_lengths % chunk_count != 0).item()):
        raise ValueError(
            "Pre-sharded packed CP inputs require every physical segment length to be divisible by 2 * cp_size."
        )
    return True


def _get_cp_local_vision_embed_indices(
    vision_mask: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    *,
    vision_embed_count: int,
    cp_group: torch.distributed.ProcessGroup,
    embed_device: torch.device,
) -> torch.Tensor:
    """Map full-sequence vision embeddings to one pre-sharded CP rank.

    The local THD row contains two chunks from each packed segment: chunk
    ``cp_rank`` and chunk ``2 * cp_size - 1 - cp_rank``. Gathering only the
    per-chunk vision-token counts reconstructs offsets into the full, natural
    vision-embedding order without communicating token IDs.
    """
    _, physical_cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if not isinstance(physical_cu_seqlens, torch.Tensor):
        raise ValueError(
            "Pre-sharded packed CP vision selection requires physical cu_seqlens metadata."
        )

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    chunk_count = 2 * cp_size
    cu_seqlens = physical_cu_seqlens.tolist()
    local_vision_mask = vision_mask.reshape(-1)
    expected_local_tokens = cu_seqlens[-1] // cp_size
    if local_vision_mask.numel() != expected_local_tokens:
        raise ValueError(
            "Pre-sharded packed CP vision mask length does not match the global packed-sequence metadata."
        )

    segment_count = len(cu_seqlens) - 1
    local_counts = torch.zeros(segment_count, 2, dtype=torch.long, device=local_vision_mask.device)
    for segment_idx in range(segment_count):
        segment_length = cu_seqlens[segment_idx + 1] - cu_seqlens[segment_idx]
        chunk_length = segment_length // chunk_count
        local_start = cu_seqlens[segment_idx] // cp_size
        local_counts[segment_idx, 0] = local_vision_mask[
            local_start : local_start + chunk_length
        ].sum()
        local_counts[segment_idx, 1] = local_vision_mask[
            local_start + chunk_length : local_start + 2 * chunk_length
        ].sum()

    gathered_counts = [torch.empty_like(local_counts) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered_counts, local_counts, group=cp_group)

    full_counts = torch.zeros(
        segment_count, chunk_count, dtype=torch.long, device=local_vision_mask.device
    )
    for rank, rank_counts in enumerate(gathered_counts):
        full_counts[:, rank] = rank_counts[:, 0]
        full_counts[:, chunk_count - 1 - rank] = rank_counts[:, 1]

    gathered_vision_count = int(full_counts.sum().item())
    if gathered_vision_count != vision_embed_count:
        raise ValueError(
            f"Packed CP ranks contain {gathered_vision_count} vision tokens, but the vision encoder produced "
            f"{vision_embed_count} embeddings."
        )

    flattened_counts = full_counts.reshape(-1)
    offsets = (torch.cumsum(flattened_counts, dim=0) - flattened_counts).reshape(
        segment_count, chunk_count
    )
    index_parts = []
    for segment_idx in range(segment_count):
        for local_chunk_idx, full_chunk_idx in enumerate((cp_rank, chunk_count - 1 - cp_rank)):
            count = int(local_counts[segment_idx, local_chunk_idx].item())
            if count == 0:
                continue
            start = int(offsets[segment_idx, full_chunk_idx].item())
            index_parts.append(
                torch.arange(start, start + count, dtype=torch.long, device=embed_device)
            )

    if not index_parts:
        return torch.empty(0, dtype=torch.long, device=embed_device)
    return torch.cat(index_parts)


class Qwen3VLModel(MegatronModule):
    """Qwen3VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
        vision_transformer_config (Qwen3VLConfigHF): HF config for the vision model.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This
            is typically True for training and False for inference.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings
            in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline
            parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        model_config (ModelConfig, optional): Builder-backed model config. It owns the construction settings
            (token IDs, vocabulary, RoPE and vision-encoder options) that a provider carries on
            ``language_transformer_config``. Defaults to None, which reads them from ``language_transformer_config``.
        language_model_kwargs (dict, optional): Additional decoder constructor arguments supplied by the builder,
            such as a Hybrid layer pattern. Defaults to None.
    """

    language_model_class = Qwen3VLGPTModel

    def __init__(
        self,
        language_transformer_config: Qwen3VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: Qwen3VLConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
        model_config: Optional[ModelConfig] = None,
        language_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)
        # A builder-backed model config also exposes its nested transformer settings, so the
        # construction settings below are read from it.
        settings = language_transformer_config if model_config is None else model_config

        if hasattr(getattr(language_transformer_layer_spec, "submodules", None), "self_attention"):
            language_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

        self.vision_transformer_config = vision_transformer_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = settings.image_token_id
        self.video_token_id = settings.video_token_id
        self.vision_start_token_id = settings.vision_start_token_id

        self.square_merge_size = vision_transformer_config.spatial_merge_size**2

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        # process groups
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp
        assert hasattr(self.pg_collection, "embd"), (
            "pg_collection must have a embd. In previous version, it used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            "If you are using the default process group, please use"
            "`parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        self.embd_group = pg_collection.embd
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        if (
            hasattr(self.config, "dist_train")
            and getattr(self.config.dist_train, "use_dist_train", False) is True
        ):
            self.use_dist_train = True
            self.vision_to_llm_dp_ratio = self.config.dist_train.vision_to_llm_dp_ratio
            self.vision_embeds = None
            self.deepstack_feature_lists = None
            assert not (self.add_encoder and self.add_decoder) and (
                self.add_encoder or self.add_decoder
            ), (
                "add_encoder and add_decoder should not be both True or both False "
                f"if use_dist_train is True, got {self.add_encoder} and {self.add_decoder}"
            )
            assert (
                self.pg_collection.cp.size() == 1
            ), "currently, dist train does not support context parallelism for encoder."
        else:
            self.use_dist_train = False

        if self.pre_process and self.add_encoder:
            if settings.use_hf_vision_model:
                raise ValueError("use_hf_vision_model is not supported for Qwen3VLModel for now")
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            )

            vision_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention
            megatron_vision_transformer_config = get_vision_model_config(
                vision_transformer_config, megatron_config=settings
            )
            megatron_vision_transformer_config.pipeline_model_parallel_size = 1
            megatron_vision_transformer_config.first_pipeline_num_layers = None

            self.vision_model = Qwen3VLVisionModel(
                megatron_vision_transformer_config,
                vision_transformer_layer_spec,
                vision_patch_merger_spec,
                pre_process=True,
                post_process=True,
                pg_collection=pg_collection,
            )
        if self.add_decoder:
            assert (
                settings.vocab_size is not None
            ), "vocab_size must be configured before constructing the Qwen3-VL language model"
            if settings.should_pad_vocab:
                language_model_vocab_size = calculate_padded_vocab_size(
                    settings.vocab_size,
                    settings.make_vocab_size_divisible_by,
                    settings.tensor_model_parallel_size,
                )
            else:
                language_model_vocab_size = settings.vocab_size

            self.language_model = self.language_model_class(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_model_vocab_size,
                max_sequence_length=settings.language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type="mrope",
                rotary_percent=settings.rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=settings.rotary_base,
                fp16_lm_cross_entropy=settings.fp16_lm_cross_entropy,
                logit_dtype=settings.logit_dtype,
                share_embeddings_and_output_weights=settings.share_embeddings_and_output_weights,
                scatter_embedding_sequence_parallel=False,
                mtp_block_spec=mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=pg_collection,
                **(language_model_kwargs or {}),
            )
            if pre_process:
                deepstack_indexes = getattr(
                    vision_transformer_config, "deepstack_visual_indexes", []
                )
                assert len(deepstack_indexes) <= len(self.language_model.decoder.layers), (
                    "the deepstack_visual_embeds should on the first pp-stage of language model",
                    f"got {len(deepstack_indexes)} deepstack_visual_indexes, "
                    f" {len(self.language_model.decoder.layers)} language model layers",
                )

            self.share_embeddings_and_output_weights = (
                self.language_model.share_embeddings_and_output_weights
            )

        if self.pg_collection.cp.size() > 1:
            assert (
                self.config.calculate_per_token_loss
            ), "Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled"

        self._expose_language_model_for_cuda_graph_helper()

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        """Expose LM fields on the VLM root for the CUDA graph helper when cuda_graph_impl is enabled.

        The CUDA graph helper expects ``position_embedding_type``, ``rotary_pos_emb``, and ``decoder`` on
        the model, but in Qwen3-VL these live on ``language_model``. ``rotary_pos_emb`` and ``decoder`` are
        exposed through properties so they do not get registered as root-level module aliases, which would
        otherwise change checkpoint keys away from the ``language_model.*`` prefix.
        """
        llm_cuda_graph_enabled = (
            self.language_model is not None
            and getattr(self.language_model.config, "cuda_graph_impl", "none") != "none"
        )
        if not llm_cuda_graph_enabled:
            return
        assert not self.language_model.config.variable_seq_lengths, (
            "Qwen3-VL MoE with CUDA graph requires fixed sequence lengths (variable_seq_lengths=False). "
            "Disable variable-length / packed pipelines (e.g. in-batch packing with PP, or other modes "
            "that set variable_seq_lengths) or turn off CUDA graph."
        )
        self.position_embedding_type = self.language_model.position_embedding_type

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    @property
    def rotary_pos_emb(self):
        """Expose language model rotary embeddings without registering a root module alias."""
        return getattr(self.language_model, "rotary_pos_emb", None)

    @property
    def decoder(self):
        """Expose language model decoder for mcore inference compatibility.

        mcore's MambaInferenceStateConfig.from_model() calls get_attr_wrapped_model(model, "decoder"),
        which only traverses .module wrappers. VLM models store the decoder under language_model.decoder,
        so we expose it here to allow the Mamba check to run and correctly return None.
        """
        return getattr(self.language_model, "decoder", None)

    def set_dist_train_input_tensors(self, input_tensor) -> None:
        """Set input tensor for the model for dist train.

        Args:
            input_tensor (list): Input tensor.
        """
        assert isinstance(
            input_tensor, list
        ), f"Input tensor must be a list, but got {type(input_tensor)}"
        assert (
            len(input_tensor) == 1
        ), f"Input tensor must be a list of length 1, but got {len(input_tensor)}"
        assert isinstance(
            input_tensor[0], dict
        ), f"Input tensor[0] must be a dictionary, but got {type(input_tensor[0])}"
        input_dict = input_tensor[0]

        if "vision_module" in input_dict:
            vision_module_output_tensor = input_dict["vision_module"]
            assert (
                vision_module_output_tensor.dim() == 3
            ), f"vision_module must be 3D [b, s, h], got shape {tuple(vision_module_output_tensor.shape)}"
            # bridge communicator send and receive tensors in 3D shape, [batch, seq, hidden].
            # Qwen3VL model needs vision embeds in 2D [batch*seq, hidden].
            # So we merge leading dims, e.g. [b, s, h] -> [b*s, h].
            d0, d1, d2 = vision_module_output_tensor.shape
            vision_module_output_tensor = vision_module_output_tensor.reshape(d0 * d1, d2)
            num_chunks = len(self.vision_transformer_config.deepstack_visual_indexes) + 1
            chunks = torch.chunk(vision_module_output_tensor, chunks=num_chunks, dim=0)
            self.vision_embeds = chunks[-1]
            self.deepstack_feature_lists = chunks[:-1]
        if "language_module" in input_dict:
            self.language_model.set_input_tensor(input_dict["language_module"])

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor for the model.

        Args:
            input_tensor (list): Input tensor.
        """
        if self.use_dist_train:
            self.set_dist_train_input_tensors(input_tensor)
            return
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.decoder.deepstack_merger_list)
            modules.append(self.vision_model.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        if freeze_vision_model and not freeze_vision_projection:
            if self.vision_model is not None:
                for param in self.vision_model.decoder.deepstack_merger_list.parameters():
                    param.requires_grad = True
                for param in self.vision_model.merger.parameters():
                    param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor = None,  # can set at dataset
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        # can set at dataset
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        mm_token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the Qwen3VL model.
        # there is a workaround for supporting sequence packing with context parallelism
        # Packed batches already arrive in MCore THD layout from the collator. The
        # model only applies rank-local CP indices after merging vision embeddings.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): Optional explicit Qwen MRoPE position ids [3, batch, text_seq_len].
                Ordinary 2D text position ids are ignored and MRoPE is computed from ``input_ids`` and visual grids.
                Inputs that are already sharded across context-parallel ranks must provide rank-local MRoPE IDs.
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            mm_token_type_ids (torch.Tensor): Token type IDs from transformers >= 5.3.0 processors.
                Not used by Qwen3VL.
            padding_mask (torch.Tensor): Optional boolean MoE padding mask where
                true marks padding. For THD input, it is derived from logical and
                physical cumulative sequence boundaries when omitted.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]: Language-model loss of shape
                [b, s] when labels are provided, otherwise logits of shape [b, s, vocab_size]. CP paths that slice
                the supervision mask return ``(output, loss_mask)``. Non-last distributed-training stages return
                ``{"language_module": output}``.
        """
        del inference_context, mm_token_type_ids  # Unused, kept for API compatibility
        assert inference_params is None, "not support inference"

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        vision_embeds = None
        deepstack_feature_lists = None

        if not _is_mrope_position_ids(position_ids):
            position_ids = None

        torch.cuda.nvtx.range_push("Qwen3VLModel.forward.pre_process")

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()
        # A singleton BSHD row is shape-identical to THD. The legacy Qwen step
        # supplies a keep mask and lets this model construct MRoPE, while THD
        # producers supply explicit 3D MRoPE positions.
        legacy_packed_bshd = (
            packed_seq_params is not None
            and input_ids is not None
            and input_ids.dim() == 2
            and (input_ids.size(0) > 1 or (attention_mask is not None and position_ids is None))
        )
        packed_input_pre_sharded = _is_packed_input_pre_sharded(
            input_ids, packed_seq_params, cp_size=cp_size
        )
        if packed_input_pre_sharded and position_ids is None:
            raise ValueError(
                "Pre-sharded packed CP inputs require explicit rank-local 3D MRoPE position_ids."
            )
        packed_cp_index = (
            get_packed_seq_cp_partition_indices(
                packed_seq_params,
                total_tokens=input_ids.size(1),
                cp_size=cp_size,
                cp_rank=cp_rank,
                device=input_ids.device,
                cp_group=self.pg_collection.cp,
            )
            if packed_seq_params is not None
            and cp_size > 1
            and input_ids is not None
            and not legacy_packed_bshd
            and not packed_input_pre_sharded
            else None
        )

        # input_ids to pass to the language model for MTP (Multi-Token Prediction).
        # MTP's _get_embeddings rolls input_ids to generate future-token embeddings,
        # so it must be a real tensor. Packed input IDs already use THD layout and
        # are indexed below only when CP is enabled.
        lm_input_ids = input_ids
        full_sequence_length = input_ids.size(1) if input_ids is not None else None
        if padding_mask is None and packed_seq_params is not None and not legacy_packed_bshd:
            _, physical_cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
            if not isinstance(physical_cu_seqlens, torch.Tensor):
                raise ValueError(
                    "Packed Qwen3-VL input requires physical query cu_seqlens metadata."
                )
            packed_total_tokens = int(physical_cu_seqlens[-1].item())
            padding_mask = _get_packed_seq_padding_mask(
                packed_seq_params, total_tokens=packed_total_tokens, device=input_ids.device
            )
            if padding_mask is not None:
                if packed_cp_index is not None:
                    padding_mask = _select_sequence(padding_mask, packed_cp_index, seq_dim=1)
                elif packed_input_pre_sharded:
                    padding_mask = _select_sequence(
                        padding_mask,
                        get_packed_seq_cp_partition_indices(
                            packed_seq_params,
                            total_tokens=packed_total_tokens,
                            cp_size=cp_size,
                            cp_rank=cp_rank,
                            device=input_ids.device,
                            cp_group=self.pg_collection.cp,
                        ),
                        seq_dim=1,
                    )
        if self.language_model is not None:
            self.language_model.rotary_pos_emb.is_thd_format = False

        if self.pre_process:
            # can reorganize_inputs at dataset
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                square_merge_size=self.square_merge_size,
            )

            vision_embeds = None

            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    if cp_img_num is None:
                        assert images_padded is None
                        vision_data, vision_grid_thw, cp_img_num, images_padded = qwen3vl_cp_split(
                            cp_size, vision_data, vision_grid_thw
                        )
                    vision_data, vision_grid_thw, seqlen_on_cp_ranks = get_vision_cp_data(
                        vision_data,
                        vision_grid_thw,
                        self.square_merge_size,
                        cp_img_num,
                        images_padded,
                        cp_rank,
                        cp_size,
                    )
                    vision_grid_thw = collapse_thw(vision_grid_thw)
                if vision_data.shape[0] > 0:
                    if self.use_dist_train:
                        if self.vision_model is not None:
                            vision_data, vision_grid_thw = get_dist_train_vision_dp_data(
                                vision_data,
                                vision_grid_thw,
                                num_chunks=self.vision_to_llm_dp_ratio,
                                dp_rank=self.pg_collection.dp.rank(),
                            )
                            vision_embeds, deepstack_feature_lists = self.vision_model(
                                hidden_states=vision_data, grid_thw=vision_grid_thw
                            )
                            output_vision_module = pack_dist_train_vision_module_output(
                                vision_embeds, deepstack_feature_lists
                            )
                            torch.cuda.nvtx.range_pop()
                            return output_vision_module
                        else:
                            vision_embeds = self.vision_embeds
                            deepstack_feature_lists = self.deepstack_feature_lists
                    else:
                        vision_embeds, deepstack_feature_lists = self.vision_model(
                            hidden_states=vision_data, grid_thw=vision_grid_thw
                        )
                else:
                    vision_embeds = torch.zeros(
                        (0, self.config.hidden_size),
                        device=vision_data.device,
                        dtype=self.config.params_dtype,
                    )
                    deepstack_feature_lists = []
                    for _ in self.vision_transformer_config.deepstack_visual_indexes:
                        deepstack_feature_lists.append(
                            torch.zeros(
                                (0, self.language_model.config.hidden_size),
                                device=vision_data.device,
                                dtype=self.config.params_dtype,
                            )
                        )
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    ensure_requires_grad_for_cp_collective(
                        (vision_embeds, *deepstack_feature_lists)
                    )
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds, seqlen_on_cp_ranks, self.pg_collection.cp
                    )
                    for i in range(len(deepstack_feature_lists)):
                        deepstack_feature_lists[i] = AllGatherVisionEmbeddings.apply(
                            deepstack_feature_lists[i], seqlen_on_cp_ranks, self.pg_collection.cp
                        )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if packed_input_pre_sharded:
                    if vision_mask is None:
                        raise ValueError(
                            "Pre-sharded packed CP vision inputs require a rank-local vision mask."
                        )
                    vision_embed_indices = _get_cp_local_vision_embed_indices(
                        vision_mask,
                        packed_seq_params,
                        vision_embed_count=vision_embeds.size(0),
                        cp_group=self.pg_collection.cp,
                        embed_device=vision_embeds.device,
                    )
                    vision_embeds = vision_embeds.index_select(0, vision_embed_indices)
                    if deepstack_feature_lists is not None:
                        deepstack_feature_lists = [
                            deepstack_embeds.index_select(0, vision_embed_indices)
                            for deepstack_embeds in deepstack_feature_lists
                        ]
                combined_embeddings[vision_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if packed_seq_params is not None:
                if legacy_packed_bshd:
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    lm_input_ids = preprocess_packed_seqs(
                        input_ids,
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    _, _, vision_mask_thd = reorganize_inputs(
                        input_ids=lm_input_ids,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_input_mask=image_input_mask,
                        video_input_mask=video_input_mask,
                        image_token_id=self.image_token_id,
                        video_token_id=self.video_token_id,
                        square_merge_size=self.square_merge_size,
                    )

                    if deepstack_feature_lists is not None:
                        tmp_embeddings = torch.zeros_like(combined_embeddings.transpose(0, 1))
                        new_deepstack_feature_lists = []
                        for deepstack_visual_embed in deepstack_feature_lists:
                            tmp_embeddings[vision_mask] = deepstack_visual_embed
                            tmp_embeddings_thd = preprocess_packed_seqs(
                                tmp_embeddings.contiguous(),
                                attention_mask,
                                pre_process=True,
                                pg_collection=self.pg_collection,
                            )[0]
                            new_deepstack_feature_lists.append(
                                tmp_embeddings_thd[vision_mask_thd].contiguous()
                            )
                        deepstack_feature_lists = new_deepstack_feature_lists

                    vision_mask = vision_mask_thd
                    combined_embeddings = (
                        preprocess_packed_seqs(
                            combined_embeddings.transpose(0, 1).contiguous(),
                            attention_mask,
                            pre_process=True,
                            pg_collection=self.pg_collection,
                        )[0]
                        .transpose(0, 1)
                        .contiguous()
                    )
                elif cp_size > 1 and packed_cp_index is None and not packed_input_pre_sharded:
                    raise ValueError("Qwen3VLModel requires input_ids for packed CP slicing")
                elif packed_cp_index is not None:
                    lm_input_ids = _select_sequence(input_ids, packed_cp_index, seq_dim=1)
                    vision_mask_local = _select_sequence(vision_mask, packed_cp_index, seq_dim=1)
                    if deepstack_feature_lists is not None:
                        tmp_embeddings = torch.zeros_like(combined_embeddings.transpose(0, 1))
                        new_deepstack_feature_lists = []
                        for deepstack_visual_embed in deepstack_feature_lists:
                            tmp_embeddings[vision_mask] = deepstack_visual_embed
                            tmp_embeddings_local = _select_sequence(
                                tmp_embeddings, packed_cp_index, seq_dim=1
                            )
                            new_deepstack_feature_lists.append(
                                tmp_embeddings_local[vision_mask_local].contiguous()
                            )
                        deepstack_feature_lists = new_deepstack_feature_lists

                    vision_mask = vision_mask_local
                    combined_embeddings = _select_sequence(
                        combined_embeddings, packed_cp_index, seq_dim=0
                    )
            elif combined_embeddings is not None and cp_size > 1:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings, group=self.pg_collection.tp
                )
                combined_embeddings = combined_embeddings.contiguous()
                if padding_mask is not None:
                    padding_mask = (
                        tensor_parallel.scatter_to_sequence_parallel_region(
                            padding_mask.transpose(0, 1).contiguous(), group=self.pg_collection.tp
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )

        else:
            combined_embeddings = None
            if legacy_packed_bshd:
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                lm_input_ids = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )[0]
            elif packed_cp_index is not None:
                lm_input_ids = _select_sequence(input_ids, packed_cp_index, seq_dim=1)

        visual_pos_masks = vision_mask
        deepstack_visual_embeds = deepstack_feature_lists
        if self.config.sequence_parallel or cp_size > 1:
            if packed_seq_params is None:  # BSHD
                visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    tp_size=self.pg_collection.tp.size(),
                    tp_rank=self.pg_collection.tp.rank(),
                    cp_size=cp_size,
                    cp_rank=self.pg_collection.cp.rank(),
                    sequence_parallel=self.config.sequence_parallel,
                )
            elif self.config.sequence_parallel:  # THD and SP
                visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    tp_size=self.pg_collection.tp.size(),
                    tp_rank=self.pg_collection.tp.rank(),
                    cp_size=1,
                    cp_rank=0,
                    sequence_parallel=self.config.sequence_parallel,
                )

        if position_ids is None:
            if input_ids is None:
                raise ValueError(
                    "Qwen3VLModel requires input_ids when explicit 3D MRoPE position_ids are not provided"
                )
            # BSHD
            # Megatron uses 4D bool masks ([B|1,1,S,S], True=masked); HF uses 2D keep masks ([B,S], 1=keep)
            # For simplicity, we set hf_attention_mask to None.
            hf_attention_mask = None
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=hf_attention_mask,
                packed_seq_params=None if legacy_packed_bshd else packed_seq_params,
            )  #  [3*b*s]
        if packed_seq_params is not None:
            if legacy_packed_bshd:
                position_ids = (
                    preprocess_packed_seqs(
                        position_ids.permute(1, 2, 0),
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    .permute(2, 0, 1)
                    .contiguous()
                )
            elif packed_cp_index is not None:
                position_ids = _select_sequence(position_ids, packed_cp_index, seq_dim=2)
            attention_mask = None
            if self.language_model is not None:
                self.language_model.rotary_pos_emb.is_thd_format = True
        elif cp_size > 1:
            lm_input_ids, _ = _split_if_full_sequence(
                lm_input_ids,
                cp_size=cp_size,
                seq_dim=1,
                cp_rank=cp_rank,
                full_sequence_length=full_sequence_length,
            )
            position_ids, position_ids_were_split = _split_if_full_sequence(
                position_ids,
                cp_size=cp_size,
                seq_dim=2,
                cp_rank=cp_rank,
                full_sequence_length=full_sequence_length,
            )
            if position_ids_were_split and self.language_model is not None:
                self.language_model.rotary_pos_emb.is_thd_format = True

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Qwen3VLModel.forward.language_model")

        return_sliced_loss_mask = False
        if packed_seq_params is not None:
            if packed_cp_index is not None:
                labels = _select_sequence(labels, packed_cp_index, seq_dim=1)
                if loss_mask is not None:
                    loss_mask = _select_sequence(loss_mask, packed_cp_index, seq_dim=1)
                    return_sliced_loss_mask = True
        elif cp_size > 1:
            labels, _ = _split_if_full_sequence(
                labels,
                cp_size=cp_size,
                seq_dim=1,
                cp_rank=cp_rank,
                full_sequence_length=full_sequence_length,
            )
            if loss_mask is not None:
                loss_mask, return_sliced_loss_mask = _split_if_full_sequence(
                    loss_mask,
                    cp_size=cp_size,
                    seq_dim=1,
                    cp_rank=cp_rank,
                    full_sequence_length=full_sequence_length,
                )

        output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,  # Added for THD training compatibility
            padding_mask=padding_mask,
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
        torch.cuda.nvtx.range_pop()
        if self.use_dist_train:
            if not is_pp_last_stage(self.pg_collection.pp):
                return {"language_module": output}
        if return_sliced_loss_mask:
            return output, loss_mask
        return output
