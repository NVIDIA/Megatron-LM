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

"""
VisionTECudaGraphHelper: CUDA Graph helper for vision encoder using TransformerEngine.

This module provides CUDA graph capture and replay functionality specifically for
vision encoders (like ViT) using TransformerEngine's make_graphed_callables().
"""

import logging
import os
import time
from contextlib import nullcontext
import torch

logger = logging.getLogger(__name__)

try:
    from transformer_engine.pytorch import make_graphed_callables

    HAVE_TE_GRAPHS = True
except ImportError:
    HAVE_TE_GRAPHS = False

try:
    from megatron.core.transformer.cuda_graphs import CudaGraphScope
    from megatron.core.utils import get_attr_wrapped_model
    from megatron.core.parallel_state import get_cuda_rng_tracker
except ImportError:
    CudaGraphScope = None


def _vision_layer_is_graphable(layer, config):
    """
    Check if a vision encoder layer is graphable.
    Similar to _layer_is_graphable but simplified for vision encoder.
    """
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if not isinstance(layer, TransformerLayer):
        return False

    # Check if CUDA graph is enabled
    if config.cuda_graph_impl != "transformer_engine":
        return False

    return True


def _wrap_graph_for_vision(graph_fn):
    """Wrap a graphed callable to filter out None outputs.

    During make_graphed_callables warmup, vision encoder layers go through their
    normal forward() path which returns (output, context=None). _te_cuda_graph_replay
    asserts len(output) == 1 but gets 2 elements. This wrapper filters out None
    values so replay sees (output,) instead of (output, None).
    """

    def wrapped(*args, **kwargs):
        result = graph_fn(*args, **kwargs)
        if isinstance(result, tuple):
            filtered = tuple(r for r in result if r is not None)
            return filtered if filtered else result
        return result

    # Preserve TE-specific attributes needed for CUDA graph management
    for attr in ('backward_dw', 'reset'):
        if hasattr(graph_fn, attr):
            setattr(wrapped, attr, getattr(graph_fn, attr))
    return wrapped


class VisionTECudaGraphHelper:
    """
    Helper class to capture CUDA Graphs for vision encoder using TE make_graphed_callables().

    This is designed specifically for vision encoders (ViT) which have:
    - Fixed sequence length (based on max image/video tokens)
    - Simpler pipeline structure (no pipeline parallelism for vision)

    Usage:
        1. Create the helper: `helper = VisionTECudaGraphHelper(model, vision_config, ...)`
        2. Create CUDA graphs: `helper.create_cudagraphs()`
        3. Set manual hooks: `helper.cuda_graph_set_manual_hooks(model_make_forward_pre_hook)`

    Args:
        model: The full model containing vision_model
        vision_config: The TransformerConfig for the vision encoder
        vision_seq_length: The sequence length for vision encoder (max vision tokens)
        micro_batch_size: Micro batch size for training
        num_microbatches: Number of microbatches per step (default 1 for vision)
    """

    def __init__(
        self,
        model,
        vision_config,
        vision_seq_length: int,
        micro_batch_size: int,
        num_microbatches: int = 1,
    ):
        assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported without TransformerEngine."
        assert (
            vision_config.cuda_graph_impl == "transformer_engine"
        ), "vision_config.cuda_graph_impl must be 'transformer_engine' to use VisionTECudaGraphHelper."
        assert (
            "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
            or os.getenv("NCCL_GRAPH_REGISTER", "") == "0"
        ), (
            "Setting NCCL_GRAPH_REGISTER=0 to avoid illegal memory access when using "
            "CUDA Graph with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
        )

        self.model = model
        self.vision_config = vision_config
        self.vision_seq_length = vision_seq_length
        self.micro_batch_size = micro_batch_size
        self.num_microbatches = num_microbatches

        # Get vision encoder layers
        self.vision_layers = []
        self.vision_model = None

        for model_chunk in model:
            # Try to get vision_model from the model chunk
            try:
                unwrapped = get_attr_wrapped_model(
                    model_chunk, 'vision_model', allow_none=True, return_model_obj=True
                )
                if unwrapped is not None and hasattr(unwrapped, 'vision_model'):
                    self.vision_model = unwrapped.vision_model
                    break
            except (RuntimeError, AttributeError):
                continue

        if self.vision_model is None:
            logger.warning(
                "VisionTECudaGraphHelper: No vision_model found in model. "
                "CUDA graphs will not be captured for vision encoder."
            )
            self.callables = []
            self._graphs_created = False
            return

        # Get the vision encoder layers
        if hasattr(self.vision_model, 'decoder') and hasattr(self.vision_model.decoder, 'layers'):
            for layer in self.vision_model.decoder.layers:
                if _vision_layer_is_graphable(layer, vision_config):
                    self.vision_layers.append(layer)

        self.callables = self.vision_layers
        self.num_layers = len(self.callables)

        logger.info(
            f"VisionTECudaGraphHelper: Found {self.num_layers} graphable vision encoder layers. "
            f"seq_length={vision_seq_length} (all images concatenated, batch_dim=1)"
        )

        self._graphs_created = False

    def graphs_created(self):
        """Returns whether the CUDA Graphs have been created."""
        return self._graphs_created

    def _get_sample_args(self):
        """
        Generate sample arguments for CUDA Graph capturing.

        Returns:
            Tuple of (sample_args, sample_kwargs) lists for each layer and microbatch.
        """
        if not self.callables:
            return [], {}

        sample_args = []
        sample_kwargs_list = []

        # Vision encoder hidden size
        hidden_size = self.vision_config.hidden_size

        for microbatch_idx in range(self.num_microbatches):
            for layer in self.callables:
                # Create static input tensors for the layer
                # Vision encoder concatenates all images along sequence dimension with batch=1
                # So shape is [total_patches, 1, hidden_size], not [seq, mbs, hidden]
                hidden_states = torch.zeros(
                    self.vision_seq_length,
                    1,  # Batch dim is always 1 for vision encoder (images concatenated in seq dim)
                    hidden_size,
                    dtype=torch.bfloat16,
                    device='cuda',
                    requires_grad=True,
                )

                # Get layer-specific static inputs if available
                if hasattr(layer, 'get_layer_static_inputs'):
                    static_inputs = layer.get_layer_static_inputs(
                        self.vision_seq_length, 1  # Batch dim is always 1 for vision encoder
                    )
                    hidden_states = static_inputs.pop("hidden_states", hidden_states)
                    sample_args.append((hidden_states,))
                    sample_kwargs_list.append(static_inputs)
                else:
                    sample_args.append((hidden_states,))
                    sample_kwargs_list.append({})

        return sample_args, sample_kwargs_list

    def _start_capturing(self):
        """Prepare for CUDA graph capturing."""
        torch.cuda.synchronize()
        start_time = time.time()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"Rank {rank}: Starting vision encoder CUDA graph capture...")

        return start_time

    def _finish_capturing(self, start_time):
        """Finalize CUDA graph capturing."""
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(
            f"Rank {rank}: Vision encoder CUDA graph capture completed in {elapsed:.2f}s. "
            f"Captured {len(self.callables)} layers."
        )

        self._graphs_created = True

    def create_cudagraphs(self):
        """
        Capture CUDA Graphs for vision encoder layers per microbatch.

        This method uses TE's make_graphed_callables to capture the forward pass
        of each vision encoder layer.

        IMPORTANT: Before capturing, we must remove any existing `cudagraph_manager`
        from the vision layers. If present, the local CUDA graph path
        (_should_call_local_cudagraph) would take priority over the TE path during
        both the make_graphed_callables warmup AND normal training replay. This causes:
        1. Warmup calls go through the local graph manager with incomplete inputs,
           corrupting the local graph state.
        2. The TE graphs (self.cuda_graphs) are never replayed because the local
           path takes priority in __call__.
        Removing cudagraph_manager ensures the TE path (_should_call_te_cudagraph)
        is used consistently.
        """
        if not self.callables:
            logger.warning(
                "VisionTECudaGraphHelper: No graphable layers found. Skipping CUDA graph capture."
            )
            return

        # Remove any existing cudagraph_manager to avoid conflict with TE CUDA graph path.
        for layer in self.callables:
            if hasattr(layer, 'cudagraph_manager'):
                delattr(layer, 'cudagraph_manager')

        # Build _order for make_graphed_callables using the actual pipeline schedule.
        #
        # With _order, make_graphed_callables returns forward FUNCTIONS instead of
        # modules with replaced forward methods. Without _order, the returned modules
        # would trigger recursive __call__ -> _te_cuda_graph_replay calls.
        #
        # CRITICAL: Vision encoder layers live on the first pipeline stage alongside
        # the first LM decoder layers. They follow the exact same pipeline schedule,
        # so we compute _order identically to the LM helper (TECudaGraphHelper).
        # This ensures make_graphed_callables can reuse static buffers across
        # microbatches whose lifetimes don't overlap in the actual schedule.
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import (
            get_pp_rank_microbatches,
            get_schedule_table,
        )
        from megatron.core.transformer.cuda_graphs import convert_schedule_table_to_order

        num_model_chunks = 1

        # If PP is not enabled, we only need to capture one microbatch.
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            self.num_microbatches = 1
        # else: keep self.num_microbatches as passed from train.py

        _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
            self.num_microbatches,
            num_model_chunks,
            getattr(self.vision_config, 'microbatch_group_size_per_vp_stage', None),
            False,
        )
        schedule_table = get_schedule_table(
            self.num_microbatches,
            num_model_chunks,
            getattr(self.vision_config, 'microbatch_group_size_per_vp_stage', None),
        )
        order = convert_schedule_table_to_order(
            num_warmup_microbatches, num_model_chunks, schedule_table
        )

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(
            f"Rank {rank}: Vision CUDA graph order: "
            f"num_microbatches={self.num_microbatches}, "
            f"num_warmup={num_warmup_microbatches}, "
            f"order_len={len(order)}"
        )

        start_time = self._start_capturing()

        # Prepare sample arguments (includes attention_mask and rotary_pos_emb)
        # Must be called AFTER num_microbatches is finalized above.
        sample_args, sample_kwargs_list = self._get_sample_args()

        # Capture CUDA graphs using TE
        try:
            from transformer_engine.pytorch.utils import is_te_min_version
        except ImportError:
            is_te_min_version = lambda v: False

        kwargs = {
            "num_warmup_iters": self.vision_config.cuda_graph_warmup_steps
            if hasattr(self.vision_config, 'cuda_graph_warmup_steps')
            else 3,
            "allow_unused_input": True,
            "_order": order,
            "_num_layers_per_chunk": [len(self.callables)],
        }

        # Reuse input/output buffers between layers to reduce peak memory
        # during capture. Critical to avoid OOM on memory-constrained GPUs.
        if is_te_min_version("2.7.0"):
            kwargs['_reuse_graph_input_output_buffers'] = True

        # FP8 is not used for vision encoder — explicitly disable to avoid
        # any TE default behavior
        kwargs['fp8_enabled'] = False

        # Add sample_kwargs (TE >= 1.10.0 required for kwarg support)
        if is_te_min_version("1.10.0") and sample_kwargs_list:
            kwargs["sample_kwargs"] = tuple(sample_kwargs_list)

        # Use RNG context for sequence parallel (matches LM helper behavior)
        if hasattr(self.vision_config, 'sequence_parallel') and self.vision_config.sequence_parallel:
            rng_context = get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            graphs = make_graphed_callables(
                tuple(self.callables),
                tuple(sample_args),
                **kwargs,
            )

        # Assign captured graphs to layers.
        # Wrap each graph to filter out None from (output, None) returned by forward()
        # so that _te_cuda_graph_replay sees (output,) and its len==1 assertion passes.
        for layer_idx, layer in enumerate(self.callables):
            layer.cuda_graphs = []
            for microbatch_idx in range(self.num_microbatches):
                graph_idx = microbatch_idx * len(self.callables) + layer_idx
                layer.cuda_graphs.append(_wrap_graph_for_vision(graphs[graph_idx]))

        self._finish_capturing(start_time)

    def cuda_graph_set_manual_hooks(self, make_forward_pre_hook_fn=None):
        """
        Set CUDA Graph manual hooks for vision encoder layers.

        Args:
            make_forward_pre_hook_fn: Optional function to create forward pre hooks.
                If None, will try to use the model's _make_forward_pre_hook.
        """
        if not self.callables or not self._graphs_created:
            return

        for layer in self.callables:
            if hasattr(layer, 'setup_manual_hooks') and make_forward_pre_hook_fn is not None:
                layer.setup_manual_hooks(make_forward_pre_hook_fn)

    def delete_cuda_graphs(self):
        """Delete CUDA graphs to free resources."""
        if not self._graphs_created:
            return

        for layer in self.callables:
            if hasattr(layer, 'cuda_graphs'):
                for cuda_graph in layer.cuda_graphs:
                    del cuda_graph
                del layer.cuda_graphs

        self._graphs_created = False
        logger.info("VisionTECudaGraphHelper: CUDA graphs deleted.")


def get_vision_cuda_graph_seq_length(vision_config, default_seq_length: int = 4096) -> int:
    """
    Calculate the sequence length for vision encoder CUDA graphs.

    For vision encoders, the sequence length depends on:
    - max_vision_cuda_graph_seq_length: explicit maximum (if set)
    - num_position_embeddings: maximum number of patches
    - spatial_merge_size: pooling factor that reduces sequence length

    Args:
        vision_config: The TransformerConfig for vision encoder
        default_seq_length: Default sequence length if cannot be calculated

    Returns:
        The sequence length to use for CUDA graph capture
    """
    # Check for explicit max sequence length setting
    if hasattr(vision_config, 'max_vision_cuda_graph_seq_length') and vision_config.max_vision_cuda_graph_seq_length:
        return vision_config.max_vision_cuda_graph_seq_length

    if hasattr(vision_config, 'num_position_embeddings'):
        # Vision encoder sequence length based on patch positions
        seq_length = vision_config.num_position_embeddings
        if hasattr(vision_config, 'spatial_merge_size'):
            # Account for spatial merging
            merge_factor = vision_config.spatial_merge_size ** 2
            seq_length = seq_length // merge_factor
        return seq_length

    return default_seq_length
