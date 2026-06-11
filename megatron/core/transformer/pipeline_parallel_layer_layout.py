# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import logging
import re
from functools import lru_cache
from typing import Optional

from megatron.core import parallel_state
from megatron.core.transformer.enums import LayerType

logger = logging.getLogger(__name__)


class PipelineParallelLayerLayout:
    """Configuration of custom pipeline parallel layer partitioning."""

    def __repr__(self) -> str:
        if isinstance(self.input_data, str):
            return self.input_data
        else:
            return str(self.input_data)

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize PipelineParallelLayerLayout from a list or a str.
        Format validation will be done here.
        """

        self.input_data = layout
        if isinstance(layout, str):
            layout = PipelineParallelLayerLayout.parse_str_to_list(layout)
        else:
            layout = copy.deepcopy(layout)
        assert all(isinstance(row, list) for row in layout), (
            f"pipeline_model_parallel_layout must be a list of lists, but got"
            f" {[type(row) for row in layout]=}"
        )

        # Check PP size and get VPP size
        assert len(layout) % pipeline_model_parallel_size == 0, (
            f"pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({len(layout)=},"
            f" {pipeline_model_parallel_size=})"
        )
        virtual_pipeline_model_parallel_size = len(layout) // pipeline_model_parallel_size

        # Convert 1D layout to 2D layout
        layout = [
            [
                layout[vpp_rank * pipeline_model_parallel_size + pp_rank]
                for vpp_rank in range(virtual_pipeline_model_parallel_size)
            ]
            for pp_rank in range(pipeline_model_parallel_size)
        ]

        # Convert all strings in pipeline_model_parallel_layout to LayerType
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(virtual_pipeline_model_parallel_size):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    assert isinstance(layer_type, LayerType) or isinstance(layer_type, str), (
                        f"elements in pipeline_model_parallel_layout must be LayerType or str,"
                        f" but got {type(layer_type)}."
                    )
                    if isinstance(layer_type, str):
                        layer_type = layer_type.strip().lower()
                        assert (
                            layer_type in LayerType.__members__
                        ), f"{layer_type} is not a valid LayerType"
                        layer_type = LayerType[layer_type]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        # Flatten the pipeline layout in layer id order.
        flatten_layout = []
        for vpp_rank in range(virtual_pipeline_model_parallel_size):
            for row in layout:
                flatten_layout.extend(row[vpp_rank])

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.layout = layout
        self.flatten_layout = flatten_layout

    def validate_layer_layout(self, num_layers: int, mtp_num_layers: int):
        """Check whether the layout is valid."""

        # Check whether the input layer id is valid
        assert all(
            isinstance(x, LayerType) for x in self.flatten_layout
        ), "All layers must be a valid LayerType."

        # Embedding layer and loss layer must be specified
        assert (
            self.flatten_layout[0] == LayerType.embedding
        ), f"The first layer must be embedding, but got {self.flatten_layout[0]}"
        assert (
            self.flatten_layout[-1] == LayerType.loss
        ), f"The last layer must be loss, but got {self.flatten_layout[-1]}"

        # Layer number verification
        assert (
            self.flatten_layout.count(LayerType.embedding) == 1
        ), "Embedding must be specified exactly once"
        # 新方案: 允许多个 loss 层 (rank6/rank7 双 loss, 按 MTP 拼接维切分各自算一半).
        # 仍要求至少 1 个 loss, 且最后一层必须是 loss (见上方 flatten_layout[-1] 检查).
        assert (
            self.flatten_layout.count(LayerType.loss) >= 1
        ), "At least one loss layer must be specified"
        # When the loss is split across multiple pipeline stages, every loss slot (`L`) maps to
        # exactly one of the N = 1 + mtp_num_layers hidden-state chunks (main + each MTP layer).
        # So the total number of loss slots across all loss stages must equal N. A single loss
        # stage (legacy) handles all chunks internally and is exempt.
        loss_stage_counts = [count for _, _, count in self.get_loss_stages()]
        if len(loss_stage_counts) > 1:
            num_chunks = 1 + (mtp_num_layers or 0)
            assert sum(loss_stage_counts) == num_chunks, (
                f"When loss is split across {len(loss_stage_counts)} pipeline stages, the total "
                f"number of loss slots ({sum(loss_stage_counts)}) must equal 1 + mtp_num_layers "
                f"({num_chunks}). Got per-stage loss counts {loss_stage_counts}."
            )
        assert self.flatten_layout.count(LayerType.decoder) == num_layers, (
            f"Number of decoder layers {self.flatten_layout.count(LayerType.decoder)}"
            f"must match num_layers {num_layers}"
        )
        # MTP layer verification
        assert self.flatten_layout.count(LayerType.mtp) == mtp_num_layers or (
            mtp_num_layers is None and self.flatten_layout.count(LayerType.mtp) == 0
        ), "Number of mtp layers in layout must match mtp_num_layers"
        for i in range(len(self.flatten_layout)):
            if self.flatten_layout[i] == LayerType.mtp:
                assert (
                    self.flatten_layout[i:].count(LayerType.decoder) == 0
                ), "decoder layers must be placed before MTP layers"
                break
        for pp_rank in range(self.pipeline_model_parallel_size):
            for vpp_rank in range(self.virtual_pipeline_model_parallel_size - 1):
                assert (
                    LayerType.mtp not in self.layout[pp_rank][vpp_rank]
                ), f"Currently we restrict that the MTP should be always in the last "
                f"virtual pipeline stage of that rank. But got {self.layout[pp_rank][vpp_rank]}"
        # Note: MTP standalone allows the MTP layers to be distributed across multiple
        # pipeline stages, so we no longer require all MTP layers to live in a single
        # virtual pipeline stage. The total MTP count is still validated above.
        for vpp_rank in range(self.virtual_pipeline_model_parallel_size - 1):
            assert LayerType.mtp not in self.layout[0][vpp_rank], (
                f"Currently we restrict that the MTP should not be in the first pp rank."
                f"But got {self.layout[0]} for the first pp rank."
            )
        ## Detect MTP standalone usage.
        mtp_standalone = False
        for pp_rank in range(self.pipeline_model_parallel_size):
            if (
                LayerType.mtp in self.layout[pp_rank][-1]
                and pp_rank != self.pipeline_model_parallel_size - 1
            ):
                mtp_standalone = True
                break

        # TODO: remove them in the future once they are supported
        if self.flatten_layout.count(LayerType.encoder) > 0:
            raise NotImplementedError("Encoder layer is not supported for flexible pipeline layout")

        return mtp_standalone

    def get_num_layers_to_build(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the number of layers to build in the pipeline stage"""
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
        else:
            vp_stage = 0

        # Count layer numbers in this stage.
        num_layers_to_build = self.layout[pp_rank][vp_stage].count(layer_type)
        return num_layers_to_build

    def get_layer_offset(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the layer offset in the pipeline stage"""
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
        else:
            vp_stage = 0

        # Calculate the offset by summing up the number of
        # layers in all the previous pipeline stages.
        offset = 0
        for _vpp_rank in range(vp_stage + 1):
            for _pp_rank in range(
                self.pipeline_model_parallel_size if _vpp_rank < vp_stage else pp_rank
            ):
                offset += self.layout[_pp_rank][_vpp_rank].count(layer_type)
        return offset

    def get_layer_id_list(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the list of layer_id for each layer in the pipeline stage."""
        offset = self.get_layer_offset(layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank)
        num_layers_to_build = self.get_num_layers_to_build(
            layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank
        )
        return list(range(offset, offset + num_layers_to_build))

    def get_loss_stages(self):
        """Return the loss stages in pipeline execution order.

        Each entry is a tuple ``(pp_rank, vp_stage, loss_count)`` for every (pp_rank, vp_stage)
        whose layout contains at least one ``LayerType.loss``. The order follows the pipeline
        execution order (vpp-major, then pp), matching ``flatten_layout``.
        """
        loss_stages = []
        for vpp_rank in range(self.virtual_pipeline_model_parallel_size):
            for pp_rank in range(self.pipeline_model_parallel_size):
                loss_count = self.layout[pp_rank][vpp_rank].count(LayerType.loss)
                if loss_count > 0:
                    loss_stages.append((pp_rank, vpp_rank, loss_count))
        return loss_stages

    def is_loss_split(self) -> bool:
        """Whether the loss computation is split across more than one pipeline stage."""
        return len(self.get_loss_stages()) > 1

    def get_loss_chunk_assignment(
        self,
        mtp_num_layers: Optional[int],
        pp_rank: Optional[int] = None,
        vp_stage: Optional[int] = None,
    ):
        """Resolve which hidden-state chunks a loss stage owns and whether it is the final stage.

        The hidden states arriving at the loss stages are the concatenation
        ``[main, mtp_0, ..., mtp_{mtp_num_layers-1}]`` along dim 0, i.e. ``N = 1 + mtp_num_layers``
        chunks with index ``j`` requiring labels rolled ``j`` times (``j=0`` is the main model).

        Assignment rule (see specs/mtp_loss_split.md §6): walk the loss stages in pipeline order
        and hand the highest-index chunks to the earliest loss stage, peeling downward, so that
        the final loss stage owns the low indices including chunk ``0`` (the main model).

        Single-loss-stage layouts (legacy) own all chunks and are always the final stage.

        Args:
            mtp_num_layers: Number of MTP layers (``None``/0 means no MTP).
            pp_rank: Pipeline rank to query. Defaults to the current rank.
            vp_stage: Virtual pipeline stage to query. Defaults to 0 / current.

        Returns:
            Tuple ``(owned_indices, is_final_stage)``. ``owned_indices`` is the sorted list of
            chunk indices this stage computes loss for; ``is_final_stage`` is True iff it owns
            chunk 0 (the main model loss).
        """
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        if vp_stage is None:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            else:
                vp_stage = 0

        num_chunks = 1 + (mtp_num_layers or 0)
        loss_stages = self.get_loss_stages()

        # Legacy single loss stage: it owns every chunk and is the final stage.
        if len(loss_stages) <= 1:
            return list(range(num_chunks)), True

        # Split mode: assign high-index chunks to earlier loss stages, peeling downward.
        hi = num_chunks - 1
        assignment = {}
        for pr, vs, loss_count in loss_stages:
            assignment[(pr, vs)] = list(range(hi - loss_count + 1, hi + 1))
            hi -= loss_count

        owned = assignment.get((pp_rank, vp_stage), [])
        is_final_stage = 0 in owned
        return owned, is_final_stage

    def pretty_repr(self):
        """Pretty representation of the custom layout, showing the layers held by each stage.
        Example:
                            VPP rank 0                 VPP rank 1
        PP rank 0           embedding,decoder*2        decoder*2
        PP rank 1-13        decoder*2                  decoder*2
        PP rank 14          decoder*2                  mtp
        PP rank 15          decoder*2                  loss
        """

        matrix = []
        if self.virtual_pipeline_model_parallel_size > 1:
            header = [""] + [
                f"VPP rank {vpp_rank}"
                for vpp_rank in range(self.virtual_pipeline_model_parallel_size)
            ]
            matrix.append(header)

        prev_row_repr, prev_row_start_pp_rank = None, None
        for pp_rank in range(self.pipeline_model_parallel_size + 1):
            row_repr = []
            if pp_rank < self.pipeline_model_parallel_size:
                for vpp_rank in range(self.virtual_pipeline_model_parallel_size):
                    stage = self.layout[pp_rank][vpp_rank]
                    stage_repr = []
                    prev_layer, prev_layer_cnt = None, 0
                    for layer_type in stage + [None]:
                        if layer_type == prev_layer:
                            prev_layer_cnt += 1
                        else:
                            if prev_layer_cnt > 1:
                                stage_repr.append(f"{prev_layer.name}*{prev_layer_cnt}")
                            elif prev_layer_cnt == 1:
                                stage_repr.append(f"{prev_layer.name}")
                            prev_layer, prev_layer_cnt = layer_type, 1
                    if len(stage_repr) == 0:
                        stage_repr.append(f"(empty stage)")
                    row_repr.append(",".join(stage_repr))

            if row_repr != prev_row_repr:
                if prev_row_start_pp_rank == pp_rank - 1:
                    matrix.append([f"PP rank {pp_rank - 1}"] + prev_row_repr)
                elif prev_row_repr is not None:
                    matrix.append(
                        [f"PP rank {prev_row_start_pp_rank}-{pp_rank - 1}"] + prev_row_repr
                    )
                prev_row_repr, prev_row_start_pp_rank = row_repr, pp_rank

        # Indent the matrix to make it more readable
        lens = [max(map(len, col)) for col in zip(*matrix)]
        indents = 8 if self.virtual_pipeline_model_parallel_size <= 4 else 4
        fmt = (" " * indents).join('{{:{}}}'.format(x) for x in lens)
        return "\n".join([fmt.format(*row) for row in matrix])

    @staticmethod
    @lru_cache()
    def from_str(layout, pipeline_model_parallel_size):
        """Parse the pipeline model parallel layout from a string."""
        parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size)
        # Pretty print the layout distribution.
        from megatron.core.utils import log_single_rank

        log_single_rank(
            logger,
            logging.INFO,
            f"Parse pipeline model parallel layout {layout} to:\n" + parsed_layout.pretty_repr(),
        )
        return parsed_layout

    @staticmethod
    def get_num_stages_from_str(layout: str):
        """Get the number of PP * VPP stages from a layout string."""
        layout_list = PipelineParallelLayerLayout.parse_str_to_list(layout)
        return len(layout_list)

    @staticmethod
    def parse_str_to_list(layout_str: str):
        """Parse a layout string to a list of lists.
        Example: "Ettt|(tt|)*29,m|L" will be parsed to
        [["E","t","t","t"]]+[["t","t"]]*29+[["m"],["L"]]"""

        layout_str = layout_str.replace(",", "")  # remove purely cosmetic commas

        # unroll multiplications in the expression
        patterns = [
            # unroll expression in parentheses ()*n. Examples:
            # xy(ab|cd|ef)*2,pq -> xyab|cd|efab|cd|efpq
            # (ab)*3 -> ababab
            # ab,(cd|)*2 -> abcd|cd|
            # (|ab)*2,cd -> |ab|abcd
            r'\(([^)]+)\)\*(\d+)',
            r'(.)\*(\d+)',  # unroll x*n to n xs
        ]
        for pattern in patterns:
            layout_str = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout_str)

        char2layer_type = {
            "E": LayerType.embedding,
            "L": LayerType.loss,
            "t": LayerType.decoder,  # t denotes "transformer"
            "m": LayerType.mtp,
        }

        # parse the layout string
        layout_list = []
        for stage in layout_str.split('|'):
            layout_list.append([])
            for layer_char in stage:
                assert layer_char in char2layer_type, (
                    f"Invalid layer character: {layer_char} ({stage=}, {layout_str=}),"
                    f" known layer characters: {list(char2layer_type.keys())}"
                )

                layout_list[-1].append(char2layer_type[layer_char])
        return layout_list
