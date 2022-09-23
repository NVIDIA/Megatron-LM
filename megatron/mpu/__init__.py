# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model parallel utility interface."""

from .initialize import is_unitialized
from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_embedding_group
from .initialize import get_position_embedding_group
from .initialize import get_model_parallel_group
from .initialize import get_tensor_model_parallel_group
from .initialize import get_pipeline_model_parallel_group
from .initialize import get_tensor_model_parallel_rank, set_tensor_model_parallel_rank
from .initialize import get_pipeline_model_parallel_rank, set_pipeline_model_parallel_rank
from .initialize import is_pipeline_first_stage, is_pipeline_last_stage
from .initialize import is_rank_in_embedding_group
from .initialize import is_rank_in_position_embedding_group
from .initialize import is_pipeline_stage_before_split, is_pipeline_stage_after_split
from .initialize import is_pipeline_stage_at_split
from .initialize import get_num_layers
from .initialize import get_tensor_model_parallel_src_rank
from .initialize import get_data_parallel_src_rank
from .initialize import get_pipeline_model_parallel_first_rank
from .initialize import get_pipeline_model_parallel_last_rank
from .initialize import get_pipeline_model_parallel_next_rank
from .initialize import get_pipeline_model_parallel_prev_rank
from .initialize import get_tensor_model_parallel_world_size, set_tensor_model_parallel_world_size
from .initialize import get_pipeline_model_parallel_world_size, set_pipeline_model_parallel_world_size
from .initialize import get_virtual_pipeline_model_parallel_rank, set_virtual_pipeline_model_parallel_rank
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized


from .utils import divide
from .utils import split_tensor_along_last_dim
