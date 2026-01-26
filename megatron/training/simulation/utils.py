# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Utility functions for VPP training simulation."""

from megatron.core import parallel_state
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.training.global_vars import get_args


class SimulationArgsOverride:
    """
    Context manager for temporarily overriding pipeline parallelism arguments
    during simulation initialization.

    In simulation mode, we use EP GPUs to simulate a model that normally requires
    EP×PP GPUs. To make the network initialization succeed with fewer GPUs, we
    temporarily set PP=1 and clear the pipeline layout. After initialization, we
    restore the user-defined PP configuration for simulation logic to use.

    Args:
        parsed_args: The parsed arguments object from parse_args()
        enable (bool): Whether to enable parameter override (typically args.simulate_global_step)

    Usage:
        >>> with SimulationArgsOverride(parsed_args, enable=args.simulate_global_step):
        ...     initialize_megatron(parsed_args=parsed_args)
        ...     args = get_args()
        ...     timers = get_timers()
    """

    def __init__(self, parsed_args, enable=True):
        self.parsed_args = parsed_args
        self.enable = enable

        # Store original values
        self.original_pp_size = None
        self.original_pp_layout = None

    def __enter__(self):
        """Override pipeline parallelism arguments for simulation initialization."""
        print(f"[DEBUG] SimulationArgsOverride.__enter__ called, enable={self.enable}")

        if not self.enable:
            return self

        # Debug: Log original parsed args
        print(f"[DEBUG] Parsed args before override:")
        print(f"  - pipeline_model_parallel_size: {self.parsed_args.pipeline_model_parallel_size}")
        print(f"  - pipeline_model_parallel_layout type: {type(self.parsed_args.pipeline_model_parallel_layout)}")
        print(f"  - pipeline_model_parallel_layout: {self.parsed_args.pipeline_model_parallel_layout}")

        # Save original user-defined values
        self.original_pp_size = self.parsed_args.pipeline_model_parallel_size
        self.original_pp_layout = PipelineParallelLayerLayout.from_str(
            layout=self.parsed_args.pipeline_model_parallel_layout,
            pipeline_model_parallel_size=self.parsed_args.pipeline_model_parallel_size
        )

        print(f"[DEBUG] Saved original values:")
        print(f"  - original_pp_size: {self.original_pp_size}")
        print(f"  - original_pp_layout type: {type(self.original_pp_layout)}")
        print(f"  - original_pp_layout: {self.original_pp_layout}")

        # Override with simulation values to enable initialization with fewer GPUs
        # Set PP=1 to allow network initialization with only EP GPUs
        self.parsed_args.pipeline_model_parallel_size = 1
        self.parsed_args.pipeline_model_parallel_layout = None

        print(f"[DEBUG] After override for simulation:")
        print(f"  - pipeline_model_parallel_size: {self.parsed_args.pipeline_model_parallel_size}")
        print(f"  - pipeline_model_parallel_layout: {self.parsed_args.pipeline_model_parallel_layout}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original pipeline parallelism configuration."""
        print(f"[DEBUG] SimulationArgsOverride.__exit__ called, enable={self.enable}")

        if not self.enable:
            return False

        # Debug: Log saved original values
        print(f"[DEBUG] Original saved values:")
        print(f"  - original_pp_size: {self.original_pp_size}")
        print(f"  - original_pp_layout type: {type(self.original_pp_layout)}")
        print(f"  - original_pp_layout: {self.original_pp_layout}")

        # Get global args object and restore user-defined PP configuration
        args = get_args()
        print(f"[DEBUG] Before restoration:")
        print(f"  - args.pipeline_model_parallel_size: {args.pipeline_model_parallel_size}")
        print(f"  - args.pipeline_model_parallel_layout: {args.pipeline_model_parallel_layout}")
        print(f"  - args.virtual_pipeline_model_parallel_size: {getattr(args, 'virtual_pipeline_model_parallel_size', 'NOT SET')}")

        args.pipeline_model_parallel_size = self.original_pp_size
        args.pipeline_model_parallel_layout = self.original_pp_layout

        print(f"[DEBUG] After restoration:")
        print(f"  - args.pipeline_model_parallel_size: {args.pipeline_model_parallel_size}")
        print(f"  - args.pipeline_model_parallel_layout: {args.pipeline_model_parallel_layout}")

        # Recalculate virtual_pipeline_model_parallel_size from restored layout
        # This is necessary because during __enter__(), layout was set to None,
        # causing virtual_pipeline_model_parallel_size to be set to None in arguments.py
        if args.pipeline_model_parallel_layout is not None:
            print(f"[DEBUG] Calculating virtual_pipeline_model_parallel_size:")
            layout_str = str(args.pipeline_model_parallel_layout)
            print(f"  - layout_str: {layout_str}")

            num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(layout_str)
            print(f"  - num_stages: {num_stages}")
            print(f"  - pp_size: {args.pipeline_model_parallel_size}")

            assert num_stages % args.pipeline_model_parallel_size == 0, (
                f"The length of pipeline_model_parallel_layout must be divisible"
                f" by pipeline_model_parallel_size ({num_stages=},"
                f" {args.pipeline_model_parallel_size=})"
            )
            args.virtual_pipeline_model_parallel_size = num_stages // args.pipeline_model_parallel_size
            print(f"  - calculated virtual_pipeline_model_parallel_size: {args.virtual_pipeline_model_parallel_size}")

            if args.virtual_pipeline_model_parallel_size == 1:
                args.virtual_pipeline_model_parallel_size = None
                print(f"  - set to None because it's 1")
        else:
            args.virtual_pipeline_model_parallel_size = None
            print(f"[DEBUG] pipeline_model_parallel_layout is None, setting virtual_pipeline_model_parallel_size to None")

        print(f"[DEBUG] Final virtual_pipeline_model_parallel_size: {args.virtual_pipeline_model_parallel_size}")

        # IMPORTANT: Update the global variable in parallel_state module
        # This is critical because is_pipeline_last_stage() checks the global variable,
        # not args.virtual_pipeline_model_parallel_size
        print(f"[DEBUG] Before updating parallel_state global variable:")
        print(f"  - parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: {parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE}")

        parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = args.virtual_pipeline_model_parallel_size

        print(f"[DEBUG] After updating parallel_state global variable:")
        print(f"  - parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: {parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE}")
        print(f"  - get_virtual_pipeline_model_parallel_world_size(): {parallel_state.get_virtual_pipeline_model_parallel_world_size()}")

        # prepare for data build
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0
        args.eval_interval = args.train_iters
        return False
    

class MockPipelineProcessGroup:
    """Mock ProcessGroup for simulation mode.

    In simulation mode, we use fewer physical GPUs than the model requires.
    This mock allows us to simulate a larger pipeline parallel configuration
    without actual distributed process groups.

    Args:
        size: Virtual world size (e.g., 4 for PP=4)
        rank: Virtual rank in the group (0 to size-1)

    Example:
        # Simulate PP=4 with rank 2
        mock_pp = MockProcessGroup(size=4, rank=2)
        assert mock_pp.size() == 4
        assert mock_pp.rank() == 2
    """

    def __init__(self, size: int = 1, rank: int = 0):
        """Initialize MockProcessGroup with virtual size and rank."""
        self._size = size
        self._rank = rank

    def size(self) -> int:
        """Return the virtual world size of the process group."""
        return self._size

    def rank(self) -> int:
        """Return the virtual rank of the current process in the group."""
        return self._rank

    def __repr__(self):
        return f"MockProcessGroup(size={self._size}, rank={self._rank})"




__all__ = ['SimulationArgsOverride']
