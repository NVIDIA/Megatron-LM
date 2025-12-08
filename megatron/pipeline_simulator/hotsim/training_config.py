from dataclasses import dataclass

from .model import Model


@dataclass
class TrainingConfig:
    """Configuration for a distributed training experiment.

    This class calculates memory requirements and efficiency metrics for training
    large language models across multiple GPUs using various parallelism techniques.
    """

    # Input configuration
    model: Model  # Model architecture details
    num_gpus: int  # Total GPUs available
    microbatch_size: int  # Batch size per microbatch

    # Parallelism dimensions
    tensor_parallel_size: int  # Number of tensor parallel groups
    context_parallel_size: int  # Number of context parallel groups
    data_parallel_size: int  # Number of data parallel groups
    pipeline_parallel_size: int  # Number of pipeline stages
    expert_parallel_size: int  # Number of expert parallel groups

    # Model execution strategy
    num_model_chunks: int | None = None  # Number of model chunks per pipeline stage
    num_layers_per_virtual_stage: int | None = (
        None  # Transformer layers per virtual pipeline stage
    )
    ckpt: str = "no"  # Whether to use activation checkpointing
    offload_ratio: float = 0.0  # Ratio of activations to offload

    def __post_init__(self):
        # Validate parallelism configuration
        if (
            self.tensor_parallel_size
            * self.context_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
            != self.num_gpus
        ):
            raise ValueError(
                "Product of parallel dimensions must equal the number of GPUs"
            )

        # Validate model chunking configuration
        if self.num_model_chunks is None and self.num_layers_per_virtual_stage is None:
            raise ValueError(
                "Either num_model_chunks or num_layers_per_virtual_stage must be specified"
            )

        # Compute derived parameters for model chunking
        if self.num_layers_per_virtual_stage:
            self.num_model_chunks = (
                self.model.num_hidden_layers
                // self.pipeline_parallel_size
                // self.num_layers_per_virtual_stage
            )
        else:
            self.num_layers_per_virtual_stage = (
                self.model.num_hidden_layers
                // self.pipeline_parallel_size
                // self.num_model_chunks
            )

        # Calculate number of transformer layers per pipeline stage
        self.num_layers_per_stage = (
            self.model.num_hidden_layers // self.pipeline_parallel_size
        )

    def __repr__(self):
        """Generate a human-readable representation of the experiment configuration."""
        return (
            f"#GPUs={self.num_gpus},\n"
            f"Model={self.model},\n"
            f"B={self.global_batch_size}, b={self.microbatch_size},\n"
            f"t={self.tensor_parallel_size}, c={self.context_parallel_size}, "
            f"d={self.data_parallel_size}, p={self.pipeline_parallel_size}\n"
            f"v={self.num_model_chunks}, l={self.num_layers_per_virtual_stage}\n"
            f"ckpt={self.ckpt}, offload={self.offload_ratio:.2%}"
        )
