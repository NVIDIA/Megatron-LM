from .model import Model
from .schedule import (
    Action,
    Chunk,
    Op,
    build_splitfuse_schedule,
    build_hybrid_schedule,
    build_1f1b_schedule,
    plot_combined_visualization,
)
from .training_config import TrainingConfig


class MemoryModel:
    """
    MemoryModel simulates memory usage during pipeline parallel training of large language models.

    This class tracks memory allocation and deallocation for model parameters, optimizer states,
    activations, KV caches, and gradients during the forward and backward passes.

    The model supports various parallelism strategies (tensor, pipeline, expert, context, data)
    and checkpointing configurations to optimize memory usage.

    Methods:
        setup(chunks_list): Prepares the memory model with chunk information and builds the execution schedule
        run(rank): Simulates execution on the specified rank and calculates memory usage
    """

    def __init__(self, config: TrainingConfig):
        self.model = config.model
        self.config = config

        if self.config.microbatch_size != 1:
            raise ValueError("This memory model only supports a microbatch size of 1.")

        self._init_memory()

    def _init_memory(self) -> None:
        # Model parameters (FP16) and gradients (FP32): 2 + 4 = 6 bytes per parameter
        self.parameter_size = (
            6
            * self.model.params_per_layer
            * self.config.num_layers_per_stage
            / self.config.tensor_parallel_size
        )

        # Optimizer states: weights (4) + momentum (4) + variance (4) = 12 bytes per parameter
        self.optimizer_size = (
            12
            * self.model.params_per_layer
            * self.config.num_layers_per_stage
            / self.config.tensor_parallel_size
            / self.config.context_parallel_size
            / self.config.data_parallel_size
        )

        # Model states and optimizer states for the embedding layer
        self.embedding_size = (
            6
            * self.model.vocab_size
            * self.model.hidden_size
            / self.config.tensor_parallel_size
        ) + (
            12
            * self.model.vocab_size
            * self.model.hidden_size
            / self.config.tensor_parallel_size
            / self.config.context_parallel_size
            / self.config.data_parallel_size
        )

        # TODO(Wei Zhang): Implement output size calculation
        # Output logits memory (FP32): 4 bytes per element
        # self.output_size = (
        #     4
        #     * self.config.microbatch_size
        #     * self.config.seq_length
        #     * self.model.vocab_size
        #     / self.config.tensor_parallel_size
        #     / self.config.context_parallel_size
        # )
        self.output_size = 0

    def _init_cache(self) -> None:
        num_layers = self.model.num_hidden_layers
        # key: stage_id, value: Dict[(batch_id, chunk_id), size]
        self.activations: list[dict[tuple[int, int], float]] = [
            dict() for _ in range(num_layers)
        ]
        self.kv_caches: list[dict[tuple[int, int], float]] = [
            dict() for _ in range(num_layers)
        ]
        self.kv_gradients: list[dict[tuple[int, int], float]] = [
            dict() for _ in range(num_layers)
        ]
        self.offload_caches: list[dict[tuple[int, int], float]] = [
            dict() for _ in range(num_layers)
        ]
        # There will be two buffers for offload and reload
        self.offload_buffer_size = 0.0
        # Memory changes for every forward/backward pass
        self.memory_histogram: list[float] = []
        self.peak_memory_histogram: list[float] = []
        CC = 90
        self.matmul_buffer = 2 * {80: 8320 * 1024, 90: 32 * 1024 * 1024}[CC]

    def _get_activation_size(self, chunk: Chunk, recompute=False) -> float:
        # Activation memory (FP16): 2 bytes per activation
        activation_size = (
            2
            * self.model.acts_per_layer(
                chunk.batch_size, chunk.length, self.config.ckpt, recompute
            )
            / self.config.tensor_parallel_size
            / self.config.context_parallel_size
        )
        return activation_size

    def _get_kv_cache_size(self, chunk: Chunk) -> float:
        num_chunks = self.num_chunks[chunk.batch_id]
        if num_chunks == 1:
            # No kv cache if not sliced
            return 0
        kv_cache_size = (
            2
            * self.model.kv_acts_per_layer(chunk.batch_size, chunk.length)
            / self.config.tensor_parallel_size
            / self.config.context_parallel_size
        )
        return kv_cache_size

    def _get_kv_gradient_size(self, chunk: Chunk) -> float:
        batch_id = chunk.batch_id
        chunk_id = chunk.chunk_id
        num_chunks = self.num_chunks[batch_id]
        if num_chunks == 1 or chunk_id == 0:
            # No kv gradient if not sliced or first chunk
            return 0

        # dKdV will be used for all previous chunks
        length = sum(self.chunks_list[batch_id][:chunk_id])

        kv_gradient_size = (
            2
            * self.model.kv_acts_per_layer(chunk.batch_size, length)
            / self.config.tensor_parallel_size
            / self.config.context_parallel_size
        )
        return kv_gradient_size

    def _get_offload_cache_size(self, activation_size: float) -> float:
        offload_cache_size = activation_size * self.config.offload_ratio
        return offload_cache_size

    def _forward_layer(self, action: Action, layer_id: int) -> None:
        chunk = action.chunk
        stage_id, batch_id, chunk_id = action.stage_id, chunk.batch_id, chunk.chunk_id

        key = (batch_id, chunk_id)

        # Forward generates activations and kv caches
        kv_cache_size = self._get_kv_cache_size(action.chunk)
        activation_size = self._get_activation_size(action.chunk)
        # We have duplicate KV in activations unless full checkpointing is used
        if self.config.ckpt != "full":
            activation_size -= kv_cache_size

        # If the new activation size is larger than the offload buffer size,
        # we need to expand the offload buffer
        offload_cache_size = self._get_offload_cache_size(activation_size)
        offload_buffer_expansion = 0.0
        expanded_buffer_size = 2 * 3 * offload_cache_size * self.config.offload_ratio
        if self.offload_buffer_size < expanded_buffer_size:
            offload_buffer_expansion = expanded_buffer_size - self.offload_buffer_size
            self.offload_buffer_size = expanded_buffer_size

        tp_all_gather_buffer = (
            self.config.microbatch_size
            * action.chunk.length
            // self.config.context_parallel_size
            * self.model.hidden_size
            * 2
            if self.config.tensor_parallel_size >= 2
            else 0
        )
        # The peak memory occurs just before we start offloading
        peak_memory = (
            self.memory_histogram[-1]
            + activation_size
            + kv_cache_size
            + offload_buffer_expansion
            + tp_all_gather_buffer
        )
        self.peak_memory_histogram.append(peak_memory)

        # Some activations are offloaded
        remaining_activation_size = activation_size - offload_cache_size

        self.activations[layer_id][key] = remaining_activation_size
        self.kv_caches[layer_id][key] = kv_cache_size
        self.offload_caches[layer_id][key] = offload_cache_size

        # Add up what's left in memory
        memory = (
            self.memory_histogram[-1]
            + remaining_activation_size
            + kv_cache_size
            + offload_buffer_expansion
        )
        self.memory_histogram.append(memory)

    def _forward(self, action: Action) -> None:
        for i in range(self.config.num_layers_per_stage):
            before = self.memory_histogram[-1]
            self._forward_layer(
                action, action.stage_id * self.config.num_layers_per_stage + i
            )
            after = self.memory_histogram[-1]

    def _backward_layer(self, action: Action, layer_id: int) -> None:
        chunk = action.chunk
        stage_id, batch_id, chunk_id = action.stage_id, chunk.batch_id, chunk.chunk_id

        key = (batch_id, chunk_id)

        # Reload activations from offload cache
        offload_cache_size = self.offload_caches[layer_id].pop(key)
        activation_size = self.activations[layer_id].pop(key)
        kv_cache_size = self.kv_caches[layer_id].pop(key)
        kv_gradient_size = self._get_kv_gradient_size(chunk)
        if chunk_id > 0:
            self.kv_gradients[layer_id][key] = kv_gradient_size

        recomputed_activation_size = self._get_activation_size(chunk, recompute=True)
        # With full checkpointing, KV are regenerated during recompute
        if self.config.ckpt == "full":
            recomputed_activation_size -= kv_cache_size

        tp_all_gather_buffer = (
            self.config.microbatch_size
            * action.chunk.length
            // self.config.context_parallel_size
            * self.model.hidden_size
            * 2
            if self.config.tensor_parallel_size >= 2
            else 0
        )
        # The peak memory occurs just before we discard activations, kv cache and kv gradients
        peak_memory = (
            self.memory_histogram[-1]
            + recomputed_activation_size
            + offload_cache_size
            + kv_gradient_size
            + tp_all_gather_buffer
            + (1 + 2) * 2 * chunk.batch_size * chunk.length * self.model.intermediate_size  # fc2 bwd & swiglu_back
        )
        self.peak_memory_histogram.append(peak_memory)

        prev_kv_gradient_size = 0.0
        if chunk_id < self.num_chunks[batch_id] - 1:
            prev_key = (batch_id, chunk_id + 1)
            prev_kv_gradient_size = self.kv_gradients[layer_id].pop(prev_key)

        # Release activations, kv cache and previous kv gradients
        memory = (
            self.memory_histogram[-1]
            - activation_size
            - kv_cache_size
            + kv_gradient_size
            - prev_kv_gradient_size
        )
        self.memory_histogram.append(memory)

    def _backward(self, action: Action) -> None:
        for i in reversed(range(self.config.num_layers_per_stage)):
            self._backward_layer(
                action, action.stage_id * self.config.num_layers_per_stage + i
            )

    def _validate_cache(self) -> None:
        storage_list = [
            self.activations,
            self.kv_caches,
            self.kv_gradients,
            self.offload_caches,
        ]
        for storage in storage_list:
            for cache in storage:
                if len(cache) != 0:
                    raise ValueError("Memory leakage detected.")

    def _simulate_execution(self, rank: int = 0) -> None:
        self._init_cache()
        base_memory = self.parameter_size + self.optimizer_size + self.matmul_buffer
        if rank == 0:
            base_memory += self.embedding_size
        if rank == self.config.pipeline_parallel_size - 1:
            base_memory += self.output_size + self.embedding_size
        self.memory_histogram = [base_memory]
        self.peak_memory_histogram = [base_memory]

        actions = self.actions_by_rank[rank]
        for i, action in enumerate(actions):
            if action is None:
                # Pipeline bubble
                memory = self.memory_histogram[-1]
                self.peak_memory_histogram.append(memory)
                self.memory_histogram.append(memory)
            elif action.op == Op.FORWARD:
                self._forward(action)
            elif action.op == Op.BACKWARD:
                self._backward(action)
            else:
                raise ValueError(f"Unknown operation: {action.op}")

        self._validate_cache()

    def setup(
        self, chunks_list: list[list[int]], actions_by_rank: list[list[Action]]
    ) -> None:
        """
        Setup the memory model with chunk information and build the execution schedule.
        Parameters
        ----------
        chunks_list : list[list[int]]
            List of micro-batch slice counts for each batch to process.
            Each number must be divisible by the number of ranks (p).
        kfkb : bool, optional
            Flag to indicate if KFKB scheduling is enabled. Default is False.
        """
        self.chunks_list = chunks_list
        self.num_microbatches = len(chunks_list)
        self.num_chunks = [len(chunks) for chunks in chunks_list]
        self.actions_by_rank = actions_by_rank

    def run(self, rank: int = 0):
        """
        Simulate execution on the specified rank and calculate memory usage.
        Parameters
        ----------
        rank : int
            The rank to simulate execution for. Default is 0.
        """
        if rank < 0 or rank >= self.config.pipeline_parallel_size:
            raise ValueError(
                f"Rank {rank} is out of range. Must be between 0 and {self.config.pipeline_parallel_size - 1}."
            )
        if self.actions_by_rank is None:
            raise ValueError("The schedule has not been built yet. Call setup() first.")

        self._simulate_execution(rank)


def test_splitfuse_schedule():
    model = Model(
        name="Llama 7B",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
    )
    config = TrainingConfig(
        model=model,
        num_gpus=8,
        microbatch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=2,
        pipeline_parallel_size=4,
        expert_parallel_size=1,
        num_model_chunks=1,
        ckpt="partial",
        offload_ratio=0.0,
    )
    print(model)
    # Example usage
    p = config.pipeline_parallel_size  # Number of ranks
    chunks_list = [
        [4545, 4432],
        [4545, 4542],
        [4564],
        [4566],
        [4594],
        [4624],
        [4638],
        [4616],
        [4629],
        [4645],
        [4644],
        [4647],
        [4644],
        [4673],
        [4667],
        [4672],
        [4671],
        [4817],
        [4948],
        [4910],
        [4971],
        [4959],
    ]
    actions_by_rank = build_splitfuse_schedule(p, chunks_list=chunks_list)

    memory_model = MemoryModel(config)
    memory_model.setup(chunks_list, actions_by_rank)
    memory_model.run(rank=1)
    print(f"Initial memory usage: {memory_model.memory_histogram[0] / 1024**2:.2f} MiB")
    print(
        f"Peak memory usage: {max(memory_model.peak_memory_histogram) / 1024**2:.2f} MiB"
    )
    print(
        f"Delta memory usage: {(max(memory_model.peak_memory_histogram) - memory_model.memory_histogram[0]) / 1024**2:.2f} MiB"
    )
    print(f"offload buffer size: {memory_model.offload_buffer_size / 1024**2:.2f} MiB")
    plot_combined_visualization(
        memory_model.actions_by_rank,
        memory_model.memory_histogram,
        memory_model.peak_memory_histogram,
    )


def test_hybrid_schedule():
    model = Model(
        name="Llama 13B",
        vocab_size=128000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
    )
    config = TrainingConfig(
        model=model,
        num_gpus=32,
        microbatch_size=1,
        tensor_parallel_size=8,
        context_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=4,
        expert_parallel_size=1,
        num_model_chunks=1,
        ckpt="no",
        offload_ratio=0,
    )
    # Example usage
    p = config.pipeline_parallel_size  # Number of ranks
    k = 2
    num_chunks = [2] * 8  # Number of slices for each batch
    seq_length = 32 * 1024  # Sequence length
    chunks_list = []
    for num in num_chunks:
        chunks_list.append([seq_length // num] * num)

    actions_by_rank = build_hybrid_schedule(
        p, k, fwd_switch=(3, 0), bwd_switch=(3, 1), chunks_list=chunks_list
    )

    memory_model = MemoryModel(config)
    memory_model.setup(chunks_list, actions_by_rank)
    memory_model.run()
    print(
        f"Peak memory usage: {max(memory_model.peak_memory_histogram) / 1024**3:.2f} GiB"
    )
    plot_combined_visualization(
        memory_model.actions_by_rank,
        memory_model.memory_histogram,
        memory_model.peak_memory_histogram,
    )


def test_1f1b_schedule():
    model = Model(
        name="Llama 7B",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
    )
    config = TrainingConfig(
        model=model,
        num_gpus=16,
        microbatch_size=1,
        tensor_parallel_size=2,
        context_parallel_size=1,
        data_parallel_size=2,
        pipeline_parallel_size=4,
        expert_parallel_size=1,
        num_model_chunks=1,
        ckpt="no",
        offload_ratio=0.0,
    )
    # Example usage
    p = config.pipeline_parallel_size  # Number of ranks
    chunks = [16 * 1024] * 8
    chunks_list = [[16 * 1024]] * 8
    actions_by_rank = build_1f1b_schedule(p, chunks=chunks)

    memory_model = MemoryModel(config)
    memory_model.setup(chunks_list, actions_by_rank)
    for i in range(p):
        memory_model.run(rank=i)
        print(f"Rank {i} memory usage:")
        print(
            f"Initial memory usage: {memory_model.memory_histogram[0] / 1024**2:.2f} MiB"
        )
        print(
            f"Peak memory usage: {max(memory_model.peak_memory_histogram) / 1024**2:.2f} MiB"
        )
        print(
            f"Delta memory usage: {(max(memory_model.peak_memory_histogram) - memory_model.memory_histogram[0]) / 1024**2:.2f} MiB"
        )
        print(
            f"offload buffer size: {memory_model.offload_buffer_size / 1024**2:.2f} MiB"
        )
        print("=" * 80)


if __name__ == "__main__":
    test_splitfuse_schedule()
    test_hybrid_schedule()
    test_1f1b_schedule()
