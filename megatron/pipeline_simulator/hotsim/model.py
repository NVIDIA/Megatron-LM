from dataclasses import dataclass, field


@dataclass
class Model:
    """Configuration class for a LLaMA-like model architecture.

    Defines parameters for a transformer model including dimensions, layers, and attention configuration.

    Attributes:
        name: Model name (auto-generated based on parameters if not provided)
        vocab_size: Size of token vocabulary
        hidden_size: Dimension of hidden representations
        intermediate_size: Dimension of MLP layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key/value heads for grouped query attention
        num_experts: Number of expert layers in mixture of experts
        num_active_experts: Number of experts activated per token
        moe_layer_interval: Interval between MoE layers
        head_dim: Dimension of each attention head (calculated automatically)
    """

    name: str = field(default=None, repr=True)
    vocab_size: int = 128000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = field(default=None, repr=True)
    num_key_value_heads: int = field(default=None, repr=True)
    num_experts: int = 1
    num_active_experts: int = 1
    moe_layer_interval: int = 1
    head_dim: int = field(init=False, repr=False)

    def __post_init__(self):
        # Set default attention heads if not provided
        if self.num_attention_heads is None:
            self.num_attention_heads = self.hidden_size // 128

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Set default KV heads if not provided
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        detailed_params = self.calc_detailed_parameters()
        self.total_params = detailed_params["total_params"]
        self.active_params = detailed_params["active_params"]
        self.params_per_dense_layer = detailed_params["params_per_dense_layer"]
        self.params_per_sparse_layer = detailed_params["params_per_sparse_layer"]
        self.active_params_per_sparse_layer = detailed_params[
            "active_params_per_sparse_layer"
        ]
        self.expert_params_per_sparse_layer = detailed_params[
            "expert_params_per_sparse_layer"
        ]
        self.non_expert_params_per_sparse_layer = detailed_params[
            "non_expert_params_per_sparse_layer"
        ]

        is_moe = self.num_experts > 1
        self.params_per_layer = (
            self.params_per_sparse_layer if is_moe else self.params_per_dense_layer
        )

        # Auto-generate name if not provided
        if self.name is None:
            if is_moe:
                self.name = (
                    f"Mixtral-{self.num_experts}x{self.active_params / 1e9:.0f}B"
                )
            else:
                gqa_suffix = (
                    "-GQA"
                    if self.num_key_value_heads < self.num_attention_heads
                    else ""
                )
                self.name = f"LLaMA-{self.total_params / 1e9:.0f}B{gqa_suffix}"

    def kv_acts_per_layer(self, batch_size: int, seq_len: int) -> float:
        """Calculates activations for key-value cache per layer.

        Args:
            batch_size: Batch size of input
            seq_length: Sequence length of input

        Returns:
            float: Number of key-value activations
        """
        kv_ratio = self.num_key_value_heads / self.num_attention_heads
        return batch_size * seq_len * self.hidden_size * (2 * kv_ratio)

    def acts_per_layer(
        self, batch_size: int, seq_len: int, ckpt: str = "no", recompute: bool = False
    ) -> float:
        """Calculates activations needed per transformer layer.

        Args:
            batch_size: Number of sequences in batch
            seq_length: Length of each sequence
            ckpt: Activation checkpointing strategy ("no", "partial", "partial+fc1", or "full")
            recompute: Whether to return recomputed activations instead of stored ones

        Returns:
            float: Number of activation elements needed
        """
        # Model dimensions
        h = self.hidden_size
        H = self.intermediate_size
        b = batch_size
        s = seq_len
        kv_ratio = self.num_key_value_heads / self.num_attention_heads

        # If full checkpointing is used, we only need to store layer input
        if ckpt == "full" and not recompute:
            return b * s * h

        # Calculate individual activation components
        attn_input = b * s * h  # Input to attention block
        attn_norm = b * s * h  # Layer norm output (not counted separately)
        attn = b * s * h * (2 + 2 * kv_ratio)  # QKV projections and output
        ffn_input = b * s * h  # Input to feed-forward network
        ffn_norm = b * s * h  # Layer norm output (not counted separately)
        fc1 = 2 * b * s * H * self.num_active_experts  # First FFN projection for SwiGLU
        swiglu = b * s * H * self.num_active_experts  # SwiGLU activations

        # Sum all activations
        total_acts = attn_input + attn + ffn_input + fc1 + swiglu

        # Return appropriate activations based on checkpointing strategy
        if ckpt == "no":
            return total_acts if not recompute else 0
        elif ckpt == "partial":
            return total_acts - swiglu if not recompute else swiglu
        elif ckpt == "partial+fc1":
            return total_acts - fc1 - swiglu if not recompute else fc1 + swiglu
        elif ckpt == "full":
            return attn_input if not recompute else total_acts - attn_input

        # Handle invalid checkpointing strategy
        raise ValueError(f"Unknown checkpointing strategy: {ckpt}")

    def tflops(self, batch_size: int, seq_len: int) -> float:
        """Calculates total TeraFLOPs for a complete forward and backward pass.

        Args:
            batch_size: Batch size of input
            seq_length: Sequence length of input

        Returns:
            float: Total TFLOPs required
        """
        detailed_flops = self.calc_detailed_flops(seq_len)
        return 3 * batch_size * detailed_flops["flops_per_forward"] / 1e12

    def calc_detailed_parameters(self) -> dict:
        """Calculate detailed model parameters breakdown.

        Provides a comprehensive breakdown of parameter counts for both standard
        transformer models and mixture-of-experts architectures.

        Returns:
            dict: Parameter statistics including total, active, and per-layer counts
        """
        # Basic model configuration
        is_moe = self.num_experts > 1
        h = self.hidden_size
        H = self.intermediate_size
        V = self.vocab_size
        kv_ratio = self.num_key_value_heads / self.num_attention_heads

        # Calculate layer distribution
        num_sparse_layers = (
            self.num_hidden_layers // self.moe_layer_interval if is_moe else 0
        )
        num_dense_layers = self.num_hidden_layers - num_sparse_layers

        # Calculate attention parameters (same for both dense and sparse layers)
        params_attention = (
            h + h * h * 2 + h * (h * kv_ratio) * 2
        )  # Q/K/V bias + Q,O,K,V projections

        # Calculate dense layer parameters
        params_dense_layer = (
            params_attention + h + 3 * h * H
        )  # Attention + layer norm + MLP

        # Calculate MoE layer parameters (total and active)
        params_sparse_layer = 0
        active_params_sparse_layer = 0
        expert_params_sparse_layer = 0
        non_expert_params_sparse_layer = 0

        if is_moe:
            params_sparse_layer = (
                params_attention
                + h
                + h * self.num_experts
                + 3 * h * H * self.num_experts
            )  # Attention + norm + router + MLP
            active_params_sparse_layer = (
                params_attention
                + h
                + h * self.num_experts
                + 3 * h * H * self.num_active_experts
            )
            expert_params_sparse_layer = 3 * h * H * self.num_experts
            non_expert_params_sparse_layer = (
                params_sparse_layer - expert_params_sparse_layer
            )

        # Calculate total parameters
        total_params = (
            num_dense_layers * params_dense_layer
            + num_sparse_layers * params_sparse_layer
            + h  # Final layer norm
            + 2 * h * V  # Embedding and output layers
        )

        # Calculate active parameters during forward pass
        active_params = (
            num_dense_layers * params_dense_layer
            + num_sparse_layers * active_params_sparse_layer
            + h  # Final layer norm
            + 2 * h * V  # Embedding and output layers
        )

        return {
            "total_params": total_params,
            "active_params": active_params,
            "params_per_dense_layer": params_dense_layer,
            "params_per_sparse_layer": params_sparse_layer,
            "active_params_per_sparse_layer": active_params_sparse_layer,
            "expert_params_per_sparse_layer": expert_params_sparse_layer,
            "non_expert_params_per_sparse_layer": non_expert_params_sparse_layer,
        }

    def calc_detailed_flops(self, seq_len: int) -> dict:
        """Calculate detailed FLOPs breakdown for model computation.

        Provides FLOP counts for model components during the forward pass.

        Args:
            seq_len: Sequence length for FLOP calculations

        Returns:
            dict: FLOP counts for dense layers, sparse layers, and complete forward pass
        """
        # Basic model configuration
        is_moe = self.num_experts > 1
        h = self.hidden_size
        H = self.intermediate_size
        V = self.vocab_size
        s = seq_len
        kv_ratio = self.num_key_value_heads / self.num_attention_heads

        # Calculate layer distribution
        num_sparse_layers = (
            self.num_hidden_layers // self.moe_layer_interval if is_moe else 0
        )
        num_dense_layers = self.num_hidden_layers - num_sparse_layers

        # Calculate FLOPs for attention mechanism
        flops_attention = (
            4 * s * h * h  # Q, K, V projections
            + 4 * s * h * (h * kv_ratio)  # Attention computations
            + 2 * s * s * h  # Attention matrix multiplications
        )

        # Calculate FLOPs for dense layer
        flops_dense_layer = flops_attention + 6 * s * h * H  # MLP operations

        # Calculate FLOPs for MoE layer
        flops_sparse_layer = 0
        if is_moe:
            flops_sparse_layer = (
                flops_attention
                + 2 * s * h * self.num_experts  # Router
                + 6 * s * h * H * self.num_active_experts  # MLP operations
            )

        # Calculate total forward pass FLOPs
        flops_forward = (
            num_dense_layers * flops_dense_layer
            + num_sparse_layers * flops_sparse_layer
            + 2 * s * h * V  # Embedding and final projection
        )

        return {
            "flops_per_dense_layer": flops_dense_layer,
            "flops_per_sparse_layer": flops_sparse_layer,
            "flops_per_forward": flops_forward,
        }

    def __repr__(self) -> str:
        """String representation of the model configuration."""
        return (
            f"({self.name}, V={self.vocab_size}, h={self.hidden_size}, "
            f"H={self.intermediate_size}, L={self.num_hidden_layers}, "
            f"a={self.num_attention_heads}, g={self.num_key_value_heads}, "
            f"d={self.head_dim}, e={self.num_experts}, topk={self.num_active_experts})"
        )
