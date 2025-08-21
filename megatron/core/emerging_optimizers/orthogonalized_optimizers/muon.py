from functools import partial

from torch.optim.optimizer import ParamsT

from llm_shower.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from llm_shower.orthogonalized_optimizers.muon_utils import newton_schulz


class Muon(OrthogonalizedOptimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz

    Muon runs standard SGD-momentum, and then performs an orthogonalization post-processing step,
    in which each 2D parameter's update is replaced with the nearest orthogonal matrix. To efficiently
    orthogonalize each update, we use 5 iterations of Newton-Schulz iteration, which has the advantage that
    it may be stably run in bfloat16 tensor cores on GPUs. Based on https://github.com/KellerJordan/Muon/blob/master/muon.py

    Warnings and Notes:
    - This optimizer requires that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).

    Args:
        lr: The learning rate used by the internal SGD.
        momentum_beta: The momentum used by the internal SGD.
        use_nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
        use_decoupled_weight_decay: Whether to use decoupled weight decay, default to be True.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        split_qkv: Whether parameter is fused attention parameters (QKV, GQA, etc.), default to be False.
        qkv_split_shapes: For grouped attention parameters (QKV, GQA, etc.), specify the shapes as a tuple of 3 integers
            representing the sizes of Q, K, V components along the first dimension.
            If None, auto-detects first dimension's integer multiple of second dimension and splits equally.
            If is_fused_attention is True, qkv_split_shapes must be provided or Muon will assume it is MHSA.
        fp32_matmul_prec: Matmul precision to use for the Newton-Schulz iteration. Defaults to "medium" (bf16).
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
            - "quintic": Quintic iteration with optimized coefficients.
            - "polar_express": Polar Express iteration with optimized coefficients.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update. Defaults to "spectral" style scaling.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        orthogonalize_fn = partial(newton_schulz, steps=num_ns_steps, coefficient_type=coefficient_type)
        scale_factor_fn = partial(get_muon_scale_factor, mode=scale_mode)

        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov,
            weight_decay,
            use_decoupled_weight_decay,
            split_qkv,
            qkv_split_shapes,
            fp32_matmul_prec,
            orthogonalize_fn,
            scale_factor_fn,
        )


def get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Get the scale for the update.

    Default mode is "spectral", which is the mode that allows for learning rate transferability from AdamW.

    Args:
        size_out: The size of the output tensor.
        size_in: The size of the input tensor.
        mode: The mode to use for the scale.

    Returns:
        The scale factor for the update.
    """
    if mode == "shape_scaling":
        # Suggested by Muon (https://kellerjordan.github.io/posts/muon/)
        return max(1, size_out / size_in) ** 0.5
    elif mode == "spectral":
        # Suggested by Scion (https://arxiv.org/abs/2502.07529) and Kimi (https://arxiv.org/abs/2502.16982)
        return max(size_out, size_in) ** 0.5
    elif mode == "unit_rms_norm":
        # Suggested by Bernstein et al. (https://jeremybernste.in/writing/deriving-muon)
        return (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")
