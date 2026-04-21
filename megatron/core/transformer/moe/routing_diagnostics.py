"""MoE Routing Diagnostics.

This module provides utilities for monitoring MoE routing patterns during training.
Enable with --log-moe-routing-diagnostics flag.

Metrics logged to TensorBoard (at tensorboard_log_interval):
    - moe_routing/dead_experts: Experts receiving 0 tokens (healthy: 0-5)
    - moe_routing/expert_utilization: Fraction of active experts (healthy: >0.9)
    - moe_routing/tokens_per_expert_cov: Coefficient of variation (healthy: <0.3)
    - moe_routing/gini_coefficient: Load inequality measure (healthy: <0.3)
    - moe_routing/expert_bias_std: Std of bias values (healthy: <1.0)

For vision-specific routing analysis (vision vs text token routing patterns),
use compute_routing_stats_simple() with router logits and vision mask.
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional


class MoERoutingMonitor:
    """Monitor MoE routing patterns for vision vs text tokens."""

    def __init__(
        self,
        model,
        num_layers: int = 52,
        sample_layers: Optional[list] = None,
        num_experts: int = 128,
    ):
        """Initialize the routing monitor.

        Args:
            model: The LLaVA model (or language model with MoE layers).
            num_layers: Total number of layers in the model.
            sample_layers: Which layers to monitor. Default: early/mid/late.
            num_experts: Number of experts in MoE layers.
        """
        self.model = model
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Sample 3 representative layers by default (early, mid, late)
        if sample_layers is None:
            sample_layers = [
                num_layers // 4,      # Early: ~layer 13
                num_layers // 2,      # Mid: ~layer 26
                3 * num_layers // 4,  # Late: ~layer 39
            ]
        self.sample_layers = sample_layers

        self._hooks = []
        self._captured_logits = {}

    def _find_router_modules(self):
        """Find router modules in the model."""
        routers = {}
        # Navigate to language model decoder layers
        lm = getattr(self.model, 'language_model', self.model)
        decoder = getattr(lm, 'decoder', None)
        if decoder is None:
            return routers

        layers = getattr(decoder, 'layers', [])
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.sample_layers:
                continue
            # Look for MoE layer with router
            mlp = getattr(layer, 'mlp', None)
            if mlp is not None:
                router = getattr(mlp, 'router', None)
                if router is not None:
                    routers[layer_idx] = router
        return routers

    def _make_hook(self, layer_idx: int):
        """Create a forward hook to capture router logits."""
        def hook(module, args, output):
            # Router forward returns (scores, routing_map)
            # We want the logits before top-k, which requires capturing from gating()
            # For now, capture the input to routing() by storing in module
            pass
        return hook

    def _make_gating_hook(self, layer_idx: int):
        """Hook to capture logits after gating (before routing decision)."""
        def hook(module, args, output):
            # output is the logits tensor from gating()
            self._captured_logits[layer_idx] = output.detach()
        return hook

    @contextmanager
    def capture(self):
        """Context manager to capture router logits during forward pass."""
        self._captured_logits = {}
        self._hooks = []

        # Find and hook router gating
        routers = self._find_router_modules()
        for layer_idx, router in routers.items():
            # Hook the gating method by wrapping it
            # Since gating is a method, we'll hook the router's forward and
            # capture logits from there
            original_forward = router.forward

            def make_capturing_forward(orig_fwd, idx):
                def capturing_forward(input):
                    # Call gating to get logits
                    input_jittered = router.apply_input_jitter(input)
                    logits = router.gating(input_jittered)
                    # Capture the logits
                    self._captured_logits[idx] = logits.detach()
                    # Continue with normal forward (will recompute, but that's ok)
                    return orig_fwd(input)
                return capturing_forward

            router.forward = make_capturing_forward(original_forward, layer_idx)
            self._hooks.append((router, original_forward))

        try:
            yield
        finally:
            # Restore original forwards
            for router, original_forward in self._hooks:
                router.forward = original_forward
            self._hooks = []

    def compute_stats(self, vision_mask: torch.Tensor) -> dict:
        """Compute routing statistics for vision vs text tokens.

        Args:
            vision_mask: Boolean tensor of shape (batch, seq_len) where True = vision token.

        Returns:
            Dictionary of routing statistics.
        """
        if not self._captured_logits:
            return {}

        stats = {}
        vision_entropies = []
        text_entropies = []
        similarities = []

        for layer_idx, logits in self._captured_logits.items():
            # logits shape: (batch * seq_len, num_experts) or (batch, seq_len, num_experts)
            if logits.dim() == 2:
                # Reshape to (batch, seq_len, num_experts) if needed
                # This depends on how the model processes sequences
                batch_size = vision_mask.shape[0]
                seq_len = vision_mask.shape[1]
                if logits.shape[0] == batch_size * seq_len:
                    logits = logits.view(batch_size, seq_len, -1)

            # Flatten batch dimension for easier processing
            if logits.dim() == 3:
                flat_logits = logits.view(-1, logits.shape[-1])
                flat_mask = vision_mask.view(-1)
            else:
                flat_logits = logits
                flat_mask = vision_mask.view(-1)

            # Ensure mask matches logits
            if flat_mask.shape[0] != flat_logits.shape[0]:
                # Skip this layer if shapes don't match (can happen with packing)
                continue

            probs = F.softmax(flat_logits.float(), dim=-1)

            # Compute stats for vision tokens
            vision_probs = probs[flat_mask]
            if vision_probs.shape[0] > 0:
                vision_mean = vision_probs.mean(dim=0)
                # Entropy: -sum(p * log(p))
                vision_entropy = -(vision_mean * (vision_mean + 1e-9).log()).sum()
                vision_entropies.append(vision_entropy.item())

            # Compute stats for text tokens
            text_probs = probs[~flat_mask]
            if text_probs.shape[0] > 0:
                text_mean = text_probs.mean(dim=0)
                text_entropy = -(text_mean * (text_mean + 1e-9).log()).sum()
                text_entropies.append(text_entropy.item())

            # Compute vision-text similarity
            if vision_probs.shape[0] > 0 and text_probs.shape[0] > 0:
                cos_sim = F.cosine_similarity(
                    vision_mean.unsqueeze(0),
                    text_mean.unsqueeze(0)
                )
                similarities.append(cos_sim.item())

        # Aggregate across sampled layers
        if vision_entropies:
            stats['vision_routing_entropy'] = sum(vision_entropies) / len(vision_entropies)
        if text_entropies:
            stats['text_routing_entropy'] = sum(text_entropies) / len(text_entropies)
        if similarities:
            stats['vision_text_routing_similarity'] = sum(similarities) / len(similarities)

        # Also compute top-1 probability for vision tokens (from last captured layer)
        if self._captured_logits:
            last_layer = max(self._captured_logits.keys())
            logits = self._captured_logits[last_layer]
            if logits.dim() == 3:
                flat_logits = logits.view(-1, logits.shape[-1])
            else:
                flat_logits = logits
            flat_mask = vision_mask.view(-1)
            if flat_mask.shape[0] == flat_logits.shape[0]:
                probs = F.softmax(flat_logits.float(), dim=-1)
                vision_probs = probs[flat_mask]
                if vision_probs.shape[0] > 0:
                    vision_mean = vision_probs.mean(dim=0)
                    stats['vision_top1_prob'] = vision_mean.max().item()

        return stats


def compute_routing_stats_simple(
    router_logits: torch.Tensor,
    vision_mask: torch.Tensor,
) -> dict:
    """Compute routing statistics without hooks (for use with captured logits).

    This is a standalone function for computing stats from router logits.

    Args:
        router_logits: Tensor of shape (batch, seq_len, num_experts) or (tokens, num_experts).
        vision_mask: Boolean tensor indicating vision tokens.

    Returns:
        Dictionary with routing statistics.
    """
    # Flatten if needed
    if router_logits.dim() == 3:
        flat_logits = router_logits.view(-1, router_logits.shape[-1])
    else:
        flat_logits = router_logits

    flat_mask = vision_mask.view(-1).bool()

    # Handle shape mismatch
    if flat_mask.shape[0] != flat_logits.shape[0]:
        return {}

    probs = F.softmax(flat_logits.float(), dim=-1)
    stats = {}

    # Vision stats
    vision_probs = probs[flat_mask]
    if vision_probs.shape[0] > 0:
        vision_mean = vision_probs.mean(dim=0)
        vision_entropy = -(vision_mean * (vision_mean + 1e-9).log()).sum()
        stats['vision_routing_entropy'] = vision_entropy.item()
        stats['vision_top1_prob'] = vision_mean.max().item()
        stats['vision_active_experts'] = (vision_mean > 0.01).sum().item()

    # Text stats
    text_probs = probs[~flat_mask]
    if text_probs.shape[0] > 0:
        text_mean = text_probs.mean(dim=0)
        text_entropy = -(text_mean * (text_mean + 1e-9).log()).sum()
        stats['text_routing_entropy'] = text_entropy.item()

    # Cross-modal similarity
    if vision_probs.shape[0] > 0 and text_probs.shape[0] > 0:
        cos_sim = F.cosine_similarity(
            vision_mean.unsqueeze(0),
            text_mean.unsqueeze(0)
        )
        stats['vision_text_routing_similarity'] = cos_sim.item()

    return stats


# =============================================================================
# Routing Anomaly Diagnostics
# =============================================================================

def compute_expert_utilization(tokens_per_expert: torch.Tensor, top_k: int = 6) -> dict:
    """Compute expert utilization metrics from tokens_per_expert counts.

    Args:
        tokens_per_expert: Tensor of shape [num_experts] with token counts.
        top_k: Number of experts selected per token.

    Returns:
        Dictionary with utilization metrics.
    """
    tokens_per_expert = tokens_per_expert.float()
    num_experts = tokens_per_expert.shape[0]
    total_tokens = tokens_per_expert.sum()

    stats = {}

    # Dead experts (received 0 tokens)
    dead_experts = (tokens_per_expert == 0).sum().item()
    stats['dead_experts'] = dead_experts
    stats['expert_utilization'] = (num_experts - dead_experts) / num_experts

    # Load imbalance metrics
    if total_tokens > 0:
        # Expected tokens per expert with perfect balance
        expected = total_tokens / num_experts

        # Coefficient of variation (std / mean)
        mean_tokens = tokens_per_expert.mean()
        std_tokens = tokens_per_expert.std()
        stats['tokens_per_expert_cov'] = (std_tokens / (mean_tokens + 1e-9)).item()

        # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        sorted_tokens = tokens_per_expert.sort()[0]
        cumsum = sorted_tokens.cumsum(0)
        gini = 1 - 2 * cumsum.sum() / (num_experts * total_tokens + 1e-9)
        stats['gini_coefficient'] = gini.item()

        # Max load factor (how overloaded is the busiest expert)
        stats['max_load_factor'] = (tokens_per_expert.max() / expected).item()

        # Min load factor (how underloaded is the least busy expert)
        nonzero_min = tokens_per_expert[tokens_per_expert > 0].min() if dead_experts < num_experts else torch.tensor(0.0)
        stats['min_load_factor'] = (nonzero_min / expected).item()

    return stats


def compute_expert_bias_stats(expert_bias: torch.Tensor) -> dict:
    """Compute statistics on expert bias values.

    Large or drifting bias values can indicate routing instability.

    Args:
        expert_bias: Tensor of shape [num_experts] with bias values.

    Returns:
        Dictionary with bias statistics.
    """
    bias = expert_bias.float()

    stats = {
        'expert_bias_mean': bias.mean().item(),
        'expert_bias_std': bias.std().item(),
        'expert_bias_min': bias.min().item(),
        'expert_bias_max': bias.max().item(),
        'expert_bias_range': (bias.max() - bias.min()).item(),
    }

    # Warning flags
    if bias.std() > 1.0:
        stats['expert_bias_warning'] = 'high_std'
    elif bias.abs().max() > 5.0:
        stats['expert_bias_warning'] = 'extreme_values'

    return stats


def get_all_routing_diagnostics(
    tokens_per_expert: torch.Tensor,
    expert_bias: Optional[torch.Tensor] = None,
    capacity: Optional[int] = None,
    top_k: int = 6,
) -> dict:
    """Compute all routing diagnostics in one call.

    Args:
        tokens_per_expert: Tensor of shape [num_experts] with token counts.
        expert_bias: Optional tensor of shape [num_experts] with bias values.
        capacity: Optional capacity limit per expert.
        top_k: Number of experts per token.

    Returns:
        Dictionary with all routing diagnostics.
    """
    stats = {}

    # Expert utilization
    stats.update(compute_expert_utilization(tokens_per_expert, top_k))

    # Expert bias
    if expert_bias is not None:
        stats.update(compute_expert_bias_stats(expert_bias))

    return stats


# =============================================================================
# Extract diagnostics from model (no forward pass modification needed)
# =============================================================================

def _unwrap_model(model):
    """Unwrap model from DDP/FSDP wrappers and list."""
    # Handle list of model chunks (pipeline parallelism)
    if isinstance(model, list):
        if len(model) == 0:
            return None
        model = model[0]

    # Unwrap DDP/FSDP wrappers
    while hasattr(model, 'module'):
        model = model.module

    return model


def extract_router_diagnostics_from_model(
    model,
    sample_layers: Optional[list] = None,
    verbose: bool = False,
) -> dict:
    """Extract routing diagnostics from model's router buffers.

    This function reads the tokens_per_expert_for_logging and expert_bias buffers
    from router modules without requiring any forward pass modifications.
    Call this periodically during training (e.g., at logging intervals).

    Args:
        model: The model (LLaVA or language model with MoE layers).
        sample_layers: Which layers to sample (default: early/mid/late).
        verbose: Print debug info.

    Returns:
        Dictionary with routing diagnostics aggregated across sampled layers.
    """
    # Unwrap model from DDP/list wrappers
    model = _unwrap_model(model)
    if model is None:
        if verbose:
            print("[routing_diagnostics] Model is None after unwrapping")
        return {}

    # Navigate to language model (for multimodal models)
    lm = model
    if hasattr(lm, 'language_model'):
        lm = lm.language_model
        if verbose:
            print(f"[routing_diagnostics] Found language_model: {type(lm)}")

    decoder = getattr(lm, 'decoder', None)
    if decoder is None:
        if verbose:
            print(f"[routing_diagnostics] No decoder found in {type(lm)}, attrs: {[a for a in dir(lm) if not a.startswith('_')][:20]}")
        return {}

    layers = getattr(decoder, 'layers', [])
    num_layers = len(layers)
    if verbose:
        print(f"[routing_diagnostics] Found {num_layers} layers")

    if sample_layers is None:
        # Sample 3 representative layers
        sample_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    all_stats = []
    routers_found = 0
    tokens_buffers_found = 0

    for layer_idx in sample_layers:
        if layer_idx >= num_layers:
            continue

        layer = layers[layer_idx]
        mlp = getattr(layer, 'mlp', None)
        if mlp is None:
            if verbose:
                print(f"[routing_diagnostics] Layer {layer_idx}: no mlp, attrs: {[a for a in dir(layer) if not a.startswith('_')][:10]}")
            continue

        router = getattr(mlp, 'router', None)
        if router is None:
            if verbose:
                print(f"[routing_diagnostics] Layer {layer_idx}: no router in mlp, attrs: {[a for a in dir(mlp) if not a.startswith('_')][:10]}")
            continue
        routers_found += 1

        # Get tokens_per_expert buffer (prefer tokens_per_expert_for_logging which is
        # saved before zeroing in the expert bias update, fall back to local_tokens_per_expert)
        tokens_per_expert = getattr(router, 'tokens_per_expert_for_logging', None)
        if tokens_per_expert is None or tokens_per_expert.sum() == 0:
            # Fall back to local_tokens_per_expert (may be zero if read after bias update)
            tokens_per_expert = getattr(router, 'local_tokens_per_expert', None)
        if tokens_per_expert is None:
            if verbose:
                print(f"[routing_diagnostics] Layer {layer_idx}: router has no token count buffers")
            continue
        tokens_buffers_found += 1

        # Get expert_bias buffer
        expert_bias = getattr(router, 'expert_bias', None)

        # Compute stats for this layer
        layer_stats = get_all_routing_diagnostics(
            tokens_per_expert=tokens_per_expert,
            expert_bias=expert_bias,
        )
        layer_stats['layer'] = layer_idx
        all_stats.append(layer_stats)

    if verbose:
        print(f"[routing_diagnostics] Found {routers_found} routers, {tokens_buffers_found} with token buffers, {len(all_stats)} with stats")

    if not all_stats:
        return {}

    # Aggregate across layers (average for most metrics)
    aggregated = {}
    numeric_keys = [k for k in all_stats[0].keys() if k != 'layer' and isinstance(all_stats[0][k], (int, float))]

    for key in numeric_keys:
        values = [s[key] for s in all_stats if key in s]
        if values:
            aggregated[key] = sum(values) / len(values)

    # Also include worst-case values for some metrics
    aggregated['dead_experts_max'] = max(s.get('dead_experts', 0) for s in all_stats)
    aggregated['gini_coefficient_max'] = max(s.get('gini_coefficient', 0) for s in all_stats)

    return aggregated


def log_routing_diagnostics(
    model,
    writer,
    iteration: int,
    prefix: str = "moe_routing",
    verbose: bool = False,
):
    """Extract and log routing diagnostics to tensorboard/wandb.

    Example usage in training loop:
        if iteration % 100 == 0:
            log_routing_diagnostics(model, writer, iteration)

    Args:
        model: The model with MoE routers.
        writer: TensorBoard SummaryWriter or similar.
        iteration: Current training iteration.
        prefix: Prefix for metric names.
        verbose: Print debug info if no stats found.
    """
    stats = extract_router_diagnostics_from_model(model, verbose=verbose)

    if not stats:
        if verbose:
            print(f"[Iteration {iteration}] No routing diagnostics extracted")
        return

    # Log each metric
    for name, value in stats.items():
        if isinstance(value, (int, float)) and not name.endswith('_warning'):
            writer.add_scalar(f"{prefix}/{name}", value, iteration)
