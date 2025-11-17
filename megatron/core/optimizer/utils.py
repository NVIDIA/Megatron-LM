def clip_output_grad_hook(module, grad_input, grad_output, *, scale=3.0, eps=1e-6):
    """Clip output gradients based on its median and spread."""
    d_input = grad_input[0]
    if d_input is None:
        return grad_input

    flat = d_input.view(-1).abs()
    median = flat.median()

    # Median absolute deviation or spread.
    spread = (flat - median).abs().median().clamp_min(eps)
    threshold = median + scale * spread

    # Clip and return.
    clipped = d_input.clamp(min=-threshold, max=threshold)
    return (clipped,) + grad_input[1:]
