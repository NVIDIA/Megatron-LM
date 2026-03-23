def token_dispatch_comm_volume(
    batch_size,
    seq_len,
    hidden_size,
    topk,
    dtype_bytes=2,  # Default FP16/BF16
):
    """Calculate communication volume for token dispatch in MoE.

    In token dispatch, tokens are routed to different experts.
    Assuming uniform distribution, communication volume is:
    batch_size * seq_len * topk * hidden_size * dtype_bytes

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        topk: Number of experts each token is routed to
        dtype_bytes: Bytes per element (2 for FP16/BF16, 4 for FP32)

    Returns:
        Communication volume in bytes
    """
    return batch_size * seq_len * topk * hidden_size * dtype_bytes


def token_combine_comm_volume(
    batch_size,
    seq_len,
    hidden_size,
    topk,
    dtype_bytes=2,  # Default FP16/BF16
):
    """Calculate communication volume for token combine in MoE.

    In token combine, expert outputs are gathered and combined.
    Assuming uniform distribution, communication volume is:
    batch_size * seq_len * topk * hidden_size * dtype_bytes

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        topk: Number of experts each token is routed to
        dtype_bytes: Bytes per element (2 for FP16/BF16, 4 for FP32)

    Returns:
        Communication volume in bytes
    """
    return batch_size * seq_len * topk * hidden_size * dtype_bytes
