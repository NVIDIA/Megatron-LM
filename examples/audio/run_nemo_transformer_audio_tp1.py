# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TP=1 smoke check for the vendored NeMo transformer audio encoder.

Importing this module does not initialize distributed state or build the encoder.
``main`` runs only under ``__main__``. The flash-attention compare stays inside
``main`` and runs only when CUDA and flash-attn are available, so pytest collection
does not execute it.
"""


def main() -> None:
    """Check a tiny legacy SDPA mask, then flash-attention when the stack is present."""
    _initialize_model_parallel_for_tp1()
    _check_sdpa_output_mask()
    _compare_flash_attention_when_available()


def _initialize_model_parallel_for_tp1() -> None:
    """Initialize model parallel only when torch.distributed is already up."""
    import torch.distributed as dist

    from megatron.core.parallel_state import (
        get_tensor_model_parallel_world_size,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    distributed_up = dist.is_available() and dist.is_initialized()
    if distributed_up and not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    if model_parallel_is_initialized():
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size != 1:
            raise RuntimeError(
                "examples/audio/run_nemo_transformer_audio_tp1.py requires tensor "
                f"model parallel world size 1, got {tp_world_size}."
            )


def _check_sdpa_output_mask() -> None:
    """Run two unequal mel lengths and compare the mask to conv subsample lengths."""
    import torch

    from megatron.core.models.audio.nemo_transformer_audio_model import (
        NemoTransformerAudioConfig,
        NemoTransformerAudioModel,
    )
    from megatron.core.models.audio.packed_audio import PackedAudioEmbeddings

    torch.manual_seed(0)
    n_mels = 8
    config = NemoTransformerAudioConfig(
        n_mels=n_mels,
        d_model=16,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        pre_encode="conv",
        attn_impl="sdpa",
        nan_debug=False,
    )
    model = NemoTransformerAudioModel(config)
    model.eval()

    mel_lengths = torch.tensor([16, 12], dtype=torch.long)
    max_frames = int(mel_lengths.max())
    features = torch.randn(mel_lengths.numel(), max_frames, n_mels)
    attention_mask = torch.arange(max_frames).unsqueeze(0) < mel_lengths.unsqueeze(1)
    features = features * attention_mask.unsqueeze(-1)

    _hidden_states, output_mask = model(features, attention_mask)
    post_lengths = torch.div(
        torch.div(mel_lengths, 2, rounding_mode="floor"), 2, rounding_mode="floor"
    )
    expected_mask = torch.arange(output_mask.shape[1]).unsqueeze(0) < post_lengths.unsqueeze(1)
    if not torch.equal(output_mask, expected_mask):
        raise RuntimeError(
            "output_mask does not match post-subsample lengths "
            f"{post_lengths.tolist()}: {output_mask.tolist()}"
        )

    packed = model.forward_packed(features, attention_mask)
    if not isinstance(packed, PackedAudioEmbeddings):
        raise TypeError(
            "forward_packed must return PackedAudioEmbeddings, " f"got {type(packed).__name__}"
        )
    if int(packed.embeddings.shape[0]) != int(post_lengths.sum()):
        raise RuntimeError(
            "Packed embeddings must contain one row per valid post-subsample frame, "
            f"got {int(packed.embeddings.shape[0])} rows for lengths {post_lengths.tolist()}"
        )
    if not torch.equal(packed.lengths.cpu(), post_lengths.to(dtype=torch.int32)):
        raise RuntimeError(
            f"Packed lengths {packed.lengths.tolist()} != post-subsample {post_lengths.tolist()}"
        )


def _compare_flash_attention_when_available() -> None:
    """Match padded varlen rows to two unpadded runs when CUDA and flash-attn exist."""
    import torch

    if not torch.cuda.is_available():
        return
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        return
    if not callable(flash_attn_varlen_func):
        return

    from megatron.core.models.audio.nemo_transformer_encoder import MultiHeadAttentionWithFA

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attention = MultiHeadAttentionWithFA(
        dim_in=32, dim_out=32, dropout=0.0, num_heads=4, causal_mask=False
    )
    attention = attention.to(device="cuda", dtype=dtype)
    attention.eval()

    torch.manual_seed(0)
    left = torch.randn(1, 5, 32, device="cuda", dtype=dtype)
    right = torch.randn(1, 2, 32, device="cuda", dtype=dtype)
    padded = left.new_zeros(2, 5, 32)
    padded[0] = left[0]
    padded[1, :2] = right[0]
    lengths = torch.tensor([5, 2], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        batched = attention(padded, lengths=lengths)
        unpadded_left = attention(left)
        unpadded_right = attention(right)

    torch.testing.assert_close(batched[0], unpadded_left[0], rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(batched[1, :2], unpadded_right[0], rtol=1.0e-2, atol=1.0e-2)
    if not torch.equal(batched[1, 2:], torch.zeros_like(batched[1, 2:])):
        raise RuntimeError("Padded flash-attention rows must stay zero.")


if __name__ == "__main__":
    main()
