"""Batch generation utilities for Megatron Lite runtime."""

from __future__ import annotations

import os

import torch  # pyright: ignore[reportMissingImports]

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _read_bool_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _resolve_thd_padding(seq_len: int, cp_size: int) -> tuple[int, int, bool]:
    """Return (padded_seq_len, alignment_multiple, pad_enabled) for THD input."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")

    pad_to_alignment = _read_bool_env("MEGATRON_LITE_THD_PAD_TO_ALIGNMENT")
    if pad_to_alignment is None:
        pad_to_alignment = cp_size > 1

    pad_multiple_env = os.environ.get("MEGATRON_LITE_THD_PAD_MULTIPLE", "auto").strip().lower()
    if pad_multiple_env in ("", "auto", "cp", "min_cp"):
        align_size = cp_size * 2 if cp_size > 1 else 1
    else:
        try:
            align_size = int(pad_multiple_env)
        except ValueError as exc:
            raise ValueError(
                "MEGATRON_LITE_THD_PAD_MULTIPLE must be 'auto' or a positive integer, "
                f"got {pad_multiple_env!r}"
            ) from exc
        if align_size <= 0:
            raise ValueError(
                "MEGATRON_LITE_THD_PAD_MULTIPLE must be 'auto' or a positive integer, "
                f"got {pad_multiple_env!r}"
            )

    if not pad_to_alignment:
        return seq_len, align_size, False

    seq_len_padded = ((seq_len + align_size - 1) // align_size) * align_size
    return seq_len_padded, align_size, True


def fixed_batches(
    vocab_size: int,
    seq_len: int,
    num_steps: int,
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic (input_ids, labels) pairs, identical across all ranks."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g).to(device)
        batches.append((ids, labels))
    return batches


def infinite_batches(
    vocab_size: int,
    seq_len: int,
    batch_size: int = 1,
    device: str = "cuda",
    seed: int = 42,
):
    """Infinite deterministic batch generator (for throughput benchmarks)."""
    g = torch.Generator(device=device).manual_seed(seed)
    while True:
        yield {
            "input_ids": torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                device=device,
                generator=g,
            ),
            "labels": torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len),
                device=device,
                generator=g,
            ),
        }


def infinite_batches_thd(
    vocab_size: int,
    seq_len: int,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
    device: str = "cuda",
    seed: int = 42,
):
    """Infinite deterministic batch generator in THD (packed variable-length) format.

    Produces plain dicts consumed directly by native lite model forwards
    (input_ids 2-D batch=1, mrope position_ids (3,1,T), pre-built PackedSeqParams).

    cp_size/cp_rank should be supplied explicitly by the caller (for example
    from handle._parallel_state). The defaults keep older single-CP call sites
    working, but CP runs must pass the real CP rank and world size.

    When context parallelism is active (cp_size > 1), each rank's packed tokens are
    split via zigzag striping. position_ids stay FULL because lite's RoPE
    (is_thd_format=False) auto-slices emb internally, same as BSH path.
    """
    from megatron.core.packed_seq_params import PackedSeqParams  # pyright: ignore[reportMissingImports]  # noqa: I001

    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")

    g = torch.Generator(device=device).manual_seed(seed)
    seq_len_padded, _align_size, _pad_to_alignment = _resolve_thd_padding(seq_len, cp_size)

    # position_ids always stay full length; RoPE auto-slices for CP.
    position_ids = (
        torch.arange(seq_len_padded, device=device)
        .view(1, 1, -1)
        .expand(3, 1, seq_len_padded)
        .contiguous()
    )
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    cu_seqlens_padded = torch.tensor([0, seq_len_padded], dtype=torch.int32, device=device)

    if cp_size > 1:
        from megatron.lite.primitive.parallel.cp import zigzag_split_for_cp  # noqa: I001

        # cu_seqlens stays full; MC GDN internally divides by cp_size.
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len_padded,
            max_seqlen_kv=seq_len_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        while True:
            ids_full = torch.randint(0, vocab_size, (seq_len,), device=device, generator=g)
            lbl_full = torch.randint(0, vocab_size, (seq_len,), device=device, generator=g)
            if seq_len_padded != seq_len:
                ids_padded = torch.zeros(seq_len_padded, dtype=ids_full.dtype, device=device)
                lbl_padded = torch.zeros(seq_len_padded, dtype=lbl_full.dtype, device=device)
                ids_padded[:seq_len] = ids_full
                lbl_padded[:seq_len] = lbl_full
            else:
                ids_padded = ids_full
                lbl_padded = lbl_full
            ids_cp = zigzag_split_for_cp(ids_padded, cp_rank, cp_size, seq_dim=0)
            lbl_cp = zigzag_split_for_cp(lbl_padded, cp_rank, cp_size, seq_dim=0)
            yield {
                "input_ids": ids_cp.unsqueeze(0),   # (1, T/cp)
                "labels": lbl_cp.unsqueeze(0),       # (1, T/cp)
                "position_ids": position_ids,         # FULL (3, 1, T)
                "packed_seq_params": packed_seq_params,
            }
    else:
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len_padded,
            max_seqlen_kv=seq_len_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        while True:
            input_ids = torch.zeros((1, seq_len_padded), dtype=torch.long, device=device)
            labels = torch.zeros((1, seq_len_padded), dtype=torch.long, device=device)
            input_ids[:, :seq_len] = torch.randint(
                0, vocab_size, (1, seq_len), device=device, generator=g
            )
            labels[:, :seq_len] = torch.randint(
                0, vocab_size, (1, seq_len), device=device, generator=g
            )
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "packed_seq_params": packed_seq_params,
            }


__all__ = [
    "fixed_batches",
    "infinite_batches",
    "infinite_batches_thd",
]
