# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic multimodal varlen mock — the packed-document paradigm.

Megatron ships two pure-text variable-length paths, and this package is
deliberately neither:

* ``megatron/training/datasets/varlen_dataset.py`` emits one unpacked
  sample per item and defers packing to the upstream packing scheduler.
* ``megatron/training/datasets/sft_dataset.py`` packs multiple
  conversations per item at dataset time, truncating the overflowing one.
* This package packs at **plan time**: every item is one precomputed
  window of whole documents (sample-atomic, never split, never
  truncated; the overflowing document closes the window and opens the
  next one). Once the fully resolved profile and the packing parameters
  (seq_length and the segment alignment) are fixed,
  ``plan_seed`` is the only source of layout randomness — the layout is
  deterministic and prefix-stable, while ``--seed`` varies token/pixel
  content only.

Deferring to the packing scheduler is not an option here: the core
scheduler currently transports token-axis tensors and length metadata,
but has no contract for rerouting ragged vision payloads alongside
their placeholder tokens. See the README for the full rationale.

Modules (import them explicitly; this package intentionally re-exports
nothing so that the numpy-only modules stay importable without torch):

* ``distributions`` — generator-agnostic numeric helpers (numpy/math).
* ``packed_document`` — the torch-free window plan kernel and the
  context-scaled default profile.
* ``qwen35_vl`` — the torch dataset, provider, config resolver, and
  Qwen3.5-VL geometry adapter.
"""
