# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixture-of-experts operations with per-token router probabilities.

Router weighting consumes activations shaped ``[tokens, features]`` and
probabilities shaped ``[tokens, 1]``, returning the activation dtype. The eager
path also accepts other broadcastable layouts. Weighted GLU operations consume
FC1 outputs and return the row-local activation's feature width.

These operations own no parameters or persistent state and perform no
communication. Callers supply tensors that are already local to an expert rank.
"""
