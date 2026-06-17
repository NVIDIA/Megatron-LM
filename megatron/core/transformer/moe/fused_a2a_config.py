# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
FusedA2AConfig dataclass for user-tunable fused all-to-all MoE parameters.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FusedA2AConfig:
    """
    Configuration for fused all-to-all MoE (FusedDispatch/FusedCombine).

    Fields:
        chunk_size: Optional[int] - Chunk size for all-to-all (must be positive if set)
        num_sms: Optional[int] - Number of SMs for kernel (must be positive and even if set)
        # Future tunables can be added here

    Precedence: CLI > ENV > CONFIG FILE > DEFAULTS
    """
    chunk_size: Optional[int] = None
    num_sms: Optional[int] = None
    # Future tunables can be added here

    def validate(self):
        if self.chunk_size is not None:
            if not (self.chunk_size > 0):
                raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.num_sms is not None:
            if not (self.num_sms > 0):
                raise ValueError(f"num_sms must be positive, got {self.num_sms}")
            # DeepEP's Buffer.set_num_sms asserts new_num_sms % 2 == 0
            # and the C++ kernel asserts config.num_sms % 2 == 0. An odd value
            # would crash deep inside the kernel with an opaque assertion;
            # fail fast at validate_args time instead.
            if self.num_sms % 2 != 0:
                raise ValueError(
                    f"num_sms must be even (DeepEP requirement), got {self.num_sms}"
                )

    @staticmethod
    def from_dict(cfg: dict) -> 'FusedA2AConfig':
        allowed = {'chunk_size', 'num_sms'}
        unknown = set(cfg.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown FusedA2AConfig keys: {unknown}")
        return FusedA2AConfig(
            chunk_size=cfg.get('chunk_size'),
            num_sms=cfg.get('num_sms'),
        )

    def __repr__(self):
        return f"FusedA2AConfig(chunk_size={self.chunk_size}, num_sms={self.num_sms})"
