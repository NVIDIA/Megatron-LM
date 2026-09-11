"""Public wrapper for the SM100 fused GDR forward CuTe DSL kernel."""

from .fused_fwd import chunk_gated_delta_rule_prefill_cute

__all__ = ["chunk_gated_delta_rule_prefill_cute"]
