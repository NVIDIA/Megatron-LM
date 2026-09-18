# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Resolve MLA and DSA Q/KV norm configuration from a layer specification.
"""

from typing import NoReturn

from megatron.core.models.backends import get_backend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig

__all__ = []

_QKNormResolvedConfig = dict[str, ModuleSpec | type | LayerNormBuilder]


class QKNormConfigResolver:
    """Validate and resolve Q/KV norm placement for MLA and DSA.

    Q/KV norm can be represented either by a standalone norm module or by
    a fused norm+linear projection. MLA can use the fused form; DSA cannot
    because it needs the normalized Q/KV values outside the projection.

    Constraints:
    - `qk_l2_norm` is unsupported for MLA/DSA.
    - A standalone Q norm is only usable when `q_lora_rank` is set.
    - Explicit norm modules cannot be paired with fused norm+linear projections.
    - Disabled QK norm rejects both explicit norms and fused norm+linear projections.
    - DSA with QK norm requires non-fused projections and standalone Q/KV norms.
    """

    def __init__(self, config: MLATransformerConfig, submodules) -> None:
        """Capture the configuration, requested modules, and backend implementations."""
        self.config = config
        self.submodules = submodules
        self.has_q_lora = config.q_lora_rank is not None
        self.is_dsa = config.experimental_attention_variant == "dsa"
        self.variant_str = "DSA" if self.is_dsa else "MLA"

        backend = get_backend(config.transformer_impl)
        self.qk_norm_impl = backend.layer_norm(
            rms_norm=config.normalization == "RMSNorm", for_qk=True
        )
        self.linear_impl = backend.column_parallel_linear()
        self.fused_norm_linear_impl = backend.column_parallel_layer_norm_linear()

    def resolve(self) -> _QKNormResolvedConfig:
        """Validate the specification and return the modules to instantiate.

        Returns:
            The Q/KV norms and projections after applying the MLA or DSA constraints.

        Raises:
            ValueError: If the requested norm placement is unsupported or conflicting.
        """
        if self.config.qk_l2_norm:
            raise ValueError(f"qk_l2_norm is not supported with {self.variant_str}.")

        self._reject_common_spec_conflicts()
        if not self.config.qk_layernorm:
            return self._resolve_disabled_qk_layernorm()
        if self.is_dsa:
            return self._resolve_dsa_qk_layernorm()
        return self._resolve_mla_qk_layernorm()

    def _resolve_disabled_qk_layernorm(self) -> _QKNormResolvedConfig:
        """Resolve projections when Q/KV normalization is disabled.

        Explicit norm modules and fused norm-linear projections are rejected because
        they would still introduce Q/KV normalization.
        """
        linear_q_proj_cls = IdentityOp
        linear_q_up_proj_cls = IdentityOp

        if self.has_q_lora:
            self._reject_disabled_norm(
                self.submodules.linear_q_up_proj,
                self.submodules.q_layernorm,
                "linear_q_up_proj",
                "q_layernorm",
            )
            linear_q_up_proj_cls = self.submodules.linear_q_up_proj or self.linear_impl
        else:
            if self._is_fused_norm_linear(self.submodules.linear_q_proj):
                raise ValueError(
                    f"spec sets linear_q_proj={self.submodules.linear_q_proj}, but "
                    "qk_layernorm/qk_l2_norm are supposed to be disabled"
                )
            linear_q_proj_cls = self.submodules.linear_q_proj or self.linear_impl

        self._reject_disabled_norm(
            self.submodules.linear_kv_up_proj,
            self.submodules.kv_layernorm,
            "linear_kv_up_proj",
            "kv_layernorm",
        )
        return self._result(
            linear_q_proj=linear_q_proj_cls,
            linear_q_up_proj=linear_q_up_proj_cls,
            linear_kv_up_proj=self.submodules.linear_kv_up_proj or self.linear_impl,
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        )

    def _resolve_dsa_qk_layernorm(self) -> _QKNormResolvedConfig:
        """Resolve DSA's standalone Q/KV norms and non-fused projections.

        DSA consumes the normalized Q/KV values outside the projection, so it cannot
        use fused norm-linear projections.
        """
        if not self.has_q_lora:
            raise ValueError(
                "`qk_layernorm=True` with `q_lora_rank is None` is not supported for DSA "
                "because DSA cannot fuse Q norm into `linear_q_proj`."
            )

        return self._result(
            linear_q_proj=IdentityOp,
            linear_q_up_proj=self._dsa_linear_or_default(
                self.submodules.linear_q_up_proj, "linear_q_up_proj"
            ),
            linear_kv_up_proj=self._dsa_linear_or_default(
                self.submodules.linear_kv_up_proj, "linear_kv_up_proj"
            ),
            q_layernorm=self._default_if_trivial(self.submodules.q_layernorm, self.qk_norm_impl),
            kv_layernorm=self._default_if_trivial(self.submodules.kv_layernorm, self.qk_norm_impl),
        )

    def _resolve_mla_qk_layernorm(self) -> _QKNormResolvedConfig:
        """Resolve MLA norms, fusing them into projections when no norm is explicit."""
        q_norm_cls = self.submodules.q_layernorm or IdentityOp
        linear_q_proj_cls = IdentityOp
        linear_q_up_proj_cls = IdentityOp

        if self.has_q_lora:
            if self._is_trivial(q_norm_cls):
                linear_q_up_proj_cls = self._mla_fused_linear_or_default(
                    self.submodules.linear_q_up_proj, "linear_q_up_proj"
                )
            else:
                linear_q_up_proj_cls = self._non_fused_or_default(
                    self.submodules.linear_q_up_proj, "linear_q_up_proj"
                )
        else:
            linear_q_proj_cls = self._mla_fused_linear_or_default(
                self.submodules.linear_q_proj, "linear_q_proj"
            )

        kv_norm_cls = self.submodules.kv_layernorm or IdentityOp
        if self._is_trivial(kv_norm_cls):
            linear_kv_up_proj_cls = self._mla_fused_linear_or_default(
                self.submodules.linear_kv_up_proj, "linear_kv_up_proj"
            )
        else:
            linear_kv_up_proj_cls = self._non_fused_or_default(
                self.submodules.linear_kv_up_proj, "linear_kv_up_proj"
            )

        return self._result(
            linear_q_proj=linear_q_proj_cls,
            linear_q_up_proj=linear_q_up_proj_cls,
            linear_kv_up_proj=linear_kv_up_proj_cls,
            q_layernorm=q_norm_cls,
            kv_layernorm=kv_norm_cls,
        )

    def _reject_common_spec_conflicts(self) -> None:
        """Reject conflicts that apply regardless of the selected attention variant."""
        if not self.has_q_lora and not self._is_trivial(self.submodules.q_layernorm):
            self._raise_unused_q_norm()
        if self.has_q_lora:
            self._reject_explicit_norm_with_fused_linear(
                self.submodules.linear_q_up_proj,
                self.submodules.q_layernorm,
                "linear_q_up_proj",
                "q_layernorm",
            )
        self._reject_explicit_norm_with_fused_linear(
            self.submodules.linear_kv_up_proj,
            self.submodules.kv_layernorm,
            "linear_kv_up_proj",
            "kv_layernorm",
        )

    def _reject_disabled_norm(self, module_spec, norm_spec, module_name, norm_name) -> None:
        """Reject a norm module or fused projection when Q/KV norm is disabled."""
        if self._is_fused_norm_linear(module_spec) or not self._is_trivial(norm_spec):
            raise ValueError(
                f"spec sets {module_name}={module_spec} and "
                f"{norm_name}={norm_spec}, but "
                "qk_layernorm/qk_l2_norm are supposed to be disabled"
            )

    def _reject_explicit_norm_with_fused_linear(
        self, module_spec, norm_spec, module_name, norm_name
    ) -> None:
        """Reject specifying the same norm both explicitly and inside a projection."""
        if not self._is_trivial(norm_spec) and self._is_fused_norm_linear(module_spec):
            raise ValueError(
                f"`{norm_name}={norm_spec}` is non-trivial "
                f"and `{module_name}={module_spec}` is a "
                f"fused norm+linear; either unset `{norm_name}` or use a "
                f"linear layer without norm fusion for `{module_name}`"
            )

    def _non_fused_or_default(self, module_spec, module_name):
        """Return a linear implementation, requiring it not to fuse normalization."""
        linear_cls = module_spec or self.linear_impl
        self._require_linear(linear_cls, module_name)
        if self._is_fused_norm_linear(linear_cls):
            raise ValueError(
                f"`{module_name}={module_spec}` is fused norm+linear, but a non-fused linear "
                f"is required"
            )
        return linear_cls

    def _dsa_linear_or_default(self, module_spec, module_name):
        """Return DSA's non-fused projection implementation.

        This uses a DSA-specific diagnostic so the rejected constraint is clear.
        """
        linear_cls = module_spec or self.linear_impl
        self._require_linear(linear_cls, module_name)
        if self._is_fused_norm_linear(linear_cls):
            raise ValueError(
                f"`{module_name}={module_spec}` is fused norm+linear, "
                f"which is not supported for DSA."
            )
        return linear_cls

    def _mla_fused_linear_or_default(self, module_spec, module_name):
        """Return a fused MLA projection, using the backend default when available."""
        if self._is_fused_norm_linear(module_spec):
            return module_spec
        return self._require_linear(self.fused_norm_linear_impl, module_name)

    def _require_linear(self, module_spec, module_name):
        """Return a configured projection or report that no viable implementation exists."""
        if module_spec is None:
            raise RuntimeError(
                "qk_layernorm requires TransformerEngine or "
                "q_layernorm/kv_layernorm to be set in the spec "
                f"to build `{module_name}`."
            )
        return module_spec

    def _raise_unused_q_norm(self) -> NoReturn:
        """Report an explicit Q norm that has no Q-LoRA projection to consume it."""
        help_msg = ""
        if not self._is_fused_norm_linear(self.submodules.linear_q_proj):
            help_msg = (
                f"Please use a fused norm+linear for "
                f"`linear_q_proj={self.submodules.linear_q_proj}` if "
                f"you intend to have a Q-norm."
            )
        raise ValueError(
            f"`q_layernorm={self.submodules.q_layernorm}` is non-trivial, "
            f"but `q_lora_rank is None`, meaning it will not be used."
            f"{help_msg}"
        )

    def _is_fused_norm_linear(self, module_spec) -> bool:
        """Return whether a module specification selects the backend fused projection."""
        module_cls = module_spec.module if isinstance(module_spec, ModuleSpec) else module_spec
        return self.fused_norm_linear_impl is not None and module_cls is self.fused_norm_linear_impl

    @staticmethod
    def _is_trivial(module_spec) -> bool:
        """Return whether a norm slot is unset or explicitly an identity operation."""
        return module_spec in (None, IdentityOp)

    @classmethod
    def _default_if_trivial(cls, module_spec, default):
        """Replace an unset or identity specification with the supplied default."""
        if cls._is_trivial(module_spec):
            return default
        return module_spec

    @staticmethod
    def _result(
        *, linear_q_proj, linear_q_up_proj, linear_kv_up_proj, q_layernorm, kv_layernorm
    ) -> _QKNormResolvedConfig:
        """Package the resolved Q/KV norms and projections in the caller's schema."""
        return dict(
            linear_q_proj=linear_q_proj,
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_up_proj=linear_kv_up_proj,
            q_layernorm=q_layernorm,
            kv_layernorm=kv_layernorm,
        )
