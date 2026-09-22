from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.model import (
    DeepseekV4Layer as LiteDeepseekV4Layer,
    DeepseekV4Model as LiteDeepseekV4Model,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
    VLLMAttention,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime import (
    AttentionKernelMetadata,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.logprob import (
    aligned_selected_log_probs,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    mhc_kernel,
    mhc_head,
    mhc_post,
    mhc_pre_broadcast,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import DeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import rms_norm
from megatron.lite.primitive.modules.attention.hca import HyperConnection
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.mhc import (
    fold_mhc_hidden_for_pipeline,
    unfold_mhc_hidden_from_pipeline,
)

class DeepseekV4Layer(LiteDeepseekV4Layer):
    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        layer_idx: int = 0,
        *,
        indexer_loss_coeff: float = 0.0,
        cache_deployment_weights: bool = False,
        moe_token_dispatcher_type: str = "deepep",
        hybridep_max_tokens_per_rank: int | None = None,
    ):
        self.config = config
        self._vllm_indexer_loss_coeff = indexer_loss_coeff
        self._cache_deployment_weights = cache_deployment_weights
        self._moe_token_dispatcher_type = moe_token_dispatcher_type
        self._hybridep_max_tokens_per_rank = hybridep_max_tokens_per_rank
        super().__init__(
            config,
            ps or ParallelState(),
            layer_idx,
            use_deepep=True,
        )

    def _build_attention(
        self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState
    ) -> nn.Module:
        return VLLMAttention(
            config,
            ps=ps,
            layer_idx=layer_idx,
            indexer_loss_coeff=self._vllm_indexer_loss_coeff,
            cache_deployment_weights=self._cache_deployment_weights,
        )

    def _build_moe(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        *,
        layer_idx: int,
        use_deepep: bool,
    ) -> nn.Module:
        del use_deepep
        return DeepseekV4MoE(
            config,
            ps,
            layer_idx=layer_idx,
            cache_deployment_weights=self._cache_deployment_weights,
            moe_token_dispatcher_type=self._moe_token_dispatcher_type,
            hybridep_max_tokens_per_rank=self._hybridep_max_tokens_per_rank,
        )

    def _mhc_pre(
        self,
        hidden_states: torch.Tensor,
        hc: HyperConnection,
        norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = hc.fn.float().contiguous()
        scale = hc.scale.float().contiguous()
        base = hc.base.float().contiguous()
        broadcast = hidden_states.ndim == 2
        kernel = "pre_broadcast" if broadcast else "pre"

        def visible_pre(hidden, fn_, scale_, base_, norm_weight_):
            common = (
                hidden,
                fn_,
                scale_,
                base_,
                self.config.rms_norm_eps,
                self.config.hc_eps,
                self.config.hc_eps,
                2.0,
                self.config.hc_sinkhorn_iters,
            )
            kwargs = {
                "norm_weight": norm_weight_,
                "norm_eps": self.config.rms_norm_eps,
            }
            if broadcast:
                kwargs["fn_broadcast"] = (
                    fn_.view(-1, self.config.hc_mult, self.config.hidden_size)
                    .sum(dim=1)
                    .contiguous()
                )
                return mhc_kernel(kernel, *common, **kwargs)
            return (hidden, *mhc_kernel(kernel, *common, **kwargs))

        return mhc_pre_broadcast(
            visible_pre,
            hidden_states,
            fn,
            scale,
            base,
            norm_weight,
            mult=self.config.hc_mult,
            iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
            norm_eps=self.config.rms_norm_eps,
        )

    @staticmethod
    def _mhc_post(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_post(
            lambda *args: mhc_kernel("post", *args),
            hidden_states,
            residual,
            post_mix,
            res_mix,
        )

    def _attention_block(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        packed_seq_params: Any,
        metadata: AttentionKernelMetadata | None = None,
    ) -> torch.Tensor:
        del position_ids, packed_seq_params
        residual, post_mix, res_mix, hidden_states = self._mhc_pre(
            x, self.attn_hc, self.input_layernorm.weight
        )
        hidden_states = self.self_attn(hidden_states, metadata=metadata)
        return self._mhc_post(hidden_states, residual, post_mix, res_mix)

    def _mlp_block(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual, post_mix, res_mix, hidden_states = self._mhc_pre(
            x, self.ffn_hc, self.post_attention_layernorm.weight
        )
        hidden_states = self.mlp(hidden_states, input_ids=input_ids)
        return self._mhc_post(hidden_states, residual, post_mix, res_mix)


class DeepseekV4Model(LiteDeepseekV4Model):
    def __init__(
        self,
        config: DeepseekV4Config,
        train_config=None,
        ps=None,
        *,
        indexer_loss_coeff: float = 0.0,
        logprob_chunk_size: int = 8192,
        cache_deployment_weights: bool = False,
        moe_token_dispatcher_type: str = "deepep",
        hybridep_max_tokens_per_rank: int | None = None,
    ):
        ps = ps or ParallelState()
        self._vllm_indexer_loss_coeff = indexer_loss_coeff
        self._cache_deployment_weights = cache_deployment_weights
        self._moe_token_dispatcher_type = moe_token_dispatcher_type
        self._hybridep_max_tokens_per_rank = hybridep_max_tokens_per_rank
        if logprob_chunk_size <= 0:
            raise ValueError("logprob_chunk_size must be positive")
        self._logprob_chunk_size = int(logprob_chunk_size)
        train_config = train_config or SimpleNamespace(vpp=1, fp8=False)
        super().__init__(
            config,
            train_config,
            ps,
            mtp_enable=False,
            use_deepep=True,
        )
        # Match the release BF16-master/FP32-coefficient dtype boundary.
        self.to(torch.bfloat16)
        fp32_suffixes = (
            ".attn_hc.fn",
            ".attn_hc.base",
            ".attn_hc.scale",
            ".ffn_hc.fn",
            ".ffn_hc.base",
            ".ffn_hc.scale",
            ".hc_fn",
            ".hc_base",
            ".hc_scale",
            ".sinks",
            ".ape",
            ".expert_bias",
        )
        for name, parameter in self.named_parameters():
            if name.endswith(fp32_suffixes):
                parameter.data = parameter.data.float()
        self._shared_projection_streams: list[torch.cuda.Stream] | None = None

    def _build_layer(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool,
    ) -> nn.Module:
        del use_deepep
        return DeepseekV4Layer(
            config,
            ps,
            layer_idx=layer_idx,
            indexer_loss_coeff=self._vllm_indexer_loss_coeff,
            cache_deployment_weights=self._cache_deployment_weights,
            moe_token_dispatcher_type=self._moe_token_dispatcher_type,
            hybridep_max_tokens_per_rank=self._hybridep_max_tokens_per_rank,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_metadata: dict[
            int, AttentionKernelMetadata
        ] | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        pipeline_streams = False
        if hidden_states is None:
            if not self.pre_process:
                hidden_states = self._input_tensor
                if hidden_states is not None:
                    hidden_states = unfold_mhc_hidden_from_pipeline(
                        hidden_states, hc_mult=self.config.hc_mult
                    ).reshape(-1, self.config.hc_mult, self.config.hidden_size)
                    pipeline_streams = True
            elif input_ids is not None:
                assert self.embed_tokens is not None
                hidden_states = self.embed_tokens.embedding(input_ids)
        if hidden_states is None:
            raise ValueError("input_ids or hidden_states is required.")
        if not hidden_states.is_cuda:
            raise RuntimeError("DeepSeek V4 vLLM training requires CUDA tensors")
        if hidden_states.ndim != 2 and not pipeline_streams:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self._shared_projection_streams is None:
            self._shared_projection_streams = [torch.cuda.Stream() for _ in range(3)]
        for layer in self.layers.values():
            layer.self_attn._projection_streams = self._shared_projection_streams
        for local_idx, layer_idx in enumerate(self.layer_indices):
            layer = self.layers[str(local_idx)]
            layer_attention_metadata = (
                None if attention_metadata is None else attention_metadata[layer_idx]
            )
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_metadata=layer_attention_metadata,
                input_ids=input_ids,
            )
        if not self.post_process:
            return {
                "hidden_states": fold_mhc_hidden_for_pipeline(
                    hidden_states.unsqueeze(1)
                )
            }
        if self.norm is None or self.hc_head is None or self.lm_head is None:
            raise RuntimeError("final pipeline stage is missing the output head")
        hidden_states = mhc_head(
            lambda *args: mhc_kernel(
                "head",
                *args, self.config.rms_norm_eps, self.config.hc_eps
            ),
            hidden_states,
            self.hc_head.hc_fn.float().contiguous(),
            self.hc_head.hc_scale.float().contiguous(),
            self.hc_head.hc_base.float().contiguous(),
            eps=self.config.hc_eps,
        )
        from vllm.model_executor.layers.batch_invariant import (
            rms_norm_batch_invariant,
        )

        hidden_states = rms_norm(
            rms_norm_batch_invariant,
            hidden_states,
            self.norm.weight,
            self.config.rms_norm_eps,
        )
        result = {"hidden_states": hidden_states}
        if labels is not None:
            temperature_value = float(
                temperature.detach().float().item()
                if isinstance(temperature, torch.Tensor)
                else temperature
            )
            if temperature_value <= 0:
                raise ValueError("temperature must be positive")
            flat_labels = labels.reshape(-1).long()
            if flat_labels.numel() != hidden_states.shape[0]:
                raise ValueError("labels must contain one target per visible token")
            selected_log_probs, entropy = aligned_selected_log_probs(
                hidden_states,
                self.lm_head,
                flat_labels,
                temperature_value,
                self._logprob_chunk_size,
                calculate_entropy=calculate_entropy,
            )
            token_loss = -selected_log_probs
            result["log_probs"] = selected_log_probs
            if calculate_entropy:
                assert entropy is not None
                result["entropy"] = entropy
            mask = (
                torch.ones_like(token_loss)
                if loss_mask is None
                else loss_mask.reshape(-1).to(token_loss.dtype)
            )
            denominator = mask.sum()
            if not bool(denominator > 0):
                raise ValueError("loss_mask must select at least one token")
            result["loss"] = (token_loss * mask).sum() / denominator
        return result
