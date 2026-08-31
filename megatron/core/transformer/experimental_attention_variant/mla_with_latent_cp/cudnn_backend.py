# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Direct public cuDNN Frontend Graph backend for MLA latent CP."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Final

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend

from .layout import PhaseSpec
from .utils import (
    BackendNotQualifiedError,
    BackendPlanNotSupportedError,
    QualifiedBackendTuple,
    _require,
    cudnn_backward_proxy,
)


class _CudnnUid(IntEnum):
    Q = 0
    K = 1
    V = 2
    O = 3
    STATS = 4
    DQ = 5
    DK = 6
    DV = 7
    DO = 8
    SEQ_Q = 9
    SEQ_KV = 10
    Q_OFFSET = 11
    K_OFFSET = 12
    V_OFFSET = 13
    O_OFFSET = 14
    STATS_OFFSET = 15


@dataclass(frozen=True)
class _CudnnPlanKey:
    process_id: int
    device_index: int
    frontend_version: str
    runtime_version: str
    dtype: torch.dtype
    sm: tuple[int, int]
    batch: int
    heads: int
    qk_dim: int
    v_dim: int
    max_q: int
    max_kv: int
    capacity_q: int
    capacity_kv: int
    causal: bool
    scale: float


@dataclass(frozen=True)
class _CudnnPlan:
    forward_graph: Any
    backward_graph: Any
    key: _CudnnPlanKey


@dataclass(frozen=True)
class _CudnnBindingKey:
    """Identity of immutable phase metadata prepared before layer execution."""

    cu_q_id: int
    cu_kv_id: int
    dtype: torch.dtype
    heads: int
    qk_dim: int
    v_dim: int
    max_q: int
    max_kv: int
    causal: bool
    scale: float


@dataclass(frozen=True)
class _CudnnBinding:
    """Prepared plan and canonical ragged buffers for one phase metadata pair."""

    plan: _CudnnPlan
    metadata: dict[_CudnnUid, Tensor]
    cu_q: Tensor
    cu_kv: Tensor
    total_q: int
    total_kv: int


def _packed_bshd_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Pinned cuDNN Frontend ragged BSHD stride for a logical BHSD shape."""

    _, heads, sequence, dimension = shape
    return (sequence * heads * dimension, dimension, heads * dimension, 1)


def _aligned_token_capacity(tokens: int) -> int:
    return max(64, ((tokens + 63) // 64) * 64)


def _pad_token_rows(tensor: Tensor, capacity: int) -> Tensor:
    _require(tensor.size(0) <= capacity, "cuDNN token capacity is too small")
    if tensor.size(0) == capacity and tensor.is_contiguous():
        return tensor
    padded = torch.empty(
        (capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.size(0)].copy_(tensor)
    return padded


class _CudnnSDPAFunction(torch.autograd.Function):
    """Public cuDNN Graph SDPA with explicit support for the outer LSE gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
        adapter: "CudnnFusedAttentionAdapter",
    ) -> tuple[Tensor, Tensor]:
        """Execute public cuDNN Graph SDPA and save its backward inputs."""
        raw_output, stats = adapter._execute_forward(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale
        )
        ctx.save_for_backward(q, k, v, raw_output, stats, cu_q, cu_kv)
        ctx.max_q = max_q
        ctx.max_kv = max_kv
        ctx.causal = causal
        ctx.scale = scale
        ctx.adapter = adapter
        return raw_output.float(), stats.float()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor | None, grad_lse: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None, None, None, None]:
        """Execute corrected public cuDNN Graph SDPA backward."""
        q, k, v, raw_output, stats, cu_q, cu_kv = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(raw_output, dtype=torch.float32)
        if grad_lse is None:
            grad_lse = torch.zeros(
                raw_output.shape[:2], dtype=torch.float32, device=raw_output.device
            )
        corrected_output, local_grad_output = cudnn_backward_proxy(
            raw_output, grad_output, grad_lse
        )
        dq, dk, dv = ctx.adapter._execute_backward(
            q,
            k,
            v,
            corrected_output.to(torch.bfloat16),
            local_grad_output.to(torch.bfloat16),
            stats,
            cu_q,
            cu_kv,
            ctx.max_q,
            ctx.max_kv,
            ctx.causal,
            ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class _CudnnRecomputedPhaseFunction(torch.autograd.Function):
    """Run cuDNN SDPA while recomputing only latent-KV expansion in backward.

    Retaining cuDNN's local output and statistics avoids replaying the attention
    forward while still discarding expanded K/V at the phase boundary.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        payload: Tensor,
        phase: PhaseSpec,
        scale: float,
        adapter: "CudnnFusedAttentionAdapter",
        expand_phase_kv: Callable[[Tensor, PhaseSpec], tuple[Tensor, Tensor]],
        *projection_parameters: Tensor,
    ) -> tuple[Tensor, Tensor]:
        key, value = expand_phase_kv(payload, phase)
        raw_output, stats = adapter._execute_forward(
            query,
            key,
            value,
            phase.cu_seqlens_q,
            phase.cu_seqlens_kv,
            phase.max_seqlen_q,
            phase.max_seqlen_kv,
            phase.causal,
            scale,
        )
        ctx.save_for_backward(query, payload, raw_output, stats, *projection_parameters)
        ctx.phase = phase
        ctx.scale = scale
        ctx.adapter = adapter
        ctx.expand_phase_kv = expand_phase_kv
        return raw_output.float(), stats.float()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor | None, grad_lse: Tensor | None
    ) -> tuple[Any, ...]:
        query, payload, raw_output, stats, *projection_parameters = ctx.saved_tensors
        _require(
            projection_parameters, "latent-KV projection has no trainable parameters"
        )
        phase = ctx.phase
        if grad_output is None:
            grad_output = torch.zeros_like(raw_output, dtype=torch.float32)
        if grad_lse is None:
            grad_lse = torch.zeros(
                raw_output.shape[:2], dtype=torch.float32, device=raw_output.device
            )

        with torch.enable_grad():
            replay_payload = payload.detach().requires_grad_(True)
            key, value = ctx.expand_phase_kv(replay_payload, phase)
            corrected_output, local_grad_output = cudnn_backward_proxy(
                raw_output, grad_output, grad_lse
            )
            dq, dk, dv = ctx.adapter._execute_backward(
                query,
                key,
                value,
                corrected_output.to(torch.bfloat16),
                local_grad_output.to(torch.bfloat16),
                stats,
                phase.cu_seqlens_q,
                phase.cu_seqlens_kv,
                phase.max_seqlen_q,
                phase.max_seqlen_kv,
                phase.causal,
                ctx.scale,
            )
            projection_gradients = torch.autograd.grad(
                (key, value),
                (replay_payload, *projection_parameters),
                grad_outputs=(dk, dv),
                allow_unused=True,
            )
            grad_payload, *grad_parameters = projection_gradients
        _require(grad_payload is not None, "latent-KV replay lost its payload gradient")
        return dq, grad_payload, None, None, None, None, *grad_parameters


def _resolve_cudnn_frontend_version(cudnn: Any) -> str:
    try:
        version = importlib.metadata.version("nvidia-cudnn-frontend")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(cudnn, "__version__", None)
    if version is None:
        raise BackendNotQualifiedError(
            "cuDNN Frontend package version metadata is missing"
        )
    return str(version)


class CudnnFusedAttentionAdapter:
    """Direct public cuDNN Frontend Graph adapter for fully ragged BF16 SDPA.

    One adapter (and therefore one public cuDNN handle and plan cache) is shared by all latent-CP
    layers in a process on one CUDA device. Plan building and handle stream mutation are protected
    by a reentrant lock. Invocation workspaces and graph variant packs remain call-local.
    """

    def __init__(
        self,
        expected_identity: QualifiedBackendTuple | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        try:
            self.cudnn = importlib.import_module("cudnn")
        except ImportError as error:
            raise BackendNotQualifiedError(
                "fused latent CP requires the public nvidia-cudnn-frontend package"
            ) from error

        self.process_id = os.getpid()
        self.device_index = (
            torch.cuda.current_device() if device_index is None else int(device_index)
        )
        self.frontend_version = _resolve_cudnn_frontend_version(self.cudnn)
        self.runtime_version = str(self.cudnn.backend_version_string())
        with torch.cuda.device(self.device_index):
            capability = torch.cuda.get_device_capability(self.device_index)
            self._handle = self.cudnn.create_handle()
        self.identity: QualifiedBackendTuple = (
            AttnBackend.fused,
            self.frontend_version,
            self.runtime_version,
            capability,
        )
        if expected_identity is not None and self.identity != expected_identity:
            self.cudnn.destroy_handle(self._handle)
            self._handle = None
            raise BackendNotQualifiedError(
                f"cuDNN adapter identity {self.identity!r} != {expected_identity!r}"
            )
        self._plans: dict[_CudnnPlanKey, _CudnnPlan] = {}
        self._bindings: OrderedDict[_CudnnBindingKey, _CudnnBinding] = OrderedDict()
        self._workspaces: dict[tuple[_CudnnPlanKey, int, bool], Tensor] = {}
        self._execution_lock = threading.RLock()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        cudnn = getattr(self, "cudnn", None)
        if (
            handle is not None
            and cudnn is not None
            and getattr(self, "process_id", None) == os.getpid()
        ):
            try:
                with torch.cuda.device(self.device_index):
                    cudnn.destroy_handle(handle)
            except Exception:
                # Interpreter shutdown may unload CUDA before module destruction.
                pass
            self._handle = None

    def _assert_bound_device(self, device: torch.device) -> None:
        _require(
            os.getpid() == self.process_id,
            "a forked process cannot reuse a cuDNN adapter",
        )
        _require(device.type == "cuda", "cuDNN plans require a CUDA device")
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        _require(
            device_index == self.device_index,
            "cuDNN adapter/handle cannot be reused across CUDA devices",
        )

    def _plan_key_from_metadata(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cu_q: Tensor,
        cu_kv: Tensor,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> _CudnnPlanKey:
        self._assert_bound_device(device)
        _require(dtype == torch.bfloat16, "cuDNN adapter supports BF16 only")
        _require(
            cu_q.device == device and cu_kv.device == device,
            "cuDNN metadata must be on the bound CUDA device",
        )
        total_q = int(cu_q[-1].item())
        total_kv = int(cu_kv[-1].item())
        _require(total_q > 0 and total_kv > 0, "cuDNN phase totals must be positive")
        return _CudnnPlanKey(
            process_id=self.process_id,
            device_index=self.device_index,
            frontend_version=self.frontend_version,
            runtime_version=self.runtime_version,
            dtype=dtype,
            sm=torch.cuda.get_device_capability(self.device_index),
            batch=cu_q.numel() - 1,
            heads=num_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            max_q=max_q,
            max_kv=max_kv,
            capacity_q=_aligned_token_capacity(total_q),
            capacity_kv=_aligned_token_capacity(total_kv),
            causal=causal,
            scale=float(scale),
        )

    @staticmethod
    def _metadata(
        cu_q: Tensor, cu_kv: Tensor, heads: int, qk_dim: int, v_dim: int
    ) -> dict[_CudnnUid, Tensor]:
        def canonical_rank4(tensor: Tensor) -> Tensor:
            return tensor.contiguous().view(-1, 1, 1, 1)

        return {
            _CudnnUid.SEQ_Q: canonical_rank4((cu_q[1:] - cu_q[:-1]).to(torch.int32)),
            _CudnnUid.SEQ_KV: canonical_rank4((cu_kv[1:] - cu_kv[:-1]).to(torch.int32)),
            _CudnnUid.Q_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads * qk_dim),
            _CudnnUid.K_OFFSET: canonical_rank4(cu_kv.to(torch.int64) * heads * qk_dim),
            _CudnnUid.V_OFFSET: canonical_rank4(cu_kv.to(torch.int64) * heads * v_dim),
            _CudnnUid.O_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads * v_dim),
            _CudnnUid.STATS_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads),
        }

    def _new_graph(self, key: _CudnnPlanKey) -> Any:
        _require(
            key.process_id == self.process_id
            and key.device_index == self.device_index
            and key.frontend_version == self.frontend_version
            and key.runtime_version == self.runtime_version
            and key.dtype == torch.bfloat16,
            "cuDNN plan key is not bound to this adapter identity",
        )
        with torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(self.device_index).cuda_stream,
            )
            return self.cudnn.pygraph(
                io_data_type=self.cudnn.data_type.BFLOAT16,
                intermediate_data_type=self.cudnn.data_type.FLOAT,
                compute_data_type=self.cudnn.data_type.FLOAT,
                handle=self._handle,
                sm_version=key.sm[0] * 10 + key.sm[1],
            )

    def _ragged_descriptors(
        self, graph: Any, key: _CudnnPlanKey, *, backward: bool
    ) -> dict[_CudnnUid, Any]:
        q_shape = (key.batch, key.heads, key.max_q, key.qk_dim)
        k_shape = (key.batch, key.heads, key.max_kv, key.qk_dim)
        v_shape = (key.batch, key.heads, key.max_kv, key.v_dim)
        o_shape = (key.batch, key.heads, key.max_q, key.v_dim)
        stats_shape = (key.batch, key.heads, key.max_q, 1)
        tensor_specs: list[tuple[_CudnnUid, tuple[int, ...], tuple[int, ...], Any]] = [
            (
                _CudnnUid.Q,
                q_shape,
                _packed_bshd_stride(q_shape),
                self.cudnn.data_type.BFLOAT16,
            ),
            (
                _CudnnUid.K,
                k_shape,
                _packed_bshd_stride(k_shape),
                self.cudnn.data_type.BFLOAT16,
            ),
            (
                _CudnnUid.V,
                v_shape,
                _packed_bshd_stride(v_shape),
                self.cudnn.data_type.BFLOAT16,
            ),
        ]
        if backward:
            tensor_specs.extend(
                [
                    (
                        _CudnnUid.O,
                        o_shape,
                        _packed_bshd_stride(o_shape),
                        self.cudnn.data_type.BFLOAT16,
                    ),
                    (
                        _CudnnUid.DO,
                        o_shape,
                        _packed_bshd_stride(o_shape),
                        self.cudnn.data_type.BFLOAT16,
                    ),
                    (
                        _CudnnUid.STATS,
                        stats_shape,
                        _packed_bshd_stride(stats_shape),
                        self.cudnn.data_type.FLOAT,
                    ),
                ]
            )
        tensors = {
            uid: graph.tensor(uid=int(uid), dim=shape, stride=stride, data_type=dtype)
            for uid, shape, stride, dtype in tensor_specs
        }
        for uid in (
            _CudnnUid.Q_OFFSET,
            _CudnnUid.K_OFFSET,
            _CudnnUid.V_OFFSET,
            _CudnnUid.O_OFFSET,
            _CudnnUid.STATS_OFFSET,
        ):
            tensors[uid] = graph.tensor(
                uid=int(uid),
                dim=(key.batch + 1, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=self.cudnn.data_type.INT64,
            )
        tensors[_CudnnUid.SEQ_Q] = graph.tensor(
            uid=int(_CudnnUid.SEQ_Q),
            dim=(key.batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=self.cudnn.data_type.INT32,
        )
        tensors[_CudnnUid.SEQ_KV] = graph.tensor(
            uid=int(_CudnnUid.SEQ_KV),
            dim=(key.batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=self.cudnn.data_type.INT32,
        )
        tensors[_CudnnUid.Q].set_ragged_offset(tensors[_CudnnUid.Q_OFFSET])
        tensors[_CudnnUid.K].set_ragged_offset(tensors[_CudnnUid.K_OFFSET])
        tensors[_CudnnUid.V].set_ragged_offset(tensors[_CudnnUid.V_OFFSET])
        if backward:
            tensors[_CudnnUid.O].set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
            tensors[_CudnnUid.DO].set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
            tensors[_CudnnUid.STATS].set_ragged_offset(tensors[_CudnnUid.STATS_OFFSET])
        return tensors

    def _build_forward_graph(self, key: _CudnnPlanKey) -> Any:
        graph = self._new_graph(key)
        tensors = self._ragged_descriptors(graph, key, backward=False)
        output, stats = graph.sdpa(
            name="mla_latent_cp_sdpa_forward",
            q=tensors[_CudnnUid.Q],
            k=tensors[_CudnnUid.K],
            v=tensors[_CudnnUid.V],
            generate_stats=True,
            attn_scale=key.scale,
            use_padding_mask=True,
            seq_len_q=tensors[_CudnnUid.SEQ_Q],
            seq_len_kv=tensors[_CudnnUid.SEQ_KV],
            diagonal_band_right_bound=0 if key.causal else None,
            diagonal_alignment=self.cudnn.diagonal_alignment.TOP_LEFT,
        )
        o_shape = (key.batch, key.heads, key.max_q, key.v_dim)
        output.set_uid(int(_CudnnUid.O)).set_output(True).set_dim(o_shape).set_stride(
            _packed_bshd_stride(o_shape)
        )
        output.set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
        stats_shape = (key.batch, key.heads, key.max_q, 1)
        stats.set_uid(int(_CudnnUid.STATS)).set_output(True).set_data_type(
            self.cudnn.data_type.FLOAT
        ).set_dim(stats_shape).set_stride(_packed_bshd_stride(stats_shape))
        stats.set_ragged_offset(tensors[_CudnnUid.STATS_OFFSET])
        self._build_graph(graph)
        return graph

    def _build_backward_graph(self, key: _CudnnPlanKey) -> Any:
        graph = self._new_graph(key)
        tensors = self._ragged_descriptors(graph, key, backward=True)
        dq, dk, dv = graph.sdpa_backward(
            name="mla_latent_cp_sdpa_backward",
            q=tensors[_CudnnUid.Q],
            k=tensors[_CudnnUid.K],
            v=tensors[_CudnnUid.V],
            o=tensors[_CudnnUid.O],
            dO=tensors[_CudnnUid.DO],
            stats=tensors[_CudnnUid.STATS],
            attn_scale=key.scale,
            use_padding_mask=True,
            seq_len_q=tensors[_CudnnUid.SEQ_Q],
            seq_len_kv=tensors[_CudnnUid.SEQ_KV],
            max_total_seq_len_q=key.capacity_q,
            max_total_seq_len_kv=key.capacity_kv,
            diagonal_band_right_bound=0 if key.causal else None,
            diagonal_alignment=self.cudnn.diagonal_alignment.TOP_LEFT,
        )
        q_shape = (key.batch, key.heads, key.max_q, key.qk_dim)
        k_shape = (key.batch, key.heads, key.max_kv, key.qk_dim)
        v_shape = (key.batch, key.heads, key.max_kv, key.v_dim)
        for tensor, uid, shape, offset_uid in (
            (dq, _CudnnUid.DQ, q_shape, _CudnnUid.Q_OFFSET),
            (dk, _CudnnUid.DK, k_shape, _CudnnUid.K_OFFSET),
            (dv, _CudnnUid.DV, v_shape, _CudnnUid.V_OFFSET),
        ):
            tensor.set_uid(int(uid)).set_output(True).set_dim(shape).set_stride(
                _packed_bshd_stride(shape)
            )
            tensor.set_ragged_offset(tensors[offset_uid])
        self._build_graph(graph)
        return graph

    def _build_graph(self, graph: Any) -> None:
        try:
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans(
                [self.cudnn.heur_mode.A, self.cudnn.heur_mode.FALLBACK]
            )
            graph.check_support()
            graph.build_plans()
        except self.cudnn.cudnnGraphNotSupportedError as error:
            raise BackendPlanNotSupportedError(str(error)) from error

    def _prepare_plan(self, key: _CudnnPlanKey) -> _CudnnPlan:
        with self._execution_lock:
            plan = self._plans.get(key)
            if plan is None:
                plan = _CudnnPlan(
                    forward_graph=self._build_forward_graph(key),
                    backward_graph=self._build_backward_graph(key),
                    key=key,
                )
                self._plans[key] = plan
            return plan

    def _get_prepared_plan(self, key: _CudnnPlanKey) -> _CudnnPlan:
        with self._execution_lock:
            plan = self._plans.get(key)
        if plan is None:
            raise BackendPlanNotSupportedError(
                "cuDNN phase plan was not prepared before transformer block execution"
            )
        return plan

    @staticmethod
    def _binding_key(
        *,
        cu_q: Tensor,
        cu_kv: Tensor,
        dtype: torch.dtype,
        heads: int,
        qk_dim: int,
        v_dim: int,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> _CudnnBindingKey:
        _require(
            cu_q.dtype == cu_kv.dtype == torch.int32,
            "cuDNN cumulative lengths must be INT32",
        )
        _require(
            cu_q.is_contiguous() and cu_kv.is_contiguous(),
            "cuDNN cumulative lengths must be contiguous",
        )
        return _CudnnBindingKey(
            cu_q_id=id(cu_q),
            cu_kv_id=id(cu_kv),
            dtype=dtype,
            heads=heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            max_q=max_q,
            max_kv=max_kv,
            causal=causal,
            scale=float(scale),
        )

    def _get_binding(self, key: _CudnnBindingKey) -> _CudnnBinding | None:
        with self._execution_lock:
            binding = self._bindings.get(key)
            if binding is not None:
                self._bindings.move_to_end(key)
            return binding

    def _remember_binding(self, key: _CudnnBindingKey, binding: _CudnnBinding) -> None:
        with self._execution_lock:
            self._bindings[key] = binding
            self._bindings.move_to_end(key)
            while len(self._bindings) > 128:
                self._bindings.popitem(last=False)

    def _require_binding(self, key: _CudnnBindingKey) -> _CudnnBinding:
        binding = self._get_binding(key)
        if binding is None:
            raise BackendPlanNotSupportedError(
                "cuDNN phase metadata was not prepared before transformer block execution"
            )
        return binding

    def _workspace(
        self, plan: _CudnnPlan, *, backward: bool, device: torch.device
    ) -> Tensor:
        stream_id = int(torch.cuda.current_stream(self.device_index).cuda_stream)
        cache_key = (plan.key, stream_id, backward)
        with self._execution_lock:
            workspace = self._workspaces.get(cache_key)
            if workspace is None:
                graph = plan.backward_graph if backward else plan.forward_graph
                workspace = torch.empty(
                    graph.get_workspace_size(), dtype=torch.uint8, device=device
                )
                self._workspaces[cache_key] = workspace
            return workspace

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Build or reuse public cuDNN Graph plans and ragged bindings."""
        device = torch.device("cuda", self.device_index)
        for phase in phases:
            binding_key = self._binding_key(
                cu_q=phase.cu_seqlens_q,
                cu_kv=phase.cu_seqlens_kv,
                dtype=torch.bfloat16,
                heads=num_heads,
                qk_dim=qk_dim,
                v_dim=v_dim,
                max_q=phase.max_seqlen_q,
                max_kv=phase.max_seqlen_kv,
                causal=phase.causal,
                scale=scale,
            )
            if self._get_binding(binding_key) is not None:
                continue
            key = self._plan_key_from_metadata(
                device=device,
                dtype=torch.bfloat16,
                cu_q=phase.cu_seqlens_q,
                cu_kv=phase.cu_seqlens_kv,
                num_heads=num_heads,
                qk_dim=qk_dim,
                v_dim=v_dim,
                max_q=phase.max_seqlen_q,
                max_kv=phase.max_seqlen_kv,
                causal=phase.causal,
                scale=scale,
            )
            plan = self._prepare_plan(key)
            self._remember_binding(
                binding_key,
                _CudnnBinding(
                    plan=plan,
                    metadata=self._metadata(
                        phase.cu_seqlens_q,
                        phase.cu_seqlens_kv,
                        num_heads,
                        qk_dim,
                        v_dim,
                    ),
                    cu_q=phase.cu_seqlens_q,
                    cu_kv=phase.cu_seqlens_kv,
                    total_q=phase.q_indices.numel(),
                    total_kv=phase.kv_indices.numel(),
                ),
            )

    def _execution_binding(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> _CudnnBinding:
        _require(
            q.dtype == k.dtype == v.dtype == torch.bfloat16,
            "cuDNN Q/K/V must all be BF16",
        )
        _require(
            q.device == k.device == v.device, "cuDNN Q/K/V must use one CUDA device"
        )
        _require(
            q.ndim == k.ndim == v.ndim == 3
            and q.size(1) == k.size(1) == v.size(1)
            and q.size(2) == k.size(2),
            "invalid cuDNN THD Q/K/V shapes",
        )
        self._assert_bound_device(q.device)
        _require(
            cu_q.device == q.device and cu_kv.device == q.device,
            "cuDNN metadata must be on the Q/K/V device",
        )
        binding_key = self._binding_key(
            cu_q=cu_q,
            cu_kv=cu_kv,
            dtype=q.dtype,
            heads=q.size(1),
            qk_dim=q.size(2),
            v_dim=v.size(2),
            max_q=max_q,
            max_kv=max_kv,
            causal=causal,
            scale=scale,
        )
        binding = self._require_binding(binding_key)
        _require(
            binding.cu_q is cu_q
            and binding.cu_kv is cu_kv
            and q.size(0) == binding.total_q
            and k.size(0) == v.size(0) == binding.total_kv,
            "cuDNN tensors disagree with prepared cumulative lengths",
        )
        return binding

    def _execute_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        binding = self._execution_binding(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale
        )
        plan = binding.plan
        key = plan.key
        metadata = binding.metadata
        q_buffer = _pad_token_rows(q, key.capacity_q)
        k_buffer = _pad_token_rows(k, key.capacity_kv)
        v_buffer = _pad_token_rows(v, key.capacity_kv)
        o_buffer = torch.empty(
            (key.capacity_q, key.heads, key.v_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
        stats_buffer = torch.empty(
            (key.capacity_q, key.heads, 1), dtype=torch.float32, device=q.device
        )
        pack = {
            int(_CudnnUid.Q): q_buffer,
            int(_CudnnUid.K): k_buffer,
            int(_CudnnUid.V): v_buffer,
            int(_CudnnUid.O): o_buffer,
            int(_CudnnUid.STATS): stats_buffer,
            **{int(uid): tensor for uid, tensor in metadata.items()},
        }
        workspace = self._workspace(plan, backward=False, device=q.device)
        with self._execution_lock, torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(self.device_index).cuda_stream,
            )
            plan.forward_graph.execute(pack, workspace, self._handle)
        return o_buffer[: q.size(0)], stats_buffer[: q.size(0), :, 0]

    def _execute_backward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        grad_output: Tensor,
        stats: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        binding = self._execution_binding(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale
        )
        plan = binding.plan
        key = plan.key
        metadata = binding.metadata
        q_buffer = _pad_token_rows(q, key.capacity_q)
        k_buffer = _pad_token_rows(k, key.capacity_kv)
        v_buffer = _pad_token_rows(v, key.capacity_kv)
        o_buffer = _pad_token_rows(output, key.capacity_q)
        do_buffer = _pad_token_rows(grad_output, key.capacity_q)
        stats_buffer = _pad_token_rows(stats.unsqueeze(-1), key.capacity_q)
        dq_buffer = torch.empty_like(q_buffer)
        dk_buffer = torch.empty_like(k_buffer)
        dv_buffer = torch.empty_like(v_buffer)
        pack = {
            int(_CudnnUid.Q): q_buffer,
            int(_CudnnUid.K): k_buffer,
            int(_CudnnUid.V): v_buffer,
            int(_CudnnUid.O): o_buffer,
            int(_CudnnUid.DO): do_buffer,
            int(_CudnnUid.STATS): stats_buffer,
            int(_CudnnUid.DQ): dq_buffer,
            int(_CudnnUid.DK): dk_buffer,
            int(_CudnnUid.DV): dv_buffer,
            **{int(uid): tensor for uid, tensor in metadata.items()},
        }
        workspace = self._workspace(plan, backward=True, device=q.device)
        with self._execution_lock, torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle,
                stream=torch.cuda.current_stream(self.device_index).cuda_stream,
            )
            plan.backward_graph.execute(pack, workspace, self._handle)
        return dq_buffer[: q.size(0)], dk_buffer[: k.size(0)], dv_buffer[: v.size(0)]

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Execute one phase through the differentiable cuDNN Graph wrapper."""
        return _CudnnSDPAFunction.apply(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale, self
        )

    def forward_recomputed_phase(
        self,
        query: Tensor,
        payload: Tensor,
        phase: PhaseSpec,
        scale: float,
        expand_phase_kv: Callable[[Tensor, PhaseSpec], tuple[Tensor, Tensor]],
        *projection_parameters: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Execute SDPA while retaining only latent payload plus cuDNN O/LSE state."""

        return _CudnnRecomputedPhaseFunction.apply(
            query,
            payload,
            phase,
            scale,
            self,
            expand_phase_kv,
            *projection_parameters,
        )


_CUDNN_ADAPTER_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_CUDNN_ADAPTER_CACHE: dict[
    tuple[int, int, QualifiedBackendTuple], CudnnFusedAttentionAdapter
] = {}


def _shared_cudnn_adapter(
    runtime_identity: QualifiedBackendTuple,
) -> CudnnFusedAttentionAdapter:
    """Return the process/device-scoped adapter shared by every local MLA layer."""

    cache_key = (os.getpid(), torch.cuda.current_device(), runtime_identity)
    with _CUDNN_ADAPTER_CACHE_LOCK:
        adapter = _CUDNN_ADAPTER_CACHE.get(cache_key)
        if adapter is None:
            adapter = CudnnFusedAttentionAdapter(
                runtime_identity, device_index=cache_key[1]
            )
            _CUDNN_ADAPTER_CACHE[cache_key] = adapter
        return adapter
