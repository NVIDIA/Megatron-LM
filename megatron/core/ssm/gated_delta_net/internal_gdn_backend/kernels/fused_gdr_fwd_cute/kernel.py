# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
# This local mcore_gdn_opt CuTeDSL source is referred from FlashInfer commit e8d31317bedb4efd52559a2234f4cb9e83428cb9; keep the license above when updating.

Chunked Gated Delta Net (GDN) forward kernel for Blackwell SM100.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], gate[BT] (scalar gate), beta[BT] (scalar LR)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (compute warp group 0):
    cumsumlog[t]     = sum_{l=0}^{t} log(gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates
    T_pairwise[i,j]  = cumprod[i] / cumprod[j]  (i>=j)       inter-token transfer weights
    (stored in registers; 128 regs/thread)

  GEMM 1 - kk   : W_kk[BT,BT]  = K  @ K^T       (lower-triangular intra scores)
  GEMM 2 - qk   : W_qk[BT,BT]  = Q  @ K^T       (output attention scores)
  GEMM 3 - k*state : KS[BT,DV] = K  @ S_prev    (key applied to state)
  GEMM 4 - q*state : QS[BT,DV] = Q  @ S_prev    (inter-chunk output, before T scaling)
  GEMM 5 - new v   : NV[BT,DV] = A_inv @ V       (corrected value vectors)
                      where A_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*beta[i]*W_kk[i,j]  (lower-tri, hierarchical blockwise inverse)
  GEMM 6 - qkv  : O_intra[BT,DV] = W_qkv @ NV   (intra-chunk output)
                   where W_qkv = T*beta*W_qk (scaled qk scores)
  GEMM 7 - kv update : dS[DK,DV] = K^T @ delta       (state update, BT contraction)
                        where delta[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    O[BT,DV]  = O_intra + T_col * QS             (combine intra + inter)
    S_next    = cumprod[BT-1] * S_prev + dS        (update state in TMEM)

SMEM layout (225.5 KB total):
  Buffer                           Size (B)  Stages
  q                                32768     1
  k                                32768     2      <-- double-buffered (prefetch next chunk)
  v                                32768     1
  A_inverse / new_v                32768     1      <-- A_inv result, then overwritten with fp16 NV
  QK output / O store              32768     1      <-- W_qk scores, then O epilogue staging
  state / decay_v                  32768     1      <-- state GMEM<->TMEM staging, shared with decay_v ALU
  cumsumlog                          512     1      <-- BT x fp32 scalars
  cumprod                            512     1
  cumprod_scale                      512     1

TMEM layout (256 KB total):
  Buffer                  Size (B)  Stages
  state (S)               65536     1      <-- DKxDV fp32 = 128x128x4B
  q*state acc             65536     1      <-- BTxDV fp32 accumulator
  qk/kk/new_v/k*state/    65536     2      <-- shared accumulator for all other GEMMs
    kv/qkv acc

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise, kk_epi, qk_epi, inverse
  warps 4-7     : compute group 1 - kv_decay_v, v-k*state, state*q_epi,
                                    new_v_epi, kv_update_epi, qkv_epilogue
  warp  8       : MMA warp       - issues all 7 GEMMs
  warp  9       : TMA load warp  - loads q, k (double-buf), v
  warp  10      : TMA gate warp  - loads gate, beta
  warp  11      : epilogue warp   - store O to global memory
"""

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import TensorMapManager, TensorMapUpdateMode

# ---------------------------------------------------------------------------
# cutlass-dsl 4.4.2 compatibility: TmaInfo was removed; make_tiled_tma_atom_*
# now returns a plain (CopyAtom, Tensor) tuple instead of TmaInfo.
# ---------------------------------------------------------------------------
try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import extract_mlir_attributes as _ema
    from cutlass.base_dsl import extract_mlir_values as _emv
    from cutlass.base_dsl import get_mlir_types as _gmt
    from cutlass.base_dsl import new_from_mlir_values as _nfmv

    class TmaInfo:  # type: ignore[no-redef]
        """Compatibility shim replacing cpasync.TmaInfo for cutlass-dsl >= 4.4.2."""

        def __init__(self, atom, tma_tensor, smem_layout=None):
            self._atom = atom
            self._tma_tensor = tma_tensor

        @property
        def atom(self):
            return self._atom

        @property
        def tma_tensor(self):
            return self._tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self._atom) + _emv(self._tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self._atom) + _ema(self._tma_tensor)

        def __new_from_mlir_values__(self, values):
            n = len(_gmt(self._atom))
            return TmaInfo(
                _nfmv(self._atom, values[:n]),
                _nfmv(self._tma_tensor, values[n:]),
            )

        def __iter__(self):
            yield self._atom
            yield self._tma_tensor

        def __getitem__(self, i):
            return (self._atom, self._tma_tensor)[i]

        def __len__(self):
            return 2


@dsl_user_op
def _gdn_timeline_clock64(*, loc=None, ip=None):
    """Device timestamp helper for Python CuTeDSL debug instrumentation."""
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _wrap_tma(ret):
    """Wrap make_tiled_tma_atom_* return value in TmaInfo if not already."""
    if isinstance(ret, TmaInfo):
        return ret
    # 4.4.2: returns (CopyAtom, Tensor) tuple
    return TmaInfo(ret[0], ret[1])


from .tile_scheduler import GDNTileScheduler, GDNTileSchedulerParams

# ---------------------------------------------------------------------------
# Combined configuration + execution class
# ---------------------------------------------------------------------------


class GatedDeltaNetChunkedKernel:
    """
    Configuration and execution class for the Chunked GDN kernel.

    Follows the same class-based structure:
      - __init__    : warp IDs, barriers, tile shapes, SMEM/TMEM sizes
      - __call__    : @cute.jit host entry point (TMA setup, kernel launch)
      - kernel      : @cute.kernel device entry point (warp dispatch)
      - per-warp methods called from kernel's chunk loop

    Args:
        io_dtype   : input/output dtype (Float16 or BFloat16)
        acc_dtype  : accumulator dtype  (Float32)
        BT         : chunk size / block tile  (64)
        DK         : key/query hidden dim     (128)
        DV         : value hidden dim         (128)
    """

    # TMA descriptor size in bytes
    bytes_per_tensormap = 128
    num_tensormaps = 9

    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        state_dtype: Type[cutlass.Numeric],
        mma_tiler_qk: Tuple[int, int, int],
        mma_tiler_qs: Tuple[int, int, int],
        mma_tiler_qkv: Tuple[int, int, int],
        mma_tiler_kv: Tuple[int, int, int],
        max_active_clusters: int,
        num_sm: int,
        is_GQA: bool,
        use_initial_state: bool,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        input_A: bool = False,
        store_A: bool = False,
        store_g: bool = False,
        store_v_new: bool = False,
        store_w: bool = False,
        store_h: bool = False,
        w_rhs_precomputed: bool = False,
        training_side_outputs_only: bool = False,
        gate_is_log_cumsum: bool = False,
        gate_is_log_decay: bool = False,
        is_persistent: bool = True,
        enable_timeline: bool = False,
        enable_varlen_tail: bool = False,
    ):
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.state_dtype = state_dtype
        self.mma_tiler_qk = mma_tiler_qk
        self.mma_tiler_qs = mma_tiler_qs
        self.mma_tiler_qkv = mma_tiler_qkv
        self.mma_tiler_kv = mma_tiler_kv
        self.max_active_clusters = max_active_clusters
        self.num_sm = num_sm
        self.is_GQA = is_GQA
        self.use_initial_state = use_initial_state
        self.store_final_state = store_final_state
        self.enable_checkpoints = enable_checkpoints
        if gate_is_log_cumsum and gate_is_log_decay:
            raise ValueError("gate_is_log_cumsum and gate_is_log_decay are mutually exclusive")
        self.use_input_A = input_A
        self.store_A = store_A
        self.store_g = store_g
        self.store_v_new = store_v_new
        self.store_w = store_w
        self.store_h = store_h
        self.use_bf16_h_tma = (
            self.store_h
            and self.io_dtype == cutlass.BFloat16
            and self.state_dtype == cutlass.BFloat16
        )
        assert not w_rhs_precomputed, (
            "w_rhs_precomputed=True is not supported; pass w_rhs=None"
        )
        self.w_rhs_precomputed = w_rhs_precomputed
        # Keep the broad training-only branch disabled; it skips too many
        # pipeline tokens at once.  This flag enables verified drop-output
        # compute skips while preserving the full forward synchronization shape.
        self.drop_output_store_only = training_side_outputs_only
        self.training_side_outputs_only = False
        self.gate_is_log_cumsum = gate_is_log_cumsum
        self.gate_is_log_decay = gate_is_log_decay
        self.enable_varlen_tail = enable_varlen_tail
        self.is_persistent = is_persistent
        self.enable_timeline = enable_timeline

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # T-pairwise / kk_epi / qk_epi / inverse
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # kv_decay_v / v-k*state / epi ops
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        self.load_gate_beta_warp_id = 10
        # store O
        self.epilogue_warp_id = 11

        self.num_regs_compute_group_0 = 224
        self.num_regs_compute_group_1 = 256
        self.num_regs_other = 24

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.load_gate_beta_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.inverse_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        self.inverse_barrier_inner = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * 2,
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )

    def _setup_attributes(self):
        # ------------------------------------------------------------------
        # SMEM sizes (bytes per stage) and stage counts
        # ------------------------------------------------------------------
        self.smem_q_stages = 1
        self.smem_k_stages = 2
        self.smem_v_stages = 1
        # Dedicated 2 x 16 KiB W-RHS staging buffer. Keeping this separate
        # from sV lets the TMA warp prefetch W without extending sV's lifetime.
        self.smem_w_stages = 2
        self.smem_ainv_stages = 1
        self.smem_qk_stages = 1
        self.smem_o_stages = 1
        self.smem_side_output_stages = 2
        self.smem_group_order_stages = 1
        # Cumulative gate buffers - placed last in SMEM
        self.smem_gate_stages = 1
        self.smem_beta_stages = 1

        # ------------------------------------------------------------------
        # TMEM column offsets and buffer sizes (fp32, 32B per column)
        # ------------------------------------------------------------------
        self.tmem_kv_acc_stages = 1
        self.tmem_q_state_acc_stages = 1
        self.tmem_state_inp_stages = 1
        self.tmem_shared_inp_stages = 2
        self.tmem_shared_acc_stages = 2

        self.tmem_state_offset = 0
        self.tmem_q_state_offset = (
            self.tmem_state_offset + self.tmem_kv_acc_stages * 128
        )
        self.tmem_state_inp_offset = (
            self.tmem_q_state_offset + self.tmem_q_state_acc_stages * 64
        )
        self.tmem_shared_acc_offset = (
            self.tmem_state_inp_offset + self.tmem_state_inp_stages * 64
        )
        self.tmem_shared_inp_offset = (
            self.tmem_shared_acc_offset + self.tmem_shared_acc_stages * 64
        )

        self.buffer_align_bytes = 1024

    @cute.jit
    def _timeline_tag(
        self,
        debug_timing: Optional[cute.Tensor],
        lane: cutlass.Int32,
        chunk_idx: cutlass.Int32,
        tag: cutlass.Int32,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        tidx: cutlass.Int32,
        record_tidx: cutlass.Int32,
    ):
        if cutlass.const_expr(self.enable_timeline):
            assert debug_timing is not None, "debug_timing must be provided when timeline is enabled"
            if batch_idx == 0 and head_idx == 0 and chunk_idx < 2 and tidx == record_tidx:
                debug_timing[lane, chunk_idx, tag] = _gdn_timeline_clock64()

    @cute.jit
    def _timeline_tag_from_args(self, timeline_args: tuple, tag: cutlass.Int32):
        debug_timing, lane, chunk_idx, batch_idx, head_idx, tidx, record_tidx = timeline_args
        self._timeline_tag(debug_timing, lane, chunk_idx, tag, batch_idx, head_idx, tidx, record_tidx)

    # -----------------------------------------------------------------------
    # Capability check
    # -----------------------------------------------------------------------

    @staticmethod
    def can_implement(
        io_dtype,
        acc_dtype,
        mma_tiler_qk,
        mma_tiler_qs,
        mma_tiler_qkv,
        mma_tiler_kv,
    ):
        """Raise CantImplementError if this configuration is not supported."""
        if io_dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise testing.CantImplementError(
                f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported"
            )
        if acc_dtype != cutlass.Float32:
            raise testing.CantImplementError(
                f"acc_dtype={acc_dtype} not supported; only Float32 is supported"
            )
        if mma_tiler_qk != (64, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qk={mma_tiler_qk} not supported; only (64, 64, 128) is supported"
            )
        if mma_tiler_qs != (128, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qs={mma_tiler_qs} not supported; only (128, 64, 128) is supported"
            )
        if mma_tiler_qkv != (128, 64, 64):
            raise testing.CantImplementError(
                f"mma_tiler_qkv={mma_tiler_qkv} not supported; only (128, 64, 64) is supported"
            )
        if mma_tiler_kv != (128, 128, 64):
            raise testing.CantImplementError(
                f"mma_tiler_kv={mma_tiler_kv} not supported; only (128, 128, 64) is supported"
            )

    # -----------------------------------------------------------------------
    # Host entry point
    # -----------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        w_rhs: Optional[cute.Tensor],
        gate: cute.Tensor,
        beta: cute.Tensor,
        output_g: Optional[cute.Tensor],
        o: cute.Tensor,
        input_A: Optional[cute.Tensor],
        output_A: Optional[cute.Tensor],
        output_v_new: Optional[cute.Tensor],
        output_w: Optional[cute.Tensor],
        output_h: Optional[cute.Tensor],
        cu_seqlens: cute.Tensor,
        s_in: Optional[cute.Tensor],
        s_out: Optional[cute.Tensor],
        s_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        tensormap_workspace: cute.Tensor,
        debug_timing: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # chunk size
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        batch_size = cu_seqlens.shape[0] - 1
        raw_k = k

        self._setup_attributes()

        if cutlass.const_expr(self.is_GQA):
            h_r = h_q // h_v
            h_qv = h_v
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_v)),
                    stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_v)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_v)),
                    stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
                ),
            )
        else:
            h_r = h_v // h_q
            h_qv = h_q
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_q)),
                    stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_q)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_q)),
                    stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
                ),
            )

        if cutlass.const_expr(self.store_w):
            if cutlass.const_expr(self.w_rhs_precomputed):
                assert w_rhs is not None, "w_rhs must be provided when w_rhs_precomputed is True"
                if cutlass.const_expr(self.is_GQA):
                    w_rhs = cute.make_tensor(
                        w_rhs.iterator,
                        cute.make_layout(
                            (w_rhs.shape[2], w_rhs.shape[0], (h_r, h_v)),
                            stride=(w_rhs.stride[2], w_rhs.stride[0], (0, w_rhs.stride[1])),
                        ),
                    )
                else:
                    w_rhs = cute.make_tensor(
                        w_rhs.iterator,
                        cute.make_layout(
                            (w_rhs.shape[2], w_rhs.shape[0], (h_r, h_q)),
                            stride=(
                                w_rhs.stride[2],
                                w_rhs.stride[0],
                                (w_rhs.stride[1], h_r * w_rhs.stride[1]),
                            ),
                        ),
                    )
            else:
                assert w_rhs is None, "w_rhs must be None when generated inside the kernel"
                if cutlass.const_expr(self.is_GQA):
                    w_rhs = cute.make_tensor(
                        raw_k.iterator,
                        cute.make_layout(
                            (raw_k.shape[2], raw_k.shape[0], (h_r, h_v)),
                            stride=(raw_k.stride[2], raw_k.stride[0], (0, raw_k.stride[1])),
                        ),
                    )
                else:
                    w_rhs = cute.make_tensor(
                        raw_k.iterator,
                        cute.make_layout(
                            (raw_k.shape[2], raw_k.shape[0], (h_r, h_q)),
                            stride=(
                                raw_k.stride[2],
                                raw_k.stride[0],
                                (raw_k.stride[1], h_r * raw_k.stride[1]),
                            ),
                        ),
                    )
        else:
            assert w_rhs is None, "w_rhs must be None when store_w is False"

        gate = cute.make_tensor(
            gate.iterator,
            cute.make_layout(
                (gate.shape[0], (h_r, h_qv)),
                stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1])),
            ),
        )
        beta = cute.make_tensor(
            beta.iterator,
            cute.make_layout(
                (beta.shape[0], (h_r, h_qv)),
                stride=(beta.stride[0], (beta.stride[1], h_r * beta.stride[1])),
            ),
        )
        if cutlass.const_expr(self.store_g):
            assert output_g is not None, "output_g must be provided when store_g is True"
            output_g = cute.make_tensor(
                output_g.iterator,
                cute.make_layout(
                    (output_g.shape[0], (h_r, h_qv)),
                    stride=(
                        output_g.stride[0],
                        (output_g.stride[1], h_r * output_g.stride[1]),
                    ),
                ),
            )
        else:
            assert output_g is None, "output_g must be None when store_g is False"
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
        if cutlass.const_expr(self.use_input_A):
            assert input_A is not None, "input_A must be provided when use_input_A is True"
            input_A = cute.make_tensor(
                input_A.iterator,
                cute.make_layout(
                    (input_A.shape[0], input_A.shape[2], (h_r, h_qv)),
                    stride=(
                        input_A.stride[0],
                        input_A.stride[2],
                        (input_A.stride[1], h_r * input_A.stride[1]),
                    ),
                ),
            )
        else:
            assert input_A is None, "input_A must be None when use_input_A is False"
        if cutlass.const_expr(self.store_A):
            assert output_A is not None, "output_A must be provided when store_A is True"
            output_A = cute.make_tensor(
                output_A.iterator,
                cute.make_layout(
                    (output_A.shape[0], output_A.shape[2], (h_r, h_qv)),
                    stride=(
                        output_A.stride[0],
                        output_A.stride[2],
                        (output_A.stride[1], h_r * output_A.stride[1]),
                    ),
                ),
            )
        else:
            assert output_A is None, "output_A must be None when store_A is False"
        if cutlass.const_expr(self.store_v_new):
            assert output_v_new is not None, "output_v_new must be provided when store_v_new is True"
            output_v_new = cute.make_tensor(
                output_v_new.iterator,
                cute.make_layout(
                    (output_v_new.shape[2], output_v_new.shape[0], (h_r, h_qv)),
                    stride=(
                        output_v_new.stride[2],
                        output_v_new.stride[0],
                        (output_v_new.stride[1], h_r * output_v_new.stride[1]),
                    ),
                ),
            )
        else:
            assert output_v_new is None, "output_v_new must be None when store_v_new is False"
        if cutlass.const_expr(self.store_w):
            assert output_w is not None, "output_w must be provided when store_w is True"
            output_w = cute.make_tensor(
                output_w.iterator,
                cute.make_layout(
                    (output_w.shape[2], output_w.shape[0], (h_r, h_qv)),
                    stride=(
                        output_w.stride[2],
                        output_w.stride[0],
                        (output_w.stride[1], h_r * output_w.stride[1]),
                    ),
                ),
            )
        else:
            assert output_w is None, "output_w must be None when store_w is False"
        if cutlass.const_expr(self.store_h):
            assert output_h is not None, "output_h must be provided when store_h is True"
            if cutlass.const_expr(self.use_bf16_h_tma):
                h_tma_output = cute.make_tensor(
                    output_h.iterator,
                    cute.make_layout(
                        (
                            output_h.shape[2],
                            output_h.shape[3],
                            (h_r, h_qv),
                            output_h.shape[0],
                        ),
                        stride=(
                            1,
                            output_h.shape[2],
                            (output_h.stride[1], h_r * output_h.stride[1]),
                            output_h.stride[0],
                        ),
                    ),
                )
            else:
                h_tma_output = None
            output_h = cute.make_tensor(
                output_h.iterator,
                cute.make_layout(
                    (
                        output_h.shape[2],
                        output_h.shape[3],
                        (h_r, h_qv),
                        output_h.shape[0],
                    ),
                    stride=(
                        output_h.stride[2],
                        output_h.stride[3],
                        (output_h.stride[1], h_r * output_h.stride[1]),
                        output_h.stride[0],
                    ),
                ),
            )
        else:
            assert output_h is None, "output_h must be None when store_h is False"
            h_tma_output = None
        if cutlass.const_expr(s_in is not None):
            s_in = cute.make_tensor(
                s_in.iterator,
                cute.make_layout(
                    (s_in.shape[2], s_in.shape[3], (h_r, h_qv), s_in.shape[0]),
                    stride=(
                        s_in.stride[2],
                        s_in.stride[3],
                        (s_in.stride[1], h_r * s_in.stride[1]),
                        s_in.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(s_out is not None):
            s_out = cute.make_tensor(
                s_out.iterator,
                cute.make_layout(
                    (s_out.shape[2], s_out.shape[3], (h_r, h_qv), s_out.shape[0]),
                    stride=(
                        s_out.stride[2],
                        s_out.stride[3],
                        (s_out.stride[1], h_r * s_out.stride[1]),
                        s_out.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(self.enable_checkpoints):
            s_checkpoints = cute.make_tensor(
                s_checkpoints.iterator,
                cute.make_layout(
                    (
                        s_checkpoints.shape[2],
                        s_checkpoints.shape[3],
                        (h_r, h_qv),
                        s_checkpoints.shape[0],
                    ),
                    stride=(
                        s_checkpoints.stride[2],
                        s_checkpoints.stride[3],
                        (s_checkpoints.stride[1], h_r * s_checkpoints.stride[1]),
                        s_checkpoints.stride[0],
                    ),
                ),
            )

        # ------------------------------------------------------------------
        # Build tiled MMAs  (one per logical GEMM group, differing in operand major modes)
        # ------------------------------------------------------------------
        def _mma_op(mma_tiler, a_major, b_major, OperandSourceA):
            # Derive MMA atom (M, N) from the first two dims of the tile shape;
            # K=16 is the hardware fp16 atom depth (fixed for SM100 tcgen05).
            return tcgen05.MmaF16BF16Op(
                self.io_dtype,
                self.acc_dtype,
                (mma_tiler[0], mma_tiler[1], 16),
                self.cta_group,
                OperandSourceA,
                a_major,
                b_major,
            )

        # GEMM 1 (kk: K@K^T) + GEMM 2 (qk: Q@K^T)          - KK-major: A=K, B=K
        tiled_mma_qk = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qk,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 3 (k*state: K@S) + GEMM 4 (q*state: Q@S)     - KN-major: A=K, B=MN
        tiled_mma_qs = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qs,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # GEMM 5 (new_v: A_inv@V) + GEMM 6 (qkv: W_qkv@NV)  - KN-major: A=K, B=MN
        tiled_mma_qkv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # for v_smem_layout_staged
        tiled_mma_qkv_ss = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.MN,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 7 (kv_update: K^T@delta -> dS)                    - MN-major: A=MN, B=MN
        tiled_mma_kv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_kv,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                tcgen05.OperandSource.TMEM,
            )
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # ------------------------------------------------------------------
        # SMEM layouts - computed before SharedStorage so cosize() is available
        # ------------------------------------------------------------------
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Q is A operand for tiled_mma_qk (GEMM 2: qk), K is B operand (GEMM 1+2: kk/qk)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_k_stages
        )
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_kv, self.mma_tiler_kv, self.io_dtype, self.smem_k_stages
        )
        # V is A operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv)
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_v_stages
        )
        w_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_w_stages
        )
        # A_inv is B operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv);
        ainv_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_ainv_stages
        )
        # W_qkv is A operand for tiled_mma_qkv (GEMM 6: qkv: NV @ qk)
        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_qk_stages
        )

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.from_tensor(o),
            self.mma_tiler_qkv[:2],
            self.smem_o_stages,
        )
        side_output_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.from_tensor(o),
            self.mma_tiler_qkv[:2],
            self.smem_side_output_stages,
        )
        if cutlass.const_expr(self.use_bf16_h_tma):
            assert output_h is not None
            h_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.io_dtype,
                utils.LayoutEnum.COL_MAJOR,
                self.mma_tiler_kv[:2],
                1,
            )
            assert cute.size_in_bytes(self.io_dtype, h_smem_layout_staged) == 32768
        else:
            h_smem_layout_staged = side_output_smem_layout_staged
        # Gate scalar arrays (1D Float32, flat layout - no swizzle needed)
        cumsumlog_smem_layout_staged = cute.make_layout(
            (self.b_t, 1, self.smem_gate_stages)
        )
        beta_smem_layout_staged = cute.make_layout((self.b_t, 1, self.smem_beta_stages))
        ainput_smem_layout_staged = cute.make_layout(
            (self.b_t, self.b_t, self.smem_ainv_stages),
            stride=(self.b_t, 1, self.b_t * self.b_t),
        )

        # ------------------------------------------------------------------
        # Shared memory struct  (defined here to capture layout cosizes)
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers - one entry per stage, 2 Int64 words per barrier
            # TMA load warp -> MMA warp (K double-buffered)
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # TMA load warp -> MMA warp
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            # TMA load warp -> MMA warp
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            # TMA load warp -> CG1 (dedicated double-buffered W RHS)
            load_w_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_w_stages * 2]
            # TMA load warp -> CG0 (saved A into sAinv)
            load_A_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # TMA gate warp -> CG0/CG1
            load_gate_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_gate_stages * 2
            ]
            # TMA beta warp -> CG0
            load_beta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_beta_stages * 2
            ]
            # MMA warp -> CG1 (Q*state acc ready in TMEM)
            q_state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_q_state_acc_stages * 2
            ]
            # MMA warp -> CG1 (GEMM 7 done)
            kv_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_kv_acc_stages * 2
            ]
            # MMA warp -> CG0/CG1 (GEMM 1-6 done)
            shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_shared_acc_stages * 2
            ]
            # CG0 -> MMA warp (A_inv ready in SMEM)
            ainv_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # CG0 -> MMA warp (QK ready in SMEM)
            qk_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_qk_stages * 2
            ]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_inp_stages * 2
            ]
            # CG1 -> MMA warp (state input ready in SMEM)
            shared_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_shared_inp_stages * 2
            ]
            # CG1 -> epilogue warp
            o_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_o_stages * 2
            ]
            # CG1 -> epilogue warp: VNew/W side-output staging readiness
            side_output_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, 2
            ]
            # CG0 -> CG1 (group order)
            group_order_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_group_order_stages * 2
            ]
            # TMEM allocation token
            tmem_holding_buf: cutlass.Int32
            # SMEM tensor buffers (aligned, in SMEM layout order)
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype,
                    cute.cosize(w_smem_layout_staged) if self.store_w else 1,
                ],
                self.buffer_align_bytes,
            ]
            # A_inv result, then overwritten with fp16 NV
            sAinv: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(ainv_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # Row-major saved-A TMA staging. CG0 converts this into sAinv layout.
            sAinput: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(ainput_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # W_qk scores
            sQk: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Dedicated VNew/W staging: two 128x64 BF16 tiles, never aliased with sO.
            sSideOutput: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype,
                    cute.cosize(side_output_smem_layout_staged)
                    if self.store_v_new or self.store_w else 1,
                ],
                self.buffer_align_bytes,
            ]
            # Dedicated 128x128 BF16 H staging (32 KiB), never aliased with VNew/W.
            sH: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype,
                    cute.cosize(h_smem_layout_staged) if self.use_bf16_h_tma else 1,
                ],
                self.buffer_align_bytes,
            ]
            # Cumulative gate scalars - placed last in SMEM
            cumsumlog: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            cumprod: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            beta: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(beta_smem_layout_staged)
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Build TMA atoms
        # ------------------------------------------------------------------
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        w_smem_layout = cute.select(w_smem_layout_staged, mode=[0, 1, 2])
        ainv_smem_layout = cute.select(ainv_smem_layout_staged, mode=[0, 1, 2])
        ainput_smem_layout = cute.select(ainput_smem_layout_staged, mode=[0, 1])

        tma_q = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q,
                q_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_k = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_v = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                v_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv_ss,
                self.cluster_layout_vmnk.shape,
            )
        )
        if cutlass.const_expr(self.store_w):
            assert w_rhs is not None, "w_rhs must be provided when store_w is True"
            tma_w = _wrap_tma(
                cute.nvgpu.make_tiled_tma_atom_A(
                    tma_load_op,
                    w_rhs,
                    w_smem_layout,
                    self.mma_tiler_qkv,
                    tiled_mma_qkv_ss,
                    self.cluster_layout_vmnk.shape,
                )
            )
        else:
            tma_w = tma_v
        if cutlass.const_expr(self.use_input_A):
            tma_A = _wrap_tma(
                cpasync.make_tiled_tma_atom(
                    tma_load_op,
                    input_A,
                    ainput_smem_layout,
                    (self.b_t, self.b_t),
                )
            )
        else:
            tma_A = tma_k
        cumsumlog_smem_layout = cute.select(cumsumlog_smem_layout_staged, mode=[0])  # noqa: F841
        beta_smem_layout = cute.select(beta_smem_layout_staged, mode=[0])  # noqa: F841

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        side_output_smem_layout = cute.select(side_output_smem_layout_staged, mode=[0, 1])
        h_smem_layout = cute.select(h_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.mma_tiler_qkv[:2],
            )
        )
        if cutlass.const_expr(self.store_v_new):
            assert output_v_new is not None, "output_v_new must be provided when store_v_new is True"
            tma_v_new_out = _wrap_tma(
                cpasync.make_tiled_tma_atom(
                    tma_store_op,
                    output_v_new,
                    side_output_smem_layout,
                    self.mma_tiler_qkv[:2],
                )
            )
        else:
            tma_v_new_out = tma_o
        if cutlass.const_expr(self.store_w):
            assert output_w is not None, "output_w must be provided when store_w is True"
            tma_w_out = _wrap_tma(
                cpasync.make_tiled_tma_atom(
                    tma_store_op,
                    output_w,
                    side_output_smem_layout,
                    self.mma_tiler_qkv[:2],
                )
            )
        else:
            tma_w_out = tma_o
        if cutlass.const_expr(self.use_bf16_h_tma):
            assert h_tma_output is not None
            tma_h_out = _wrap_tma(
                cpasync.make_tiled_tma_atom(
                    tma_store_op,
                    h_tma_output,
                    h_smem_layout,
                    self.mma_tiler_kv[:2],
                )
            )
        else:
            tma_h_out = tma_o

        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem_layout)
        self.tma_A_bytes = cute.size_in_bytes(self.io_dtype, ainput_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        scheduler_params = GDNTileSchedulerParams(
            num_seqs=batch_size,
            num_q_heads=h_q,
            num_v_heads=h_v,
            is_GQA=self.is_GQA,
            is_persistent=self.is_persistent,
        )
        grid_shape = GDNTileScheduler.get_grid_shape(
            scheduler_params, self.max_active_clusters
        )

        self.kernel(
            tiled_mma_qk,
            tiled_mma_qs,
            tiled_mma_qkv,
            tiled_mma_qkv_ss,
            tiled_mma_kv,
            tma_q,
            tma_k,
            tma_v,
            tma_w,
            tma_A,
            gate,
            beta,
            output_g,
            tma_o,
            tma_v_new_out,
            tma_w_out,
            tma_h_out,
            input_A,
            output_A,
            output_v_new,
            output_w,
            output_h,
            h_tma_output,
            cu_seqlens,
            s_in,
            s_out,
            s_checkpoints,
            cu_checkpoints,
            checkpoint_every_n_tokens,
            scale,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            v_smem_layout_staged,
            w_smem_layout_staged,
            cumsumlog_smem_layout_staged,
            beta_smem_layout_staged,
            ainput_smem_layout_staged,
            ainv_smem_layout_staged,
            qk_smem_layout_staged,
            o_smem_layout_staged,
            side_output_smem_layout_staged,
            h_smem_layout_staged,
            scheduler_params,
            q,
            k,
            v,
            w_rhs,
            o,
            tensormap_workspace,
            debug_timing,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -----------------------------------------------------------------------
    # Device kernel
    # -----------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # GEMM 1 (kk) + GEMM 2 (qk)
        tiled_mma_qk: cute.TiledMma,
        # GEMM 3 (k*state) + GEMM 4 (q*state)
        tiled_mma_qs: cute.TiledMma,
        # GEMM 5 (new_v) + GEMM 6 (qkv)
        tiled_mma_qkv: cute.TiledMma,
        # GEMM 5 (new_v: A_inv@V) first tile
        tiled_mma_qkv_ss: cute.TiledMma,
        # GEMM 7 (kv_update)
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        tma_w: TmaInfo,
        tma_A: TmaInfo,
        mGate: cute.Tensor,
        mBeta: cute.Tensor,
        mGateOut: Optional[cute.Tensor],
        tma_o: TmaInfo,
        tma_v_new_out: TmaInfo,
        tma_w_out: TmaInfo,
        tma_h_out: TmaInfo,
        mA_in: Optional[cute.Tensor],
        mA_out: Optional[cute.Tensor],
        mVNew_out: Optional[cute.Tensor],
        mW_out: Optional[cute.Tensor],
        mH_out: Optional[cute.Tensor],
        mH_tma_out: Optional[cute.Tensor],
        # (B+1,)  int32  cumulative seq lengths
        cu_seqlens: cute.Tensor,
        # initial state (fp32) from GMEM; None if not used
        mS_init: Optional[cute.Tensor],
        # final state output (fp32) to GMEM; None if not stored
        mS_out: Optional[cute.Tensor],
        mS_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        beta_smem_layout_staged: cute.Layout,
        ainput_smem_layout_staged: cute.Layout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        side_output_smem_layout_staged: cute.ComposedLayout,
        h_smem_layout_staged: cute.ComposedLayout,
        # Scheduler
        scheduler_params: GDNTileSchedulerParams,
        # TMA descriptor workspace in GMEM (one set of 9 slots per CTA).
        # Slots: Q=0, K=1, V=2, O=3, A=4, W_RHS=5, VNEW_OUT=6, W_OUT=7, H_OUT=8.
        mQ,
        mK,
        mV,
        mW: Optional[cute.Tensor],
        # used for TMA descriptor update
        mO,
        # (num_ctas, 3+smem_k_stages, 16) Int64
        tensormap_workspace: cute.Tensor,
        # Optional debug timeline tensor: [lane, chunk(0..1), tag] int64 globaltimer stamps.
        debug_timing: Optional[cute.Tensor],
    ):
        """
        Main GDN chunked kernel.

        Warp specialization is the outermost control flow: each warp role owns
        its own persistent tile-scheduler loop, iterating over (batch, head)
        tiles and then over chunks within each tile.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.use_initial_state):
            assert mS_init is not None, (
                "mS_init must be provided if use_initial_state is True"
            )
        else:
            assert mS_init is None, "mS_init must be None if use_initial_state is False"
        if cutlass.const_expr(self.store_final_state):
            assert mS_out is not None, (
                "mS_out must be provided if store_final_state is True"
            )
        else:
            assert mS_out is None, "mS_out must be None if store_final_state is False"
        if cutlass.const_expr(self.store_v_new):
            assert mVNew_out is not None, (
                "mVNew_out must be provided if store_v_new is True"
            )
        else:
            assert mVNew_out is None, "mVNew_out must be None if store_v_new is False"
        if cutlass.const_expr(self.store_w):
            assert mW_out is not None, "mW_out must be provided if store_w is True"
            assert mW is not None, "mW must be provided if store_w is True"
        else:
            assert mW_out is None, "mW_out must be None if store_w is False"
            assert mW is None, "mW must be None if store_w is False"
        if cutlass.const_expr(self.store_h):
            assert mH_out is not None, "mH_out must be provided if store_h is True"
            if cutlass.const_expr(self.use_bf16_h_tma):
                assert mH_tma_out is not None
            else:
                assert mH_tma_out is None
        else:
            assert mH_out is None, "mH_out must be None if store_h is False"
            assert mH_tma_out is None

        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one set of 9 ptrs per CTA.
        # Slots: Q=0, K=1, V=2, O=3, A=4, W_RHS=5, VNEW_OUT=6, W_OUT=7, H_OUT=8.
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 3, None)].iterator
        )
        tensormap_A_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 4, None)].iterator
        )
        tensormap_w_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 5, None)].iterator
        )
        tensormap_v_new_out_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 6, None)].iterator
        )
        tensormap_w_out_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 7, None)].iterator
        )
        tensormap_h_out_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 8, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.store_w):
            sW = storage.sW.get_tensor(
                w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner
            )
        else:
            sW = sV
        # A_inverse / new_v  (A_inv written first, then overwritten with fp16 NV)
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        sAinput = storage.sAinput.get_tensor(ainput_smem_layout_staged)
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.store_v_new or self.store_w):
            sSideOutput = storage.sSideOutput.get_tensor(
                side_output_smem_layout_staged.outer,
                swizzle=side_output_smem_layout_staged.inner,
            )
        else:
            sSideOutput = sO
        if cutlass.const_expr(self.use_bf16_h_tma):
            h_smem_layout = cute.select(h_smem_layout_staged, mode=[0, 1])
            sH = storage.sH.get_tensor(
                h_smem_layout.outer,
                swizzle=h_smem_layout.inner,
            )
        else:
            sH = sSideOutput
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)
        sBeta = storage.beta.get_tensor(beta_smem_layout_staged)

        if warp_idx == self.mma_warp_id:
            if cutlass.const_expr(not self.drop_output_store_only):
                cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            if cutlass.const_expr(self.store_w):
                cpasync.prefetch_descriptor(tma_w.atom)
            if cutlass.const_expr(self.use_input_A):
                cpasync.prefetch_descriptor(tma_A.atom)
            if cutlass.const_expr(not self.drop_output_store_only):
                cpasync.prefetch_descriptor(tma_o.atom)
            if cutlass.const_expr(self.store_v_new):
                cpasync.prefetch_descriptor(tma_v_new_out.atom)
            if cutlass.const_expr(self.store_w):
                cpasync.prefetch_descriptor(tma_w_out.atom)
            if cutlass.const_expr(self.use_bf16_h_tma):
                cpasync.prefetch_descriptor(tma_h_out.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # Correction warp is the last one that accesses tmem
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 warp (gate/beta ldg/ldgsts warp)
        cg_gate = _cg(self.threads_per_warp * len([self.load_gate_beta_warp_id]))
        # 1 thread (UMMA issuer)
        cg_mma = _cg(len([self.mma_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # One participating thread per CG0 warp for TMA async waits.
        cg_cg0_tma = _cg(len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V/W load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # 256 threads (CG0 + CG1)
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp * len(self.compute_group_1_warp_ids)
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load -> consumers: K (double-buffered), Q, V, and W.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_w_producer, load_w_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_w_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_A_producer, load_A_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_tma,
            tx_count=self.tma_A_bytes,
            barrier_storage=storage.load_A_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Gate warp (warp 10) -> CG0 / CG1:  gate/beta (PipelineAsync, software-signaled)
        # ldg/ldgsts paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_beta_producer, load_beta_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.smem_beta_stages,
            producer_group=cg_gate,
            consumer_group=cg_cg0,
            barrier_storage=storage.load_beta_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG0/CG1:  shared_acc
        shared_acc_producer, shared_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_shared_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.shared_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp:  state_inp_ready
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp:  shared_inp_ready
        shared_inp_ready_producer, shared_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_shared_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.shared_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        side_output_store_producer, side_output_store_consumer = (
            pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=cg_cg1,
                consumer_group=cg_epi,
                barrier_storage=storage.side_output_store_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        group_order_producer, group_order_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_group_order_stages,
            producer_group=cg_cg0,
            consumer_group=cg_cg1,
            barrier_storage=storage.group_order_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        # ------------------------------------------------------------------
        # 2. Warp specialization - each warp role owns its own scheduler loop
        # ------------------------------------------------------------------

        # ==============================================================
        # COMPUTE WARP GROUP 0 (warps 0-3)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                sQk_pisl = self._transform_to_position_independent_layout(
                    sQk, qk_smem_layout_staged.inner
                )
                # First chunk: no previous state (S_prev = 0), skip GEMMs 3/4
                for chunk_idx in cutlass.range(num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    self._timeline_tag(
                        debug_timing, 0, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.compute_group_0_warp_ids[0] * self.threads_per_warp,
                    )
                    (
                        load_gate_consumer,
                        load_beta_consumer,
                        load_A_consumer,
                        shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                        group_order_producer,
                    ) = self.compute_group_0(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_qk,),
                        (sCumsumlog, sBeta, sAinv, sAinput, sQk_pisl),
                        (
                            load_gate_consumer,
                            load_beta_consumer,
                            load_A_consumer,
                            shared_acc_consumer,
                            a_inv_ready_producer,
                            qk_ready_producer,
                            group_order_producer,
                        ),
                        (
                            mA_in,
                            mA_out,
                            head_idx,
                            chunk_offset,
                            chunk_idx == 0,
                            batch_end,
                        ),
                        (
                            debug_timing,
                            cutlass.Int32(0),
                            chunk_idx,
                            batch_idx,
                            head_idx,
                            tidx,
                            self.compute_group_0_warp_ids[0] * self.threads_per_warp,
                        ),
                    )
                    self._timeline_tag(
                        debug_timing, 0, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.compute_group_0_warp_ids[0] * self.threads_per_warp,
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()
            o_store_producer.tail()
            group_order_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 1 (warps 4-7)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            # Total TMEM columns: state(128) + q_state_acc(128) + shared_acc(128x2) = 512
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                checkpoint_offset = 0
                if cutlass.const_expr(self.enable_checkpoints or self.store_h):
                    checkpoint_offset = cu_checkpoints[batch_idx]
                if cutlass.const_expr(self.use_initial_state):
                    kv_acc_producer = self._load_initial_state(
                        tidx,
                        mS_init,
                        head_idx,
                        batch_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        kv_acc_producer,
                    )
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                if cutlass.const_expr(self.store_w):
                    sW_pisl = self._transform_to_position_independent_layout(
                        sW, w_smem_layout_staged.inner
                    )
                else:
                    sW_pisl = sV_pisl
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                self._timeline_tag(
                    debug_timing, 1, 0, 0, batch_idx, head_idx, tidx,
                    self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                )
                (
                    load_v_consumer,
                    load_w_consumer,
                    load_gate_consumer,
                    shared_acc_consumer,
                    kv_acc_consumer,
                    q_state_acc_consumer,
                    group_order_consumer,
                    kv_acc_producer,
                    state_inp_ready_producer,
                    shared_inp_ready_producer,
                    o_store_producer,
                    side_output_store_producer,
                    checkpoint_idx,
                ) = self.compute_group_1(
                    tidx,
                    tmem_ptr,
                    scale,
                    (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                    (
                        sV_pisl,
                        sW_pisl,
                        sCumsumlog,
                        sCumprod,
                        sBeta,
                        sO_pisl,
                        sSideOutput,
                        sH,
                    ),
                    (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                    (
                        load_v_consumer,
                        load_w_consumer,
                        load_gate_consumer,
                        shared_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        group_order_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        side_output_store_producer,
                    ),
                    (mVNew_out, mW_out, mH_out, True, 0, head_idx, batch_start),
                    (
                        debug_timing,
                        cutlass.Int32(1),
                        cutlass.Int32(0),
                        batch_idx,
                        head_idx,
                        tidx,
                        self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                    ),
                )
                self._timeline_tag(
                    debug_timing, 1, 0, 1, batch_idx, head_idx, tidx,
                    self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                )
                for chunk_idx in cutlass.range(1, num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    self._timeline_tag(
                        debug_timing, 1, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                    )
                    (
                        load_v_consumer,
                        load_w_consumer,
                        load_gate_consumer,
                        shared_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        group_order_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        side_output_store_producer,
                        checkpoint_offset,
                    ) = self.compute_group_1(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sW_pisl,
                            sCumsumlog,
                            sCumprod,
                            sBeta,
                            sO_pisl,
                            sSideOutput,
                            sH,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_w_consumer,
                            load_gate_consumer,
                            shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            group_order_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                            side_output_store_producer,
                        ),
                        (mVNew_out, mW_out, mH_out, False, chunk_idx, head_idx, chunk_offset),
                        (
                            debug_timing,
                            cutlass.Int32(1),
                            chunk_idx,
                            batch_idx,
                            head_idx,
                            tidx,
                            self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                        ),
                    )
                    self._timeline_tag(
                        debug_timing, 1, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.compute_group_1_warp_ids[0] * self.threads_per_warp,
                    )
                if cutlass.const_expr(
                    self.store_final_state or self.enable_checkpoints
                ):
                    kv_acc_consumer = self._store_final_state(
                        tidx,
                        mS_out,
                        head_idx,
                        batch_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        kv_acc_consumer,
                        seqlen_b,
                        mS_checkpoints,
                        checkpoint_offset,
                        checkpoint_every_n_tokens,
                    )
                else:
                    kv_acc_handle = kv_acc_consumer.wait_and_advance()
                    kv_acc_handle.release()

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            shared_inp_ready_producer.tail()
            o_store_producer.tail()
            side_output_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # MMA WARP (warp 8)
        # ==============================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                # First chunk: no previous state (S_prev = 0), skip GEMMs 3/4.
                chunk_offset = batch_start
                self._timeline_tag(
                    debug_timing, 2, 0, 0, batch_idx, head_idx, tidx,
                    self.mma_warp_id * self.threads_per_warp,
                )
                (
                    shared_acc_producer,
                    q_state_acc_producer,
                    kv_acc_producer,
                    load_k_consumer,
                    load_q_consumer,
                    load_v_consumer,
                    a_inv_ready_consumer,
                    qk_ready_consumer,
                    state_inp_ready_consumer,
                    shared_inp_ready_consumer,
                ) = self.mma_warp(
                    tmem_ptr,
                    (
                        tiled_mma_qk,
                        tiled_mma_qs,
                        tiled_mma_qkv,
                        tiled_mma_qkv_ss,
                        tiled_mma_kv,
                    ),
                    (sQ, sK, sK_trans, sV, sAinv, sQk),
                    (
                        shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                    ),
                    (True,),
                    (
                        debug_timing,
                        cutlass.Int32(2),
                        cutlass.Int32(0),
                        batch_idx,
                        head_idx,
                        tidx,
                        self.mma_warp_id * self.threads_per_warp,
                    ),
                )
                self._timeline_tag(
                    debug_timing, 2, 0, 1, batch_idx, head_idx, tidx,
                    self.mma_warp_id * self.threads_per_warp,
                )

                # Main loop: chunks 1..num_chunks_b-1 with previous state.
                for chunk_idx in cutlass.range(1, num_chunks_b):  # noqa: B007
                    self._timeline_tag(
                        debug_timing, 2, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.mma_warp_id * self.threads_per_warp,
                    )
                    (
                        shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                    ) = self.mma_warp(
                        tmem_ptr,
                        (
                            tiled_mma_qk,
                            tiled_mma_qs,
                            tiled_mma_qkv,
                            tiled_mma_qkv_ss,
                            tiled_mma_kv,
                        ),
                        (sQ, sK, sK_trans, sV, sAinv, sQk),
                        (
                            shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            load_v_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            shared_inp_ready_consumer,
                        ),
                        (False,),
                        (
                            debug_timing,
                            cutlass.Int32(2),
                            chunk_idx,
                            batch_idx,
                            head_idx,
                            tidx,
                            self.mma_warp_id * self.threads_per_warp,
                        ),
                    )
                    self._timeline_tag(
                        debug_timing, 2, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.mma_warp_id * self.threads_per_warp,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # ==============================================================
        # TMA LOAD WARP (warp 9)
        # ==============================================================
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init base descriptors once into GMEM (copies embedded atom descriptor)
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
                )
                if cutlass.const_expr(self.store_w):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_w.atom, tensormap_w_ptr, self.tma_qkv_warp_id
                    )
                if cutlass.const_expr(self.use_input_A):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_A.atom, tensormap_A_ptr, self.tma_qkv_warp_id
                    )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                # Build bounded tensors: same ptr/strides, token dim capped to batch_end
                bounded_q = cute.make_tensor(
                    mQ.iterator,
                    cute.make_layout(
                        (batch_end, mQ.shape[1], mQ.shape[2]),
                        stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                    ),
                )
                bounded_k = cute.make_tensor(
                    mK.iterator,
                    cute.make_layout(
                        (batch_end, mK.shape[1], mK.shape[2]),
                        stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                    ),
                )
                bounded_v = cute.make_tensor(
                    mV.iterator,
                    cute.make_layout(
                        (mV.shape[0], batch_end, mV.shape[2]),
                        stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                    ),
                )
                # Update K/Q/V descriptors
                tensormap_manager.update_tensormap(
                    (bounded_q, bounded_k, bounded_v),
                    (tma_q.atom, tma_k.atom, tma_v.atom),
                    (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                    self.tma_qkv_warp_id,
                    (None, None, None),
                )
                if cutlass.const_expr(self.store_w):
                    assert mW is not None, "mW must be provided when store_w is True"
                    bounded_w = cute.make_tensor(
                        mW.iterator,
                        cute.make_layout(
                            (mW.shape[0], batch_end, mW.shape[2]),
                            stride=(mW.stride[0], mW.stride[1], mW.stride[2]),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_w,),
                        (tma_w.atom,),
                        (tensormap_w_ptr,),
                        self.tma_qkv_warp_id,
                        (None,),
                    )
                if cutlass.const_expr(self.use_input_A):
                    assert mA_in is not None, "mA_in must be provided when use_input_A is True"
                    bounded_A = cute.make_tensor(
                        mA_in.iterator,
                        cute.make_layout(
                            (batch_end, mA_in.shape[1], mA_in.shape[2]),
                            stride=(mA_in.stride[0], mA_in.stride[1], mA_in.stride[2]),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_A,),
                        (tma_A.atom,),
                        (tensormap_A_ptr,),
                        self.tma_qkv_warp_id,
                        (None,),
                    )

                for chunk_idx in cutlass.range(num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    self._timeline_tag(
                        debug_timing, 3, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.tma_qkv_warp_id * self.threads_per_warp,
                    )
                    (
                        load_q_producer,
                        load_k_producer,
                        load_v_producer,
                        load_w_producer,
                        load_A_producer,
                    ) = (
                        self.tma_qkv_warp(
                            (tiled_mma_qk, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv),
                            (tma_q, tma_k, tma_v, tma_w, tma_A),
                            (sQ, sK, sV, sW, sAinput),
                            (
                                load_q_producer,
                                load_k_producer,
                                load_v_producer,
                                load_w_producer,
                                load_A_producer,
                            ),
                            (chunk_offset, chunk_idx, batch_idx, head_idx),
                            (
                                tensormap_manager,
                                tensormap_q_ptr,
                                tensormap_k_ptr,
                                tensormap_v_ptr,
                                tensormap_w_ptr,
                                tensormap_A_ptr,
                            ),
                            (
                                debug_timing,
                                cutlass.Int32(3),
                                chunk_idx,
                                batch_idx,
                                head_idx,
                                tidx,
                                self.tma_qkv_warp_id * self.threads_per_warp,
                            ),
                        )
                    )
                    self._timeline_tag(
                        debug_timing, 3, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.tma_qkv_warp_id * self.threads_per_warp,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            if cutlass.const_expr(self.store_w):
                load_w_producer.tail()
            if cutlass.const_expr(self.use_input_A):
                load_A_producer.tail()

        # ==============================================================
        # GATE/BETA LOAD WARP (warp 10)
        # ==============================================================
        if warp_idx == self.load_gate_beta_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                batch_end = batch_start + seqlen_b
                # Full tiles (all but the last): unconditional loads
                for chunk_idx in cutlass.range(num_chunks_b - 1):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    self._timeline_tag(
                        debug_timing, 4, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.load_gate_beta_warp_id * self.threads_per_warp,
                    )
                    load_gate_producer, load_beta_producer = self.load_gate_beta_warp(
                        tidx,
                        (mGate, mBeta, mGateOut),
                        (sCumsumlog, sCumprod, sBeta),
                        (load_gate_producer, load_beta_producer),
                        (chunk_offset, head_idx, False, batch_end),
                        (
                            debug_timing,
                            cutlass.Int32(4),
                            chunk_idx,
                            batch_idx,
                            head_idx,
                            tidx,
                            self.load_gate_beta_warp_id * self.threads_per_warp,
                        ),
                    )
                    self._timeline_tag(
                        debug_timing, 4, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.load_gate_beta_warp_id * self.threads_per_warp,
                    )
                # Last tile: is_last_tile=True, valid_tokens derived inside
                chunk_idx = num_chunks_b - 1
                self._timeline_tag(
                    debug_timing, 4, chunk_idx, 0, batch_idx, head_idx, tidx,
                    self.load_gate_beta_warp_id * self.threads_per_warp,
                )
                load_gate_producer, load_beta_producer = self.load_gate_beta_warp(
                    tidx,
                    (mGate, mBeta, mGateOut),
                    (sCumsumlog, sCumprod, sBeta),
                    (load_gate_producer, load_beta_producer),
                    (
                        batch_start + (num_chunks_b - 1) * self.b_t,
                        head_idx,
                        True,
                        batch_end,
                    ),
                    (
                        debug_timing,
                        cutlass.Int32(4),
                        chunk_idx,
                        batch_idx,
                        head_idx,
                        tidx,
                        self.load_gate_beta_warp_id * self.threads_per_warp,
                    ),
                )
                self._timeline_tag(
                    debug_timing, 4, chunk_idx, 1, batch_idx, head_idx, tidx,
                    self.load_gate_beta_warp_id * self.threads_per_warp,
                )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            load_gate_producer.tail()
            load_beta_producer.tail()
        # ==============================================================
        # EPILOGUE WARP (warp 11)
        # ==============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # All epilogue descriptors are updated per packed-varlen work tile.
            if work.is_valid_tile:
                if cutlass.const_expr(not self.drop_output_store_only):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
                    )
                if cutlass.const_expr(self.store_v_new):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_v_new_out.atom,
                        tensormap_v_new_out_ptr,
                        self.epilogue_warp_id,
                    )
                if cutlass.const_expr(self.store_w):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_w_out.atom, tensormap_w_out_ptr, self.epilogue_warp_id
                    )
                if cutlass.const_expr(self.use_bf16_h_tma):
                    tensormap_manager.init_tensormap_from_atom(
                        tma_h_out.atom,
                        tensormap_h_out_ptr,
                        self.epilogue_warp_id,
                    )
                if cutlass.const_expr(
                    not self.drop_output_store_only or self.store_v_new or self.store_w or self.use_bf16_h_tma
                ):
                    tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                if cutlass.const_expr(not self.drop_output_store_only):
                    # Build bounded O tensor: token dim capped to batch_end
                    bounded_o = cute.make_tensor(
                        mO.iterator,
                        cute.make_layout(
                            (mO.shape[0], batch_end, mO.shape[2]),
                            stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                        ),
                    )

                    # Update O descriptor independently
                    tensormap_manager.update_tensormap(
                        (bounded_o,),
                        (tma_o.atom,),
                        (tensormap_o_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

                if cutlass.const_expr(self.store_v_new):
                    assert mVNew_out is not None
                    bounded_v_new_out = cute.make_tensor(
                        mVNew_out.iterator,
                        cute.make_layout(
                            (
                                mVNew_out.shape[0],
                                batch_end,
                                mVNew_out.shape[2],
                            ),
                            stride=(
                                mVNew_out.stride[0],
                                mVNew_out.stride[1],
                                mVNew_out.stride[2],
                            ),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_v_new_out,),
                        (tma_v_new_out.atom,),
                        (tensormap_v_new_out_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_v_new_out_ptr)

                if cutlass.const_expr(self.store_w):
                    assert mW_out is not None
                    bounded_w_out = cute.make_tensor(
                        mW_out.iterator,
                        cute.make_layout(
                            (mW_out.shape[0], batch_end, mW_out.shape[2]),
                            stride=(
                                mW_out.stride[0],
                                mW_out.stride[1],
                                mW_out.stride[2],
                            ),
                        ),
                    )
                    tensormap_manager.update_tensormap(
                        (bounded_w_out,),
                        (tma_w_out.atom,),
                        (tensormap_w_out_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_w_out_ptr)
                if cutlass.const_expr(self.use_bf16_h_tma):
                    assert mH_tma_out is not None
                    tensormap_manager.update_tensormap(
                        (mH_tma_out,),
                        (tma_h_out.atom,),
                        (tensormap_h_out_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_h_out_ptr)

                for chunk_idx in cutlass.range(num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    self._timeline_tag(
                        debug_timing, 5, chunk_idx, 0, batch_idx, head_idx, tidx,
                        self.epilogue_warp_id * self.threads_per_warp,
                    )
                    if cutlass.const_expr(self.use_bf16_h_tma):
                        h_checkpoint_idx = (
                            cu_checkpoints[batch_idx]
                            + (chunk_idx * self.b_t) // checkpoint_every_n_tokens
                        )
                        side_output_store_consumer = self.output_h_tma_warp(
                            (sH,),
                            (tma_h_out,),
                            (side_output_store_consumer,),
                            (head_idx, h_checkpoint_idx),
                            (
                                tensormap_manager,
                                tensormap_h_out_ptr,
                            ),
                            (),
                        )
                    if cutlass.const_expr(self.store_v_new or self.store_w):
                        side_output_store_consumer = self.side_output_tma_warp(
                            (sSideOutput,),
                            (tma_v_new_out, tma_w_out),
                            (side_output_store_consumer,),
                            (head_idx, chunk_offset),
                            (
                                tensormap_manager,
                                tensormap_v_new_out_ptr,
                                tensormap_w_out_ptr,
                            ),
                            (),
                        )
                    if cutlass.const_expr(not self.drop_output_store_only):
                        o_store_consumer = self.epilogue_warp(
                            (sO,),
                            (tma_o,),
                            (o_store_consumer,),
                            (head_idx, chunk_offset),
                            (tensormap_manager, tensormap_o_ptr),
                            (
                                debug_timing,
                                cutlass.Int32(5),
                                chunk_idx,
                                batch_idx,
                                head_idx,
                                tidx,
                                self.epilogue_warp_id * self.threads_per_warp,
                            ),
                        )
                    else:
                        # Consume O readiness without materializing the forward output.
                        self._timeline_tag(
                            debug_timing, 5, chunk_idx, 2, batch_idx, head_idx, tidx,
                            self.epilogue_warp_id * self.threads_per_warp,
                        )
                        o_handle = o_store_consumer.wait_and_advance()
                        self._timeline_tag(
                            debug_timing, 5, chunk_idx, 3, batch_idx, head_idx, tidx,
                            self.epilogue_warp_id * self.threads_per_warp,
                        )
                        o_handle.release()
                    self._timeline_tag(
                        debug_timing, 5, chunk_idx, 1, batch_idx, head_idx, tidx,
                        self.epilogue_warp_id * self.threads_per_warp,
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

    # -----------------------------------------------------------------------
    # Per-warp methods  (called from kernel's chunk loop)
    # -----------------------------------------------------------------------
    @cute.jit
    def tma_qkv_warp(
        self,
        mma_args: tuple,
        tma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        tensormap_args: tuple,
        timeline_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warp 9: load Q, K, V, and the optional W RHS for the current chunk.

        Pattern (following fmha.py / dense_gemm_persistent.py):
          1. domain_offset the TMA tensor to (chunk_offset, head_idx, 0) so that
             the logical tile (0, ...) maps to the current chunk.
          2. flat_divide to obtain the tiled global view.
          3. thr_mma.partition_{A,B} to get the TMA-compatible per-thread view.
          4. cpasync.tma_partition -> (tXsX, tXgX) SMEM/global pairs.
          5. acquire pipeline stage, issue cute.copy, signal mbarrier.

        Note on head coordinate: head_idx is the flat KV-head index in [0, h_qv).
        For the hierarchical head layout (h_r, h_qv) with h_r having stride 0
        (broadcast), the flat index maps correctly as long as head_idx < h_qv.
        """
        tiled_mma_qk, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv = mma_args
        tma_q, tma_k, tma_v, tma_w, tma_A = tma_args
        sQ, sK, sV, sW, sAinput = smem_args
        (
            load_q_producer,
            load_k_producer,
            load_v_producer,
            load_w_producer,
            load_A_producer,
        ) = pipeline_args
        chunk_offset, chunk_idx, batch_idx, head_idx = work_args
        (
            tensormap_manager,
            tensormap_q_ptr,
            tensormap_k_ptr,
            tensormap_v_ptr,
            tensormap_w_ptr,
            tensormap_A_ptr,
        ) = tensormap_args

        # Single-CTA mode: no multicast, cta_v = 0.
        cta_layout = cute.make_layout(1)

        # Per-thread MMA slices (cta_v=0 for ONE-CTA mode).
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_qkv = tiled_mma_qkv.get_slice(0)
        thr_mma_qkv_ss = tiled_mma_qkv_ss.get_slice(0)
        thr_mma_kv = tiled_mma_kv.get_slice(0)  # noqa: F841

        # Tile shapes from the MMA tiler (128, 128, 128):
        #   mode[0,2] = (BT, DK) - M,K tile for A (Q) and for B (K) of tiled_mma_qk
        #   mode[1,2] = (BT, DV) - tile shape for loading V (B operand in GEMMs 5/6)
        # (BT, DK)
        qk_tile = cute.select(self.mma_tiler_qk, mode=[0, 2])
        # (DV, BT)
        v_tile = cute.select(self.mma_tiler_qkv, mode=[0, 2])
        # (BT, BT)
        ainv_tile = (self.b_t, self.b_t)

        # ------------------------------------------------------------------
        # K  (B operand of GEMM-kk / GEMM-qk, double-buffered)
        # Tensor shape: (total_tokens, H_hier, DK)
        # TMA tile:     (BT, DK)
        # ------------------------------------------------------------------
        mK = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_k.tma_tensor[None, None, head_idx]
        )
        # (..., num_k_tiles, ...)
        gK = cute.flat_divide(mK, qk_tile)
        tCgK = thr_mma_qk.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            tma_k.atom,
            0,
            cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )

        # Load K for the current chunk into the next available pipeline stage.
        self._timeline_tag_from_args(timeline_args, 2)
        k_handle = load_k_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_k_ptr)

        cute.copy(
            tma_k.atom,
            tKgK[(None, 0, 0)],
            tKsK[(None, k_handle.index)],
            tma_bar_ptr=k_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_k_ptr, cute.AddressSpace.generic
            ),
        )

        self._timeline_tag_from_args(timeline_args, 3)

        if cutlass.const_expr(self.use_input_A):
            self._timeline_tag_from_args(timeline_args, 4)
            mA = cute.domain_offset(
                (chunk_offset, cutlass.Int32(0)), tma_A.tma_tensor[None, None, head_idx]
            )
            gA = cute.flat_divide(mA, ainv_tile)
            tAsA, tAgA = cpasync.tma_partition(
                tma_A.atom,
                0,
                cta_layout,
                cute.group_modes(sAinput, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            a_handle = load_A_producer.acquire_and_advance()
            if chunk_idx == 0:
                tensormap_manager.fence_tensormap_update(tensormap_A_ptr)
            cute.copy(
                tma_A.atom,
                tAgA[(None, 0, 0)],
                tAsA[(None, a_handle.index)],
                tma_bar_ptr=a_handle.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_A_ptr, cute.AddressSpace.generic
                ),
            )

            self._timeline_tag_from_args(timeline_args, 5)

        if cutlass.const_expr(not self.drop_output_store_only):
            # --------------------------------------------------------------
            # Q  (A operand of GEMM-qk/O path, single-buffered)
            # --------------------------------------------------------------
            mQ = cute.domain_offset(
                (chunk_offset, cutlass.Int32(0)), tma_q.tma_tensor[None, None, head_idx]
            )
            gQ = cute.flat_divide(mQ, qk_tile)
            tCgQ = thr_mma_qk.partition_A(gQ)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_q.atom,
                0,
                cta_layout,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tCgQ, 0, 3),
            )

            self._timeline_tag_from_args(timeline_args, 6)
            q_handle = load_q_producer.acquire_and_advance()
            if chunk_idx == 0:
                tensormap_manager.fence_tensormap_update(tensormap_q_ptr)
            cute.copy(
                tma_q.atom,
                tQgQ[(None, 0, 0)],
                tQsQ[(None, q_handle.index)],
                tma_bar_ptr=q_handle.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_q_ptr, cute.AddressSpace.generic
                ),
            )

            self._timeline_tag_from_args(timeline_args, 7)

        # ------------------------------------------------------------------
        # V  (B operand of GEMM-new_v / GEMM-qkv, single-buffered)
        # ------------------------------------------------------------------
        mV = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset), tma_v.tma_tensor[None, None, head_idx]
        )
        gV = cute.flat_divide(mV, v_tile)
        tCgV = thr_mma_qkv_ss.partition_A(gV)
        tVsV, tVgV = cpasync.tma_partition(
            tma_v.atom,
            0,
            cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tCgV, 0, 3),
        )

        self._timeline_tag_from_args(timeline_args, 8)
        v_handle = load_v_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_v_ptr)
        cute.copy(
            tma_v.atom,
            tVgV[(None, 0, 0)],
            tVsV[(None, v_handle.index)],
            tma_bar_ptr=v_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_v_ptr, cute.AddressSpace.generic
            ),
        )

        self._timeline_tag_from_args(timeline_args, 9)

        if cutlass.const_expr(self.store_w):
            mW = cute.domain_offset(
                (cutlass.Int32(0), chunk_offset),
                tma_w.tma_tensor[None, None, head_idx],
            )
            gW = cute.flat_divide(mW, v_tile)
            tCgW = thr_mma_qkv_ss.partition_A(gW)
            tWsW, tWgW = cpasync.tma_partition(
                tma_w.atom,
                0,
                cta_layout,
                cute.group_modes(sW, 0, 3),
                cute.group_modes(tCgW, 0, 3),
            )
            w_handle = load_w_producer.acquire_and_advance()
            if chunk_idx == 0:
                tensormap_manager.fence_tensormap_update(tensormap_w_ptr)
            cute.copy(
                tma_w.atom,
                tWgW[(None, 0, 0)],
                tWsW[(None, w_handle.index)],
                tma_bar_ptr=w_handle.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_w_ptr, cute.AddressSpace.generic
                ),
            )

        return (
            load_q_producer,
            load_k_producer,
            load_v_producer,
            load_w_producer,
            load_A_producer,
        )

    @cute.jit
    def load_gate_beta_warp(
        self,
        tidx: cutlass.Int32,
        gmem_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        timeline_args: tuple,
    ) -> tuple[pipeline.PipelineProducer, pipeline.PipelineProducer]:
        """Warp 10: load gate[BT] and beta[BT] for the current chunk.

        Gate is loaded via ldg (sync G->R), converted to chunk-local cumulative
        log2 in registers, then stored to sCumsumlog SMEM (stride-32 layout
        matching compute_group_0 reads). If gate_is_log_cumsum=True, gate is
        already chunk-local cumulative log2 and the conversion is skipped.

        Beta is loaded via ldgsts (async G->S, cp.async) into sBeta SMEM.

        The last tile uses predicated copies: elements with linear index >= valid_tokens
        are out-of-bounds and receive neutral values (gate=1 -> ln=0, beta=0).

        Thread tidx (lane 0..31) owns positions tidx, tidx+32, tidx+64, tidx+96.
        """
        gate, beta, gate_out = gmem_args
        sCumsumlog, sCumprod, sBeta = smem_args
        load_gate_producer, load_beta_producer = pipeline_args
        chunk_offset, head_idx, is_last_tile, batch_end = work_args

        # lane index
        lidx = tidx % self.threads_per_warp

        mGateHead = gate[None, head_idx]
        gGate = cute.domain_offset((chunk_offset,), mGateHead)
        cGate = cute.domain_offset(
            (chunk_offset,), cute.make_identity_tensor(mGateHead.shape)
        )
        gBeta = cute.domain_offset((chunk_offset,), beta[None, head_idx])
        if cutlass.const_expr(self.store_g):
            assert gate_out is not None, "gate_out must be provided when store_g is True"
            mGateOutHead = gate_out[None, head_idx]
            gGateOut = cute.domain_offset((chunk_offset,), mGateOutHead)
            gGateOut = cute.flat_divide(gGateOut, (self.b_t,))[None, 0]
        else:
            gGateOut = None
        gGate = cute.flat_divide(gGate, (self.b_t,))[None, 0]
        cGate = cute.flat_divide(cGate, (self.b_t,))[None, 0]
        gBeta = cute.flat_divide(gBeta, (self.b_t,))[None, 0]

        # Tiled copy: 1D thread/value layouts; partition_S/D handle element mapping.
        # thread_layout (32,): each of the 32 lanes maps to one row of the b_t tile.
        # value_layout  (4,) : each lane owns 4 elements strided by threads_per_warp.
        thread_layout = cute.make_layout((self.threads_per_warp,), stride=(1,))
        value_layout = cute.make_layout((1,), stride=(1,))

        # Gate: sync G->R (ldg), apply ln + prefix sum, then R->S (sts)
        atom_gate_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        tiled_copy_gate_g2r = cute.make_tiled_copy_tv(
            atom_gate_g2r, thread_layout, value_layout
        )

        # Beta: async G->S (ldgsts / cp.async)
        atom_beta_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.ALWAYS
            ),
            cutlass.Float32,
            num_bits_per_copy=32,
        )
        tiled_copy_beta_g2s = cute.make_tiled_copy_tv(
            atom_beta_g2s, thread_layout, value_layout
        )

        # Per-thread partitions (1D tensors; no manual 2D reshaping needed)
        thr_copy_gate_g2r = tiled_copy_gate_g2r.get_slice(lidx)
        tGgGate = thr_copy_gate_g2r.partition_S(gGate)
        tGsCumsumlog = thr_copy_gate_g2r.partition_D(sCumsumlog)
        tGsCumprod = thr_copy_gate_g2r.partition_D(sCumprod)
        if cutlass.const_expr(self.store_g):
            assert gGateOut is not None, "gGateOut must be available when store_g is True"
            tGgGateOut = thr_copy_gate_g2r.partition_D(gGateOut)
        else:
            tGgGateOut = None

        thr_copy_beta_g2s = tiled_copy_beta_g2s.get_slice(lidx)
        tBgBeta = thr_copy_beta_g2s.partition_S(gBeta)
        tBsBeta = thr_copy_beta_g2s.partition_D(sBeta)

        self._timeline_tag_from_args(timeline_args, 2)
        gate_handle = load_gate_producer.acquire_and_advance()

        rGate = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrGate = tiled_copy_gate_g2r.retile(rGate)
        rCumprod = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrCumprod = tiled_copy_gate_g2r.retile(rCumprod)

        # --- Predicate (last tile only): compute once, reuse for gate and beta ---
        if cutlass.const_expr(is_last_tile):
            valid_tokens = batch_end  # noqa: F841
            tGcGate = thr_copy_gate_g2r.partition_S(cGate)
            tGpGate = cute.make_rmem_tensor(
                ((tGcGate.shape[0][1],), tGcGate.shape[1]), cutlass.Boolean
            )
            for i in range(cute.size(tGpGate)):
                tGpGate[i] = cute.elem_less(tGcGate[i][0], batch_end)

        # --- Gate load ---
        if cutlass.const_expr(is_last_tile):
            if cutlass.const_expr(self.gate_is_log_cumsum):
                if cutlass.const_expr(self.enable_varlen_tail):
                    # Extend the last valid cumulative value across padded rows.
                    tGrGate.fill(mGateHead[batch_end - 1])
                else:
                    tGrGate.fill(0.0)
            else:
                if cutlass.const_expr(self.gate_is_log_decay):
                    # OOB neutral for raw Megatron log-decay: 0.0 contributes no decay.
                    tGrGate.fill(0.0)
                else:
                    # OOB neutral: 1.0 -> log2 ~= 0.0 (no decay contribution).
                    tGrGate.fill(1.0)
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate, pred=tGpGate)
        else:
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate)

        # --- Convert gate to chunk-local cumulative log2 if needed ---
        if not cutlass.const_expr(self.gate_is_log_cumsum):
            for i in range(cute.size(tGrGate)):
                if cutlass.const_expr(self.gate_is_log_decay):
                    tGrGate[i] = tGrGate[i] * 1.4426950408889634
                else:
                    tGrGate[i] = cute.math.log2(tGrGate[i] + 1e-10, fastmath=True)
            for offset in [1, 2, 4, 8, 16]:
                for col in range(cute.size(tGrGate)):
                    n = cute.arch.shuffle_sync_up(
                        tGrGate[col], offset, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lidx >= offset:
                        tGrGate[col] = tGrGate[col] + n
            sum_v = 0.0  # noqa: F841
            for col in range(1, cute.size(tGrGate)):
                last_v = cute.arch.shuffle_sync(
                    tGrGate[col - 1],
                    self.threads_per_warp - 1,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=self.threads_per_warp - 1,
                )
                tGrGate[col] += last_v
        if cutlass.const_expr(self.store_g):
            assert tGgGateOut is not None, "tGgGateOut must be available when store_g is True"
            if cutlass.const_expr(is_last_tile):
                cute.copy(tiled_copy_gate_g2r, tGrGate, tGgGateOut, pred=tGpGate)
            else:
                cute.copy(tiled_copy_gate_g2r, tGrGate, tGgGateOut)
        for col in range(cute.size(tGrGate)):
            tGrCumprod[col] = cute.math.exp2(tGrGate[col], fastmath=True)
        cute.copy(
            tiled_copy_gate_g2r, tGrGate, tGsCumsumlog[None, None, 0, gate_handle.index]
        )
        cute.copy(
            tiled_copy_gate_g2r,
            tGrCumprod,
            tGsCumprod[None, None, 0, gate_handle.index],
        )

        gate_handle.commit()
        self._timeline_tag_from_args(timeline_args, 3)

        # --- Beta load ---
        self._timeline_tag_from_args(timeline_args, 4)
        beta_handle = load_beta_producer.acquire_and_advance()
        if cutlass.const_expr(is_last_tile):
            # clear OOB slots before predicated cp.async
            tBsBeta.fill(0.0)
            cute.copy(
                tiled_copy_beta_g2s,
                tBgBeta,
                tBsBeta[None, None, 0, beta_handle.index],
                pred=tGpGate,
            )
        else:
            cute.copy(
                tiled_copy_beta_g2s, tBgBeta, tBsBeta[None, None, 0, beta_handle.index]
            )
        beta_handle.commit()
        self._timeline_tag_from_args(timeline_args, 5)

        return load_gate_producer, load_beta_producer

    @cute.jit
    def mma_warp(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        timeline_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Warp 8: issue all 7 GEMMs in dependency order."""
        tiled_mma_qk, tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv = (
            mma_args
        )
        sQ, sK, sK_trans, sV, sAinv, sQk = smem_args
        (
            shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            shared_inp_ready_consumer,
        ) = pipeline_args
        (is_first_chunk,) = work_args

        valid_state = not is_first_chunk or self.use_initial_state

        # ------------------------------------------------------------------
        # Build TMEM accumulator views
        # ------------------------------------------------------------------
        # Shared acc (GEMMs 1/2/3/5/6) - 2 stages, layout from tiled_mma_qk
        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_shared_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tCtAcc_fake.layout
        )

        shared_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(shared_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )

        # q*state acc (GEMM 4 only) - 1 stage, layout from tiled_mma_qs
        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )

        # state acc (GEMM 7 only) - 1 stage, layout from tiled_mma_kv
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )

        # ------------------------------------------------------------------
        # Pre-create operand fragments (stage dim preserved; sliced at GEMM time)
        # tiled_mma_qk operands (GEMMs 1, 2)
        # K as A for GEMM 1 (kk)
        tCrK_A = tiled_mma_qk.make_fragment_A(sK)
        # K as B for GEMMs 1+2 (kk, qk)
        tCrK_B = tiled_mma_qk.make_fragment_B(sK)
        # Q as A for GEMM 2 (qk)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ)
        # tiled_mma_qs operands (GEMMs 3, 4)
        # K as A for GEMM 3 (k*state)
        tCrS_A = tCtState_inp
        # Q as A for GEMM 4 (q*state)
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ)
        # S_prev as B for GEMMs 3+4
        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sK)
        # tiled_mma_qkv operands (GEMMs 5, 6)
        # V-KS as A for GEMM 5
        if cutlass.const_expr(valid_state):
            tCrV_A = tCtShared_inp
        else:
            tCrV_A = tiled_mma_qkv_ss.make_fragment_A(sV)
        # A_inv as B for GEMM 5
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)

        # W_qkv as A for GEMM 6
        tCrQkv_A = tCtShared_inp
        # NV as B for GEMM 6
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        # tiled_mma_kv operands (GEMM 7)
        # delta as A for GEMM 7
        tCrDecayV_A = tCtShared_inp
        # K^T as B for GEMM 7
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)

        # ---- GEMM 1: kk  (K @ K^T -> shared acc) ----------------------------
        # Both A and B are K; valid because tiled_mma_qk has K-major for both operands.
        self._timeline_tag_from_args(timeline_args, 2)
        k_handle = load_k_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 3)
        kk_handle = shared_acc_producer.acquire_and_advance()

        num_kphases = cute.size(tCrK_A, mode=[2])
        if cutlass.const_expr(not self.use_input_A):
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qk,
                    tCtShared[None, None, None, kk_handle.index],
                    tCrK_A[None, None, kphase_idx, k_handle.index],
                    tCrK_B[None, None, kphase_idx, k_handle.index],
                    tCtShared[None, None, None, kk_handle.index],
                )

        # Signal W_kk ready -> CG0 kk_epi, or a dummy token when saved A is used.
        kk_handle.commit()

        # In drop-output mode Q no longer feeds qk/GEMM4 math. Skip qk compute
        # and the Q TMA producer/consumer path while preserving shared_acc order.
        if cutlass.const_expr(not self.drop_output_store_only):
            self._timeline_tag_from_args(timeline_args, 4)
            q_handle = load_q_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 5)
            qk_handle = shared_acc_producer.acquire_and_advance()

            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qk,
                    tCtShared[None, None, None, qk_handle.index],
                    tCrQ_A[None, None, kphase_idx, q_handle.index],
                    tCrK_B[None, None, kphase_idx, k_handle.index],
                    tCtShared[None, None, None, qk_handle.index],
                )

            # Signal W_qk ready -> CG0 qk_epi
            qk_handle.commit()
        else:
            # Keep the shared_acc barrier token that GEMM 2 used in the full forward path.
            qk_handle = shared_acc_producer.acquire_and_advance()
            qk_handle.commit()

        # ---- GEMM 3: k*state  (K @ S_prev -> shared acc) --------------------
        # ---- GEMM 4: q*state  (Q @ S_prev -> tmem_q_state) ------------------
        # Skipped on the first chunk when use_initial_state is False (S_prev = 0, outputs are zero).
        if valid_state:
            self._timeline_tag_from_args(timeline_args, 6)
            s_handle = state_inp_ready_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 7)
            ks_handle = shared_acc_producer.acquire_and_advance()

            num_kphases_qs = cute.size(tCrS_A, mode=[2])
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtShared[None, None, None, ks_handle.index],
                    tCrS_A[None, None, kphase_idx, s_handle.index],
                    tCrK_B_qs[None, None, kphase_idx, k_handle.index],
                    tCtShared[None, None, None, ks_handle.index],
                )
            ks_handle.commit()

            if cutlass.const_expr(not self.drop_output_store_only):
                q_state_acc_handle = q_state_acc_producer.acquire_and_advance()
                # S_prev still loaded (same s_handle.index as GEMM 3).
                num_kphases_qs = cute.size(tCrS_A, mode=[2])
                for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                    tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_qs,
                        tCtQState[None, None, None, q_state_acc_handle.index],
                        tCrS_A[None, None, kphase_idx, s_handle.index],
                        tCrQ_B_qs[None, None, kphase_idx, q_handle.index],
                        tCtQState[None, None, None, q_state_acc_handle.index],
                    )
                q_state_acc_handle.commit()
            else:
                # Preserve the q_state_acc barrier token that GEMM 4 used in full forward.
                q_state_acc_handle = q_state_acc_producer.acquire_and_advance()
                q_state_acc_handle.commit()
            # Release state SMEM (S_prev fully consumed by training path GEMM 3;
            # full forward additionally consumed it in GEMM 4 above).
            s_handle.release()

        if cutlass.const_expr(not self.drop_output_store_only):
            q_handle.release()

        # ---- GEMM 5: new_v  (A_inv @ (V-KS) -> shared acc) ------------------
        # A_inv from CG0 (a_inv_ready); V-KS from CG1 (v_ks_ready, stored in sV).
        self._timeline_tag_from_args(timeline_args, 8)
        vks_handle = shared_inp_ready_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 9)
        self._timeline_tag_from_args(timeline_args, 10)
        ainv_handle = a_inv_ready_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 11)
        nv_handle = shared_acc_producer.acquire_and_advance()

        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        cur_tiled_mma_qkv = tiled_mma_qkv if valid_state else tiled_mma_qkv_ss
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            cur_tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                cur_tiled_mma_qkv,
                tCtShared[None, None, None, nv_handle.index],
                tCrV_A[None, None, kphase_idx, vks_handle.index],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtShared[None, None, None, nv_handle.index],
            )

        nv_handle.commit()
        vks_handle.release()

        # Optional training side output: W = A_beta @ (K * cumprod).
        # CG1 stages K*cumprod in the shared-input TMEM pipeline. Reuse the same
        # A_beta SMEM tile before releasing it to avoid a second forward replay.
        if cutlass.const_expr(self.store_w):
            self._timeline_tag_from_args(timeline_args, 12)
            wks_handle = shared_inp_ready_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 13)
            w_handle = shared_acc_producer.acquire_and_advance()
            for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
                tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qkv,
                    tCtShared[None, None, None, w_handle.index],
                    tCrQkv_A[None, None, kphase_idx, wks_handle.index],
                    tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                    tCtShared[None, None, None, w_handle.index],
                )
            w_handle.commit()
            wks_handle.release()

        ainv_handle.release()

        if cutlass.const_expr(not self.training_side_outputs_only):
            # ---- GEMM 6: qkv  (W_qkv @ NV -> q * state acc) ------------------
            # W_qkv from CG0 (qk_ready, stored in sQk); NV from CG1 (new_v_ready, stored in sNv).
            self._timeline_tag_from_args(timeline_args, 14)
            qkv_qk_handle = qk_ready_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 15)
            self._timeline_tag_from_args(timeline_args, 16)
            qkv_nv_handle = shared_inp_ready_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 17)
            q_state_acc_handle = q_state_acc_producer.acquire_and_advance()

            if cutlass.const_expr(not self.drop_output_store_only):
                num_kphases_qkv = cute.size(tCrQkv_A, mode=[2])
                for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
                    tiled_mma_qkv.set(
                        tcgen05.Field.ACCUMULATE, valid_state or (kphase_idx != 0)
                    )
                    cute.gemm(
                        tiled_mma_qkv,
                        tCtQState[None, None, None, q_state_acc_handle.index],
                        tCrQkv_A[None, None, kphase_idx, qkv_nv_handle.index],
                        tCrNv_B[None, None, kphase_idx, qkv_qk_handle.index],
                        tCtQState[None, None, None, q_state_acc_handle.index],
                    )

            qkv_qk_handle.release()
            qkv_nv_handle.release()
            q_state_acc_handle.commit()
        else:
            # Preserve the q_state_acc/shared_inp slots that GEMM 6 used in full forward.
            q_state_acc_producer.advance()
            shared_inp_ready_consumer.advance()

        # ---- GEMM 7: kv_update  (K^T @ delta -> state TMEM) ---------------------
        # delta from CG1 (decay_v_ready, stored in sDecayV); K^T reuses k_handle slot.
        # First chunk: zero-init on kphase 0. Subsequent chunks: always accumulate.
        if cutlass.const_expr(self.use_initial_state and is_first_chunk):
            kv_acc_producer.advance()
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        self._timeline_tag_from_args(timeline_args, 18)
        dv_handle = shared_inp_ready_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 19)

        num_kphases_kv = cute.size(tCrKt_B, mode=[2])
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, valid_state or (kphase_idx != 0))
            cute.gemm(
                tiled_mma_kv,
                tCtState[None, None, None, kv_acc_handle.index],
                tCrDecayV_A[None, None, kphase_idx, dv_handle.index],
                tCrKt_B[None, None, kphase_idx, k_handle.index],
                tCtState[None, None, None, kv_acc_handle.index],
            )
        kv_acc_handle.commit()
        dv_handle.release()
        # K SMEM slot now free for next chunk
        k_handle.release()

        return (  # type: ignore[return-value]
            shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            shared_inp_ready_consumer,
        )

    @cute.jit
    def compute_group_0(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        timeline_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 0-3: T-pairwise, kk_epi, inverse, qk_epi."""
        (tiled_mma_qk,) = mma_args
        sCumsumlog, sBeta, sAinv, sAinput, sQk = smem_args
        (
            load_gate_consumer,
            load_beta_consumer,
            load_A_consumer,
            shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
            group_order_producer,
        ) = pipeline_args
        (mA_in, mA_out, head_idx, chunk_offset, is_first_chunk, batch_end) = (
            work_args
        )

        # ------------------------------------------------------------------
        # Preamble: per-thread ID within CG0 and TMEM copy setup
        # ------------------------------------------------------------------
        # Local thread ID within CG0 (0..127)
        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0

        # Build TMEM tensor view of shared_acc (both stages):
        #   shape = (per_thread_acc_shape..., tmem_shared_acc_stages)
        tAcc_shape = tiled_mma_qk.partition_shape_C(
            (self.mma_tiler_qk[0], self.mma_tiler_qk[1])
        )
        tAcc_wo_stages = tiled_mma_qk.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(
            tAcc_wo_stages.iterator,
            cute.flat_product(
                tAcc_wo_stages.layout,
                cute.make_layout((self.tmem_shared_acc_stages,), stride=(1,)),
            ),
        )
        tStS_staged = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tAcc.layout
        )
        tStS_staged_mn_view = self.transform_partitioned_tensor_layout(tStS_staged)
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        # TMEM load copy atom and tiled copy (loads fp32 accum from TMEM -> registers)
        tStS_for_t2r = tStS_staged[(None, None), 0, 0, 0]
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tStS_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg0_tidx)

        # Per-thread TMEM source (staged).
        # tTR_tStS shape: (A, B, NUM_SUBS, NUM_STAGES)
        # Each subtile is tTR_tStS[None, None, sub, stage] with shape (A, B).
        tTR_tStS = thr_shared_t2r.partition_S(tStS_staged_mn_view)
        tTR_tScS = thr_shared_t2r.partition_D(cS)

        sub_tile_size = 32

        # SMEM store copies: fp32 registers -> fp16 SMEM (A-operands for GEMM 5 and 6).
        # make_tiled_copy_D mirrors tmem_tiled_copy's thread-value mapping so the
        # register layout from the TMEM load aligns with the SMEM destination partition.
        atom_ainv_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_ainv_r2s = cute.make_tiled_copy_D(atom_ainv_r2s, tiled_shared_t2r)
        sAinv_mn_view = self.transform_partitioned_tensor_layout(sAinv)
        thr_ainv_r2s = tiled_ainv_r2s.get_slice(cg0_tidx)
        tCsAI = thr_ainv_r2s.partition_D(sAinv_mn_view)

        atom_ainv_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_ainv_s2r = cute.make_tiled_copy_D(atom_ainv_s2r, tiled_shared_t2r)
        thr_ainv_s2r = tiled_ainv_s2r.get_slice(cg0_tidx)

        atom_qk_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_qk_r2s = cute.make_tiled_copy_D(atom_qk_r2s, tiled_shared_t2r)
        sQk_mn_view = self.transform_partitioned_tensor_layout(sQk)
        tCsQK = tiled_qk_r2s.get_slice(cg0_tidx).partition_D(sQk_mn_view)


        # ------------------------------------------------------------------
        # Step 1: T-pairwise - wait for cumsumlog, compute T row, release
        #   sCumsumlog[t] = sum_{l=0}^{t} log(gate_l)  (prefix sum done by gate warp)
        #   T[i,j]        = exp(cumsumlog[i] - cumsumlog[j])  for i>=j, else 0
        #
        #   The gate warp (warp 10) stores the inclusive prefix sum directly into
        #   sCumsumlog before committing the gate pipeline handle, so CG0 can read
        #   the final cumsumlog values immediately after wait_and_advance().
        # ------------------------------------------------------------------
        self._timeline_tag_from_args(timeline_args, 2)
        gate_handle = load_gate_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 3)

        # Pre-compute T[i,*] for all columns into registers BEFORE waiting for TMEM.
        # This overlaps the 128 exp2f calls with the MMA warp's GEMMs so that
        # kk_epi and qk_epi only need a register multiply - no SMEM reads in the
        # hot loop.  Register cost: BT fp32 = 128 regs (live until after qk_epi).
        #   tGrT[j] = exp2(cumsumlog[i] - cumsumlog[j])  for i >= j, else 0.0
        #   where i = cg0_tidx (this thread's row), j = k (col, compile-time).
        if cutlass.const_expr(not self.use_input_A or not self.drop_output_store_only):
            rCumsumlog = cute.make_rmem_tensor((2, 16), self.acc_dtype)
            tGrCumsumlog = thr_shared_t2r.partition_D(rCumsumlog)
            for k in cutlass.range_constexpr(cute.size(tTR_tScS)):
                coord = tTR_tScS[k]
                tGrCumsumlog[k] = (
                    cute.math.exp2(
                        sCumsumlog[coord[0]] - sCumsumlog[coord[1]], fastmath=True
                    )
                    if coord[0] >= coord[1]
                    else 0.0
                )
        gate_handle.release()

        # Wait for beta ready (used in both kk_epi and qk_epi scaling)
        self._timeline_tag_from_args(timeline_args, 4)
        beta_handle = load_beta_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 5)

        if cutlass.const_expr(not self.use_input_A):
            rBeta = cute.make_rmem_tensor((2,), self.acc_dtype)
            rBeta = cute.make_tensor(
                rBeta.iterator,
                cute.flat_product(rBeta.layout, cute.make_layout((16,), stride=(0,))),
            )
            tGrBeta = thr_shared_t2r.partition_D(rBeta)
            for k in cutlass.range_constexpr(cute.size(tTR_tScS)):
                coord = tTR_tScS[k]
                tGrBeta[k] = sBeta[coord[0], 0, beta_handle.index]

        # ------------------------------------------------------------------
        # Step 2: kk_epi - load W_kk (GEMM 1 result from TMEM), scale -> M_kk
        #   Depends on: shared_acc stage 0 (MMA warp GEMM 1 kk done)
        #   W_kk[i,j] *= T[i,j] * beta[i]  where T[i,j] = exp2(cumsumlog[i]-cumsumlog[j])
        #   TMEM load is split into num_kk_subs subtiles (each covering kk_sub_size cols).
        # ------------------------------------------------------------------
        # Acquire sAinvNv slot, fence, signal
        ainv_handle = a_inv_ready_producer.acquire_and_advance()
        if cutlass.const_expr(not self.training_side_outputs_only):
            # Full forward only: acquire sQkOstore slot, write W_qkv (fp16), fence, signal.
            qk_ready_handle = qk_ready_producer.acquire_and_advance()
            group_order_handle = group_order_producer.acquire_and_advance()
        self._timeline_tag_from_args(timeline_args, 6)
        kk_handle = shared_acc_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 7)

        # tStS_for_t2r = tStS_staged[(None, None), 0, 0, kk_handle.index]
        # tTR_tStS = thr_shared_t2r.partition_S(tStS_for_t2r)
        # tKKrKK: full-size register buffer for M_kk (inverse step), filled subtile by subtile.
        # tKKrKK[j] = M_kk[cg0_tidx, j] after the loop (thread owns its full row).
        # SMEM write is deferred to the inverse step (row-major layout, below).
        tKKrKK = cute.make_rmem_tensor_like(tTR_tScS, self.acc_dtype)

        tKKrKK_out = cute.make_rmem_tensor_like(tKKrKK, self.io_dtype)
        if cutlass.const_expr(not self.use_input_A):
            tCrAI = tiled_ainv_r2s.retile(tKKrKK_out)
            for sub in cutlass.range(tKKrKK.shape[2]):
                cute.copy(
                    tiled_shared_t2r,
                    tTR_tStS[None, 0, sub, kk_handle.index],
                    tKKrKK[None, 0, sub],
                )
                for k in cutlass.range(sub_tile_size):
                    tKKrKK[k, 0, sub] = (
                        tKKrKK[k, 0, sub] * tGrCumsumlog[k, 0, sub] * tGrBeta[k, 0, sub]
                    )
                tKKrKK_out[None, 0, sub].store(
                    tKKrKK[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_ainv_r2s,
                    tCrAI[None, 0, sub],
                    tCsAI[None, 0, sub, ainv_handle.index],
                )
        # Release shared_acc stage 0 (CG1 also releases its side - collective barrier)
        kk_handle.release()

        if cutlass.const_expr(not self.drop_output_store_only):
            # ------------------------------------------------------------------
            # Step 3: qk_epi - load W_qk (GEMM 2 result from TMEM), scale -> W_qkv
            #   Depends on: shared_acc stage 1 (MMA warp GEMM 2 qk done)
            #   W_qk[i,j] *= T[i,j] * scale  where T[i,j] = exp2(cumsumlog[i]-cumsumlog[j])
            #   Same subtile structure as kk_epi (num_kk_subs / kk_sub_size reused).
            #   Done before inverse so that qk_ready signals GEMM 6 early, letting the
            #   MMA warp run GEMM 6 in parallel with the inverse computation below.
            # ------------------------------------------------------------------
            self._timeline_tag_from_args(timeline_args, 8)
            qk_handle = shared_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 9)

            tStS_for_t2r = tStS_staged[(None, None), 0, 0, qk_handle.index]
            # tTR_tStS = thr_shared_t2r.partition_S(tStS_for_t2r)

            # tQKrQK: full-size register buffer for W_qkv (SMEM write), filled subtile by subtile.
            tQKrQK = cute.make_rmem_tensor_like(tTR_tScS, self.acc_dtype)

            # Convert fp32 tQKrQK -> fp16 and store to sQk, one subtile at a time.
            tQKrQK_out = cute.make_rmem_tensor_like(tQKrQK, self.io_dtype)
            tCrQK = tiled_qk_r2s.retile(tQKrQK_out)
            for sub in cutlass.range(tQKrQK.shape[2]):
                cute.copy(
                    tiled_shared_t2r,
                    tTR_tStS[None, 0, sub, qk_handle.index],
                    tQKrQK[None, 0, sub],
                )
                for k in cutlass.range(sub_tile_size):
                    tQKrQK[k, 0, sub] = tQKrQK[k, 0, sub] * tGrCumsumlog[k, 0, sub] * scale
                tQKrQK_out[None, 0, sub].store(
                    tQKrQK[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_qk_r2s, tCrQK[None, 0, sub], tCsQK[None, 0, sub, qk_handle.index]
                )
            cute.arch.fence_view_async_shared()
            # Release shared_acc stage 1
            qk_handle.release()
            group_order_handle.commit()

            qk_ready_handle.commit()
        else:
            # Match the dummy shared_acc producer token that replaces GEMM 2.
            self._timeline_tag_from_args(timeline_args, 8)
            qk_handle = shared_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 9)
            qk_handle.release()
            if cutlass.const_expr(not self.training_side_outputs_only):
                # GEMM 6 and qkv_epilogue still consume these barriers in the
                # conservative drop-output path, but their data is discarded.
                group_order_handle.commit()
                qk_ready_handle.commit()

        # Advance past shared_acc stages used by GEMMs 3/4 (K*State, Q*State).
        # These stages form a collective barrier with CG1; CG0 must advance
        # even though it does not read the results, so CG1 can proceed.
        # First chunk without valid state skips GEMM 4 (Q*State), so only
        # one advance is needed instead of two.
        shared_acc_consumer.advance()
        valid_state = not is_first_chunk or self.use_initial_state
        if valid_state:
            shared_acc_consumer.advance()
        if cutlass.const_expr(self.store_w):
            shared_acc_consumer.advance()

        # ------------------------------------------------------------------
        # Step 4: inverse - compute A_inv = (I + M_kk)^{-1}, write to sAinvNv
        #   Done after qk_epi so the inverse computation overlaps with GEMM 6
        #   (qkv) running in the MMA warp.
        # ------------------------------------------------------------------

        if cutlass.const_expr(not self.use_input_A):
            # -- Hierarchical blockwise inverse: A_inv = (I + M_kk)^{-1} ----------
            # Thread cg0_tidx owns row cg0_tidx of the BTxBT matrix.
            # sAinv is reinterpreted as row-major (BTxBT) fp16 for the algorithm;
            # the MMA-ready A-operand layout is written back via tiled_store_ainv after.
            # NOTE: assumes io_dtype == Float16 (algorithm uses fp16 SMEM + fp32 accumulators).
            warp_id = cg0_tidx // 32
            lane_id = cg0_tidx % 32

            sA = sAinv_mn_view[None, None, ainv_handle.index]
            # Stage 1: Gauss-Jordan inversion of BT//8 = 16 diagonal 8x8 blocks.
            sM_8x8 = cute.flat_divide(sA, (8, 8))
            self.inverse_barrier.arrive_and_wait()
            if warp_id < 2:
                self._invert_diagonal_NxN(
                    sM_8x8[None, None, cg0_tidx // 8, cg0_tidx // 8], cg0_tidx, 8
                )
            self.inverse_barrier.arrive_and_wait()

            # Stage 2: off-diagonal correction 8x8 -> 16x16.
            # 8 diagonal 16x16 tiles; each warp handles 2 tiles sequentially.
            sM_16x16 = cute.flat_divide(sA, (16, 16))
            self._blockwise_diagonal_8x8_to_16x16(
                sM_16x16[None, None, warp_id, warp_id], lane_id
            )
            self.inverse_barrier.arrive_and_wait()

            # Stage 3: off-diagonal correction 16x16 -> 32x32.
            # 4 diagonal 32x32 tiles; one tile per warp.
            sM_32x32 = cute.flat_divide(sA, (32, 32))
            if warp_id < 2:
                self._blockwise_diagonal_16x16_to_32x32(
                    sM_32x32[None, None, warp_id, warp_id], lane_id
                )
            self.inverse_barrier.arrive_and_wait()

            # Stage 4: off-diagonal correction 32x32 -> 64x64.
            # 2 diagonal 64x64 tiles; warps 0,1 on tile 0, warps 2,3 on tile 1.
            sM_64x64 = cute.flat_divide(sA, (64, 64))
            if warp_id < 2:
                self._blockwise_diagonal_32x32_to_64x64(
                    sM_64x64[None, None, warp_id // 2, warp_id // 2], warp_id, lane_id
                )
            self.inverse_barrier.arrive_and_wait()

        # tCrAI receives unscaled A_inv either from saved input_A or freshly inverted M_kk.
        # Apply beta column-scaling: A_inv[i,j] *= beta[j]  (equivalent to A_inv @ diag(beta))
        tCsAI = thr_ainv_s2r.partition_S(sAinv_mn_view)
        tCsAI = tCsAI[None, None, None, ainv_handle.index]
        tCrAI = cute.make_rmem_tensor_like(tCsAI, self.io_dtype)
        tCrAI_acc = cute.make_rmem_tensor_like(tCrAI, self.acc_dtype)
        rBeta = cute.make_rmem_tensor((16,), self.acc_dtype)
        rBeta = cute.make_tensor(
            rBeta.iterator,
            cute.flat_product(cute.make_layout((2,), stride=(0,)), rBeta.layout),
        )
        tKKrBeta = thr_ainv_s2r.partition_D(rBeta)
        for k in cutlass.range_constexpr(cute.size(tTR_tScS)):
            coord = tTR_tScS[k]
            tKKrBeta[k] = sBeta[coord[1], 0, beta_handle.index]
        if cutlass.const_expr(self.use_input_A):
            self._timeline_tag_from_args(timeline_args, 10)
            a_load_handle = load_A_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 11)
            tCsAinput = thr_ainv_s2r.partition_D(
                sAinput[None, None, a_load_handle.index]
            )
            cute.autovec_copy(tCsAinput, tCrAI)
            a_load_handle.release()
        else:
            cute.copy(
                tiled_ainv_s2r,
                tCsAI,
                tCrAI,
            )
        if cutlass.const_expr(self.store_A):
            assert mA_out is not None, "mA_out must be provided when store_A is True"
            gA_out = cute.domain_offset(
                (chunk_offset, cutlass.Int32(0)), mA_out[None, None, head_idx]
            )
            gA_tile = cute.flat_divide(gA_out, (self.b_t, self.b_t))[None, None, 0, 0]
            tCgAI = thr_ainv_s2r.partition_D(gA_tile)
            if cutlass.const_expr(not self.enable_varlen_tail):
                cute.autovec_copy(
                    tCrAI,
                    tCgAI,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
            else:
                if chunk_offset + self.b_t <= batch_end:
                    cute.autovec_copy(
                        tCrAI,
                        tCgAI,
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                    )
                else:
                    for k in cutlass.range_constexpr(cute.size(tCrAI)):
                        coord = tTR_tScS[k]
                        if chunk_offset + coord[0] < batch_end:
                            tCgAI[k] = tCrAI[k]

        tCrAI_acc.store(tCrAI.load().to(self.acc_dtype))
        for k in cutlass.range(cute.size(tCrAI)):
            tCrAI_acc[k] = tCrAI_acc[k] * tKKrBeta[k]
        tCrAI.store(tCrAI_acc.load().to(self.io_dtype))
        cute.copy(
            tiled_ainv_r2s,
            tCrAI,
            tCsAI,
        )

        # Fence SMEM writes and signal A_inv ready to MMA warp (GEMM 5 can start)
        cute.arch.fence_view_async_shared()

        ainv_handle.commit()

        # Release beta pipeline (beta fully consumed by this chunk)
        beta_handle.release()

        return (  # type: ignore[return-value]
            load_gate_consumer,
            load_beta_consumer,
            load_A_consumer,
            shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
            group_order_producer,
        )

    # ------------------------------------------------------------------
    # Hierarchical blockwise inverse helpers (ported from gdn_inverse_verify.py)
    # Compute X = (I + M)^{-1} for a 128x128 unit lower-triangular matrix in-place
    # on a row-major fp16 SMEM buffer.  5-stage algorithm:
    #   Stage 1: Gauss-Jordan inversion of 16 diagonal 8x8 blocks (warp shuffle)
    #   Stage 2: 8x8 -> 16x16 via warp MMA  (SM80_16x8x8)
    #   Stage 3: 16x16 -> 32x32 via warp MMA (SM80_16x8x16)
    #   Stage 4: 32x32 -> 64x64 via warp MMA, 2 warps per 64x64 tile
    #   Stage 5: 64x64 -> 128x128 via warp MMA, 4 warps on full matrix
    # ------------------------------------------------------------------

    def _make_acc_tensor_into_a_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret accumulator tensor as an A-operand tensor for the next MMA.

        For SM80_16x8x8 (ratio=1) the layout is unchanged; for SM80_16x8x16 (ratio=2)
        the C-frag atom size differs from the A-frag atom size and requires a reshape.
        """
        acc_layout_divided = cute.logical_divide(acc.layout, (None, None, 2))
        acc_layout_a = cute.make_layout(
            (
                (acc_layout_divided.shape[0], acc_layout_divided.shape[2][0]),
                acc_layout_divided.shape[1],
                acc_layout_divided.shape[2][1],
            ),
            stride=(
                (acc_layout_divided.stride[0], acc_layout_divided.stride[2][0]),
                acc_layout_divided.stride[1],
                acc_layout_divided.stride[2][1],
            ),
        )
        return cute.make_tensor(acc.iterator, acc_layout_a)

    @cute.jit
    def _invert_diagonal_NxN(self, mat_NxN, tidx, N: int = 8):
        """Stage 1: Gauss-Jordan inversion of one diagonal NxN block in-place (fp16 SMEM).

        Each thread owns one row (tidx_in_group = tidx % N).
        Uses warp shuffle to broadcast pivot values; no __syncthreads inside.
        N-1 pivot steps, compile-time unrolled.
        """
        tidx_in_group = tidx % N
        row_f16 = cute.make_rmem_tensor((N,), self.io_dtype)
        cute.autovec_copy(mat_NxN[tidx_in_group, None], row_f16)
        row = cute.make_rmem_tensor_like(row_f16, self.acc_dtype)
        row.store(row_f16.load().to(cutlass.Float32))
        for i in cutlass.range_constexpr(N):
            row[i] = 1.0 if tidx_in_group == i else row[i]
        for src_row in cutlass.range_constexpr(N - 1):
            row_scale = -row[src_row]
            for i in cutlass.range_constexpr(src_row):
                shfl_val = cute.arch.shuffle_sync(
                    row[i], src_row, mask=0xFFFFFFFF, mask_and_clamp=0b1100000011111
                )
                row[i] = (
                    row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
                )
            row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

        row_f16.store(row.load().to(self.io_dtype))
        cute.autovec_copy(row_f16, mat_NxN[tidx_in_group, None])

    @cute.jit
    def _blockwise_diagonal_8x8_to_16x16(self, mat_16x16, lane_id):
        """Stage 2: off-diagonal correction for one 16x16 diagonal tile (8x8 -> 16x16).

        After Stage 1 each diagonal 8x8 is inverted.  Computes the bottom-left 8x8
        correction block: C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x8_F32F16F16F32_TN, single warp.  D^{-1} broadcast 8x8 -> 16x8.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 8)
        )
        tiled_mma = cute.make_tiled_mma(mma_atom, cute.make_layout((1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_8x8_2x2 = cute.flat_divide(mat_16x16, (8, 8))
        sDinv = mat_8x8_2x2[None, None, 1, 1]
        sC = mat_8x8_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_8x8_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_8x8_2x2[None, None, 1, 0]

        sDinv_m_bcast = cute.make_tensor(
            sDinv.iterator,
            cute.logical_product(sDinv.layout, (cute.make_layout((2,), stride=(0,)),)),
        )
        sO_m_bcast = cute.make_tensor(
            sO.iterator,
            cute.logical_product(sO.layout, (cute.make_layout((2,), stride=(0,)),)),
        )

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv_m_bcast))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))

        tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast)
        tOsDinv = cute.logical_divide(tOsDinv, (tOsDinv.shape[0], None, None))
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOrDinv_cv = cute.logical_divide(tOrDinv_cv, (tOrDinv_cv.shape[0], None, None))
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO_m_bcast)
        tOsO = cute.logical_divide(tOsO, (tOsO.shape[0], None, None))
        tOrO_cv = O_thr_copy.retile(tOrO)
        tOrO_cv = cute.logical_divide(tOrO_cv, (tOrO_cv.shape[0], None, None))
        cute.copy(
            D_tiled_copy,
            tOsDinv[(None, 0), None, None],
            tOrDinv_cv[(None, 0), None, None],
        )
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, self.io_dtype)
        tOrDC.store(tDCrDC_a.load().to(self.io_dtype))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(
            tOrO_cv[(None, 0), None, None], self.io_dtype
        )
        tOrO_cv_cvt.store(tOrO_cv[(None, 0), None, None].load().to(self.io_dtype))
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO[(None, 0), None, None])

    @cute.jit
    def _blockwise_diagonal_16x16_to_32x32(self, mat_32x32, lane_id):
        """Stage 3: off-diagonal correction for one 32x32 diagonal tile (16x16 -> 32x32).

        After Stage 2 each diagonal 16x16 is inverted.  Computes C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x16_F32F16F16F32_TN, TileShape (16,16,16), single warp.
        make_acc_into_op ratio=2: A-frag atom size (8) / C-frag atom size (4).
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_16x16_2x2 = cute.flat_divide(mat_32x32, (16, 16))
        sDinv = mat_16x16_2x2[None, None, 1, 1]
        sC = mat_16x16_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_16x16_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_16x16_2x2[None, None, 1, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_32x32.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_32x32.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_32x32.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_32x32.element_type))
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _blockwise_diagonal_32x32_to_64x64(self, mat_64x64, warp_id, lane_id):
        """Stage 4: off-diagonal correction for one 64x64 diagonal tile (32x32 -> 64x64).

        4 warps collaborate (warp_id in {0,1,2,3}); x = warp_id//2, y = warp_id%2.
        MMA: SM80_16x8x16 TileShape (16,32,32), permutation_mnk=(16,32,32).
        Ends with sync_threads() to protect the sO write from races.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 32, 32)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_32x32_2x2 = cute.flat_divide(mat_64x64, (32, 32))
        sDinv_full = mat_32x32_2x2[None, None, 1, 1]
        sC_full = mat_32x32_2x2[None, None, 1, 0]
        sAinv_full = mat_32x32_2x2[None, None, 0, 0]

        sDinv = cute.flat_divide(sDinv_full, (16, 32))[None, None, warp_id % 2, 0]
        sC = cute.make_tensor(
            sC_full.iterator, cute.select(sC_full.layout, mode=[1, 0])
        )
        sAinv = cute.make_tensor(
            sAinv_full.iterator, cute.select(sAinv_full.layout, mode=[1, 0])
        )
        sO = cute.flat_divide(sC_full, (16, 32))[None, None, warp_id % 2, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_64x64.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_64x64.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_64x64.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_64x64.element_type))
        self.inverse_barrier_inner.arrive_and_wait()
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _load_initial_state(
        self,
        tidx,
        mS_init,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        kv_acc_producer,
    ) -> pipeline.PipelineProducer:
        """Load S_init from GMEM into state TMEM (fp32).

        Two steps:
          1. GMEM fp32 -> registers
          2. registers -> state TMEM (fp32), signal kv_acc so MMA can start GEMM 7
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM store copy (registers -> state TMEM)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        cState = cute.make_identity_tensor((self.mma_tiler_kv[0], self.mma_tiler_kv[1]))
        tCtState_for_r2t = tCtState[(None, None), 0, 0, 0]
        tiled_state_r2t = tcgen05.make_tmem_copy(state_r2t_atom, tCtState_for_r2t)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tRT_tCcState = thr_state_r2t.partition_S(cState)
        tRT_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, self.acc_dtype)
        tGR_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, self.state_dtype)

        gS_init = cute.flat_divide(
            mS_init[None, None, head_idx, batch_idx],
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
        )[None, None, 0, 0]
        tGR_tCgState = thr_state_r2t.partition_S(gS_init)
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        for sub in cutlass.range(tRT_tCrState.shape[2]):
            # 1. Load S_init fp32 GMEM -> fp32 registers
            cute.autovec_copy(
                tGR_tCgState[None, 0, sub],
                tGR_tCrState[None, 0, sub],
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                tRT_tCrState[None, 0, sub].store(
                    tGR_tCrState[None, 0, sub].load().to(self.state_dtype)
                )
            else:
                tRT_tCrState = tGR_tCrState

            # 2. fp32 registers -> state TMEM; signal kv_acc (GEMM 7 accumulates dS on top)
            cute.copy(
                tiled_state_r2t,
                tRT_tCrState[None, 0, sub],
                tRT_tCtState[None, 0, sub, kv_acc_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()

        # Manually sync before committing - CG1 is not the MMA warp so uses mbarrier_arrive.
        self.init_state_store_barrier.arrive_and_wait()
        if cg1_tidx == 0:
            cute.arch.mbarrier_arrive(kv_acc_handle.barrier)

        return kv_acc_producer

    @cute.jit
    def _store_final_state(
        self,
        tidx,
        # full output-state GMEM tensor (DK, DV, (h_r, h_qv), B) fp32
        mS_out,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        # MMA -> CG1 consumer; waited+released inside this method
        kv_acc_consumer,
        seqlen_b,
        mS_checkpoints,
        checkpoint_offset,
        checkpoint_every_n_tokens,
    ):
        """Store final recurrent state from TMEM (fp32) to GMEM mS_out.

        Waits for the last GEMM-7 (kv_acc) to complete, reads state TMEM -> registers,
        writes registers -> GMEM fp32, then releases the consumer handle.
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM layout (mirrors compute_group_1 setup)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )

        # TMEM -> registers  (Ld32x32b)
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        # Wait for last GEMM-7 to finish
        kv_acc_handle = kv_acc_consumer.wait_and_advance()

        for sub in cutlass.range(tTR_rState.shape[2]):
            # Read state TMEM -> fp32 registers
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, sub, kv_acc_handle.index],
                tTR_rState[None, 0, sub],
            )
            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                tRG_rState[None, 0, sub].store(
                    tTR_rState[None, 0, sub].load().to(self.state_dtype)
                )
            else:
                tRG_rState = tTR_rState
            if cutlass.const_expr(self.enable_checkpoints):
                if seqlen_b % checkpoint_every_n_tokens == 0:
                    gS_checkpoints = cute.make_tensor(
                        mS_checkpoints[
                            None, None, head_idx, checkpoint_offset
                        ].iterator,
                        cute.make_ordered_layout(
                            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]), order=(1, 0)
                        ),
                    )
                    tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                    cute.autovec_copy(
                        tRG_rState[None, 0, sub],
                        tSgCheckpoints[None, 0, sub],
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                    )
            if cutlass.const_expr(self.store_final_state):
                gS_out = cute.flat_divide(
                    mS_out[None, None, head_idx, batch_idx],
                    (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                )[None, None, 0, 0]
                tRG_tCgState = thr_state_t2r.partition_D(gS_out)
                cute.autovec_copy(
                    tRG_rState,
                    tRG_tCgState,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
        kv_acc_handle.release()
        return kv_acc_consumer

    @cute.jit
    def compute_group_1(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        checkpoint_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        timeline_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 4-7: kv_decay_v, v-k*state, state*q_epi, new_v_epi, qkv_epilogue, kv_update_epi."""
        sV, sW, sCumsumlog, sCumprod, sBeta, sO, sSideOutput, sH = smem_args
        mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens = checkpoint_args
        (
            load_v_consumer,
            load_w_consumer,
            load_gate_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            group_order_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
            side_output_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        (mVNew_out, mW_out, mH_out, is_first_chunk, chunk_idx, head_idx, chunk_offset) = work_args

        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # -- State TMEM tensor (DKxDV, layout from tiled_mma_kv) --------------
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        tiled_state_r2t = tcgen05.make_tmem_copy(atom_state_r2t, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )
        tCtState_inp_mn_view = self.transform_partitioned_tensor_layout(tCtState_inp)
        tCcState_inp = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        atom_state_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.io_dtype
        )
        tCtState_inp_for_r2t = tCtState_inp_mn_view[None, None, 0]
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            atom_state_inp_r2t, tCtState_inp_for_r2t
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcState_inp = thr_state_inp_r2t.partition_S(tCcState_inp)
        tRT_tCtState_inp = thr_state_inp_r2t.partition_D(tCtState_inp_mn_view)
        tRT_rState_inp = cute.make_rmem_tensor_like(tRT_tCcState_inp, self.io_dtype)

        # -- Shared acc TMEM tensor (BTxDV, layout from tiled_mma_qkv) ----------
        # Needed for v-k*state (K*S, stage 0) and new_v_epi (NV, stage 1).
        # qkv_epilogue reads O_intra from q_state TMEM, not from this buffer.
        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout, cute.make_layout((self.tmem_shared_acc_stages,))
            ),
        )
        tCtShared_mn_view = self.transform_partitioned_tensor_layout(tCtShared)
        # use qk here to construct shared as it has the biggest tiler (BT, BT)
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtShared_for_t2r = tCtShared[(None, None), 0, 0, 0]
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tCtShared_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg1_tidx)
        tTR_tCtShared = thr_shared_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcShared = thr_shared_t2r.partition_D(tCcShared)

        qkv_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(qkv_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )
        tCtShared_inp_mn_view = self.transform_partitioned_tensor_layout(tCtShared_inp)
        tCcShared_inp = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        atom_shared_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tCtShared_inp_for_r2t = tCtShared_inp_mn_view[None, None, 0]
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_for_r2t
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcShared_inp = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        tRT_tCtShared_inp = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)
        tRT_rShared_inp = cute.make_rmem_tensor_like(tRT_tCcShared_inp, self.io_dtype)  # noqa: F841

        # -- Q-state TMEM tensor (BTxDV, layout from tiled_mma_qs) --------------
        # Used by state*q_epi (scale Q*S result) and qkv_epilogue (read final O).
        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )
        tCtQState_mn_view = self.transform_partitioned_tensor_layout(tCtQState)
        tCcQState = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        atom_qs_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        atom_qs_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtQState_for_t2r = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_t2r = tcgen05.make_tmem_copy(atom_qs_t2r, tCtQState_for_t2r)
        tCtQState_for_r2t = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_r2t = tcgen05.make_tmem_copy(atom_qs_r2t, tCtQState_for_r2t)
        thr_qs_t2r = tiled_qs_t2r.get_slice(cg1_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(cg1_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        tTR_rQS = cute.make_rmem_tensor_like(tTR_tCcQS, self.acc_dtype)

        # -- SMEM V tiled copy: sV has domain (DV, BT); threads over BT (dim 1) --
        # Thread cg1_tidx owns all DV features for BT token cg1_tidx.
        # Beta scaling: sBeta[cg1_tidx] - one scalar per thread.

        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        tRT_tCtV = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)  # noqa: F841
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(
            atom_v_s2r,
            tiled_shared_inp_r2t,
        )
        thr_v_s2r = tiled_v_s2r.get_slice(cg1_tidx)

        # -- SMEM store: fp32 TMEM (tiled_mma_qs layout) -> fp16 sO --
        # Used by qkv_epilogue to write final O result.
        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_for_t2r)
        thr_o_t2r = tiled_o_t2r.get_slice(cg1_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_mn_view)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(cg1_tidx)
        tCsO = thr_o_r2s.partition_D(sO)
        atom_side_output_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_side_output_r2s = cute.make_tiled_copy_D(
            atom_side_output_r2s, tiled_shared_t2r
        )
        tCsSideOutput = tiled_side_output_r2s.get_slice(cg1_tidx).partition_D(sSideOutput)

        sub_tile_size = 32

        self._timeline_tag_from_args(timeline_args, 2)
        gate_handle = load_gate_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 3)

        cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_handle.index]

        valid_state = not is_first_chunk or self.use_initial_state
        if cutlass.const_expr(valid_state):
            if cutlass.const_expr(self.use_initial_state):
                kv_acc_producer.advance()
            # Wait for previous chunk's GEMM 7 to finish writing dS.
            # This also serialises the TMEM read-modify-write vs GEMM 7.
            self._timeline_tag_from_args(timeline_args, 4)
            kv_handle = kv_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 5)

            state_inp_ready_handle = state_inp_ready_producer.acquire_and_advance()
            if cutlass.const_expr(self.use_bf16_h_tma):
                tCsH = thr_state_t2r.partition_D(sH)
                h_side_output_store_handle = side_output_store_producer.acquire_and_advance()
            for sub in cutlass.range(tRT_rState_inp.shape[2]):
                cute.copy(
                    tiled_state_t2r,
                    tTR_tCtState[None, 0, sub, kv_handle.index],
                    tTR_rState[None, 0, sub],
                )
                if cutlass.const_expr(self.enable_checkpoints and not is_first_chunk):
                    if (self.b_t * chunk_idx) % checkpoint_every_n_tokens == 0:
                        gS_checkpoints = cute.make_tensor(
                            mS_checkpoints[
                                None, None, head_idx, checkpoint_offset
                            ].iterator,
                            cute.make_ordered_layout(
                                (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                                order=(1, 0),
                            ),
                        )
                        tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                        if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                            tRG_rState[None, 0, sub].store(
                                tTR_rState[None, 0, sub].load().to(self.state_dtype)
                            )
                        else:
                            tRG_rState = tTR_rState
                        cute.autovec_copy(
                            tRG_rState[None, 0, sub],
                            tSgCheckpoints[None, 0, sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )
                if cutlass.const_expr(self.store_h):
                    assert mH_out is not None, "mH_out must be provided when store_h is True"
                    if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                        tRG_rState[None, 0, sub].store(
                            tTR_rState[None, 0, sub].load().to(self.state_dtype)
                        )
                    else:
                        tRG_rState = tTR_rState
                    if cutlass.const_expr(self.use_bf16_h_tma):
                        cute.autovec_copy(
                            tRG_rState[None, 0, sub],
                            tCsH[None, 0, sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )
                    else:
                        h_checkpoint_idx = checkpoint_offset + 1
                        gH_out = cute.make_tensor(
                            mH_out[None, None, head_idx, h_checkpoint_idx].iterator,
                            cute.make_ordered_layout(
                                (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                                order=(0, 1),
                            ),
                        )
                        tSgH = thr_state_t2r.partition_D(gH_out)
                        cute.autovec_copy(
                            tRG_rState[None, 0, sub],
                            tSgH[None, 0, sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )

                tRT_rState_inp[None, 0, sub].store(
                    tTR_rState[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_state_inp_r2t,
                    tRT_rState_inp[None, 0, sub],
                    tRT_tCtState_inp[None, 0, sub, state_inp_ready_handle.index],
                )
            if cutlass.const_expr(self.use_bf16_h_tma):
                cute.arch.fence_view_async_shared()
                h_side_output_store_handle.commit()
            cute.arch.fence_view_async_tmem_store()
            state_inp_ready_handle.commit()

            # Load S_prev -> scale by Phi -> write Phi*S_prev back to same TMEM slot.
            for sub in cutlass.range(tTR_rState.shape[2]):
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rState[k, 0, sub] = tTR_rState[k, 0, sub] * cumprod_total
                cute.copy(
                    tiled_state_r2t,
                    tTR_rState[None, 0, sub],
                    tRT_tCtState[None, 0, sub, kv_handle.index],
                )
            if cutlass.const_expr((self.enable_checkpoints or self.store_h) and not is_first_chunk):
                if (self.b_t * chunk_idx) % checkpoint_every_n_tokens == 0:
                    checkpoint_offset += 1
            cute.arch.fence_view_async_tmem_store()

            # Release slot - MMA can now acquire it for this chunk's GEMM 7
            # (which accumulates dS on top of Phi*S_prev).
            kv_handle.release()

        elif cutlass.const_expr(self.use_bf16_h_tma):
            tCsH = thr_state_t2r.partition_D(sH)
            tRG_rState.fill(0.0)
            h_side_output_store_handle = side_output_store_producer.acquire_and_advance()
            for sub in cutlass.range(tRG_rState.shape[2]):
                cute.autovec_copy(
                    tRG_rState[None, 0, sub],
                    tCsH[None, 0, sub],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
            cute.arch.fence_view_async_shared()
            h_side_output_store_handle.commit()

        # Wait for kk and the qk/dummy-qk shared_acc slots to finish.
        shared_acc_consumer.advance()
        shared_acc_consumer.advance()

        rCumprod = cute.make_rmem_tensor((1, cute.size(tTR_tCcShared)), self.acc_dtype)
        tGrCumprod = thr_shared_t2r.partition_D(rCumprod)
        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):
            coord = tTR_tCcShared[k]
            tGrCumprod[k] = sCumprod[coord[1], 0, gate_handle.index]
        rDecayScale = cute.make_rmem_tensor(
            (1, cute.size(tTR_tCcShared)), self.acc_dtype
        )
        tGrDecayScale = thr_shared_t2r.partition_D(rDecayScale)
        last_cumsumlog = sCumsumlog[self.b_t - 1, 0, gate_handle.index]
        for k in cutlass.range_constexpr(cute.size(tTR_tCcShared)):
            coord = tTR_tCcShared[k]
            tGrDecayScale[k] = cute.math.exp2(
                last_cumsumlog - sCumsumlog[coord[1], 0, gate_handle.index],
                fastmath=True,
            )
        gate_handle.release()
        # ---- v - k*state  (ALU) -----------------------------------------------
        # delta[bt, dv] = V[bt, dv] - (K*S)[bt, dv]
        # Beta is fused into A_beta in CG0; no beta scaling here.
        # Thread cg1_tidx owns all DV elements for token bt=cg1_tidx.
        # Write fp16 delta to sV (GEMM 5 via v_ks_ready) and sStateDv (GEMM 7 via decay_v_ready).
        vks_handle = shared_inp_ready_producer.acquire_and_advance()
        self._timeline_tag_from_args(timeline_args, 6)
        v_handle = load_v_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 7)

        if cutlass.const_expr(valid_state):
            # Load V[*, cg1_tidx] from sV SMEM into registers

            tTR_rKS = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
            sV_vt_view = self.transform_partitioned_tensor_layout(sV)
            tCsV = thr_v_s2r.partition_S(sV_vt_view)

            tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
            tCrV = tiled_v_s2r.retile(tRT_rV)
            cute.copy(tiled_v_s2r, tCsV[None, None, None, v_handle.index], tCrV)
            if cutlass.const_expr(not self.training_side_outputs_only):
                self._timeline_tag_from_args(timeline_args, 8)
                group_order_handle = group_order_consumer.wait_and_advance()
                self._timeline_tag_from_args(timeline_args, 9)
            # Wait for GEMM 3 (K*S) result in shared_acc
            self._timeline_tag_from_args(timeline_args, 10)
            ks_acc_handle = shared_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 11)
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, None, None, ks_acc_handle.index],
                tTR_rKS,
            )
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tTR_rKS[k] = tTR_rKS[k] * tGrCumprod[k]
            ks_acc_handle.release()
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)
            cute.copy(
                tiled_shared_inp_r2t,
                tRT_rV,
                tRT_tCtShared_inp[None, None, None, vks_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
        else:
            if cutlass.const_expr(not self.training_side_outputs_only):
                self._timeline_tag_from_args(timeline_args, 8)
                group_order_handle = group_order_consumer.wait_and_advance()
                self._timeline_tag_from_args(timeline_args, 9)
        vks_handle.commit()

        # Move the optional W RHS from its dedicated TMA buffer into TMEM.
        if cutlass.const_expr(self.store_w):
            wks_handle = shared_inp_ready_producer.acquire_and_advance()
            w_load_handle = load_w_consumer.wait_and_advance()
            sW_vt_view = self.transform_partitioned_tensor_layout(sW)
            tCsW = thr_v_s2r.partition_S(sW_vt_view)
            tRT_rWIn = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
            tCrWIn = tiled_v_s2r.retile(tRT_rWIn)
            cute.copy(
                tiled_v_s2r,
                tCsW[None, None, None, w_load_handle.index],
                tCrWIn,
            )
            w_load_handle.release()
            if cutlass.const_expr(not self.w_rhs_precomputed):
                for k in cutlass.range(cute.size(tRT_rWIn), vectorize=True):
                    tRT_rWIn[k] = (
                        tRT_rWIn[k].to(self.acc_dtype) * tGrCumprod[k]
                    ).to(self.io_dtype)
            cute.copy(
                tiled_shared_inp_r2t,
                tRT_rWIn,
                tRT_tCtShared_inp[None, None, None, wks_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            wks_handle.commit()

        # ---- state*q_epi (ALU) ------------------------------------------------
        # Scale Q*S_prev cross-chunk contribution: QS[bt, *] *= cumprod[bt]
        # Thread cg1_tidx owns token bt=cg1_tidx -> scalar multiply by cur_cumprod.
        # Write scaled result back to same q_state TMEM slot so GEMM 6 accumulates on top.
        if cutlass.const_expr(valid_state and not self.drop_output_store_only):
            self._timeline_tag_from_args(timeline_args, 12)
            qs_handle = q_state_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 13)
            for sub in cutlass.range(tTR_rQS.shape[1]):
                cute.copy(
                    tiled_qs_t2r,
                    tTR_tCtQS[None, sub, 0, qs_handle.index],
                    tTR_rQS[None, sub, 0],
                )
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rQS[k, sub, 0] = (
                        tTR_rQS[k, sub, 0] * tGrCumprod[k, sub, 0] * scale
                    )
                cute.copy(
                    tiled_qs_r2t,
                    tTR_rQS[None, sub, 0],
                    tRT_tCtQS[None, sub, 0, qs_handle.index],
                )
            cute.arch.fence_view_async_tmem_store()

            qs_handle.release()
        elif cutlass.const_expr(valid_state):
            # Match the dummy q_state_acc producer token for skipped GEMM 4.
            self._timeline_tag_from_args(timeline_args, 12)
            qs_handle = q_state_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 13)
            qs_handle.release()

        # ---- new_v_epi --------------------------------------------------------
        # NV = A_inv @ delta (GEMM 5 result) in shared_acc TMEM; fp32 -> fp16 -> sAinvNv.
        self._timeline_tag_from_args(timeline_args, 14)
        nv_handle = shared_acc_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 15)
        sV_vt_view = self.transform_partitioned_tensor_layout(sV)
        v_handle.release()
        if cutlass.const_expr(self.store_v_new or self.store_w):
            side_output_store_handle = side_output_store_producer.acquire_and_advance()

        if cutlass.const_expr(not self.training_side_outputs_only):
            nv_ready_handle = shared_inp_ready_producer.acquire_and_advance()

        tTR_rNv = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        for sub in cutlass.range(tTR_rNv.shape[1]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, sub, 0, nv_handle.index],
                tTR_rNv[None, sub, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rNv[None, sub, 0].load().to(self.io_dtype)
            )
            if cutlass.const_expr(not self.drop_output_store_only):
                cute.copy(
                    tiled_shared_inp_r2t,
                    tTR_rNv_inp[None, sub, 0],
                    tRT_tCtShared_inp[None, sub, 0, nv_ready_handle.index],
                )
        if cutlass.const_expr(self.store_v_new):
            tRSNv = tiled_side_output_r2s.retile(tTR_rNv_inp)
            cute.copy(
                tiled_side_output_r2s, tRSNv, tCsSideOutput[None, None, None, 0]
            )
        if cutlass.const_expr(not self.drop_output_store_only):
            cute.arch.fence_view_async_tmem_store()
        nv_handle.release()
        if cutlass.const_expr(not self.training_side_outputs_only):
            nv_ready_handle.commit()
        else:
            # Keep the shared_inp_ready slot that GEMM 6 consumed in full forward.
            shared_inp_ready_producer.advance()

        if cutlass.const_expr(self.store_w):
            assert mW_out is not None, "mW_out must be provided when store_w is True"
            self._timeline_tag_from_args(timeline_args, 16)
            w_handle = shared_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 17)
            tTR_rW = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
            tTR_rW_out = cute.make_rmem_tensor_like(tTR_rW, self.io_dtype)
            for sub in cutlass.range(tTR_rW.shape[1]):
                cute.copy(
                    tiled_shared_t2r,
                    tTR_tCtShared[None, sub, 0, w_handle.index],
                    tTR_rW[None, sub, 0],
                )
                tTR_rW_out[None, sub, 0].store(
                    tTR_rW[None, sub, 0].load().to(self.io_dtype)
                )
            tRSW = tiled_side_output_r2s.retile(tTR_rW_out)
            cute.copy(
                tiled_side_output_r2s, tRSW, tCsSideOutput[None, None, None, 1]
            )
            w_handle.release()

        if cutlass.const_expr(self.store_v_new or self.store_w):
            cute.arch.fence_view_async_shared()
            side_output_store_handle.commit()

        # Write decay_scale * delta to sStateDv -> GEMM 7 B-operand
        # -- kv_decay_v: S_prev *= Phi = exp(cumsumlog[BT-1]) ------------------
        # Always wait for gate (collective barrier shared with CG0).
        decay_v_handle = shared_inp_ready_producer.acquire_and_advance()
        tTR_rDv = tTR_rNv
        tRT_rDv_inp = cute.make_rmem_tensor_like(tTR_rDv, self.io_dtype)
        for sub in cutlass.range(tTR_rDv.shape[1]):
            for k in cutlass.range(sub_tile_size, vectorize=True):
                tTR_rDv[k, sub, 0] = tTR_rDv[k, sub, 0] * tGrDecayScale[k, sub, 0]
            tRT_rDv_inp[None, sub, 0].store(
                tTR_rDv[None, sub, 0].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_shared_inp_r2t,
                tRT_rDv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, decay_v_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()
        decay_v_handle.commit()

        if cutlass.const_expr(not self.training_side_outputs_only):
            # ---- qkv_epilogue -----------------------------------------------------
            # GEMM 6 accumulated W_qkv@NV into q_state TMEM on top of the scaled Q*S.
            # q_state_acc second wait (same 1-stage pipeline, wraps back to stage 0).
            o_handle = o_store_producer.acquire_and_advance()
            self._timeline_tag_from_args(timeline_args, 18)
            qs_handle2 = q_state_acc_consumer.wait_and_advance()
            self._timeline_tag_from_args(timeline_args, 19)

            if cutlass.const_expr(not self.drop_output_store_only):
                tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, self.acc_dtype)
                tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
                tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
                cute.copy(
                    tiled_o_t2r,
                    tTR_tOtO[None, None, None, qs_handle2.index],
                    tTR_tOrO,
                )
                tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
                cute.copy(tiled_o_r2s, tRS_tOrO, tCsO[None, None, None, o_handle.index])
                cute.arch.fence_view_async_shared()
            group_order_handle.release()
            qs_handle2.release()

            # O in sQkOstore ready for epilogue warp TMA store, or a dummy token when O is dropped.
            o_handle.commit()
        else:
            # Match the dummy q_state_acc/o_store producer slots for skipped GEMM 6/O store.
            q_state_acc_consumer.advance()
            o_store_producer.advance()

        # ---- kv_update_epi ----------------------------------------------------
        # None, do state update at the beginning of the next chunk.
        return (  # type: ignore[return-value]
            load_v_consumer,
            load_w_consumer,
            load_gate_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            group_order_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
            side_output_store_producer,
            checkpoint_offset,
        )

    @cute.jit
    def side_output_tma_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
        timeline_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: bulk-store VNew/W staging tiles from SMEM to GMEM via TMA."""
        (sSideOutput,) = smem_args
        (tma_v_new_out, tma_w_out) = tma_args
        (side_output_store_consumer,) = pipeline_args
        head_idx, chunk_offset = work_args
        (
            tensormap_manager,
            tensormap_v_new_out_ptr,
            tensormap_w_out_ptr,
        ) = tensormap_args

        side_output_store_handle = side_output_store_consumer.wait_and_advance()
        cta_layout = cute.make_layout(1)
        side_output_tile = cute.select(self.mma_tiler_qkv, mode=[0, 1])

        if cutlass.const_expr(self.store_v_new):
            mVNew = cute.domain_offset(
                (cutlass.Int32(0), chunk_offset),
                tma_v_new_out.tma_tensor[None, None, head_idx],
            )
            gVNew = cute.flat_divide(mVNew, side_output_tile)
            tVsVNew, tVgVNew = cpasync.tma_partition(
                tma_v_new_out.atom,
                0,
                cta_layout,
                cute.group_modes(sSideOutput, 0, 2),
                cute.group_modes(gVNew, 0, 2),
            )
            cute.copy(
                tma_v_new_out.atom,
                tVsVNew[(None, 0)],
                tVgVNew[(None, 0, 0)],
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_v_new_out_ptr, cute.AddressSpace.generic
                ),
            )

        if cutlass.const_expr(self.store_w):
            mW = cute.domain_offset(
                (cutlass.Int32(0), chunk_offset),
                tma_w_out.tma_tensor[None, None, head_idx],
            )
            gW = cute.flat_divide(mW, side_output_tile)
            tWsW, tWgW = cpasync.tma_partition(
                tma_w_out.atom,
                0,
                cta_layout,
                cute.group_modes(sSideOutput, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
            cute.copy(
                tma_w_out.atom,
                tWsW[(None, 1)],
                tWgW[(None, 0, 0)],
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_w_out_ptr, cute.AddressSpace.generic
                ),
            )

        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)
        side_output_store_handle.release()
        return side_output_store_consumer
    @cute.jit
    def output_h_tma_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
        timeline_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: bulk-store the BF16 recurrent-state side output via TMA."""
        (sH,) = smem_args
        (tma_h_out,) = tma_args
        (side_output_store_consumer,) = pipeline_args
        head_idx, h_checkpoint_idx = work_args
        (
            tensormap_manager,
            tensormap_h_out_ptr,
        ) = tensormap_args

        h_side_output_store_handle = side_output_store_consumer.wait_and_advance()
        cta_layout = cute.make_layout(1)
        h_tile = cute.select(self.mma_tiler_kv, mode=[0, 1])
        mH = cute.domain_offset(
            (cutlass.Int32(0), cutlass.Int32(0)),
            tma_h_out.tma_tensor[None, None, head_idx, h_checkpoint_idx],
        )
        gH = cute.flat_divide(mH, h_tile)
        tHsH, tHgH = cpasync.tma_partition(
            tma_h_out.atom,
            0,
            cta_layout,
            cute.group_modes(sH, 0, 2),
            cute.group_modes(gH, 0, 2),
        )
        cute.copy(
            tma_h_out.atom,
            tHsH[None],
            tHgH[(None, 0, 0)],
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_h_out_ptr, cute.AddressSpace.generic
            ),
        )
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)
        h_side_output_store_handle.release()
        return side_output_store_consumer

    @cute.jit
    def epilogue_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
        timeline_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: TMA bulk-store O from SMEM staging buffer to global memory.

        Steps:
          1. Wait for CG1 to signal O is ready in sO (via o_store_consumer).
          2. Domain-offset the TMA tensor to (chunk_offset, head_idx), flat-divide
             into tiles, tma_partition -> (tOsO, tOgO).
          3. Issue TMA S2G bulk copy using the per-work-tile updated descriptor.
          4. Commit the async group and wait for the store to land in GMEM.
          5. Release the pipeline slot back to CG1.
        """
        (sO,) = smem_args
        (tma_o,) = tma_args
        (o_store_consumer,) = pipeline_args
        head_idx, chunk_offset = work_args
        tensormap_manager, tensormap_o_ptr = tensormap_args

        self._timeline_tag_from_args(timeline_args, 2)
        o_handle = o_store_consumer.wait_and_advance()
        self._timeline_tag_from_args(timeline_args, 3)

        cta_layout = cute.make_layout(1)
        # (BT, DV)
        o_tile = cute.select(self.mma_tiler_qkv, mode=[0, 1])

        # Position global O tile at current chunk / head
        mO = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset),
            tma_o.tma_tensor[None, None, head_idx],
        )
        # (BT, DV, num_o_tiles, ...)
        gO = cute.flat_divide(mO, o_tile)

        # TMA partition: tOsO = SMEM source, tOgO = GMEM destination
        tOsO, tOgO = cpasync.tma_partition(
            tma_o.atom,
            0,
            cta_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

        # TMA bulk store SMEM -> GMEM using the descriptor updated per work tile
        self._timeline_tag_from_args(timeline_args, 4)
        cute.copy(
            tma_o.atom,
            tOsO[(None, o_handle.index)],
            tOgO[(None, 0, 0)],
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_o_ptr, cute.AddressSpace.generic
            ),
        )

        # Wait for the store to complete before releasing the SMEM slot
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)
        self._timeline_tag_from_args(timeline_args, 5)

        o_handle.release()

        return o_store_consumer

    def transform_partitioned_tensor_layout(self, tensor: cute.Tensor) -> cute.Tensor:
        """
        Transform MMA layout from ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        to ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest).

        This groups MMA_ATOM_M with MMA_M and MMA_ATOM_N with MMA_N.

        :param tensor: Input tensor with layout ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        :type tensor: cute.Tensor
        :return: Transformed tensor with layout ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
        :rtype: cute.Tensor
        """
        layout = tensor.layout
        # Save original layout in case it is a composed layout
        stored_layout = layout

        if isinstance(stored_layout, cute.ComposedLayout):
            # For composed layouts, we only modify the outer layout
            layout = layout.outer

        shape = layout.shape
        stride = layout.stride

        # Build new shape: ((shape[0][0], shape[1]), (shape[0][1], shape[2]), ...rest)
        new_shape = ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:])

        # Build new stride: ((stride[0][0], stride[1]), (stride[0][1], stride[2]), ...rest)
        new_stride = ((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:])

        new_layout = cute.make_layout(shape=new_shape, stride=new_stride)

        if isinstance(stored_layout, cute.ComposedLayout):
            # Recreate the composed layout
            new_layout = cute.make_composed_layout(
                stored_layout.inner, stored_layout.offset, new_layout
            )

        return cute.make_tensor(tensor.iterator, new_layout)

    @cute.jit
    def _transform_to_position_independent_layout(
        self, tensor: cute.Tensor, swizzle_inner: cute.Swizzle
    ) -> cute.Tensor:
        wo_swizzle_iter = cute.recast_ptr(tensor.iterator, swizzle_=None)
        pisl_swizzle_base = int(math.log2(self.io_dtype.width)) - 1
        pisl_swizzle = cute.make_swizzle(
            swizzle_inner.num_bits, pisl_swizzle_base, swizzle_inner.num_shift
        )
        tensor_pisl = cute.make_composed_layout(pisl_swizzle, 0, tensor.layout)
        return cute.make_tensor(wo_swizzle_iter, tensor_pisl)

    @staticmethod
    def get_workspace_size(num_sm: int, B: int, HQ: int, HV: int, is_persistent: bool):
        # q, k, v, o, A, w
        if is_persistent:
            return (
                GatedDeltaNetChunkedKernel.bytes_per_tensormap
                * GatedDeltaNetChunkedKernel.num_tensormaps
                * num_sm
            )
        HO = HQ if HQ >= HV else HV
        return (
            GatedDeltaNetChunkedKernel.bytes_per_tensormap
            * GatedDeltaNetChunkedKernel.num_tensormaps
            * (B * HO)
        )

    @cute.jit
    def initialize_workspace(
        self, workspace: cute.Tensor, grid_dim: Tuple[int, int, int]
    ):
        workspace = cute.make_tensor(
            workspace.iterator,
            cute.make_layout(
                (
                    grid_dim[0] * grid_dim[1] * grid_dim[2],
                    GatedDeltaNetChunkedKernel.num_tensormaps,
                    GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                ),
                stride=(
                    GatedDeltaNetChunkedKernel.num_tensormaps
                    * GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                    GatedDeltaNetChunkedKernel.bytes_per_tensormap,
                    1,
                ),
            ),
        )
        return workspace
