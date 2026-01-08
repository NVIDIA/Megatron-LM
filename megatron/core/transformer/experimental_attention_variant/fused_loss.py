# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

from megatron.core.process_groups_config import ProcessGroupCollection

import tilelang
import tilelang.language as T

# TODO: just for accuracy check, remove it later
def fused_qk_topk_native(q, weights, k, mask, index_topk):
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1).contiguous()

    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    seqlen = index_scores.size(-1)
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    return topk_indices

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, # logits computation needs
}


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def _fused_qk_topk_kernel(
    heads,
    index_dim,
    topk,
    num_stages=2,
    threads=512,
    debug=False,
    dtype=T.float8_e4m3fn,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    RADIX = 1 << 8
    BLOCK_SIZE = threads
    SMEM_INPUT_SIZE = 4096  # assume the threshold bucket size after first pass is less than 4K

    # logits compute
    block_Q = 1 # restricted
    block_N = 256
    # dtype=T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk]

    block_TOPK = topk + block_N

    @T.prim_func
    def tl_topk_kernel(
        # input: T.Tensor[(seq_len, seq_len_kv), accum_dtype],
        topk_index: T.Tensor[(seq_len, topk), index_dtype],
        topk_logits: T.Tensor[(seq_len, topk), accum_dtype],
        starts: T.Tensor[(seq_len), index_dtype],
        ends: T.Tensor[(seq_len), index_dtype],
        # logits compute
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        LogitsIdx: T.Tensor(logits_shape, index_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as (bx):
            # logits compute
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s_shared = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s_shared, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([2, RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)

            l_new_topk = topk

            # sync
            tx = T.get_thread_binding()
            copy_done = T.alloc_barrier(arrive_count=512)
            gemm_done = T.alloc_barrier(arrive_count=512)

            T.fill(s_histogram[0, :], 0)
            T.fill(s_num_input[0], 0)

            nbn_i = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)
            s_val = T.alloc_var(accum_dtype)
            s_idx = T.alloc_var(index_dtype)
            input_idx = T.alloc_var(T.int32)

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            # fill block_TOPK logits
            fill_size = T.min(topk, cu_k_e_max - cu_k_s_min)
            T.barrier_arrive(gemm_done)

            for nbn_i in T.serial(T.ceildiv(fill_size, block_N)):
                T.barrier_wait(gemm_done, nbn_i % 2)

                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "copy done")

                T.barrier_arrive(copy_done)
                T.barrier_wait(copy_done, nbn_i % 2)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "start gemm")

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s_shared,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "gemm done")

                T.barrier_arrive(gemm_done)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "pass gemm barrier")

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_shared[bn_i, bq_i * heads + h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                        bn_i
                    ]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                # update histogram for topk stage 1
                for s in T.Parallel(block_N):
                    input_idx = cu_k_s_min + nbn_i * block_N + s
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                        inval_int16 = convert_to_uint16(logits[s, 0])
                        T.atomic_add(s_histogram[0, inval_int16], 1)

                # store topk logits and index first
                for s in T.Parallel(block_N):
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and cu_k_s_min + nbn_i * block_N + s < fill_size:
                        Logits[bx, cu_k_s_min + nbn_i * block_N + s] = logits[s, 0]
                        LogitsIdx[bx, cu_k_s_min + nbn_i * block_N + s] = cu_k_s_min + nbn_i * block_N + s

            T.sync_threads(1, 512)

            if cu_k_e_max - cu_k_s_min > topk:
                # update topk for each logits block compute
                cu_k_s_min = cu_k_s_min + fill_size
                T.fill(s_histogram[1, :], 0)

                for nbn_i in T.serial(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N)):
                    T.fill(s_num_input[0], 0)

                    # logits compute
                    T.barrier_wait(gemm_done, nbn_i % 2)

                    T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                    T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "copy done")

                    T.barrier_arrive(copy_done)
                    T.barrier_wait(copy_done, nbn_i % 2)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "start gemm")

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s_shared,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "gemm done")

                    T.barrier_arrive(gemm_done)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "pass gemm barrier")

                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                            bn_i
                        ]

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    # block_Q is restricted to 1
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            T.atomic_add(s_histogram[nbn_i % 2, inval_int16], 1)

                    # maintain s_histogram for the next block
                    T.copy(s_histogram[nbn_i % 2, :], s_histogram[(nbn_i % 2) ^ 1, :])

                    # topk compute

                    # cumsum
                    s_threshold_bin_id[0] = -1
                    T.sync_threads(1, 512)
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[nbn_i % 2, tx] = l_val

                        # find threshold bin id
                        T.sync_threads(3, RADIX)
                        if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads(1, 512)
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.sync_threads(1, 512)

                    if debug and bx == 0 and tx == 0 and l_threshold_bin_id < 0:
                        T.print(l_threshold_bin_id, "stage 1l_threshold_bin_id < 0")

                    if debug and bx == 0 and tx == 0:
                        T.print(l_new_topk, "l_new_topk 0")

                    # reset counter greater than topk to topk
                    # TODO: check accuracy issue
                    s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id] = topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.fill(s_histogram[(nbn_i % 2) ^ 1, 0 : l_threshold_bin_id], 0)
                    if debug and bx == 0 and tx == 0:
                        T.print(s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id], "s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id]")

                    # collect previous topk elements with exponent ≥ threshold
                    for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        T.sync_threads(1, 512)
                        input_idx = s * BLOCK_SIZE + tx
                        if input_idx < topk:
                            bin_id = convert_to_uint16(Logits[bx, input_idx])
                            l_bin_id32 = T.Cast(T.int32, bin_id)
                            if l_bin_id32 > l_threshold_bin_id:
                                # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = LogitsIdx[bx, input_idx]
                                topk_logits[bx, pos] = Logits[bx, input_idx]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # collect current block elements with exponent ≥ threshold
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            l_bin_id32 = T.Cast(T.int32, inval_int16)
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = input_idx
                                topk_logits[bx, pos] = logits[s, 0]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # stage 2: tail pass
                    for round in T.serial(4):
                        if l_new_topk <= 0:
                            T.loop_break()

                        r_idx = round % 2
                        l_start_pos = topk - l_new_topk

                        T.sync_threads(1, 512)
                        T.fill(s_histogram[nbn_i % 2, :], 0)
                        if tx == 0:
                            s_num_input[r_idx ^ 1] = 0
                        T.sync_threads(1, 512)

                        if debug and bx == 0 and tx == 0:
                            T.print(s_num_input[r_idx], "s_num_input[r_idx]")

                        l_num_input = s_num_input[r_idx]
                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )
                                T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32], 1)
                        T.sync_threads(1, 512)

                        # cumsum
                        s_threshold_bin_id[0] = -1
                        if tx < RADIX:
                            for i in T.serial(8):
                                offset = 1 << i
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    s_histogram[nbn_i % 2, tx] = l_val

                            # find threshold bin id
                            T.sync_threads(3, RADIX)
                            if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                                s_threshold_bin_id[0] = tx
                        T.sync_threads(1, 512)

                        l_threshold_bin_id = s_threshold_bin_id[0]
                        l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                        T.sync_threads(1, 512)

                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                    s_idx = LogitsIdx[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                    s_idx = input_idx

                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )

                                if l_bin_id32 > l_threshold_bin_id:
                                    pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    topk_logits[bx, pos] = s_val
                                    topk_index[bx, pos] = s_idx
                                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                    if round == 3:
                                        l_out_pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                        if l_out_pos < topk:
                                            topk_logits[bx, l_out_pos] = s_val
                                            topk_index[bx, l_out_pos] = s_idx
                                    else:
                                        pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                        s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]



                    # dump topk to Logits
                    T.copy(topk_index[bx, :], LogitsIdx[bx, :])
                    T.copy(topk_logits[bx, :], Logits[bx, :])
            else:
                T.copy(LogitsIdx[bx, :], topk_index[bx, :])
                T.copy(Logits[bx, :], topk_logits[bx, :])


    return tl_topk_kernel


def fused_qk_topk(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
    starts, ends, topk, debug=False, kv_scales=None, input=None,
):
    assert len(cu_seqlen_ks.shape) == len(cu_seqlen_ke.shape) == len(starts.shape) == len(ends.shape) == 1
    assert weights.dtype == torch.float32

    origin_dim = len(q.shape)
    if origin_dim == 4:
        batch = q.shape[1]
        q = q.view(-1, *q.shape[-2:])
        kv = kv.view(-1, kv.shape[-1])
        weights = weights.view(-1, weights.shape[-1])
    else:
        assert len(q.shape) == 3

    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    topk_indexes = torch.zeros(seq_len, topk, device=q.device, dtype=torch.int32)
    topk_logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits_idx = torch.empty([seq_len, topk], device=q.device, dtype=torch.int32)

    if kv_scales is None:
        kv_scales = torch.ones(seq_len_kv, device=q.device, dtype=torch.float32)

    kernel = _fused_qk_topk_kernel(heads=heads, index_dim=index_dim, topk=topk, debug=debug, dtype=q.dtype)
    kernel(
        # input, 
        topk_indexes, 
        topk_logits,
        starts, 
        ends,
        # logits
        q.view(seq_len * heads, index_dim),
        kv,
        kv_scales,
        logits,
        logits_idx,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    if origin_dim == 4:
        topk_indexes = topk_indexes.reshape(seq_len, batch, topk)
    
    return topk_indexes, topk_logits, logits


@triton.jit
def _fwd_fused_indexer_loss_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Mask_ptr,
    Topk_Idx_ptr,
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Mask_ptr,
    # Q strides: [Sq, B, H, D]
    stride_qs,
    stride_qb,
    stride_qh,
    stride_qd,
    # K strides: [Sk, B, D]
    stride_ks,
    stride_kb,
    stride_kd,
    # W strides: [Sq, B, H]
    stride_ws,
    stride_wb,
    stride_wh,
    # Mask strides: [B, Sq, Sk]
    stride_mb,
    stride_ms,
    stride_mk,
    # Topk strides: [B, Sq, TopK]
    stride_tb,
    stride_ts,
    stride_tk,
    # Attn query strides: [Sq, B, H, D]
    stride_asq,
    stride_aqb,
    stride_aqh,
    stride_aqd,
    # Attn key strides: [Sk, B, H, D]
    stride_ask,
    stride_akb,
    stride_akh,
    stride_akd,
    # Loss strides: [B, Sq]
    stride_lb,
    stride_ls,
    # Index mask strides: [B, Sq, Sk]
    stride_imsq,
    stride_imsk,
    # Dimensions
    H: tl.constexpr,
    D: tl.constexpr,
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sq: tl.constexpr,
    Sk: tl.constexpr,
    ASq: tl.constexpr,
    Sq_offset: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    TopK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    # should be within (ASq_offset, ASq_offset + ASq)
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)
    sq = Sq_offset + sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = (sq < Sq) & (sq < Sq_offset + ASq)
    
    # Base pointers for this (b, sq)
    q_base = Q_ptr + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + sq * stride_ws + b * stride_wb

    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # compute the first pass for attn softmax and index softmax
    # apply causal mask by loop trunctation
    causal_sk = tl.minimum(tl.min(sq) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Compute index_scores for this chunk
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in tl.range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores + mask_vals, float("-inf"))
        else:
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores, float("-inf"))

        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            # leave all threads to topk parallelism
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = Sq_offset + sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = (sq_i_offs < Sq) & (sq_i_offs < Sq_offset + ASq)

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where(sq_i_valid & (sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        # first pass for index softmax
        m1_i_1 = m1_i
        m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
        m1_i = tl.where(m1_i <= float("-inf"), 0.0, m1_i)
        d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.exp(index_scores - m1_i[:, None]).sum(axis=1)

        # first pass for attn softmax
        h_ids = tl.arange(0, AH)

        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh

            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += sparse_mask[None, :, :]

        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(attn_scores, axis=-1))
        m_i = tl.where(m_i <= float("-inf"), 0.0, m_i)
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.exp(attn_scores - m_i[:, :, None]).sum(axis=-1)
    
    # recompute for the second pass of attn softmax
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        # Compute index_scores for this chunk
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in tl.range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores + mask_vals, float("-inf"))
        else:
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores, float("-inf"))

        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            # leave all threads to topk parallelism
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = Sq_offset + sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = (sq_i_offs < Sq) & (sq_i_offs < Sq_offset + ASq)

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where(sq_i_valid & (sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask

        # compute loss
        h_ids = tl.arange(0, AH)

        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh

            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += sparse_mask[None, :, :]

        # softmax
        softmax_attn_i = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        softmax_index_i = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]

        # reduce head dim
        softmax_attn_i = tl.sum(softmax_attn_i, axis=0) / AH

        # loss
        loss_sk = softmax_attn_i * (tl.log(softmax_attn_i + 1e-10) - tl.log(softmax_index_i + 1e-10))
        loss_i += loss_sk.sum(axis=-1)

    # Store loss
    tl.store(Loss_ptr + b * stride_lb + aq * stride_ls, loss_i, mask=aq_valid)


def fwd_fused_indexer_loss(
    q: torch.Tensor,
    weights: torch.Tensor, 
    k: torch.Tensor,
    attn_query: torch.Tensor,
    attn_key: torch.Tensor,
    topk: int,
    softmax_scale: float,
    loss_coeff: float,
    mask: Optional[torch.Tensor] = None,
    sparse_loss: Optional[bool] = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    accuracy_check: bool = False,
) -> torch.Tensor:
    """
    Fused index score computation + TopK selection using Triton.
    
    This avoids materializing the full [B, Sq, Sk] index_scores tensor.
    
    Args:
        q: [Sq, B, H, D] query tensor
        weights: [Sq, B, H] attention weights  
        k: [Sk, B, D] key tensor
        attn_query: [Sq, B, H, D] attention query tensor
        attn_key: [Sq, B, H, D] attention key tensor
        topk: number of top indices to return
        softmax_scale: softmax scale
        mask: [B, Sq, Sk] mask tensor (optional)
        sparse_loss: whether to use sparse loss
        loss_coeff: loss coefficient
    Returns:
        topk_indices: [B, Sq, TopK] int64 tensor of top-k indices
    """
    Sq, B, H, D = q.shape
    Sk = k.shape[0]

    ASq, AB, AH, AD = attn_query.shape
    ASk = attn_key.shape[0]

    # Clamp topk to valid range
    topk = min(topk, Sk)
    
    # TopK could be 2048
    BLOCK_SK = 64
    BLOCK_SQ = 16
    BLOCK_D = 64
    
    # Handle mask strides
    if mask is not None:
        stride_mb = mask.stride(0)
        stride_ms = mask.stride(1)
        stride_mk = mask.stride(2)
        has_mask = True
    else:
        stride_mb = stride_ms = stride_mk = 0
        has_mask = False

    # current_stream = torch.cuda.current_stream()
    # topk_stream = torch.cuda.Stream()
    # topk_stream.wait_stream(current_stream)
    # with torch.cuda.stream(topk_stream):
    if accuracy_check:
        topk_indices = fused_qk_topk_native(q, weights, k, mask, topk)
    else:
        # TODO: consider varlen
        # TODO: fix TopK < 1024
        starts = torch.zeros(Sq * B, dtype=torch.int32).cuda()
        ends = torch.ones(Sq * B, dtype=torch.int32).cuda() * Sk
        topk_indices, topk_logits, original_logits = fused_qk_topk(
            q,
            k,
            weights,
            starts,
            ends,
            starts,
            ends,
            topk,
            debug=False,
            kv_scales=None,
            input=None,
        )
        # [Sq, B, TopK] -> [B, Sq, TopK]
        topk_indices = topk_indices.transpose(0, 1)

    Sq_offset = 0
    if pg_collection is not None and pg_collection.tp.size() > 1:
        tp_size = pg_collection.tp.size()

        # all-to-all for attn query
        assert ASq % tp_size == 0
        # [ASq, B, H, D] -> [tp_size, H, ASq // tp_size, B, D]
        view_attn_q = (
            attn_query.permute(2, 0, 1, 3)
            .reshape(AH, tp_size, ASq // tp_size, AB, AD)
            .transpose(0, 1)
            .contiguous()
        )
        output_attn_q = torch.empty_like(view_attn_q)
        dist.all_to_all_single(
            output_attn_q, 
            view_attn_q,
            group=pg_collection.tp
        )
        # [tp_size, H, ASq // tp_size, B, D] -> [ASq // tp_size, B, H * tp_size, D]
        attn_query = (
            output_attn_q.reshape(AH * tp_size, ASq // tp_size, AB, AD)
            .permute(1, 2, 0, 3)
            .contiguous()
        )

        # all-gather for attn key
        gathered_attn_k = torch.empty(
            (tp_size, *attn_key.shape), 
            device=attn_key.device, 
            dtype=attn_key.dtype
        )
        dist.all_gather_into_tensor(gathered_attn_k, attn_key.contiguous(), group=pg_collection.tp)
        # [tp_size, Sk, B, H, D] -> [Sk, B, H * tp_size, D]
        attn_key = (
            gathered_attn_k.permute(1, 2, 0, 3, 4)
            .reshape(ASk, AB, AH * tp_size, AD)
            .contiguous()
        )

        ASq, AB, AH, AD = attn_query.shape
        AHk = attn_key.shape[2]

        assert Sq == ASq * tp_size and \
            Sk == ASk and AH == H and AHk == H and \
            AB == B and AD == D
        
        # Do not split index scores, it introduces extra problem in communication and casual mask
        # Sq should be within (Sq_offset, Sq_offset + Sq)
        tp_rank = pg_collection.tp.rank()
        Sq_offset = Sq // tp_size * tp_rank
        assert Sq_offset + ASq <= Sq

    # cuda stream synchronization
    # torch.cuda.synchronize()
    assert topk_indices.max() < Sk

    # TopK could be 2048
    BLOCK_TOPK = min(2048, topk)
    BLOCK_SK = 64
    BLOCK_SQ = 16
    BLOCK_D  = 64

    out_loss = torch.empty((B, ASq), dtype=torch.float32, device=q.device)
    attn_num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    attn_grid = (B, attn_num_sq_blocks,)

    index_mask = torch.empty((BLOCK_SQ, BLOCK_SK), dtype=torch.float32, device=q.device)
    if sparse_loss:
        stride_imsq = index_mask.stride(0)
        stride_imsk = index_mask.stride(1)
    else:
        stride_imsq = stride_imsk = 0
    
    _fwd_fused_indexer_loss_kernel[attn_grid](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Mask_ptr=mask,
        Topk_Idx_ptr=topk_indices,
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss,
        Index_Mask_ptr=index_mask,
        # Q strides
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        # K strides
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        # W strides
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        # Mask strides
        stride_mb=stride_mb,
        stride_ms=stride_ms,
        stride_mk=stride_mk,
        # Topk indices strides
        stride_tb=topk_indices.stride(0),
        stride_ts=topk_indices.stride(1),
        stride_tk=topk_indices.stride(2),
        # Attn query strides: [Sq, B, H, D]
        stride_asq=attn_query.stride(0),
        stride_aqb=attn_query.stride(1),
        stride_aqh=attn_query.stride(2),
        stride_aqd=attn_query.stride(3),
        # Attn key strides: [Sk, B, H, D]
        stride_ask=attn_key.stride(0),
        stride_akb=attn_key.stride(1),
        stride_akh=attn_key.stride(2),
        stride_akd=attn_key.stride(3),
        # Loss strides: [B, Sq]
        stride_lb=out_loss.stride(0),
        stride_ls=out_loss.stride(1),
        # Index mask strides: [B, Sq, Sk]
        stride_imsq=stride_imsq,
        stride_imsk=stride_imsk,
        # Dimensions
        H=H,
        D=D,
        AH=AH,
        AD=AD,
        Sq=Sq,
        Sk=Sk,
        ASq=ASq,
        Sq_offset=Sq_offset,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=BLOCK_TOPK,
        TopK=topk,
        HAS_MASK=has_mask,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
    )

    indexer_loss = out_loss.mean() * loss_coeff

    if pg_collection is not None and pg_collection.tp.size() > 1:
        # reduce loss
        dist.all_reduce(indexer_loss, group=pg_collection.tp)
        indexer_loss /= pg_collection.tp.size()
    
    return topk_indices, indexer_loss, out_loss


@triton.jit
def _bwd_fused_indexer_loss_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Attn_Query_ptr,
    Attn_Key_ptr,
    Topk_Idx_ptr,
    Grad_Q_ptr,
    Grad_W_ptr,
    Grad_K_ptr,
    # Q strides: [Sq, B, H, D]
    stride_qs,
    stride_qb,
    stride_qh,
    stride_qd,
    # K strides: [Sk, B, D]
    stride_ks,
    stride_kb,
    stride_kd,
    # W strides: [Sq, B, H]
    stride_ws,
    stride_wb,
    stride_wh,
    # Attn query strides: [Sq, B, H, D]
    stride_asq,
    stride_aqb,
    stride_aqh,
    stride_aqd,
    # Attn key strides: [Sk, B, H, D]
    stride_ask,
    stride_akb,
    stride_akh,
    stride_akd,
    # Topk indices strides: [B, Sq, TopK]
    stride_tb,
    stride_ts,
    stride_tk,
    # Grad Q strides: [Sq, B, H, D]
    stride_gqs,
    stride_gqb,
    stride_gqh,
    stride_gqd,
    # Grad W strides: [Sq, B, H]
    stride_gws,
    stride_gwb,
    stride_gwh,
    # Grad K strides: [B, Sk, D]
    stride_pgb,
    stride_pgk,
    stride_pgd,
    # Dimensions
    H: tl.constexpr,
    D: tl.constexpr,
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sq: tl.constexpr,
    Sk: tl.constexpr,
    TopK: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    Softmax_Scale: tl.constexpr,
    Grad_Loss_Scale: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
):
    """
    Compute gradient of KL loss w.r.t. index_scores logits (before softmax).
    This is the first step of the backward pass - compute grad_index_logits.
    """
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = sq < Sq
    
    # Base pointers
    q_base = Q_ptr + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + b * stride_wb
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb
    
    # First pass: compute softmax denominators  
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    sum_grad = tl.zeros([BLOCK_SQ, 1], dtype=tl.float32)
    
    causal_sk = tl.minimum(tl.max(sq) + 1, Sk)

    # First pass for softmax statistics
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Compute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = sq_i_offs < Sq

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where((sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        m1_i_1 = m1_i
        m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
        d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.sum(tl.exp(index_scores - m1_i[:, None]), axis=1)
        
        # Compute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(attn_scores, axis=-1))
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.sum(tl.exp(attn_scores - m_i[:, :, None]), axis=-1)
    
    # Second pass: compute gradient w.r.t. index_logits
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        # Recompute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = sq_i_offs < Sq

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where((sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        # Recompute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        # Compute softmax values
        index_scores_softmax = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]
        attn_scores_softmax = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        
        # Sum and normalize attention scores
        attn_scores_sum = tl.sum(attn_scores_softmax, axis=0) / AH
        
        # Gradient of KL divergence w.r.t. index_scores_softmax
        grad_index_softmax = -attn_scores_sum / (index_scores_softmax + 1e-10) * Grad_Loss_Scale
        
        # Backward through softmax
        sum_grad += tl.sum(grad_index_softmax * index_scores_softmax, axis=-1, keep_dims=True)

    # Third pass
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Recompute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            # leave all threads to topk parallelism
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = sq_i_offs < Sq

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where(sq_i_valid & (sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        # Recompute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        # Compute softmax values
        index_scores_softmax = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]
        attn_scores_softmax = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        
        # Sum and normalize attention scores
        attn_scores_sum = tl.sum(attn_scores_softmax, axis=0) / AH

        # Gradient of KL divergence w.r.t. index_scores_softmax
        grad_index_softmax = -attn_scores_sum / (index_scores_softmax + 1e-10) * Grad_Loss_Scale        

        grad_index_logits = index_scores_softmax * (grad_index_softmax - sum_grad)
        
        # Apply valid mask
        valid_mask = (sq[:, None] >= sk_offs[None, :])
        if SPARSE_LOSS:
            valid_mask = valid_mask & (sparse_mask == 0.0)
        grad_index_logits = tl.where(valid_mask, grad_index_logits, 0.0)

        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            # Compute scores = q @ k.T [BLOCK_SQ, BLOCK_SK]
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU activation and mask
            scores_relu = tl.maximum(dot, 0.0)
            relu_mask = (dot > 0.0).to(tl.float32)

            # grad_weights: sum(grad_logits * scores_relu, dim=sk)
            # [BLOCK_SQ, BLOCK_SK] --sum over sk--> [BLOCK_SQ]
            # [sq, b, 1, sk] * [sq, b, h, sk] -> [sq, b, h]
            grad_w_val = tl.sum(grad_index_logits * scores_relu, axis=-1)
            grad_w_ptrs = Grad_W_ptr + sq * stride_gws + b * stride_gwb + h * stride_gwh
            tl.atomic_add(grad_w_ptrs, grad_w_val, mask=sq_valid)

            # grad_scores = grad_logits * weights * relu_mask
            grad_scores = grad_index_logits * w_val[:, None] * relu_mask

            # Compute grad_q for this head and write with atomic add
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D

                k_ptrs = k_base + sk_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[:, None] & d_valid[None, :]), other=0.0)
                
                # grad_q: grad_scores @ k [BLOCK_SQ, BLOCK_SK] @ [BLOCK_SK, BLOCK_D]
                grad_q_part = tl.dot(grad_scores, k_vals)
                grad_q_base = Grad_Q_ptr + b * stride_gqb + h * stride_gqh
                grad_q_ptrs = grad_q_base + sq[:, None] * stride_gqs + d_offs[None, :] * stride_gqd
                tl.atomic_add(grad_q_ptrs, grad_q_part, mask=(sq_valid[:, None] & d_valid[None, :]))

                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)                

                # Compute partial grad_k: grad_scores.T @ q [BLOCK_SK, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
                partial_grad_k = tl.dot(tl.trans(grad_scores), q_vals)
                partial_base = Grad_K_ptr + b * stride_pgb
                partial_ptrs = partial_base + sk_offs[:, None] * stride_pgk + d_offs[None, :] * stride_pgd
                tl.atomic_add(partial_ptrs, partial_grad_k, mask=(sk_valid[:, None] & d_valid[None, :]))


def bwd_fused_indexer_loss(
    q, weights, k, query, key, topk_indices,
    softmax_scale, loss_coeff, sparse_loss,
    grad_loss
):
    """
    Fully-fused Triton implementation of backward pass.
    
    Uses three Triton kernels:
    1. Compute grad_index_logits
    2. Compute grad_q and grad_weights (no atomics needed)
    3. Compute grad_k using two-phase reduction (avoid atomics)
    
    This is more complex than the hybrid approach but potentially faster
    for large problem sizes.
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)
    h = weights.size(2)  # indexer heads
    d = q.size(3)  # indexer dimension
    
    BLOCK_SQ = 16  # Must be >= 16 for Triton tl.dot
    BLOCK_SK = 64
    BLOCK_D = 64
    
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_weights = torch.zeros_like(weights, dtype=torch.float32)
    num_sq_blocks = triton.cdiv(sq, BLOCK_SQ)
    grad_k = torch.zeros(b, sk, d, device=q.device, dtype=torch.float32)

    grid1 = (b, num_sq_blocks)
    grad_loss_scale = grad_loss.item() * loss_coeff / (b * sq)
    
    # Get topk
    topk = topk_indices.size(-1)
    BLOCK_TOPK = min(topk, 2048)
    
    _bwd_fused_indexer_loss_kernel[grid1](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Attn_Query_ptr=query,
        Attn_Key_ptr=key,
        Topk_Idx_ptr=topk_indices,
        Grad_Q_ptr=grad_q,
        Grad_W_ptr=grad_weights,
        Grad_K_ptr=grad_k,
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        stride_asq=query.stride(0),
        stride_aqb=query.stride(1),
        stride_aqh=query.stride(2),
        stride_aqd=query.stride(3),
        stride_ask=key.stride(0),
        stride_akb=key.stride(1),
        stride_akh=key.stride(2),
        stride_akd=key.stride(3),
        stride_tb=topk_indices.stride(0),
        stride_ts=topk_indices.stride(1),
        stride_tk=topk_indices.stride(2),
        stride_gqs=grad_q.stride(0),
        stride_gqb=grad_q.stride(1),
        stride_gqh=grad_q.stride(2),
        stride_gqd=grad_q.stride(3),
        stride_gws=grad_weights.stride(0),
        stride_gwb=grad_weights.stride(1),
        stride_gwh=grad_weights.stride(2),
        stride_pgb=grad_k.stride(0),
        stride_pgk=grad_k.stride(1),
        stride_pgd=grad_k.stride(2),
        H=h,
        D=d,
        AH=np,
        AD=hn,
        Sq=sq,
        Sk=sk,
        TopK=topk,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=BLOCK_TOPK,
        Softmax_Scale=softmax_scale,
        Grad_Loss_Scale=grad_loss_scale,
        SPARSE_LOSS=sparse_loss,
    )

    grad_k = grad_k.permute(1, 0, 2)
    
    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, weights, k, query, key, softmax_scale, topk, loss_coeff, sparse_loss):
        """
        Fused forward: index_scores never materialized in full.
        """
        # Run fused Triton kernel
        topk_indices, loss = fwd_fused_indexer_loss(
            q, weights, k, query, key, softmax_scale, topk, loss_coeff, sparse_loss
        )
        
        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        
        return topk_indices, loss
    
    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q , grad_weights, grad_k = bwd_fused_indexer_loss(
            q, weights, k, query, key, topk_indices, 
            ctx.softmax_scale, ctx.loss_coeff * grad_loss, ctx.sparse_loss
        )
        
        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None