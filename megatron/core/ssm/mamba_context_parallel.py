# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp
from megatron.core.utils import is_te_min_version

try:
    from einops import repeat

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn’t installed or the torch wrapper is missing
    tex = None


class MambaContextParallel:
    """
    This class provides the following functionality related to Mamba "all-to-all" context parallel:
    1. Error checking, and creation of, relevant parameters (e.g. nheads_local_tpcp)
    2. Collective operations on activations, on each context parallel rank, before and after the
       convolution and SSM
    3. A convolution operator that uses the correct slices of trainable variables on the current
       context parallel rank
    4. Sliced views of relevant trainable variables for the current context parallel rank

    This class is intentionally not a sub-class of MegatronModule. This class does not contain any
    trainable variables of its own and should not be involved in any checkpoint loading or saving.

    Args:
        cp_group (torch.distributed.ProcessGroup):
            The process group to use for context parallel.
        d_inner_local_tp (int): d_inner on the current tp rank
        nheads_local_tp (int): nheads on the current tp rank
        ngroups_local_tp (int): ngroups on the current tp rank
        d_state (int): Mamba d_state
        conv1d_weight_cp1 (torch.Tensor):
            The conv1d weight which would be used on this tp rank if cp_size was 1
        conv1d_bias_cp1 (torch.Tensor):
            The conv1d bias which would be used on this tp rank if cp_size was 1
        conv1d_padding (int):
            The conv1d padding which would be used on this tp rank if cp_size was 1
        dt_bias_cp1 (torch.Tensor):
            The dt_bias parameter which would be used on this tp rank if cp_size was 1
        A_log_cp1 (torch.Tensor):
            The A_log parameter which would be used on this tp rank if cp_size was 1
        D_cp1 (torch.Tensor): The D parameter which would be used on this tp rank if cp_size was 1
        D_has_hdim (bool): D parameter is sized to hidden dimension, rather than being per-head
        sequence_is_contiguous (bool): Whether CP ranks already hold contiguous causal intervals.
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        d_inner_local_tp: int,
        nheads_local_tp: int,
        ngroups_local_tp: int,
        d_state: int,
        conv1d_weight_cp1: torch.Tensor,
        conv1d_bias_cp1: torch.Tensor,
        conv1d_padding: int,
        dt_bias_cp1: torch.Tensor,
        A_log_cp1: torch.Tensor,
        D_cp1: torch.Tensor,
        D_has_hdim: bool,
        sequence_is_contiguous: bool = False,
    ) -> None:
        if not HAVE_EINOPS:
            raise ImportError("einops is required by the Mamba model but cannot be imported")

        self.cp_group = cp_group
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.conv1d_weight_cp1 = conv1d_weight_cp1
        self.conv1d_bias_cp1 = conv1d_bias_cp1
        self.conv1d_padding = conv1d_padding
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1
        self.D_has_hdim = D_has_hdim
        self.sequence_is_contiguous = sequence_is_contiguous

        self.cp_size = self.cp_group.size()

        if self.cp_size == 1:
            self.d_inner_local_tpcp = self.d_inner_local_tp
            self.nheads_local_tpcp = self.nheads_local_tp
            self.ngroups_local_tpcp = self.ngroups_local_tp
            return

        self.cp_rank = self.cp_group.rank()

        # Ensure that each CP rank gets at least one head:
        assert (
            self.nheads_local_tp % self.cp_size == 0
        ), "nheads must be evenly divisible by tp_size * cp_size"
        # Note that an upper-bound on cp_size is nheads // tp_size
        self.nheads_local_tpcp = self.nheads_local_tp // self.cp_size

        # Note that we do not need to confirm that `d_inner_local_tp % cp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = self.d_inner // headdim`,
        # `nheads % tp_size == 0`, `nheads_local_tp = nheads // tp_size`, and
        # `nheads_local_tp % cp_size == 0`
        self.d_inner_local_tpcp = self.d_inner_local_tp // self.cp_size

        # Ensure that each CP rank gets a positive integer number of groups:
        if self.ngroups_local_tp < self.cp_size:
            assert (
                self.cp_size % self.ngroups_local_tp == 0
            ), "cp_size must be evenly divisible by ngroups/tp_size"
            # Need to replicate the group state (shard the heads of each group) across CP ranks:
            self.group_repeat_count = self.cp_size // self.ngroups_local_tp
            self.ngroups_local_tpcp = 1
        else:
            assert (
                self.ngroups_local_tp % self.cp_size == 0
            ), "ngroups must be evenly divisible by tp_size * cp_size"
            # Group state is not replicted across CP ranks. All heads for any group are present on
            # one CP rank
            self.group_repeat_count = 1
            self.ngroups_local_tpcp = self.ngroups_local_tp // self.cp_size

        # Note that we do not need to confirm that `nheads_local_tpcp % ngroups_local_tpcp == 0`
        # because `nheads % ngroups == 0`, and therefore `nheads_local_tp % ngroups_local_tp == 0`,
        # and also `nheads_local_tpcp = nheads_local_tp // cp_size` whilst ngroups_local_tpcp is
        # either 1 or `ngroups_local_tp // cp_size`

    def pre_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """Method to be applied before the convolution and SSM"""
        if self.cp_size == 1:
            return input_

        z, x, B, C, dt = torch.split(
            input_,
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp,  # z, x: [l_global//cp, b, d_inner]
                self.ngroups_local_tp * self.d_state,  # B: [l_global//cp, b, ngroups * d_state]
                self.ngroups_local_tp * self.d_state,  # C: [l_global//cp, b, ngroups * d_state]
                self.nheads_local_tp,  # dt : [l_global//cp, b, nheads]
            ],
            dim=-1,
        )

        # TODO (duncan): Can the some or all of the all_to_alls be combined?

        # [l_global//cp, b, d_inner] -> [l_global, b, d_inner//cp]
        z = _all_to_all_cp2hp(z, self.cp_group)

        # [l_global//cp, b, d_inner] -> [l_global, b, d_inner//cp]
        x = _all_to_all_cp2hp(x, self.cp_group)

        # Below, each group state will be repeated before moving to the next group state. This
        # causes replicas of the same group state to land on consecutive context parallel ranks,
        # along with their associated heads. This is consistent with consecutive group states being
        # associated with consecutive groups of heads.
        B = repeat(
            B,
            "l b (g n) -> l b (g r n)",
            g=self.ngroups_local_tp,
            n=self.d_state,
            r=self.group_repeat_count,
        )
        C = repeat(
            C,
            "l b (g n) -> l b (g r n)",
            g=self.ngroups_local_tp,
            n=self.d_state,
            r=self.group_repeat_count,
        )

        # [l_global//cp, b, g*r*n] -> [l_global, b, g*r*n//cp]
        B = _all_to_all_cp2hp(B, self.cp_group)

        # [l_global//cp, b, g*r*n] -> [l_global, b, g*r*n//cp]
        C = _all_to_all_cp2hp(C, self.cp_group)

        # [l_global//cp, b, nheads] -> [l_global, b, nheads//cp]
        dt = _all_to_all_cp2hp(dt, self.cp_group)

        output = torch.cat([z, x, B, C, dt], dim=-1)
        if not self.sequence_is_contiguous:
            output = _undo_attention_load_balancing(output, self.cp_size, packed_seq_params)

        return output

    def post_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """Method to be applied after the convolution and SSM"""
        if self.cp_size == 1:
            return input_
        if not self.sequence_is_contiguous:
            input_ = _redo_attention_load_balancing(input_, self.cp_size, packed_seq_params)
        return _all_to_all_hp2cp(input_, self.cp_group)

    def conv1d(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Performs a conv1d on one context parallel rank, using slices of the weight and bias from
        the convolution that would be run when cp_size=1
        """
        return F.conv1d(
            input=input_,
            weight=self.get_conv1d_weight(),
            bias=self.get_conv1d_bias(),
            padding=self.conv1d_padding,
            groups=self.conv1d_channels(),  # in_channels == out_channels == groups
        )

    # TODO(duncan): Make this a class instance variable?
    def conv1d_channels(self):
        """Returns the number of convolution channels on the current context parallel rank"""
        # The number of convolution input (or output) channels, per context parallel rank, is the
        # sum of the hidden (or feature) dimensions of x, B, and C, per context parallel rank
        return self.d_inner_local_tpcp + 2 * self.ngroups_local_tpcp * self.d_state

    def get_conv1d_weight(self) -> torch.Tensor:
        """Returns a slice of the conv1d weight relevant to the current context parallel rank"""
        # weight shape: [conv_dim, 1, d_conv]
        return self._slice_conv_param(self.conv1d_weight_cp1)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Returns a slice of the conv1d bias relevant to the current context parallel rank"""
        # bias shape: [conv_dim]
        return self._slice_conv_param(self.conv1d_bias_cp1)

    def get_dt_bias(self) -> torch.Tensor:
        """Returns a slice of dt_bias relevant to the current context parallel rank"""
        return self._slice_vector_param(self.dt_bias_cp1)

    def get_A_log(self) -> torch.Tensor:
        """Returns a slice of A_log relevant to the current context parallel rank"""
        return self._slice_vector_param(self.A_log_cp1)

    def get_D(self) -> torch.Tensor:
        """Returns a slice of D relevant to the current context parallel rank"""
        return self._slice_vector_param(self.D_cp1, has_hdim=self.D_has_hdim)

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """
        Slices a cp_size=1 conv1d parameter (either weight or bias) along the first dimension,
        returning the parts of the parameter needed for convolution on the current context parallel
        rank. Parameter slicing is done in the forward path so that gradients will backpropagate to
        the cp_size=1 parameters.
        """
        if self.cp_size == 1:
            return param

        x, B, C = torch.split(
            param,
            [
                self.d_inner_local_tp,
                self.ngroups_local_tp * self.d_state,
                self.ngroups_local_tp * self.d_state,
            ],
            dim=0,
        )

        # Slicing section of parameter associated with x:
        size = self.d_inner_local_tpcp
        start = self.cp_rank * size
        end = start + size
        x_sliced = x[start:end, ...]

        # Slicing section of parameter associated with B and C:
        size = self.ngroups_local_tpcp * self.d_state
        start = (self.cp_rank // self.group_repeat_count) * size
        end = start + size
        B_sliced = B[start:end, ...]
        C_sliced = C[start:end, ...]

        return torch.cat([x_sliced, B_sliced, C_sliced], dim=0).contiguous()

    def _slice_vector_param(self, param: torch.Tensor, has_hdim: bool = False) -> torch.Tensor:
        """
        Slices a cp_size=1 vector parameter along the first dimension, returning the part of the
        parameter needed on the current context parallel rank. Parameter slicing is done in the
        forward path so that gradients will backpropagate to the cp_size=1 parameters.
        """
        if self.cp_size == 1:
            return param

        size = self.d_inner_local_tpcp if has_hdim else self.nheads_local_tpcp
        start = self.cp_rank * size
        end = start + size
        return param[start:end]


def _all_to_all_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence/context-parallel-size, batch, local-hidden] to
    [global-sequence, batch, local-hidden/context-parallel-size].

    Args:
        input_ (torch.Tensor):
            The input tensor, which is partitioned along the sequence dimension
        cp_group (torch.distributed.ProcessGroup):
            Process group to use for context parallel

    Returns:
        torch.Tensor: The output tensor with shape
            [global-sequence, batch, local-hidden/context-parallel-size].
    """
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    s_out, h_out = s_in * cp_group.size(), h_in // cp_group.size()
    output = all_to_all_sp2hp(input_, group=cp_group)
    output = output.reshape(s_out, b_in, h_out)
    return output


def _all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence, batch, local-hidden/context-parallel-size] to
    [global-sequence/context-parallel-size, batch, local-hidden].

    Args:
        input_ (torch.Tensor):
            The input tensor, which is partitioned along the hidden dimension
        cp_group (torch.distributed.ProcessGroup):
            Process group to use for context parallel

    Returns:
        torch.Tensor: The output tensor with shape
            [global-sequence/context-parallel-size, batch, local-hidden].
    """
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    s_out, h_out = s_in // cp_group.size(), h_in * cp_group.size()
    output = all_to_all_hp2sp(input_, group=cp_group)
    output = output.reshape(s_out, b_in, h_out)
    return output


_THD_TE_DEFAULT_SHARED_BYTES = 48 * 1024


def _thd_partitioned_indices(
    cu_seqlens: torch.Tensor, total_tokens: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Match TE's THD zigzag indices without its whole-cu shared-memory limit.

    TE 2.10 stages every ``cu_seqlens`` entry in dynamic shared memory but
    does not opt in above 48 KiB. Keep TE for smaller layouts; search document
    boundaries on the input device when that launch would exceed the limit.
    """
    if cu_seqlens.numel() * cu_seqlens.element_size() <= _THD_TE_DEFAULT_SHARED_BYTES:
        return tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)

    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.ndim == 1
    assert total_tokens % (2 * cp_size) == 0 and 0 <= cp_rank < cp_size
    # Every physical document is divisible by 2*cp_size. The scaled cu table
    # partitions each rank's [total_tokens/cp_size] logical output positions.
    scaled_cu = torch.div(cu_seqlens, cp_size, rounding_mode="floor").contiguous()
    token_ids = torch.arange(total_tokens // cp_size, dtype=torch.int32, device=cu_seqlens.device)
    doc_ids = torch.searchsorted(scaled_cu, token_ids, right=True, out_int32=True) - 1
    starts = scaled_cu.index_select(0, doc_ids)
    ends = scaled_cu.index_select(0, doc_ids + 1)
    relative = token_ids - starts
    half = (ends - starts) // 2
    offset = torch.where(relative < half, cp_rank, 2 * (cp_size - 1) - cp_rank)
    return (starts * cp_size + relative + half * offset).to(torch.int32)


def _undo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Undoes the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 162534 to 123456 for
    sequential processing by the convolution and SSM.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [2 * i for i in range(num_chunks_div_2)] + [
            num_chunks - 2 * i - 1 for i in range(num_chunks_div_2)
        ]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        output = torch.empty_like(input_)
        for cp_rank in range(cp_size):
            start = cp_rank * seqlen_per_rank
            end = start + seqlen_per_rank
            index = _thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
            output[index] = input_[start:end]
        return output


def _redo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Redo the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 123456 to 162534 for
    efficient processing by attention.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [None] * num_chunks
        order[::2] = range(num_chunks_div_2)  # order[even]
        order[1::2] = reversed(range(num_chunks_div_2, num_chunks))  # order[odd]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        index = torch.empty(total_tokens, device=input_.device, dtype=torch.int32)
        for cp_rank in range(cp_size):
            start = cp_rank * seqlen_per_rank
            end = start + seqlen_per_rank
            index[start:end] = _thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
        return input_.index_select(0, index)


def reconstruct_tensor_cp(
    tensor: torch.Tensor,
    packed_seq_params: Optional[PackedSeqParams] = None,
    dim: int = 0,
    differentiable: bool = False,
) -> torch.Tensor:
    """All-gather a context-parallel-sharded tensor and undo the zigzag load balancing.

    Every CP rank ends up with the full sequence in true causal (sequential) order, so
    modules that need global context — the QSA indexer block selection and the PLE n-gram
    hashing plus its causal dilated conv — can run on the gathered tensor exactly as in the
    CP=1 case. The local rank's own slice is re-pinned into the gathered buffer so the local
    autograd graph is preserved by default. Attention must pass ``differentiable=True``:
    its local queries attend to remote keys/values, whose gradients must be summed across
    ranks in the gather's reduce-scatter backward.
    """
    from megatron.core import parallel_state

    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return tensor

    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = parallel_state.get_context_parallel_rank()

    need_transpose = dim != 0
    if need_transpose:
        tensor = tensor.transpose(0, dim).contiguous()

    tensor = tensor.contiguous()
    if differentiable:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        gathered = gather_from_sequence_parallel_region(
            tensor, tensor_parallel_output_grad=True, group=cp_group
        )
    else:
        output_list = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, tensor, group=cp_group)
        # Preserve the local autograd graph for existing replicated computations.
        output_list[cp_rank] = tensor
        gathered = torch.cat(output_list, dim=0)

    gathered = _undo_attention_load_balancing(gathered, cp_size, packed_seq_params)

    if need_transpose:
        gathered = gathered.transpose(0, dim).contiguous()
    return gathered


def split_tensor_cp(
    tensor: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None, dim: int = 0
) -> torch.Tensor:
    """Inverse of :func:`reconstruct_tensor_cp`: redo the zigzag load balancing and return
    this CP rank's local slice.

    Used to split a full-sequence output — produced on the gathered tensor — back into the
    CP-local zigzag form the rest of the model expects. No collective is needed: every rank
    already holds the full tensor from a preceding ``reconstruct_tensor_cp``, so this is a
    local reorder followed by a slice. The slice preserves the autograd graph for the local
    rows, routing their gradient back through the sequential full tensor to the local slot.
    """
    from megatron.core import parallel_state

    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return tensor

    cp_rank = parallel_state.get_context_parallel_rank()

    need_transpose = dim != 0
    if need_transpose:
        tensor = tensor.transpose(0, dim).contiguous()

    redone = _redo_attention_load_balancing(tensor, cp_size, packed_seq_params)
    s_full = redone.shape[0]
    s_local = s_full // cp_size
    out = redone[cp_rank * s_local : (cp_rank + 1) * s_local]

    if need_transpose:
        out = out.transpose(0, dim).contiguous()
    return out
