# Copyright (C) 2024 Intel Corporation

import math
from typing import Union

import torch

from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_expert_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
)


class CapacityBins(torch.nn.Module):
    """CapacityBins - maps current capacity value into capacity bins.

    When using drop_tokens=false, the capacity at each iteration will differ since
    we use a capacity to accommodate for the largest number of tokens sent to an expert.
    This creates dynamic shapes tensors.

    The motivation for using bins is to reduce the dynamic shapes to a limited set, hence
    being more friendly when running in non-eager mode (e.g., using compile).

    The minimum range of capacity is the optimal capacity where all tokens are evenly routed
    among all experts. The maximum range of capacity is the worst-case capacity where all
    tokens are routed to a single expert (unlikely, but a valid upper bound).

    This class maintains the current configured capacity bins. It also tracks bins usage info
    which enables to dynamically update the capacity bins to optimize performance (i.e. to
    minimize the number of dummy extra tokens that are routed).

    Upon initialization, if configured_bins provided, use configured_bins to initialize the bins.
    Otherwise, the capacity bins are initialized to bins with exponentially growing width.

    Argument use_cpu forces capacity bins logic to be executed on the CPU (not on the accelerator).
    When using torch.compile, this prevents potential graph breaks.
    """

    def __init__(
        self,
        topk: int,
        num_experts: int,
        num_capacity_bins: int,
        capacity_bins_exp_base: float,
        capacity_bins_alignment: int,
        min_bin_size: int = 1,
        configured_bins: Union[list, None] = None,
        use_cpu=True,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.num_experts = num_experts
        self.num_capacity_bins = num_capacity_bins
        self.capacity_bins_exp_base = capacity_bins_exp_base
        self.configured_alignment = capacity_bins_alignment
        assert min_bin_size > 0, f'CapacityBins min_bin_size must be > 0, got {min_bin_size}'
        self.min_bin_size = min_bin_size
        if configured_bins is not None:
            assert (
                len(configured_bins) == self.num_capacity_bins
            ), f'Configured bins ({configured_bins}) does not match num capacity bins ({self.num_capacity_bins})'
            assert all(
                bin_edge > 0 for bin_edge in configured_bins
            ), 'Configured bin edges must be > 0'
            assert all(
                configured_bins[i] < configured_bins[i + 1] for i in range(len(configured_bins) - 1)
            ), 'Configured bin edges must be a strictly increasing list'
        self.use_cpu = use_cpu

        # initialize usage stats
        zero_bins = torch.zeros(
            num_capacity_bins, dtype=torch.long, device='cpu', requires_grad=False
        )
        self.register_buffer('bins_usage', zero_bins.clone().detach())
        self.register_buffer('bins_usage_last', zero_bins.clone().detach())

        # initialize bin edges
        if configured_bins is not None:
            self.register_buffer(
                'capacity_bins',
                torch.tensor(configured_bins, dtype=torch.long, device='cpu', requires_grad=False),
            )
        else:
            # we don't know the range of the capacity bins, therefore we create a zeroed tensor
            # when we load from checkpoint, or during the first forward, we update the bins
            # note that if the first element = 0, it marks that capacity_bins is not initialized
            self.register_buffer('capacity_bins', zero_bins.clone().detach())

        # attribute self.device is the device to use for capacity bins logic, where attribute self.model_device
        # is the device used by the model. attributes can be different in case use_cpu is configured.
        self.device = None
        self.model_device = None

        self.min_tokens_per_expert = None
        self.max_tokens_per_expert = None
        self.alignment = None

    def set_bins(self, bins: list):
        with torch.no_grad():
            # set the new capacity bins and clear the usage stats (not relevant for new bins)
            self.capacity_bins.copy_(torch.tensor(bins, dtype=torch.long, device=self.device))
            self.bins_usage.zero_()
            self.bins_usage_last.zero_()

    def get_stats(self, incremental=True):

        def is_usage_data_available(usage_tensor):
            with torch.no_grad():
                return usage_tensor.sum().item() > 0

        if not is_usage_data_available(self.bins_usage):
            return None

        with torch.no_grad():
            # reduce stats across all workers; for that, we need to temporarily move stats to model device
            bins_usage = self.bins_usage.clone().detach().to(self.model_device)

            torch.distributed.all_reduce(
                bins_usage, op=torch.distributed.ReduceOp.SUM, group=get_data_parallel_group()
            )

            bins_usage = bins_usage.to(self.device)

            # incremental returns only the diff from last activation of get_stats()
            if incremental:
                delta_bins_usage = bins_usage
                if is_usage_data_available(self.bins_usage_last):
                    delta_bins_usage -= self.bins_usage_last
                self.bins_usage_last.copy_(bins_usage)
                bins_usage = delta_bins_usage

            # stats are returned using cpu tensors
            bins_usage = bins_usage.to('cpu')
            bins_usage_list = bins_usage.tolist()
            bins_edges = self.capacity_bins.clone().detach().to('cpu')
            bins_edges_list = bins_edges.tolist()
            stats = {
                'min_range': self.min_tokens_per_expert,
                'max_range': self.max_tokens_per_expert,
                'alignment': self.alignment,
                'min_bin_size': self.min_bin_size,
                'edges': bins_edges,
                'usage': bins_usage,
                'summary': {
                    f'bin{i}_{bins_edges_list[i]}': bins_usage_list[i]
                    for i in range(len(bins_usage))
                },
            }
        return stats

    def _save_device(self, device: str):
        if self.device is None:
            # set self.device to requested device for capacity bins logic. also keep device used by model
            assert (
                self.model_device is None
            ), f'Expected model_device=None on 1st forward, but got {self.model_device}'
            self.model_device = device
            self.device = 'cpu' if self.use_cpu else self.model_device

            # move all model's buffers to device used for capacity bins logic
            self.capacity_bins = self.capacity_bins.to(self.device)
            self.bins_usage = self.bins_usage.to(self.device)
            self.bins_usage_last = self.bins_usage_last.to(self.device)

    def get_binned_capacity(self, gate_output, capacity, update_stats=True):
        with torch.no_grad():
            # on first forward, capture device used
            # then, move inputs to requested capacity bins device

            self._save_device(gate_output.device)
            gate_output, capacity = gate_output.to(self.device), capacity.to(self.device)

            # get bins; if first call, calculate bins
            bins = self._get_capacity_bins(gate_output.shape[0], gate_output.device)

            # find bin to use based on current capacity and update stats
            index = torch.searchsorted(bins, capacity, right=False)
            index = torch.min(
                index, torch.tensor(len(bins) - 1, dtype=torch.int64, device=self.device)
            )
            if update_stats:
                self._update_stats(index)

        return bins[index].to(self.model_device)

    def _update_stats(self, index):
        # currently we maintain stats for training only
        if self.training:
            self.bins_usage[index] += 1

    def _generate_bins(self, force_start_bin=False):
        # create exponentially growing width bins, and normalize width sum to 1.0
        # when force_start_bin=True, we force the first bin value = start range (aka start).
        # force_start_bin=True is handled by prepending width=0
        start = self.min_tokens_per_expert
        stop = self.max_tokens_per_expert
        exp_base = torch.tensor(self.capacity_bins_exp_base, dtype=torch.float).to(self.device)
        if force_start_bin:
            bin_widths = exp_base ** torch.arange(0, self.num_capacity_bins - 1, device=self.device)
            bin_widths = torch.cat([torch.tensor([0.0], device=bin_widths.device), bin_widths])
        else:
            bin_widths = exp_base ** torch.arange(0, self.num_capacity_bins, device=self.device)
        normalized_bin_widths = bin_widths / torch.sum(bin_widths)

        # calculate bin edges by accumulating the bins width and scaling to [start...stop] range
        # finally, align bin edges
        bin_edges = torch.cumsum(normalized_bin_widths, dim=0)
        bin_edges = start + (stop - start) * bin_edges
        bin_edges = torch.ceil(bin_edges / self.alignment).mul(self.alignment).to(torch.long)

        # verify that we got N distinct capacity bins
        assert len(set(bin_edges.tolist())) == self.num_capacity_bins, (
            f'Resulting capacity bins size != {self.num_capacity_bins}, bins={bin_edges.tolist()}. '
            f'Please try to reduce expotent base value with HL_CAPACITY_BINS_EXP_BASE '
            f'(current value: {exp_base.item()}, minimal value: 1.0). '
            f'If this is insufficient, limit the number of capacity bins with '
            f'HL_MOE_NUM_CAPACITY_BINS (set to {self.num_capacity_bins}) or reduce alignment with '
            f'HL_MOE_CAPACITY_BINS_ALIGNMENT (set to {self.alignment}).'
        )

        return bin_edges

    def _verify_configured_bins(self):
        """This method runs once (at first forward) and verifies that configured bins are valid"""
        # verify configured bins range
        if (
            self.capacity_bins[0].item() < self.min_tokens_per_expert
            or self.capacity_bins[-1].item() < self.max_tokens_per_expert
        ):
            print(
                f'Invalid capacity_bins={self.capacity_bins.clone().detach().cpu().tolist()},tokens per expert (min,max)={(self.min_tokens_per_expert, self.max_tokens_per_expert)}'
            )
            return False
        # verify configured bins alignment
        alignment = torch.tensor(self.alignment, dtype=torch.long, device=self.device)
        if torch.remainder(self.capacity_bins, alignment).sum().item() != 0:
            print(
                f'Invalid capacity_bins={self.capacity_bins.clone().detach().cpu().tolist()}, alignment={self.alignment} '
            )
            return False
        return True

    def _get_capacity_bins(self, size: int, device: str) -> Union[torch.Tensor, None]:
        """Generates capacity bins with exponential growing width.

        During training, we encourage tokens to be evenly routed (via aux loss).
        Therefore, generate bins with exponential growing bins width, i.e., bins that are
        closer to the start are smaller and thus have less extra non-required capacity.

        Alignment is required when the bins have to be aligned on a specific value.
        For example:
        1. Configured alignment (capacity_bins_alignment) due to e.g. hardware specific considerations
        2. When the non-experts are using TP and the experts ate not using TP, we
        need to align the bins on TP boundary.

        Args:
            gate_output (torch.Tensor): router gating function output tensor

        Returns:
            bins tensor (torch.Tensor dtype=torch.long)
        """
        # in case of first forward, initialize information based on gate_output
        if self.min_tokens_per_expert is None:
            # calculate optimal and worst case (min and max) tokens per expert
            n_tokens_in_micro_batch = torch.tensor(size, device=device).to(torch.long)
            n_optimal_tokens_per_expert = torch.ceil(
                self.topk * n_tokens_in_micro_batch / self.num_experts
            ).to(torch.long)
            self.min_tokens_per_expert = n_optimal_tokens_per_expert.item()
            self.max_tokens_per_expert = n_tokens_in_micro_batch.item()
            # handle bin alignment - maximum between configured alignment and TP (if used)
            tp_alignment = 1
            if get_expert_model_parallel_world_size() == 1:
                tp_alignment = get_tensor_model_parallel_world_size()
            self.alignment = max(self.configured_alignment, tp_alignment)

            # if bins configured (either configured by user or loaded from checkpoint) - verify valid bins
            # otherwise, initialize bins
            if self.capacity_bins[0] > 0:
                if self.training and not self._verify_configured_bins():
                    # temporary WA for diff in parameters such as seql, bs (number of tokens per expert change) after load from checkpoint
                    self.capacity_bins = self._generate_bins()
            else:
                self.capacity_bins = self._generate_bins()

        return self.capacity_bins


def optimize_bins(
    min_range, bins: torch.Tensor, bins_usage: torch.Tensor, alignment, min_bin_size
) -> list:
    """Optimize MOE capacity bins according to collected bins usage statistics

    The bins are optimized to minimize the cost of binning.
    The cost of each bin is defined as the additional tokens processed in this bin.
    Since we don't have the actual capacities that were mapped to each bin, we use the median of the bin.
    After we calculate the cost of all bins, we iteratively try to replace the lowest and highest cost bins
    with 2 bins: the original highest cost bin and the median of the highest cost bin.
    This way, we keep the number of bins constant while decreasing the overall cost of binning.

    For example:
        Given bins [150, 200, 250, 300] with start of range=100
        And usage  [100, 0,   50,  10 ]

        We first calculate the cost of each bin:
        Cost:      [25*100, 25*0, 25*50, 25*10] = [2500, 0, 1250, 250]

        Lowest cost bin is 200 (index=1)
        Highest cost bin is 150 (index=0)

        First iteration of optimization:
        Remove bin1 and split bin0 --> [125, 150, 250, 300]
    """

    def align_to(value):
        return int(math.ceil(value / alignment) * alignment)

    # sort bins by their cost of usage (we want to split high cost bins)
    # we assume that for each bin, the cost is 1/2 of its width * usage count
    shifted_bins = torch.cat(
        [torch.tensor([min_range], dtype=bins.dtype, device=bins.device), bins[:-1]]
    )
    width = bins - shifted_bins
    cost = bins_usage * width / 2.0
    sorted_cost = torch.argsort(cost, descending=False, stable=True).tolist()

    # sorted cost is in ascending order
    # min_sort_idx is current index into sorted_cost for candidate bin to be removed
    # max_sort_idx is current index into sorted_cost for candidate bin to be split
    bins = bins.tolist()
    n_bins = len(bins)
    min_sort_idx = 0
    max_sort_idx = n_bins - 1
    new_bins = []
    while min_sort_idx <= max_sort_idx:
        # if same cost, keep all remaining bins and exit
        # this also handles the case of min_sort_idx == max_sort_idx
        min_cost = cost[sorted_cost[min_sort_idx]]
        max_cost = cost[sorted_cost[max_sort_idx]]
        if min_cost == max_cost:
            bin_indexes = sorted_cost[min_sort_idx : max_sort_idx + 1]
            new_bins.extend([bins[idx] for idx in bin_indexes])
            break

        # last bin can't be removed
        min_bin_idx = sorted_cost[min_sort_idx]
        if min_bin_idx == n_bins - 1:
            new_bins.append(bins[min_bin_idx])
            min_sort_idx += 1
            continue

        # calculate the left & right bin's width of the candidate bin after we split it to 2
        # verify that both left & right will meet the min bin size requirement
        max_bin_idx = sorted_cost[max_sort_idx]
        max_bin_start = min_range if max_bin_idx == 0 else bins[max_bin_idx - 1]
        max_bin_end = bins[max_bin_idx]
        mid_point = (max_bin_start + max_bin_end) // 2
        mid_point = align_to(mid_point)
        left_bin_width = mid_point - max_bin_start
        right_bin_width = max_bin_end - mid_point
        if left_bin_width < min_bin_size or right_bin_width < min_bin_size:
            new_bins.append(bins[max_bin_idx])
            max_sort_idx -= 1
            continue

        # skip min cost bin and split max cost bin
        new_bins.append(mid_point)
        new_bins.append(max_bin_end)
        min_sort_idx += 1
        max_sort_idx -= 1

    # sort the bins in ascending order
    bins = sorted(new_bins)
    return bins
