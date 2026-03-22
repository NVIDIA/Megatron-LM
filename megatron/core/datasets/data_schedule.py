# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.hybrid_cp_schedule import BalancedCPScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunDataIterator


class HybridCPDataLoaderWrapper:
    """
    A wrapper class that wraps around an existing data_iterator.
    For every __next__ call,
    1. Each DP rank pulls a batch of packed samples.
    2. Extracts the sequence lengths of each sub-sample and all-gathers across the DP group.
    3. Schedules the sub-samples to the DPxCP ranks using the BalancedCPScheduler.
    4. Based on the schedule, reroutes the sub-samples to the correct rank using all-to-all.
    5. Returns the assigned sub-samples to this rank.

    Args:
        data_iterator: The original data_iterator to wrap around
        config: The config object containing the max_seqlen_per_dp_cp_rank
        dp_cp_group: Data parallel context parallel group.
    """

    def __init__(
        self, data_iterator, config, pg_collection: Optional[ProcessGroupCollection] = None
    ):
        self.data_iterator = data_iterator
        self.config = config
        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            self.dp_group = parallel_state.get_data_parallel_group()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.dp_cp_group = pg_collection.dp_cp
            self.dp_group = pg_collection.dp
            self.tp_group = pg_collection.tp
        assert (
            self.dp_cp_group is not None and self.dp_group is not None and self.tp_group is not None
        ), "dp_cp_group, dp_group, tp_group must not be None when using hybrid context parallel"

        self.cp_balancing_scheduler = BalancedCPScheduler(
            config=self.config, dp_cp_group=self.dp_cp_group
        )

        self.total_hdp_gpus = self.dp_cp_group.size()

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def get_global_seqlens(self, subsample_seqlens: torch.Tensor) -> List[int]:
        """
        Gathers the sequence lengths of all subsamples from all DP ranks.
        Each DP rank loads the same number of microbatches but each microbatch
        may have a different number of subsamples.

        We find the number of subsamples each rank holds and then gather the
        sequence lengths of all subsamples from all ranks.
        """
        # Collect the number of subsamples from all ranks
        local_len = torch.tensor([subsample_seqlens.shape[0]], dtype=torch.int32).cuda()
        dp_subsample_count = [torch.zeros_like(local_len) for _ in range(self.dp_group.size())]
        torch.distributed.all_gather(dp_subsample_count, local_len, group=self.dp_group)

        # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
        dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
        max_sub_samples = int(dp_subsample_counts.max().item())

        if local_len.item() < max_sub_samples:
            subsample_seqlens_padded = torch.cat(
                [
                    subsample_seqlens,
                    torch.zeros(max_sub_samples - local_len.item(), dtype=torch.int32).cuda(),
                ],
                dim=0,
            )
        else:
            subsample_seqlens_padded = subsample_seqlens

        # Gather the subsample_seqlens from all ranks
        seqlens_gathered = [
            torch.empty_like(subsample_seqlens_padded) for _ in range(self.dp_group.size())
        ]
        torch.distributed.all_gather(
            seqlens_gathered, subsample_seqlens_padded, group=self.dp_group
        )

        # Trim each seqlens_gathered to the length of the correct sample
        for dp_rank, seqlen in enumerate(seqlens_gathered):
            seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

        seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
        seqlens_gathered = seqlens_gathered.cpu().tolist()

        # Calculate the offsets to assign unique global ID to each subsample.
        csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum[:-1]], dim=0)

        return seqlens_gathered, offsets

    def get_global_id_seqlens(self, num_local_subsamples, offsets, seqlens_gathered):
        """
        Calculates the global ID for each subsample.

        We assign a unique global ID to each subsample.

        Returns:
        global_id_seqlens: list of (global_id, seqlen) tuples for scheduling.
        global_ids_this_rank: list of global IDs locally present on this rank.
        """
        dp_rank = self.dp_group.rank()
        global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()
        # Create a list of (global_id, seqlen) tuples for scheduling
        global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]
        # Get the global IDs locally present on this rank
        global_ids_this_rank = global_ids[
            offsets[dp_rank] : offsets[dp_rank] + num_local_subsamples
        ]

        return global_id_seqlens, global_ids_this_rank

    def _gid_to_src_rank(self, gid: int, offsets: List[int]) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        # Since the torch.distributed.get_process_group_ranks
        # provides the global rank, we need to consider TP
        hdp_rank = (
            torch.distributed.get_process_group_ranks(self.dp_group)[dp_src_rank]
            // self.tp_group.size()
        )
        return hdp_rank

    def reroute_samples_to_hdp_ranks(
        self, batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
    ):
        """
        Reroutes the sub-samples to the correct rank after scheduling.

        For each key in the batch dict, we perform an all-to-all communication
        to transfer the data to the correct ranks.
        Since all CP ranks within a DP group have the same data, we only need
        to transfer data between matching CP ranks.
        """
        gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
        hdp_rank = self.dp_cp_group.rank()
        dp_ranks = torch.distributed.get_process_group_ranks(self.dp_group)
        # Here we actually want to get the DP group's rank within the HDP group,
        # we need to consider TP
        dp_ranks = [r // self.tp_group.size() for r in dp_ranks]

        data_keys = batch[0].keys()

        # Create the send plan
        combined_sample_id_groups: List[List[int]] = [[] for _ in range(self.total_hdp_gpus)]

        for d in range(self.total_hdp_gpus):
            for sample_id_group in sample_id_groups:
                combined_sample_id_groups[d].extend(sample_id_group[d])

        for dest_rank in range(self.total_hdp_gpus):
            combined_sample_id_groups[dest_rank].sort()

        # Filter out samples that are not present on this rank
        send_ids_sorted = [
            gid
            for d in dp_ranks
            for gid in combined_sample_id_groups[d]
            if gid in global_ids_this_rank
        ]
        # send_counts = [len(combined_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

        send_lens_split = [0] * self.total_hdp_gpus
        for dest_rank in range(self.total_hdp_gpus):
            if dest_rank in dp_ranks:
                send_lens_split[dest_rank] = sum(
                    [
                        global_id_seqlens[gid][1]
                        for gid in combined_sample_id_groups[dest_rank]
                        if gid in global_ids_this_rank
                    ]
                )
            else:
                # We only need to share local data with DP ranks that have different data.
                send_lens_split[dest_rank] = 0

        # Create the recv plan
        recv_sample_id_groups = [[] for _ in range(self.total_hdp_gpus)]
        for gid in combined_sample_id_groups[hdp_rank]:
            src_rank = self._gid_to_src_rank(gid, offsets)
            recv_sample_id_groups[src_rank].append(gid)

        recv_lens_split = [0] * self.total_hdp_gpus
        for src_rank in range(self.total_hdp_gpus):
            recv_lens_split[src_rank] = sum(
                [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
            )

        recv_ids_sorted = [
            gid for d in range(self.total_hdp_gpus) for gid in recv_sample_id_groups[d]
        ]
        recv_counts = [len(recv_sample_id_groups[d]) for d in range(self.total_hdp_gpus)]

        recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

        def _pack_sample_by_key(key: str) -> torch.Tensor:
            flattened_tensors = []
            for gid in send_ids_sorted:
                t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
                flattened_tensors.append(t)
            return (
                torch.cat(flattened_tensors, dim=0)
                if flattened_tensors
                else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
            )

        def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
            cursor = 0
            for i, gid in enumerate(recv_ids_sorted):
                sample_len = global_id_seqlens[gid][1]
                recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
                cursor += sample_len

        for key in data_keys:
            send_tensor = _pack_sample_by_key(key)
            recv_tensor = torch.empty(
                sum(recv_lens_split), device=torch.cuda.current_device(), dtype=send_tensor.dtype
            )
            torch.distributed.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=recv_lens_split,
                input_split_sizes=send_lens_split,
                group=self.dp_cp_group,
            )
            _unpack_sample_by_key(key, recv_tensor)

        recv_sample_with_id = {
            recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)
        }
        return recv_sample_with_id

    def unpack_batch(self, batch):
        """
        Unpacks the packed samples into a list of sub-samples.
        Since each sub-sample may be routed to different DPxCP ranks,
        we unpack the sample here to avoid unnecessarily transferring
        the entire packed sample.
        """
        batch_unpacked = []
        for sample in batch:
            for sub_sample in range(sample["cu_seqlens"].shape[0] - 1):
                sub_sample_dict = {}
                start_idx = sample["cu_seqlens"][sub_sample]
                end_idx = sample["cu_seqlens"][sub_sample + 1]
                if end_idx - start_idx == 0:
                    continue
                for key in sample.keys():
                    if key in ["cu_seqlens", "batch_idx", "max_seqlen"]:
                        continue
                    sub_sample_dict[key] = sample[key][start_idx:end_idx]
                batch_unpacked.append(sub_sample_dict)
        return batch_unpacked

    def _pack_sequences(
        self, samples: List[Dict[str, torch.Tensor]], local_cp_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        def _pack_tensors(tensors):
            return torch.cat([t.reshape(-1) for t in tensors], dim=0)

        def _get_pad_len(seq_len, local_cp_size):
            # Step1: Calculate padding for entire sequence after packing
            # Packed Sequence should be divisible by local_cp_size * 2
            # Or local_cp_size * tp_size * 2 when using sequence parallel
            tp_size = 1
            if self.config.sequence_parallel:
                # TODO (pmannan): Remove parallel_state usage and pass pg_collection instead
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                pad_granularity = local_cp_size * tp_size * 2
            else:
                pad_granularity = local_cp_size * 2
            mod_token_count = seq_len % pad_granularity
            seq_pad_len = 0
            if mod_token_count != 0:
                seq_pad_len = (pad_granularity - mod_token_count) % pad_granularity

            total_seq_len = seq_len + seq_pad_len

            # Step2: Calculate padding required for sequence after sharding
            sharded_pad_granularity = 1
            # MXFP8 BLOCK_SIZE is 32 and sequence after sharding should be divisible
            if self.config.fp8 is not None and self.config.fp8_recipe == "mxfp8":
                sharded_pad_granularity = 32
            if (
                self.config.moe_token_dispatcher_type == "flex"
                and self.config.moe_flex_dispatcher_backend == "hybridep"
            ):
                # HybridEP requires MAX_NUM_OF_TOKENS_PER_RANK to be divisible
                # by NUM_OF_TOKENS_PER_CHUNK (128)
                sharded_pad_granularity = 128

            # tp_size is set to 1 when sequence parallel is not enabled
            sharded_tensor_shape = total_seq_len // (local_cp_size * tp_size)
            mod_token_count = sharded_tensor_shape % sharded_pad_granularity
            sharded_pad_len = 0
            if mod_token_count != 0:
                sharded_pad_len = (sharded_pad_granularity - mod_token_count) * (local_cp_size * tp_size)

            return sharded_pad_len + seq_pad_len

        # Get the padded lengths of all sub-samples being packed
        sample_padded_lens = torch.tensor(
            [s["tokens"].shape[0] for s in samples], dtype=torch.int32
        )

        tokens = _pack_tensors([sample["tokens"] for sample in samples])
        labels = _pack_tensors([sample["labels"] for sample in samples])
        loss_mask = _pack_tensors([sample["loss_mask"] for sample in samples])
        position_ids = _pack_tensors([sample["position_ids"] for sample in samples])

        # Create the cu_seqlens_padded tensor
        cu_seqlens_padded = torch.empty(
            1, sample_padded_lens.numel() + 1, device=torch.cuda.current_device(), dtype=torch.int32
        )
        cu_seqlens_padded[0, 0] = 0
        cu_seqlens_padded[0, 1:] = torch.cumsum(sample_padded_lens, dim=0)

        # We only pad after packing because SFTDataset already pads
        # individual samples
        pad_len = _get_pad_len(tokens.shape[0], local_cp_size)
        if pad_len > 0:
            tokens = torch.cat(
                [tokens, torch.zeros(pad_len, dtype=tokens.dtype, device=tokens.device)]
            )
            labels = torch.cat(
                [labels, torch.zeros(pad_len, dtype=labels.dtype, device=labels.device)]
            )
            loss_mask = torch.cat(
                [loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype, device=loss_mask.device)]
            )
            position_ids = torch.cat(
                [
                    position_ids,
                    torch.tensor(
                        range(position_ids[-1] + 1, position_ids[-1] + 1 + pad_len),
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    ),
                ]
            )
            sample_padded_lens = torch.cat(
                [
                    sample_padded_lens,
                    torch.tensor([pad_len], dtype=torch.int32, device=sample_padded_lens.device),
                ]
            )
            cu_seqlens_padded[:, -1:] = cu_seqlens_padded[:, -1:] + pad_len

        new_sample = {}
        new_sample["tokens"] = tokens
        new_sample["labels"] = labels
        new_sample["loss_mask"] = loss_mask
        new_sample["position_ids"] = position_ids
        if local_cp_size is not None:
            new_sample["local_cp_size"] = torch.tensor(
                local_cp_size, device=torch.cuda.current_device(), dtype=torch.int32
            )

        # new_sample["cu_seqlens_padded"] = cu_seqlens_padded
        # We set cu_seqlens to cu_seqlens_padded here
        # we don't provide cu_seqlens_padded here because `get_batch_on_this_tp_rank` does not
        # consider it at the moment.
        # It is assumed that any necessary padding will be after data is loaded in.
        # TODO(pmannan): We need to be able to differentiate if the original data_iterator
        # is providing padded samples or valid lengths.
        new_sample["cu_seqlens"] = cu_seqlens_padded
        max_seqlen = torch.max(torch.diff(cu_seqlens_padded[0])).to(dtype=torch.int32)
        new_sample["max_seqlen"] = max_seqlen

        return new_sample

    def _build_packed_microbatches(
        self,
        grouped_samples: List[List[Dict[str, torch.Tensor]]],
        sample_id_groups: List[List[int]],
        hdp_rank: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build packed samples for each microbatch given a pre-built list of `samples` per microbatch.

        Args:
            grouped_samples: List of length `num_microbatches`. Each element is the `samples` list
                (list[sample]) for that microbatch, where `sample` is the dict returned by
                `dataset.__getitem__`.
            sample_id_groups: List of length `num_microbatches`.
                Each element is the `sample_id_groups` list (list[sample_id]) for that microbatch,
                where `sample_id` is the id of the sub-sample.

        Returns:
            new_samples: list of packed samples (dicts) length == num_micro_batches.
        """
        num_micro_batches = len(grouped_samples)

        new_samples: List[Dict[str, torch.Tensor]] = []
        for i in range(num_micro_batches):
            samples = grouped_samples[i]
            local_cp_size = -1
            # sample_id_groups = [[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [6, 7, 8]]]
            # Indicates the sub-sample ids per microbatch for each DPxCP rank.
            for sub_sample_id in sample_id_groups[i][hdp_rank]:  # i:0 hdp_rank:0 [[0, 1, 2]]
                # sub_sample_id: 0 / 1 / 2
                partner_cp_size = len(
                    [True for sample_ids in sample_id_groups[i] if sub_sample_id in sample_ids]
                )

                if local_cp_size == -1:
                    local_cp_size = partner_cp_size
                else:
                    assert (
                        local_cp_size == partner_cp_size
                    ), f"\
                        found sample within a packed microbatch with different local_cp_size: \
                        {local_cp_size} != {partner_cp_size}"

            new_sample = self._pack_sequences(samples, local_cp_size)
            new_samples.append(new_sample)

        return new_samples

    def unpad_batch(self, batch):
        """
        Removes the end padding from the batch which could lead to an invalid sample.
        This could be a result of truncation or padding in the dataset.
        For example, a packed sample is truncated and is left with prompt tokens
        which leads to a sample with all zero loss mask.
        We do this before scheduling.
        """
        for sample in batch:
            end_sample_token_count = int(sample["cu_seqlens"][-1] - sample["cu_seqlens"][-2])
            if sample["loss_mask"][-end_sample_token_count:].sum() == 0:
                sample["cu_seqlens"][-1] = sample["cu_seqlens"][-2]
                for key in sample.keys():
                    if key in ["cu_seqlens", "batch_idx", "max_seqlen"]:
                        continue
                    sample[key] = sample[key][:-end_sample_token_count]
        return batch

    def __next__(self) -> Any:
        """
        Get the next item from the dataset, pull scheduling metadata and return it.
        """
        if self.data_iterator is None:
            # TP0 reads from data_iterator, others receive via broadcast.
            return None, None
        else:
            batch = next(self.data_iterator)
        subsample_seqlens = []
        batch = self.unpad_batch(batch)
        for sample in batch:
            subsample_seqlens.extend(
                [
                    int(sample["cu_seqlens"][i + 1] - sample["cu_seqlens"][i])
                    for i in range(0, sample["cu_seqlens"].shape[0] - 1)
                ]
            )
        subsample_seqlens = torch.tensor(subsample_seqlens, dtype=torch.int32).cuda()
        subsample_seqlens = subsample_seqlens[subsample_seqlens != 0]

        seqlens_gathered, offsets = self.get_global_seqlens(subsample_seqlens)

        global_id_seqlens, global_ids_this_rank = self.get_global_id_seqlens(
            subsample_seqlens.shape[0], offsets, seqlens_gathered
        )

        groups, sample_id_groups = self.cp_balancing_scheduler.get_groups_and_subsamples(
            global_id_seqlens, self.config
        )

        batch = self.unpack_batch(batch)
        samples_this_rank_with_id = self.reroute_samples_to_hdp_ranks(
            batch, global_ids_this_rank, global_id_seqlens, sample_id_groups, offsets
        )

        batch, sample_id_groups = samples_this_rank_with_id, sample_id_groups

        hdp_rank = self.dp_cp_group.rank()
        num_micro_batches = len(sample_id_groups)

        grouped_samples = [
            [batch[sub_sample_id] for sub_sample_id in sample_id_groups[i][hdp_rank]]
            for i in range(num_micro_batches)
        ]

        new_samples = self._build_packed_microbatches(
            grouped_samples=grouped_samples, sample_id_groups=sample_id_groups, hdp_rank=hdp_rank
        )

        new_data_iterator = RerunDataIterator(iter(new_samples))

        return new_data_iterator, sample_id_groups
