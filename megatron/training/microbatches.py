# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron number of micro-batches calculators."""

from abc import ABC
from abc import abstractmethod


def build_num_microbatches_calculator(args):

    # Constant num micro-batches.
    if args.rampup_batch_size is None:
        num_microbatches_calculator = ConstantNumMicroBatches(
            args.global_batch_size, args.micro_batch_size,
            args.data_parallel_size)
        if args.rank == 0:
            print('setting number of micro-batches to constant {}'.format(
                num_microbatches_calculator.get()), flush=True)

    else:
        assert len(args.rampup_batch_size) == 3, 'expected the following ' \
            'format: --rampup-batch-size <start batch size> ' \
            '<batch size incerement> <ramp-up samples>'
        start_batch_size = int(args.rampup_batch_size[0])
        batch_size_increment = int(args.rampup_batch_size[1])
        ramup_samples = int(args.rampup_batch_size[2])
        if args.rank == 0:
            print('will use batch size rampup starting from global batch '
                  'size {} to global batch size {} with batch size increments '
                  '{} over {} samples.'.format(start_batch_size,
                                               args.global_batch_size,
                                               batch_size_increment,
                                               ramup_samples), flush=True)
        num_microbatches_calculator = RampupBatchsizeNumMicroBatches(
            start_batch_size, batch_size_increment, ramup_samples,
            args.global_batch_size, args.micro_batch_size,
            args.data_parallel_size)

    return num_microbatches_calculator


class NumMicroBatchesCalculator(ABC):

    def __init__(self):
        self.num_micro_batches = None
        self.current_global_batch_size = None

    def get(self):
        return self.num_micro_batches

    def get_current_global_batch_size(self):
        return self.current_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check):
        pass


class ConstantNumMicroBatches(NumMicroBatchesCalculator):

    def __init__(self, global_batch_size, micro_batch_size, data_parallel_size):
        micro_batch_times_data_parallel = micro_batch_size * \
                                          data_parallel_size
        assert global_batch_size % micro_batch_times_data_parallel == 0, \
            'global batch size ({}) is not divisible by micro batch size ({})' \
            ' times data parallel size ({})'.format(global_batch_size,
                                                    micro_batch_size,
                                                    data_parallel_size)
        self.num_micro_batches = global_batch_size // \
                                 micro_batch_times_data_parallel
        assert self.num_micro_batches >= 1
        self.current_global_batch_size = global_batch_size

    def update(self, consumed_samples, consistency_check):
        pass


class RampupBatchsizeNumMicroBatches(NumMicroBatchesCalculator):

    def __init__(self, start_batch_size, batch_size_increment, ramup_samples,
                 global_batch_size, micro_batch_size, data_parallel_size):
        """Batch size ramp up.
        Over 
          steps = (global-batch-size - start-batch-size) / batch_size_increment
        increment batch size from start-batch-size to global-batch-size using
          rampup-samples / steps
        samples.
        Arguments:
            start_batch_size: global batch size to start with
            batch_size_increment: global batch size increments
            ramup_samples: number of samples to use ramp up global
               batch size from `start_batch_size` to `global_batch_size`
            global_batch_size: global batch size post rampup
            micro_batch_size: micro batch size
            data_parallel_size: data parallel size.
        """

        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * \
                                                    self.data_parallel_size
        assert self.micro_batch_times_data_parallel_size > 0
        
        assert start_batch_size > 0
        self.start_batch_size = start_batch_size

        assert global_batch_size > 0
        self.global_batch_size = global_batch_size
        diff_batch_size = self.global_batch_size - self.start_batch_size
        assert diff_batch_size >= 0
        assert batch_size_increment > 0
        self.batch_size_increment = batch_size_increment
        assert diff_batch_size % batch_size_increment == 0, 'expected ' \
            'global batch size interval ({}) to be divisible by global batch ' \
            'size increment ({})'.format(diff_batch_size, batch_size_increment)

        num_increments = diff_batch_size // self.batch_size_increment
        self.ramup_samples = ramup_samples
        assert self.ramup_samples >= 0
        self.rampup_samples_per_increment = self.ramup_samples / num_increments

        # Initialize number of microbatches.
        self.update(0, False)


    def update(self, consumed_samples, consistency_check):

        if consumed_samples > self.ramup_samples:
            self.current_global_batch_size = self.global_batch_size
        else:
            steps = int(consumed_samples / self.rampup_samples_per_increment)
            self.current_global_batch_size = self.start_batch_size + \
                steps * self.batch_size_increment
            assert self.current_global_batch_size <= self.global_batch_size

        if consistency_check:
            assert self.current_global_batch_size % \
                self.micro_batch_times_data_parallel_size == 0, 'current global ' \
                'batch size ({}) is not divisible by micro-batch-size ({}) times' \
                'data parallel size ({})'.format(self.current_global_batch_size,
                                                 self.micro_batch_size,
                                                 self.data_parallel_size)
        self.num_micro_batches = self.current_global_batch_size // \
                                 self.micro_batch_times_data_parallel_size
