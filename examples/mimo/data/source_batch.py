# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Source-global-batch iteration for replicated multimodal example datasets."""


class SourceBatchIterator:
    """Resume a deterministic dataset at a complete source-global-batch boundary.

    The dataset implements ``build_global_batch(step, global_batch_size)`` and
    returns ``(samples, media)``. This is an example dataset contract, independent
    of the core runtime and the number of decoder rounds selected by packing.
    """

    def __init__(self, dataset, global_batch_size, consumed_samples=0) -> None:
        if global_batch_size <= 0 or consumed_samples < 0 or consumed_samples % global_batch_size:
            raise ValueError("MIMO resume requires a complete source global batch")
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.step = consumed_samples // global_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples, media = self.dataset.build_global_batch(self.step, self.global_batch_size)
        result = dict(step=self.step, samples=samples, media=media)
        self.step += 1
        return result
