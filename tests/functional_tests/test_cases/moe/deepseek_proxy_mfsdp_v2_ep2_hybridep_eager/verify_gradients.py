# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reject even a single bad gradient step in the #7400 training regression."""

import argparse
import math

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    """Check every loss and pre-clipping gradient norm from one training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--train-iters", required=True, type=int)
    args = parser.parse_args()

    events = EventAccumulator(args.tensorboard_dir, size_guidance={"scalars": 0})
    events.Reload()
    for metric in ("lm loss", "grad-norm"):
        samples = events.Scalars(metric)
        assert [sample.step for sample in samples] == list(
            range(1, args.train_iters + 1)
        ), f"{metric}: expected exactly {args.train_iters} consecutive optimizer steps"
        for sample in samples:
            assert math.isfinite(
                sample.value
            ), f"{metric} is nonfinite at step {sample.step}: {sample.value}"
            if metric == "grad-norm":
                # Healthy runs peak near 6.2. #7400 produced norms of 65-107
                # before NaNs; do not let the approximate loss check's allowance
                # for an outlier step hide an isolated gradient corruption.
                assert (
                    0 < sample.value < 20
                ), f"Unexpected gradient norm at step {sample.step}: {sample.value}"
        print(f"Validated {len(samples)} {metric} samples")


if __name__ == "__main__":
    main()
