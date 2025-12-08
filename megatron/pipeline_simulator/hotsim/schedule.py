import pprint
from dataclasses import dataclass
from enum import Enum

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

pp = pprint.PrettyPrinter()


class Op(Enum):
    FORWARD = "F"
    BACKWARD = "B"


@dataclass(frozen=True)
class Chunk:
    batch_id: int
    chunk_id: int
    length: int
    # Assume that we always use a batch size of 1 for simplicity
    batch_size: int = 1


@dataclass(frozen=True)
class Action:
    stage_id: int
    op: Op
    chunk: Chunk


def plot_timeline(actions_by_rank: list[list[Action]]) -> None:
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    height = len(actions_by_rank)
    width = max([len(actions) for actions in actions_by_rank])
    blues = ["#BFDBFE", "#60A5FA", "#2563EB", "#1E40AF"]
    greens = ["#BBF7D0", "#34D399", "#16A34A", "#166534"]
    for rank, actions in enumerate(actions_by_rank):
        for step, action in enumerate(actions):
            if action is None:
                continue
            if action.op == Op.FORWARD:
                color = blues[action.chunk.batch_id % len(blues)]
            else:
                color = greens[action.chunk.batch_id % len(greens)]
            ax.add_patch(
                patches.Rectangle(
                    (step / width, rank / height),
                    1 / width,
                    1 / height,
                    facecolor=color,
                    edgecolor="black",
                )
            )
            if action is not None:
                chunk = action.chunk
                ax.text(
                    (step + 0.5) / width,
                    (rank + 0.5) / height,
                    f"{chunk.chunk_id}\n{chunk.batch_id}",
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
    ax.invert_yaxis()
    plt.show()


def plot_memory_histogram(histogram: list[int], peak_histogram: list[int]) -> None:
    histogram = np.array(histogram) / 1024**3  # Convert to GiB
    peak_histogram = np.array(peak_histogram) / 1024**3  # Convert to GiB

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(histogram, label="Memory Usage", color="blue")
    ax.plot(peak_histogram, label="Peak Memory Usage", color="red")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Memory Size (GiB)")
    ax.set_title("Memory Usage Over Time")
    ax.legend()
    plt.show()


def plot_combined_visualization(
    actions_by_rank: list[list[Action]], histogram: list[int], peak_histogram: list[int]
) -> None:
    """
    Create a combined visualization with timeline and memory histogram stacked vertically.

    Parameters:
    -----------
    actions_by_rank : list[list[Action]]
        Actions organized by ranks for timeline visualization
    histogram : list[int]
        Memory usage over time
    peak_histogram : list[int]
        Peak memory usage over time
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1]}
    )

    # Plot timeline on the top subplot
    height = len(actions_by_rank)
    width = max([len(actions) for actions in actions_by_rank])
    blues = ["#BFDBFE", "#60A5FA", "#2563EB", "#1E40AF"]
    greens = ["#BBF7D0", "#34D399", "#16A34A", "#166534"]

    for rank, actions in enumerate(actions_by_rank):
        for step, action in enumerate(actions):
            if action is None:
                continue
            if action.op == Op.FORWARD:
                color = blues[action.chunk.batch_id % len(blues)]
            else:
                color = greens[action.chunk.batch_id % len(greens)]
            ax1.add_patch(
                patches.Rectangle(
                    (step, rank),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                )
            )
            if action is not None:
                chunk = action.chunk
                ax1.text(
                    step + 0.5,
                    rank + 0.5,
                    f"{chunk.chunk_id}\n{chunk.batch_id}",
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="center",
                )

    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)  # Invert y-axis
    ax1.set_title("Pipeline Execution Timeline")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Ranks")

    # Plot memory histogram on the bottom subplot
    histogram = np.array(histogram) / 1024**3  # Convert to GiB
    peak_histogram = np.array(peak_histogram) / 1024**3  # Convert to GiB
    ax2.plot(histogram, label="Memory Usage", color="blue")
    ax2.plot(peak_histogram, label="Peak Memory Usage", color="red")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Memory Size (GiB)")
    ax2.set_title("Memory Usage Over Time")
    ax2.legend()
    ax2.set_xlim(0, len(histogram))
    ax2.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def build_slimpipe_schedule(p, v, chunks_list: list[list[int]]) -> list[list[Action]]:
    """
    Simulate pipeline parallelism with different configurations.

    This function simulates the behavior of pipeline parallelism in distributed training,
    analyzing how data flows through different pipeline stages across multiple ranks.

    Parameters
    ----------
    p : int
        Number of ranks (processors/GPUs) available for parallel processing.
    v : int
        Number of pipeline stages in the model.
    num_chunks : list[int]
        List of micro-batch slice counts for each batch to process.
        Each number must be divisible by the number of ranks (p).

    Returns
    -------
    None
    """
    num_chunks = [len(chunks) for chunks in chunks_list]

    if any(n == 0 for n in num_chunks):
        raise ValueError("Number of slices must be greater than 0.")
    if any(n % p != 0 for n in num_chunks):
        raise ValueError("Number of slices must be divisible by the number of ranks.")
    if any(num_chunks[i] > num_chunks[i - 1] for i in range(1, len(num_chunks))):
        raise ValueError("Number of slices must be non-increasing.")

    m = len(num_chunks)
    forwards = []
    for batch_id in range(m):
        n = num_chunks[batch_id]
        for start_chunk_id in range(0, n, p):
            for stage_id in range(v):
                for chunk_id in range(start_chunk_id, start_chunk_id + p):
                    chunk = Chunk(
                        batch_id=batch_id,
                        chunk_id=chunk_id,
                        length=chunks_list[batch_id][chunk_id],
                    )
                    computation = Action(
                        stage_id=stage_id,
                        op=Op.FORWARD,
                        chunk=chunk,
                    )
                    forwards.append(computation)

    backwards = []
    for batch_id in range(m):
        n = num_chunks[batch_id]
        for start_chunk_id in range(n - 1, -1, -p):
            for stage_id in range(v - 1, -1, -1):
                for chunk_id in range(start_chunk_id, start_chunk_id - p, -1):
                    chunk = Chunk(
                        batch_id=batch_id,
                        chunk_id=chunk_id,
                        length=chunks_list[batch_id][chunk_id],
                    )
                    computation = Action(
                        stage_id=stage_id,
                        op=Op.BACKWARD,
                        chunk=chunk,
                    )
                    backwards.append(computation)

    actions = []
    warmup = num_chunks[0] * v
    fwd, bwd = 1 - p, 0
    # While there are still backward slices to process on the first rank
    while bwd < len(forwards) + p - 1:
        if fwd < warmup:
            op = Op.FORWARD
        elif fwd == len(forwards):
            op = Op.BACKWARD
        elif fwd - bwd == warmup:
            op = Op.BACKWARD
        else:
            op = Op.FORWARD

        ops = []
        if op == Op.FORWARD:
            for rank in range(p):
                fwd_idx = fwd + rank
                ops.append(
                    forwards[fwd_idx]
                    if fwd_idx >= 0 and fwd_idx < len(forwards)
                    else None
                )
            fwd += 1
        else:
            for rank in range(p):
                bwd_idx = bwd - rank
                ops.append(
                    backwards[bwd_idx]
                    if bwd_idx >= 0 and bwd_idx < len(backwards)
                    else None
                )
            bwd += 1
        actions.append(ops[::-1])

    actions_by_rank = [list(row) for row in zip(*actions)]
    return actions_by_rank


def build_splitfuse_schedule(p, chunks_list: list[list[int]]) -> list[list[Action]]:
    forwards = []
    for batch_id, chunks in enumerate(chunks_list):
        for chunk_id, size in enumerate(chunks):
            chunk = Chunk(
                batch_id=batch_id,
                chunk_id=chunk_id,
                length=size,
            )
            action = Action(
                stage_id=0,
                op=Op.FORWARD,
                chunk=chunk,
            )
            forwards.append(action)

    backwards = []
    for batch_id, chunks in enumerate(chunks_list):
        for chunk_id, size in reversed(list(enumerate(chunks))):
            chunk = Chunk(
                batch_id=batch_id,
                chunk_id=chunk_id,
                length=size,
            )
            action = Action(
                stage_id=0,
                op=Op.BACKWARD,
                chunk=chunk,
            )
            backwards.append(action)

    actions = []
    warmup = len(chunks_list[0])
    fwd, bwd = 1 - p, 0
    # While there are still backward slices to process on the first rank
    while bwd < len(forwards) + p - 1:
        if fwd < warmup:
            op = Op.FORWARD
        elif fwd == len(forwards):
            op = Op.BACKWARD
        elif fwd - bwd == warmup:
            op = Op.BACKWARD
        else:
            op = Op.FORWARD

        ops = []
        if op == Op.FORWARD:
            for rank in range(p):
                fwd_idx = fwd + rank
                if fwd_idx >= 0 and fwd_idx < len(forwards):
                    action = forwards[fwd_idx]
                    ops.append(
                        Action(action.stage_id + p - 1 - rank, action.op, action.chunk)
                    )
                else:
                    ops.append(None)
            fwd += 1
        else:
            for rank in range(p):
                bwd_idx = bwd - rank
                if bwd_idx >= 0 and bwd_idx < len(backwards):
                    action = backwards[bwd_idx]
                    ops.append(
                        Action(action.stage_id + p - 1 - rank, action.op, action.chunk)
                    )
                else:
                    ops.append(None)
            bwd += 1
        actions.append(ops[::-1])

    return [list(row) for row in zip(*actions)]


def build_hybrid_schedule(
    p: int,
    k: int,
    fwd_switch: tuple[int, int],
    bwd_switch: tuple[int, int],
    chunks_list: list[list[int]],
) -> list[list[Action]]:
    forwards = []
    for microbatch_id, chunks in enumerate(chunks_list):
        for chunk_id, size in enumerate(chunks):
            chunk = Chunk(
                batch_id=microbatch_id,
                chunk_id=chunk_id,
                length=size,
            )
            action = Action(
                stage_id=0,
                op=Op.FORWARD,
                chunk=chunk,
            )
            forwards.append(action)

    backwards = []
    for microbatch_id, chunks in enumerate(chunks_list):
        for chunk_id, size in reversed(list(enumerate(chunks))):
            chunk = Chunk(
                batch_id=microbatch_id,
                chunk_id=chunk_id,
                length=size,
            )
            action = Action(
                stage_id=0,
                op=Op.BACKWARD,
                chunk=chunk,
            )
            backwards.append(action)

    num_chunks = [len(chunks) for chunks in chunks_list]
    actions_by_rank: list[list[Action]] = []

    for rank in range(p):
        fwd_switched, bwd_switched = False, False
        actions = []
        slimpipe_warmup = num_chunks[0] + 2 * (p - 1 - rank)
        kfkb_warmup = k * (p - rank)
        fwd, bwd = 0, 0

        # Warmup phase
        counter = 0
        while counter < (
            kfkb_warmup if fwd_switched else slimpipe_warmup
        ) and fwd < len(forwards):
            chunk = forwards[fwd].chunk
            data_id = (chunk.batch_id, chunk.chunk_id)
            if not fwd_switched and data_id == fwd_switch:
                fwd_switched = True
                continue
            cnt = k if fwd_switched else 1
            for _ in range(cnt):
                if fwd >= len(forwards):
                    break
                action = forwards[fwd]
                actions.append(Action(rank, action.op, action.chunk))
                fwd += 1
            counter += 1

        # Steady state phase
        while fwd < len(forwards):
            # Backward
            if not bwd_switched:
                chunk = backwards[bwd].chunk
                data_id = (chunk.batch_id, chunk.chunk_id)
                if data_id == bwd_switch:
                    bwd_switched = True
            cnt = k if bwd_switched else 1
            for _ in range(cnt):
                if bwd >= len(backwards):
                    break
                action = backwards[bwd]
                actions.append(Action(rank, action.op, action.chunk))
                bwd += 1

            # Forward
            if not fwd_switched:
                chunk = forwards[fwd].chunk
                data_id = (chunk.batch_id, chunk.chunk_id)
                if data_id == fwd_switch:
                    fwd_switched = True
            cnt = k if fwd_switched else 1
            for _ in range(cnt):
                if fwd >= len(forwards):
                    break
                action = forwards[fwd]
                actions.append(Action(rank, action.op, action.chunk))
                fwd += 1

        # Cooldown phase
        while bwd < len(backwards):
            action = backwards[bwd]
            actions.append(Action(rank, action.op, action.chunk))
            bwd += 1

        actions_by_rank.append(actions)

    return actions_by_rank


def build_1f1b_schedule(p, chunks: list[int]) -> list[list[Action]]:
    forwards = []
    for batch_id, size in enumerate(chunks):
        chunk = Chunk(
            batch_id=batch_id,
            chunk_id=0,
            length=size,
        )
        action = Action(
            stage_id=0,
            op=Op.FORWARD,
            chunk=chunk,
        )
        forwards.append(action)

    backwards = []
    for batch_id, size in enumerate(chunks):
        chunk = Chunk(
            batch_id=batch_id,
            chunk_id=0,
            length=size,
        )
        action = Action(
            stage_id=0,
            op=Op.BACKWARD,
            chunk=chunk,
        )
        backwards.append(action)

    actions_by_rank: list[list[Action]] = []

    for rank in range(p):
        actions: list[Action] = []
        warmup = min(len(chunks), p - rank)
        fwd, bwd = 0, 0

        while fwd < warmup:
            action = forwards[fwd]
            actions.append(Action(rank, action.op, action.chunk))
            fwd += 1

        while fwd < len(forwards):
            action = backwards[bwd]
            actions.append(Action(rank, action.op, action.chunk))
            bwd += 1

            action = forwards[fwd]
            actions.append(Action(rank, action.op, action.chunk))
            fwd += 1

        while bwd < len(backwards):
            action = backwards[bwd]
            actions.append(Action(rank, action.op, action.chunk))
            bwd += 1

        actions_by_rank.append(actions)

    return actions_by_rank


if __name__ == "__main__":
    # Example usage
    p = 4  # Number of ranks
    num_chunks = [8, 6, 4, 1]  # Number of slices for each batch

    chunks_list = []
    for num in num_chunks:
        chunks_list.append([1] * num)

    actions_by_rank = build_splitfuse_schedule(p, chunks_list)
    pp.pprint(actions_by_rank)
    plot_timeline(actions_by_rank)
