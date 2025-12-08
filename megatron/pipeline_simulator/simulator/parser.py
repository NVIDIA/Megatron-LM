from copy import deepcopy
from itertools import zip_longest
from typing import Mapping

from .ir import Action
from .schedules import InterleavedSchedule, SlimPipeSchedule


class Parser:
    @staticmethod
    def print(
        pipeline_order: Mapping[int, list[Action | None]],
        error_step_number: int | None = None,
    ) -> str:
        """
        Formats the pipeline order in a timestep (row) x rank (column) grid of actions.

        Args:
            pipeline_order: Dictionary mapping ranks to their action sequences
            error_step_number: Optional step number to highlight with an error marker

        Returns:
            Formatted string representation of the pipeline schedule
        """
        # Create a deep copy to avoid mutating the original
        pipeline_order = deepcopy(pipeline_order)

        # Replace None values with empty strings
        for rank_actions in pipeline_order.values():
            for i, action in enumerate(rank_actions):
                if action is None:
                    rank_actions[i] = ""  # type: ignore

        # Calculate dimensions and labels
        num_steps = max(len(actions) for actions in pipeline_order.values())
        num_ranks = len(pipeline_order)

        step_labels = [
            f"Step {i:0{len(str(num_steps - 1))}d}" for i in range(num_steps)
        ]
        rank_labels = [f"Rank {i}" for i in range(num_ranks)]

        # Get actions for each rank in sorted order
        rank_actions = [
            pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
        ]

        # Transpose to get actions by step instead of by rank
        transposed_actions = list(zip_longest(*rank_actions, fillvalue=""))

        # Calculate column widths for alignment
        max_lengths = [
            max(
                len(str(item))
                for item in [rank_labels[i], *[row[i] for row in transposed_actions]]
            )
            for i in range(num_ranks)
        ]

        # Format the header row
        label_width = len(step_labels[0])
        header_row = " " * (label_width + 2) + " ".join(
            f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
        )

        # Format each row with proper alignment
        formatted_rows = []
        for step_num, (label, actions) in enumerate(
            zip(step_labels, transposed_actions)
        ):
            row = f"{label}: " + " ".join(
                f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(actions)
            )

            # Add error indicator if needed
            if error_step_number is not None and step_num == error_step_number:
                row += " <-- ERROR HERE"

            formatted_rows.append(row)

        # Join all rows into the final table
        return header_row + "\n" + "\n".join(formatted_rows) + "\n"


def test_interleaved():
    p = 4
    v = 2
    chunks_list = [1] * 8
    schedule = InterleavedSchedule(p, v, chunks_list)
    order_deps = schedule.order_deps()
    pipeline_order = {rank: actions for rank, actions in enumerate(order_deps)}
    print(Parser.print(pipeline_order))


def test_slimpipe():
    p = 4
    v = 2
    chunks_list = [[0, 1, 2, 3], [4, 5, 6, 7]]
    schedule = SlimPipeSchedule(p, v, chunks_list)
    order_deps = schedule.order_deps()
    pipeline_order = {rank: actions for rank, actions in enumerate(order_deps)}
    print(Parser.print(pipeline_order))


if __name__ == "__main__":
    test_interleaved()
    test_slimpipe()
