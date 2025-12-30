from typing import Mapping

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colormaps

try:
    from .ir import Action, B, F, Stats
    from .schedules import AbstractSchedule
except:
    from ir import Action, B, F, Stats
    from schedules import AbstractSchedule

class Plotter:
    def __init__(self, schedule: AbstractSchedule) -> None:
        self.schedule = schedule

    def draw_timeline(self, timeline: Mapping[Action, Stats]) -> None:
        """Draw a timeline visualization of the pipeline execution."""
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

        pp = self.schedule.pipeline_parallelism()
        stage_map = self.schedule.stage_map()
        width = max(timeline[action].end_time for action in timeline)

        # Get distinct colormaps for forward and backward passes
        forward_cmap = colormaps["Blues"]
        backward_cmap = colormaps["Greens"]

        # Calculate number of colors needed based on virtual stages
        num_stages = max(stage_id // pp for stage_id in stage_map.keys()) + 1

        # Generate color lists with brightness range from 0.4 to 0.8
        blues = [
            forward_cmap(0.3 + 0.3 * i / max(1, num_stages - 1))
            for i in range(num_stages)
        ]
        greens = [
            backward_cmap(0.3 + 0.3 * i / max(1, num_stages - 1))
            for i in range(num_stages)
        ]

        for action, stats in timeline.items():
            stage_id, action_type, data_id = action
            start_time, end_time = stats
            cost = end_time - start_time

            virtual_stage_id = stage_id // pp
            if action_type == F:
                color = blues[virtual_stage_id % len(blues)]
            elif action_type == B:
                color = greens[virtual_stage_id % len(greens)]
            else:
                raise ValueError(f"Invalid action: {action}")

            rank = stage_map[stage_id]

            # Add rectangle for action
            ax.add_patch(
                patches.Rectangle(
                    (start_time / width, rank / pp),
                    cost / width,
                    1 / pp,
                    facecolor=color,
                    edgecolor="black",
                )
            )

            # Add label
            if action_type in (F, B):
                label = (
                    "\n".join(map(str, data_id))
                    if isinstance(data_id, tuple)
                    else str(data_id)
                )
                ax.text(
                    (start_time + 0.5 * cost) / width,
                    (rank + 0.5) / pp,
                    label,
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="center",
                )

        ax.invert_yaxis()
        plt.axis("off")
        plt.savefig("/m2v_model/wuguohao03/nv_teamwork/Megatron-LM/megatron/pipeline_simulator/simulator/pipeline.png")
