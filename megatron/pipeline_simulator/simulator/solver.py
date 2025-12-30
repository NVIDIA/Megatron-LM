from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx

try:
    from .ir import Action, Stats
    from .plotter import Plotter
    from .schedules import (
        AbstractSchedule,
        InterleavedSchedule,
        SlimPipeSchedule,
        SplitFuseSchedule,
        kFkBSchedule,
        HybridSlimSchedule,
    )
except:
    from ir import Action, Stats
    from plotter import Plotter
    from schedules import (
        AbstractSchedule,
        InterleavedSchedule,
        SlimPipeSchedule,
        SplitFuseSchedule,
        kFkBSchedule,
        HybridSlimSchedule,
    )


class Solver:
    def __init__(self):
        self.G = nx.DiGraph()
        self.sorted_actions = []

    def add_deps(self, actions: Iterable[Action]) -> None:
        """
        Add dependencies on actions. The actions will be executed in the order provided.
        """
        nx.add_path(self.G, actions)

    def add_costs(self, costs: Mapping[Action, float]) -> None:
        """
        Add costs to the actions in the graph.
        """
        nx.set_node_attributes(self.G, costs, "cost")

    def sort(self) -> list[Action]:
        """
        Sort the actions in the graph based on their dependencies.
        """
        self.sorted_actions = list(nx.topological_sort(self.G))
        return self.sorted_actions

    def get_sources(self) -> list[Action]:
        """
        Get the source actions in the graph (actions with no prerequisites).
        """
        return [node for node, in_degree in self.G.in_degree() if in_degree == 0]

    def solve(self) -> dict[Action, Stats]:
        """
        Solves the pipeline scheduling problem and returns the makespan.

        Sets start times of source nodes to 0, then computes end times based on costs.
        Updates successor nodes to respect precedence constraints.

        Returns:
            The makespan of the schedule (maximum end time of any node).
        """
        for node in self.get_sources():
            self.G.nodes[node]["start_time"] = 0

        for u in nx.topological_sort(self.G):
            start_time = self.G.nodes[u]["start_time"]
            cost = self.G.nodes[u]["cost"]
            end_time = start_time + cost
            self.G.nodes[u]["end_time"] = end_time

            for v in self.G.successors(u):
                v_node = self.G.nodes[v]
                if "start_time" not in v_node or v_node["start_time"] < end_time:
                    v_node["start_time"] = end_time

        timeline = {}
        for node in self.G.nodes:
            start_time = self.G.nodes[node]["start_time"]
            end_time = self.G.nodes[node]["end_time"]
            timeline[node] = Stats(start_time, end_time)

        return timeline

    def show(self) -> None:
        """Draw the dependency graph using networkx."""
        options = {
            "font_size": 8,
            "node_size": 16,
            "linewidths": 1,
        }
        nx.draw_networkx(
            self.G, with_labels=True, pos=nx.spring_layout(self.G), **options
        )
        plt.axis("off")
        plt.show()


def test_with_schedule(schedule: AbstractSchedule) -> None:
    data_deps = schedule.data_deps()
    order_deps = schedule.order_deps()
    cost_map = schedule.cost_map()

    # Set up and run solver
    solver = Solver()
    for actions in order_deps:
        # Filter out None values and add dependencies for remaining actions
        solver.add_deps(filter(None, actions))

    for actions in data_deps:
        solver.add_deps(actions)

    solver.add_costs(cost_map)
    timeline = solver.solve()
    # solver.show()

    # plotter = Plotter(schedule)
    # plotter.draw_timeline(timeline)

    # print(f"The makespan is {max(stats.end_time for stats in timeline.values())}")
    return max(stats.end_time for stats in timeline.values())


def test_interleaved() -> None:
    """Test the solver with an interleaved schedule."""
    p = 4  # Number of pipeline stages
    v = 2  # Number of microbatches
    microbatches_1 = [2, 6, 8, 4, 3, 4, 4, 2]  # Cost of each microbatch
    microbatches_2 = [4, 8, 5, 4, 10, 3, 3, 3]  # Cost of each microbatch
    # microbatches = [(microbatches_1[i] + microbatches_2[i])/2 for i in range(len(microbatches_1))]
    
    microbatches_balanced = (sum(microbatches_1) + sum(microbatches_2)) // (len(microbatches_1) + len(microbatches_2))
    microbatches = [microbatches_balanced] * len(microbatches_1)
    
    # microbatches = microbatches_2
    
    # microbatches = [3, 1, 2, 4, 8/2, 7/2, 4, 2]  # Cost of each microbatch
    # microbatches = microbatches.sort()

    # Build the schedule and dependencies
    schedule = InterleavedSchedule(p, v, microbatches)
    test_with_schedule(schedule)


def test_slimpipe():
    """Test the solver with a slimpipe schedule."""
    p = 4  # Number of pipeline stages
    v = 2  # Number of microbatches
    chunks_list = [[1] * 8, [1] * 4]  # Cost of each chunk

    # Build the schedule and dependencies
    schedule = SlimPipeSchedule(p, v, chunks_list)
    test_with_schedule(schedule)


def test_splitfuse():
    """Test the solver with a slimpipe schedule."""
    p = 4  # Number of pipeline stages
    fwd_costs = [[1] * 8, [2] * 7, [1.5] * 4, [1] * 3]  # Cost of each chunk
    bwd_costs = [[2] * 8, [2.5] * 7, [3] * 4, [1.8] * 3]  # Cost of each chunk

    # Build the schedule and dependencies
    schedule = SplitFuseSchedule(p, fwd_costs, bwd_costs)
    test_with_schedule(schedule)


def test_kfkb():
    """Test the solver with a kFkB schedule."""
    p = 4  # Number of pipeline stages
    k = 2  # Number for kFkB
    fwd_costs = [1] * 12  # Cost of each chunk

    # Build the schedule and dependencies
    schedule = kFkBSchedule(p, k, fwd_costs)
    test_with_schedule(schedule)


def test_hybrid():
    """Test the solver with a hybrid slimpipe schedule."""
    p = 4  # Number of pipeline stages
    k = 2  # Number for kFkB
    fwd_costs = [[1, 1] for i in reversed(range(40, 56, 2))]
    bwd_costs = [[2, 2] for i in reversed(range(40, 56, 2))]

    # Build the schedule and dependencies
    schedule = HybridSlimSchedule(
        p,
        k,
        fwd_switch=(3, 0),
        bwd_switch=(3, 1),
        fwd_costs=fwd_costs,
        bwd_costs=bwd_costs,
    )
    test_with_schedule(schedule)


if __name__ == "__main__":
    test_interleaved()
    # test_slimpipe()
    # test_splitfuse()
    # test_kfkb()
    # test_hybrid()
