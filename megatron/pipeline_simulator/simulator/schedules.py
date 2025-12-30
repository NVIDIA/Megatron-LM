from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Mapping

try:
    from .ir import Action, B, F
except:
    from ir import Action, B, F


class AbstractSchedule(ABC):
    @abstractmethod
    def pipeline_parallelism(self) -> int:
        """
        Returns the number of ranks in the schedule.
        """
        pass

    @abstractmethod
    def data_deps(self) -> list[list[Action]]:
        """
        Returns the data dependencies for the schedule.
        """
        pass

    @abstractmethod
    def order_deps(self) -> list[list[Action]]:
        """
        Returns the order dependencies for the schedule.
        """
        pass

    @abstractmethod
    def cost_map(self) -> Mapping[Action, int | float]:
        """
        Returns the costs associated with each action in the schedule.
        """
        pass

    @abstractmethod
    def stage_map(self) -> Mapping[int, int]:
        """
        Returns the mapping of stages to ranks.
        """
        pass


class InterleavedSchedule(AbstractSchedule):
    def __init__(self, p: int, v: int, microbatches: list[int] | list[float], backward_microbatches=None) -> None:
        self.p = p
        self.v = v
        self.microbatches = microbatches
        self.backward_microbatches = None
        if backward_microbatches is not None:
            self.backward_microbatches = backward_microbatches

        num_microbatches = len(microbatches)
        if num_microbatches % self.p != 0 or num_microbatches // self.p <= 1:
            raise ValueError(
                "Number of microbatches must be divisible by the number of ranks and twice the number of ranks."
            )

    def pipeline_parallelism(self) -> int:
        return self.p

    def data_deps(self) -> list[list[Action]]:
        num_stages = self.p * self.v
        deps = []

        # Latter stages depend on earlier stages
        for microbatch_id in range(len(self.microbatches)):
            dep = []
            for stage_id in range(num_stages):
                action = Action(
                    stage_id=stage_id,
                    action_type=F,
                    data_id=microbatch_id,
                )
                dep.append(action)
            for stage_id in reversed(range(num_stages)):
                action = Action(
                    stage_id=stage_id,
                    action_type=B,
                    data_id=microbatch_id,
                )
                dep.append(action)
            deps.append(dep)

        return deps

    def order_deps(self) -> list[list[Action]]:
        m, p, v = len(self.microbatches), self.p, self.v

        forwards = []
        for start_microbatch_id in range(0, m, p):
            for virtual_stage_id in range(v):
                for microbatch_id in range(
                    start_microbatch_id, start_microbatch_id + p
                ):
                    action = Action(
                        stage_id=virtual_stage_id * p,
                        action_type=F,
                        data_id=microbatch_id,
                    )
                    forwards.append(action)

        backwards = []
        for start_microbatch_id in range(0, m, p):
            for virtual_stage_id in range(v - 1, -1, -1):
                for microbatch_id in range(
                    start_microbatch_id, start_microbatch_id + p
                ):
                    action = Action(
                        stage_id=virtual_stage_id * p,
                        action_type=B,
                        data_id=microbatch_id,
                    )
                    backwards.append(action)

        actions_by_rank = []
        for rank in range(p):
            actions = []
            warmup = v * p + p - 1 - 2 * rank
            fwd, bwd = 0, 0
            while bwd < len(backwards):
                if fwd < warmup:
                    op = F
                elif fwd == len(forwards):
                    op = B
                elif fwd - bwd == warmup:
                    op = B
                else:
                    op = F

                action = forwards[fwd] if op == F else backwards[bwd]
                if op == F:
                    fwd += 1
                else:
                    bwd += 1
                stage_id, action_type, data_id = action
                actions.append(Action(stage_id + rank, action_type, data_id))
            actions_by_rank.append(actions)

        return actions_by_rank

    def cost_map(self) -> Mapping[Action, int | float]:
        costs = {}
        for microbatch_id, cost in enumerate(self.microbatches):
            if self.backward_microbatches is not None:
                bwd_cost = self.backward_microbatches[microbatch_id]
            else:
                bwd_cost = 2 * cost
            for stage_id in range(self.p * self.v):
                fwd = Action(
                    stage_id=stage_id,
                    action_type=F,
                    data_id=microbatch_id,
                )
                bwd = Action(
                    stage_id=stage_id,
                    action_type=B,
                    data_id=microbatch_id,
                )
                costs[fwd] = cost
                costs[bwd] = bwd_cost
        return costs

    def stage_map(self) -> Mapping[int, int]:
        return {i: i % self.p for i in range(self.p * self.v)}


class SlimPipeSchedule(AbstractSchedule):
    def __init__(
        self, p: int, v: int, chunks_list: list[list[int]] | list[list[float]]
    ) -> None:
        self.p = p
        self.v = v
        self.chunks_list = chunks_list

        num_chunks = [len(chunks) for chunks in self.chunks_list]
        if any(n == 0 for n in num_chunks):
            raise ValueError("Number of slices must be greater than 0.")
        if any(n % self.p != 0 for n in num_chunks):
            raise ValueError(
                "Number of slices must be divisible by the number of ranks."
            )
        if any(prev < curr for prev, curr in pairwise(num_chunks)):
            raise ValueError("Number of slices must be non-increasing.")

    def pipeline_parallelism(self) -> int:
        return self.p

    def data_deps(self) -> list[list[Action]]:
        num_stages = self.p * self.v
        deps = []

        # Latter stages depend on earlier stages
        for microbatch_id, chunks in enumerate(self.chunks_list):
            for chunk_id in range(len(chunks)):
                dep = []
                for stage_id in range(num_stages):
                    action = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                for stage_id in reversed(range(num_stages)):
                    action = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                deps.append(dep)

        # Latter chunks depend on earlier chunks
        for stage_id in range(num_stages):
            for microbatch_id, chunks in enumerate(self.chunks_list):
                dep = []
                for chunk_id in range(len(chunks)):
                    action = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                for chunk_id in reversed(range(len(chunks))):
                    action = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                deps.append(dep)

        return deps

    def order_deps(self) -> list[list[Action]]:
        p, v = self.p, self.v

        forwards = []
        for microbatch_id, chunks in enumerate(self.chunks_list):
            n = len(chunks)
            for start_chunk_id in range(0, n, p):
                for virtual_stage_id in range(v):
                    for chunk_id in range(start_chunk_id, start_chunk_id + p):
                        action = Action(
                            stage_id=virtual_stage_id * p,
                            action_type=F,
                            data_id=(microbatch_id, chunk_id),
                        )
                        forwards.append(action)

        backwards = []
        for microbatch_id, chunks in enumerate(self.chunks_list):
            n = len(chunks)
            for start_chunk_id in range(n - 1, -1, -p):
                for virtual_stage_id in range(v - 1, -1, -1):
                    for chunk_id in range(start_chunk_id, start_chunk_id - p, -1):
                        action = Action(
                            stage_id=virtual_stage_id * p,
                            action_type=B,
                            data_id=(microbatch_id, chunk_id),
                        )
                        backwards.append(action)

        actions = []
        warmup = len(self.chunks_list[0]) * v
        fwd, bwd = 1 - p, 0
        # While there are still backward slices to process on the first rank
        while bwd < len(forwards) + p - 1:
            if fwd < warmup:
                op = F
            elif fwd == len(forwards):
                op = B
            elif fwd - bwd == warmup:
                op = B
            else:
                op = F

            ops = []
            if op == F:
                for rank in range(p):
                    fwd_idx = fwd + rank
                    if fwd_idx >= 0 and fwd_idx < len(forwards):
                        stage_id, action_type, data_id = forwards[fwd_idx]
                        ops.append(
                            Action(stage_id + p - 1 - rank, action_type, data_id)
                        )
                    else:
                        ops.append(None)
                fwd += 1
            else:
                for rank in range(p):
                    bwd_idx = bwd - rank
                    if bwd_idx >= 0 and bwd_idx < len(backwards):
                        stage_id, action_type, data_id = backwards[bwd_idx]
                        ops.append(
                            Action(stage_id + p - 1 - rank, action_type, data_id)
                        )
                    else:
                        ops.append(None)
                bwd += 1
            actions.append(ops[::-1])

        return [list(row) for row in zip(*actions)]

    def cost_map(self) -> Mapping[Action, int | float]:
        costs = {}
        for microbatch_id, chunks in enumerate(self.chunks_list):
            for chunk_id, cost in enumerate(chunks):
                for stage_id in range(self.p * self.v):
                    fwd = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    bwd = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    costs[fwd] = cost
                    costs[bwd] = 2 * cost
        return costs

    def stage_map(self) -> Mapping[int, int]:
        return {i: i % self.p for i in range(self.p * self.v)}


class SplitFuseSchedule(AbstractSchedule):
    def __init__(
        self,
        p: int,
        fwd_costs: list[list[float]],
        bwd_costs: list[list[float]],
    ) -> None:
        self.p = p
        self.fwd_costs = fwd_costs
        self.bwd_costs = bwd_costs

        if len(fwd_costs) != len(bwd_costs):
            raise ValueError("Number of microbatches must be the same for fwd and bwd.")
        for fwds, bwds in zip(fwd_costs, bwd_costs):
            if len(fwds) != len(bwds):
                raise ValueError(
                    "Number of slices must be the same for the same microbatch."
                )

        num_chunks = [len(chunks) for chunks in self.fwd_costs]
        if any(n == 0 for n in num_chunks):
            raise ValueError("Number of slices must be greater than 0.")
        if any(prev < curr for prev, curr in pairwise(num_chunks)):
            raise ValueError("Number of slices must be non-increasing.")

    def pipeline_parallelism(self) -> int:
        return self.p

    def data_deps(self) -> list[list[Action]]:
        num_stages = self.p
        deps = []

        # Latter stages depend on earlier stages
        for microbatch_id, chunks in enumerate(self.fwd_costs):
            for chunk_id in range(len(chunks)):
                dep = []
                for stage_id in range(num_stages):
                    action = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                for stage_id in reversed(range(num_stages)):
                    action = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                deps.append(dep)

        # Latter chunks depend on earlier chunks
        for stage_id in range(num_stages):
            for microbatch_id, chunks in enumerate(self.fwd_costs):
                dep = []
                for chunk_id in range(len(chunks)):
                    action = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                for chunk_id in reversed(range(len(chunks))):
                    action = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    dep.append(action)
                deps.append(dep)

        return deps

    def order_deps(self) -> list[list[Action]]:
        p = self.p

        forwards = []
        for microbatch_id, chunks in enumerate(self.fwd_costs):
            for chunk_id, chunk in enumerate(chunks):
                action = Action(
                    stage_id=0,
                    action_type=F,
                    data_id=(microbatch_id, chunk_id),
                )
                forwards.append(action)

        backwards = []
        for microbatch_id, chunks in enumerate(self.fwd_costs):
            for chunk_id, chunk in reversed(list(enumerate(chunks))):
                action = Action(
                    stage_id=0,
                    action_type=B,
                    data_id=(microbatch_id, chunk_id),
                )
                backwards.append(action)

        actions = []
        warmup = len(self.fwd_costs[0])
        fwd, bwd = 1 - p, 0
        # While there are still backward slices to process on the first rank
        while bwd < len(forwards) + p - 1:
            if fwd < warmup:
                op = F
            elif fwd == len(forwards):
                op = B
            elif fwd - bwd == warmup:
                op = B
            else:
                op = F

            ops = []
            if op == F:
                for rank in range(p):
                    fwd_idx = fwd + rank
                    if fwd_idx >= 0 and fwd_idx < len(forwards):
                        stage_id, action_type, data_id = forwards[fwd_idx]
                        ops.append(
                            Action(stage_id + p - 1 - rank, action_type, data_id)
                        )
                    else:
                        ops.append(None)
                fwd += 1
            else:
                for rank in range(p):
                    bwd_idx = bwd - rank
                    if bwd_idx >= 0 and bwd_idx < len(backwards):
                        stage_id, action_type, data_id = backwards[bwd_idx]
                        ops.append(
                            Action(stage_id + p - 1 - rank, action_type, data_id)
                        )
                    else:
                        ops.append(None)
                bwd += 1
            actions.append(ops[::-1])

        actions_by_rank = [list(row) for row in zip(*actions)]

        bwd_fwds = []
        for rank, actions in enumerate(reversed(actions_by_rank)):
            i = 0
            while i < len(actions) and (not actions[i] or actions[i].action_type == F):
                i += 1
            while i < len(actions) and (not actions[i] or actions[i].action_type == B):
                bwd_fwds.append(actions[i])
                i += 1
            while i < len(actions) and (not actions[i] or actions[i].action_type == F):
                bwd_fwds.append(actions[i])
                i += 1

        actions_by_rank.append(bwd_fwds)

        return actions_by_rank

    def cost_map(self) -> Mapping[Action, int | float]:
        costs = {}
        for microbatch_id, (fwd_chunks, bwd_chunks) in enumerate(
            zip(self.fwd_costs, self.bwd_costs)
        ):
            for chunk_id, (fwd_cost, bwd_cost) in enumerate(
                zip(fwd_chunks, bwd_chunks)
            ):
                for stage_id in range(self.p):
                    fwd = Action(
                        stage_id=stage_id,
                        action_type=F,
                        data_id=(microbatch_id, chunk_id),
                    )
                    bwd = Action(
                        stage_id=stage_id,
                        action_type=B,
                        data_id=(microbatch_id, chunk_id),
                    )
                    costs[fwd] = fwd_cost
                    costs[bwd] = bwd_cost
        return costs

    def stage_map(self) -> Mapping[int, int]:
        return {i: i for i in range(self.p)}


class kFkBSchedule(AbstractSchedule):
    def __init__(self, p: int, k: int, microbatches: list[int] | list[float], backward_microbatches=None) -> None:
        self.p = p
        self.k = k
        self.microbatches = microbatches
        self.backward_microbatches = None
        if backward_microbatches is not None:
            self.backward_microbatches = backward_microbatches

    def pipeline_parallelism(self) -> int:
        return self.p

    def data_deps(self) -> list[list[Action]]:
        num_stages = self.p
        deps = []

        # Latter stages depend on earlier stages
        for microbatch_id in range(len(self.microbatches)):
            dep = []
            for stage_id in range(num_stages):
                action = Action(
                    stage_id=stage_id,
                    action_type=F,
                    data_id=microbatch_id,
                )
                dep.append(action)
            for stage_id in reversed(range(num_stages)):
                action = Action(
                    stage_id=stage_id,
                    action_type=B,
                    data_id=microbatch_id,
                )
                dep.append(action)
            deps.append(dep)

        return deps

    def order_deps(self) -> list[list[Action]]:
        p, k = self.p, self.k

        forwards = []
        for microbatch_id in range(len(self.microbatches)):
            action = Action(
                stage_id=0,
                action_type=F,
                data_id=microbatch_id,
            )
            forwards.append(action)

        backwards = []
        for microbatch_id in range(len(self.microbatches)):
            action = Action(
                stage_id=0,
                action_type=B,
                data_id=microbatch_id,
            )
            backwards.append(action)

        actions_by_rank: list[list[Action]] = []
        for rank in range(p):
            actions = []
            warmup = min(k * (p - rank), len(forwards))
            fwd, bwd = 0, 0
            while bwd < len(backwards):
                if fwd < warmup:
                    op = F
                elif fwd == len(forwards):
                    op = B
                elif fwd - bwd >= warmup:
                    op = B
                else:
                    op = F

                for i in range(k):
                    action = forwards[fwd] if op == F else backwards[bwd]
                    stage_id, action_type, data_id = action
                    actions.append(Action(stage_id + rank, action_type, data_id))
                    if op == F:
                        fwd += 1
                    else:
                        bwd += 1
                    if fwd >= len(forwards) or bwd >= len(backwards):
                        break
            actions_by_rank.append(actions)

        return actions_by_rank

    def cost_map(self) -> Mapping[Action, int | float]:
        costs = {}

        microbatch_id_list = list(range(len(self.microbatches)))
        for idx, microbatch_id in enumerate(microbatch_id_list):
            cost = self.microbatches[idx]
            if self.backward_microbatches:
                bwd_cost = self.backward_microbatches[idx]
            else:
                bwd_cost = 2 * cost
            for stage_id in range(self.p):
                fwd = Action(
                    stage_id=stage_id,
                    action_type=F,
                    data_id=microbatch_id,
                )
                bwd = Action(
                    stage_id=stage_id,
                    action_type=B,
                    data_id=microbatch_id,
                )
                costs[fwd] = cost
                costs[bwd] = bwd_cost

        return costs

    def stage_map(self) -> Mapping[int, int]:
        return {i: i for i in range(self.p)}


class HybridSlimSchedule(SplitFuseSchedule):
    def __init__(
        self,
        p: int,
        k: int,
        fwd_switch: tuple[int, int],
        bwd_switch: tuple[int, int],
        fwd_costs: list[list[float]],
        bwd_costs: list[list[float]],
    ) -> None:
        """
        Initializes the HybridSlimSchedule with the given parameters.

        Args:
            p (int): Number of pipeline stages.
            k (int): Number of k as in kFkB.
            fwd_switch (tuple[int, int]): Starting index of kFkB forward.
            bwd_switch (tuple[int, int]): Starting index of kFkB backward.
            fwd_costs (list[list[float]]): Forward costs for each microbatch and chunk.
            bwd_costs (list[list[float]]): Backward costs for each microbatch and chunk.
        """
        super().__init__(p, fwd_costs, bwd_costs)
        self.k = k
        self.fwd_switch = fwd_switch
        self.bwd_switch = bwd_switch

        num_chunks = [len(chunks) for chunks in self.fwd_costs]
        if any(n == 0 for n in num_chunks):
            raise ValueError("Number of slices must be greater than 0.")
        if any(prev < curr for prev, curr in pairwise(num_chunks)):
            raise ValueError("Number of slices must be non-increasing.")

        if self.k <= 1:
            raise ValueError("k for kFkB schedule must be greater than 1.")
        if self.k < num_chunks[0]:
            raise ValueError(
                "k for kFkB schedule must be greater than or equal to the maximal number of chunks."
            )

        microbatch_id, chunk_id = fwd_switch
        if (
            microbatch_id < 0
            or microbatch_id >= len(fwd_costs)
            or chunk_id < 0
            or chunk_id >= len(fwd_costs[microbatch_id])
        ):
            raise ValueError("Invalid fwd_switch indices.")

        microbatch_id, chunk_id = bwd_switch
        if (
            microbatch_id < 0
            or microbatch_id >= len(bwd_costs)
            or chunk_id < 0
            or chunk_id >= len(bwd_costs[microbatch_id])
        ):
            raise ValueError("Invalid bwd_switch indices.")

    def order_deps(self) -> list[list[Action]]:
        p, k = self.p, self.k

        forwards = []
        for microbatch_id, chunks in enumerate(self.fwd_costs):
            for chunk_id in range(len(chunks)):
                action = Action(
                    stage_id=0,
                    action_type=F,
                    data_id=(microbatch_id, chunk_id),
                )
                forwards.append(action)

        backwards = []
        for microbatch_id, chunks in enumerate(self.fwd_costs):
            for chunk_id in reversed(range(len(chunks))):
                action = Action(
                    stage_id=0,
                    action_type=B,
                    data_id=(microbatch_id, chunk_id),
                )
                backwards.append(action)

        num_chunks = [len(chunks) for chunks in self.fwd_costs]
        total_chunks = sum(num_chunks)
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
                action = forwards[fwd]
                stage_id, action_type, data_id = action
                if not fwd_switched and data_id == self.fwd_switch:
                    fwd_switched = True
                    continue
                cnt = k if fwd_switched else 1
                for _ in range(cnt):
                    if fwd >= len(forwards):
                        break
                    action = forwards[fwd]
                    stage_id, action_type, data_id = action
                    actions.append(Action(stage_id + rank, action_type, data_id))
                    fwd += 1
                counter += 1

            # Steady state phase
            while fwd < len(forwards):
                # Backward
                if not bwd_switched:
                    stage_id, action_type, data_id = backwards[bwd]
                    if data_id == self.bwd_switch:
                        bwd_switched = True
                cnt = k if bwd_switched else 1
                for _ in range(cnt):
                    if bwd >= len(backwards):
                        break
                    action = backwards[bwd]
                    stage_id, action_type, data_id = action
                    actions.append(Action(rank, action_type, data_id))
                    bwd += 1

                # Forward
                if not fwd_switched:
                    stage_id, action_type, data_id = forwards[fwd]
                    if data_id == self.fwd_switch:
                        fwd_switched = True
                cnt = k if fwd_switched else 1
                for _ in range(cnt):
                    if fwd >= len(forwards):
                        break
                    action = forwards[fwd]
                    stage_id, action_type, data_id = action
                    actions.append(Action(rank, action_type, data_id))
                    fwd += 1

            # Cooldown phase
            while bwd < len(backwards):
                action = backwards[bwd]
                stage_id, action_type, data_id = action
                actions.append(Action(rank, action_type, data_id))
                bwd += 1

            actions_by_rank.append(actions)

        bwd_fwds = []
        for rank, actions in enumerate(reversed(actions_by_rank)):
            i = 0
            while i < len(actions) and actions[i].action_type == F:
                i += 1
            while i < len(actions) and actions[i].action_type == B:
                bwd_fwds.append(actions[i])
                i += 1
            while i < len(actions) and actions[i].action_type == F:
                bwd_fwds.append(actions[i])
                i += 1

        actions_by_rank.append(bwd_fwds)

        return actions_by_rank


if __name__ == "__main__":
    p = 4
    v = 2
    chunks_list = [[2, 3, 4, 5], [4, 5, 6, 7]]
    schedule = SlimPipeSchedule(p, v, chunks_list)
    order_deps = schedule.order_deps()
    data_deps = schedule.data_deps()

    print("\nData Dependencies:")
    for rank in range(p):
        print(f"Rank {rank}:")
        for action in data_deps[rank]:
            print(action)
    print("\nOrder Dependencies:")
    for rank in range(p):
        print(f"Rank {rank}:")
        for action in order_deps[rank]:
            print(action)
    print("\nCosts:")
    costs = schedule.cost_map()
    for action, cost in costs.items():
        print(f"{action}: {cost}")
    print("\nStage Mapping:")
    stage_mapping = schedule.stage_map()
    for stage_id, rank in stage_mapping.items():
        print(f"Stage {stage_id} -> Rank {rank}")
