# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
class Counter:
    """A simple counter class

    This class is responsible for assigning request ids to incoming requests
    """

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0
