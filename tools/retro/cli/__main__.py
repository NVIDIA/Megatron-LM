# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from . import retro


if __name__ == "__main__":
    retro.init(os.environ["RETRO_WORKDIR"])
