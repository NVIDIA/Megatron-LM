# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=["megatron/core/datasets/helpers.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-Wall", "-std=c++17"],
            optional=True,
        )
    ]
)
