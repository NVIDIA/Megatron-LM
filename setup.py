import sysconfig

import pybind11
from setuptools import Extension, setup


def get_pybind_include():
    return [
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_path("include")}"
    ]

setup_args = dict(
    ext_modules=[
        Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=["megatron/core/datasets/helpers.cpp"],
            language="c++",
            extra_compile_args=(get_pybind_include()) +
                ["-O3", "-Wall", "-std=c++17"],
            optional=True,
        )
    ]
)
setup(**setup_args)
