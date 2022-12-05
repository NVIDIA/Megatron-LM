from setuptools import find_packages, setup

setup(
    name="megatron.core",
    version="0.1",
    description="Core components of Megatron.",
    packages=find_packages(include=("megatron.core")),
)
