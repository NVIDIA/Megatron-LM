from setuptools import setup, find_packages

setup(
    name="megatron.core",
    version="0.1",
    description="Core components of Megatron.",
    packages=find_packages(
        include=("megatron.core")
    )
)
