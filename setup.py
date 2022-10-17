"""Setup for pip package."""

import os
import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by Megatron.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="megatron-lm",
    version="3.0.0",
    description="Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/NVIDIA/Megatron-LM",
    author="NVIDIA INC",
    maintainer="NVIDIA INC",
    # The licence under which the project is released
    license="See https://github.com/NVIDIA/Megatron-LM/blob/master/LICENSE",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        # Indicate what your project relates to
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Additional Setting
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=setuptools.find_packages(),
    install_requires=["nltk", "six", "regex", "torch>=1.12.0", "pybind11"],
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords="deep learning, Megatron, gpu, NLP, nvidia, pytorch, torch, language",
)
