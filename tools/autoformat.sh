#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# for now we just format core

black ${SCRIPT_DIR}/../megatron/core
isort ${SCRIPT_DIR}/../megatron/core
