cmake_minimum_required(VERSION 3.12)
project(quant_cy_npu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)

# Find NPU libraries
find_library(ASCENDCL_LIB ascendcl PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)
find_library(ACLNN_LIB aclnn PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)
find_library(ACLNN_BASE_LIB aclnn_base PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)
find_library(ACLNN_COMMON_LIB aclnn_common PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)
find_library(ACLNN_MATH_LIB aclnn_math PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)
find_library(ACLNN_VECTOR_LIB aclnn_vector PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_cy_npu/base/cusrc
    /usr/local/Ascend/ascend-toolkit/latest/include
    /usr/local/Ascend/ascend-toolkit/latest/include/aclnn
    /usr/local/Ascend/ascend-toolkit/latest/include/aclnn_base
    /usr/local/Ascend/ascend-toolkit/latest/include/aclnn_common
    /usr/local/Ascend/ascend-toolkit/latest/include/aclnn_math
    /usr/local/Ascend/ascend-toolkit/latest/include/aclnn_vector
)

# Source files
set(SOURCES
    quant_cy_npu/base/cusrc/npu_quant.cpp
    quant_cy_npu/base/cusrc/npu_quantop_base.cpp
)

# Create the extension module
pybind11_add_module(quant_cy_npu_npu_ops ${SOURCES})

# Set compile definitions
target_compile_definitions(quant_cy_npu_npu_ops PRIVATE
    ASCEND_910B=1
    __DAV_C220_VEC__=1
    ASCEND_IS_AIV=1
)

# Set compile options
target_compile_options(quant_cy_npu_npu_ops PRIVATE
    -std=c++17
    -O2
    -fPIC
    -DASCEND_910B
    -D__DAV_C220_VEC__
    -DASCEND_IS_AIV
)

# Link libraries
target_link_libraries(quant_cy_npu_npu_ops PRIVATE
    ${TORCH_LIBRARIES}
    ${ASCENDCL_LIB}
    ${ACLNN_LIB}
    ${ACLNN_BASE_LIB}
    ${ACLNN_COMMON_LIB}
    ${ACLNN_MATH_LIB}
    ${ACLNN_VECTOR_LIB}
)

# Set RPATH
set_target_properties(quant_cy_npu_npu_ops PROPERTIES
    INSTALL_RPATH "/usr/local/Ascend/ascend-toolkit/latest/lib64"
    BUILD_RPATH "/usr/local/Ascend/ascend-toolkit/latest/lib64"
)

# Installation
install(TARGETS quant_cy_npu_npu_ops
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
