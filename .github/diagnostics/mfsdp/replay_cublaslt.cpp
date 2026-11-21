// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

static void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void *load(const char *path, size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != expected)
    throw std::runtime_error("Input file size mismatch");
  std::vector<char> bytes(expected);
  stream.seekg(0);
  stream.read(bytes.data(), bytes.size());
  void *device;
  cuda_check(cudaMalloc(&device, expected));
  cuda_check(cudaMemcpy(device, bytes.data(), expected, cudaMemcpyHostToDevice));
  return device;
}

int main(int argc, char **argv) {
  if (argc != 5) return 2;
  const uint64_t m = 512, n = 2048, k = std::stoull(argv[3]);
  void *a = load(argv[1], m * k * 2), *b = load(argv[2], n * k * 2);
  void *out, *workspace;
  const size_t workspace_bytes = 33555456;
  cuda_check(cudaMalloc(&out, m * n * 2));
  cuda_check(cudaMalloc(&workspace, workspace_bytes));
  cublasLtHandle_t handle;
  check(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t operation;
  check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transpose = CUBLAS_OP_T;
  check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &transpose, sizeof(transpose)));
  cublasLtMatrixLayout_t adesc, bdesc, odesc;
  check(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, m, k, m));
  check(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, n, k, n));
  check(cublasLtMatrixLayoutCreate(&odesc, CUDA_R_16BF, m, n, m));
  cublasLtMatmulPreference_t preference;
  check(cublasLtMatmulPreferenceCreate(&preference));
  check(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  check(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
      sizeof(reduction)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int count = 0;
  check(cublasLtMatmulAlgoGetHeuristic(handle, operation, adesc, bdesc, odesc,
                                     odesc, preference, 1, &heuristic, &count));
  if (count != 1) throw std::runtime_error("No algorithm returned");
  const float alpha = 1, beta = 0;
  for (int mode = 0; mode < 3; ++mode) {
    auto algo = heuristic.algo;
    const char *label = mode == 0 ? "heuristic" : mode == 1 ? "split6" : "split8";
    if (mode != 0) {
      uint32_t tile = mode == 1 ? CUBLASLT_MATMUL_TILE_256x192 : CUBLASLT_MATMUL_TILE_256x256;
      uint32_t split = mode == 1 ? 6 : 8;
      check(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                               &tile, sizeof(tile)));
      check(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                               &split, sizeof(split)));
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
      check(cublasLtMatmul(handle, operation, &alpha, a, adesc, b, bdesc,
                           &beta, out, odesc, out, odesc, &algo, workspace,
                           workspace_bytes, nullptr));
      cuda_check(cudaDeviceSynchronize());
      std::vector<char> bytes(m * n * 2);
      cuda_check(cudaMemcpy(bytes.data(), out, bytes.size(), cudaMemcpyDeviceToHost));
      std::string path = std::string(argv[4]) + "-" + label + "-" + std::to_string(repeat) + ".bin";
      std::ofstream file(path, std::ios::binary);
      file.write(bytes.data(), bytes.size());
      std::cout << path << std::endl;
    }
  }
}
