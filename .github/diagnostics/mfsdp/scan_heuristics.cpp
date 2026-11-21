// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Query the captured BF16 weight-gradient geometry without allocating operands.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

static const std::array<cublasLtMatmulAlgoConfigAttributes_t, 8> attributes = {
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
};

struct Choice {
  std::array<int, 8> configuration{};
  size_t workspace{};
  bool operator==(const Choice &other) const {
    return configuration == other.configuration && workspace == other.workspace;
  }
};

static void write_range(std::ofstream &out, int begin, int end,
                        const Choice &choice) {
  out << begin << ' ' << end;
  for (int value : choice.configuration) out << ' ' << value;
  out << ' ' << choice.workspace << '\n';
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  auto start = std::chrono::steady_clock::now();
  cudaDeviceProp device{};
  if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
    throw std::runtime_error("Cannot inspect CUDA device");
  cublasLtHandle_t handle;
  check(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t operation;
  check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transpose = CUBLAS_OP_T;
  check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &transpose, sizeof(transpose)));
  cublasLtMatrixLayout_t output;
  check(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, 512, 2048, 512));
  cublasLtMatmulPreference_t preference;
  check(cublasLtMatmulPreferenceCreate(&preference));
  const size_t workspace = 33555456;
  check(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof(workspace)));
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  check(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
      sizeof(reduction)));
  std::ofstream out(argv[1]);
  if (!out) throw std::runtime_error("Cannot open output");
  int begin = 1, ranges = 0;
  Choice previous;
  for (int k = 1; k <= 65536; ++k) {
    cublasLtMatrixLayout_t a, b;
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, 512, k, 512));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, 2048, k, 2048));
    cublasLtMatmulHeuristicResult_t result;
    int count{};
    check(cublasLtMatmulAlgoGetHeuristic(handle, operation, a, b, output,
                                       output, preference, 1, &result, &count));
    if (count != 1) throw std::runtime_error("No heuristic at K=" + std::to_string(k));
    Choice choice;
    choice.workspace = result.workspaceSize;
    for (size_t i = 0; i < attributes.size(); ++i) {
      choice.configuration[i] = -1;
      size_t written{};
      cublasLtMatmulAlgoConfigGetAttribute(
          &result.algo, attributes[i], &choice.configuration[i], sizeof(int), &written);
    }
    if (k > 1 && !(choice == previous)) {
      write_range(out, begin, k - 1, previous);
      ++ranges;
      begin = k;
    }
    previous = choice;
    check(cublasLtMatrixLayoutDestroy(a));
    check(cublasLtMatrixLayoutDestroy(b));
  }
  write_range(out, begin, 65536, previous);
  ++ranges;
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::ofstream metadata(std::string(argv[1]) + ".metadata.json");
  metadata << "{\"gpu\":\"" << device.name << "\",\"major\":" << device.major
           << ",\"minor\":" << device.minor << ",\"sm_count\":" << device.multiProcessorCount
           << ",\"cublaslt_version\":" << cublasLtGetVersion()
           << ",\"m\":512,\"n\":2048,\"k_min\":1,\"k_max\":65536,\"workspace_limit\":"
           << workspace << ",\"ranges\":" << ranges << ",\"elapsed_seconds\":" << elapsed << "}\n";
  std::cout << "Queried 65536 reduction sizes in " << elapsed << " seconds; " << ranges << " ranges\n";
}
