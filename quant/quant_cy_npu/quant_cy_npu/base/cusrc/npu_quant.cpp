#include "torch/extension.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"


void run_mxfp4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len);
void run_mxfp8e4m3_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_mxfp8e4m3_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_nvf4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N);
void run_hifx_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);
void run_hifx_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit);


void mxfp4_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp4_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void mxfp4_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp4_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void hif8_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
}

void hif8_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int len = x.numel();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hif8_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), len);
}

void mxfp8e4m3_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp8e4m3_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void mxfp8e4m3_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_mxfp8e4m3_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void nvf4_quant(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_nvf4_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void nvf4_quant_bf16(at::Tensor x, at::Tensor y){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_nvf4_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N);
}

void hifx_quant(at::Tensor x, at::Tensor y, int mant_bit){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hifx_kernel(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N, mant_bit);
}

void hifx_quant_bf16(at::Tensor x, at::Tensor y, int mant_bit){
    int devidx = x.device().index();
    int M = x.numel() / x.size(-1);
    int N = x.size(-1);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();
    run_hifx_kernel_bf16(40, aclstream, (uint8_t*)(x.storage().data()), (uint8_t*)(y.storage().data()), M, N, mant_bit);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_quant", &mxfp4_quant, "mxfp4_quant");
    m.def("mxfp4_quant_bf16", &mxfp4_quant_bf16, "mxfp4_quant_bf16");
    m.def("hif8_quant", &hif8_quant, "hif8_quant");
    m.def("hif8_quant_bf16", &hif8_quant_bf16, "hif8_quant_bf16");
    m.def("mxfp8e4m3_quant", &mxfp8e4m3_quant, "mxfp8e4m3_quant");
    m.def("mxfp8e4m3_quant_bf16", &mxfp8e4m3_quant_bf16, "mxfp8e4m3_quant_bf16");
    m.def("nvf4_quant", &nvf4_quant, "nvf4_quant");
    m.def("nvf4_quant_bf16", &nvf4_quant_bf16, "nvf4_quant_bf16");
    m.def("hifx_quant", &hifx_quant, "hifx_quant");
    m.def("hifx_quant_bf16", &hifx_quant_bf16, "hifx_quant_bf16");
}