import math
import torch
# import torch_npu
from typing import Tuple

def optimized_mxfp8_e4m3_matmul(A: torch.Tensor, B: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """优化的MXFP8矩阵乘法实现"""
    device = A.device
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # E4M3格式常量
    MAX_VAL = 448.0

    # 预分配输出张量
    C = torch.zeros((M, N), dtype=torch.float32, device=device)

    # 向量化的量化函数
    def vectorized_quantize_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """向量化的E4M3量化"""
        # x shape: (M, num_blocks, block_size) 或 (num_blocks, block_size, N)
        # scale shape: (M, num_blocks, 1) 或 (num_blocks, 1, N)

        # 缩放输入
        scaled_x = x / scale

        # 处理符号
        sign = torch.sign(scaled_x)
        abs_x = torch.abs(scaled_x)

        # 饱和处理
        abs_x = torch.clamp(abs_x, 0, MAX_VAL)

        # 近似量化 - 使用查找表或简化的量化
        # 这里使用简化版本：将值量化到2^n * (1 + k/8)的网格
        log_abs = torch.log2(torch.clamp(abs_x, min=1e-10))
        exp = torch.floor(log_abs).clamp(-6, 8)

        # 计算量化后的值
        base = torch.pow(2.0, exp)
        normalized = abs_x / base
        # 量化mantissa到8个level (3 bits)
        quantized_mant = torch.round((normalized - 1.0) * 8) / 8
        quantized_mant = torch.clamp(quantized_mant, 0, 7/8)

        result = sign * base * (1.0 + quantized_mant)
        return result

    # 批量处理A的行块
    def process_A_blocks(A_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量处理A的行块量化"""
        M, K = A_tensor.shape
        num_blocks = (K + block_size - 1) // block_size

        # 重塑为块结构
        padded_K = num_blocks * block_size
        if padded_K > K:
            A_padded = torch.zeros((M, padded_K), dtype=A_tensor.dtype, device=A_tensor.device)
            A_padded[:, :K] = A_tensor
        else:
            A_padded = A_tensor

        A_blocks = A_padded.view(M, num_blocks, block_size)

        # 计算每个块的缩放因子 (M, num_blocks, 1)
        block_max = torch.abs(A_blocks).max(dim=2, keepdim=True)[0]
        scales = torch.pow(2.0, torch.ceil(torch.log2(block_max / MAX_VAL)))
        scales = torch.where(block_max == 0, torch.ones_like(scales), scales)

        # 向量化量化
        A_quantized = vectorized_quantize_e4m3(A_blocks, scales)

        return A_quantized.view(M, padded_K)[:, :K], scales.squeeze(-1)

    # 批量处理B的列块
    def process_B_blocks(B_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量处理B的列块量化"""
        K, N = B_tensor.shape
        num_blocks = (K + block_size - 1) // block_size

        # 重塑为块结构
        padded_K = num_blocks * block_size
        if padded_K > K:
            B_padded = torch.zeros((padded_K, N), dtype=B_tensor.dtype, device=B_tensor.device)
            B_padded[:K, :] = B_tensor
        else:
            B_padded = B_tensor

        B_blocks = B_padded.view(num_blocks, block_size, N)

        # 计算每个块的缩放因子 (num_blocks, 1, N)
        block_max = torch.abs(B_blocks).max(dim=1, keepdim=True)[0]
        scales = torch.pow(2.0, torch.ceil(torch.log2(block_max / MAX_VAL)))
        scales = torch.where(block_max == 0, torch.ones_like(scales), scales)

        # 向量化量化
        B_quantized = vectorized_quantize_e4m3(B_blocks, scales)

        return B_quantized.view(padded_K, N)[:K, :], scales.squeeze(1)

    # 量化A和B
    A_quantized, A_scales = process_A_blocks(A)
    B_quantized, B_scales = process_B_blocks(B)

    # 分块矩阵乘法
    num_blocks = (K + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        start_k = block_idx * block_size
        end_k = min(start_k + block_size, K)

        # 提取当前块
        A_block = A_quantized[:, start_k:end_k]
        B_block = B_quantized[start_k:end_k, :]

        # 计算部分乘积
        partial = torch.matmul(A_block, B_block)

        # 应用缩放因子
        A_scale_block = A_scales[:, block_idx:block_idx+1]
        B_scale_block = B_scales[block_idx:block_idx+1, :]
        combined_scale = A_scale_block * B_scale_block

        # 累加到结果
        C += partial * combined_scale

    return C


def optimized_mxfp8_e5m2_matmul(A: torch.Tensor, B: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """优化的MXFP8-E5M2矩阵乘法实现"""
    device = A.device
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # E5M2格式常量
    # E5M2: 1 sign + 5 exponent + 2 mantissa
    # 指数范围: -15 到 +16 (偏置15)
    # 最大值: 2^16 * (1 + 3/4) = 2^16 * 1.75 = 114688
    MAX_VAL = 114688.0
    MIN_EXP = -15
    MAX_EXP = 16

    # 预分配输出张量
    C = torch.zeros((M, N), dtype=torch.float32, device=device)

    # 向量化的量化函数
    def vectorized_quantize_e5m2(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """向量化的E5M2量化"""
        # x shape: (M, num_blocks, block_size) 或 (num_blocks, block_size, N)
        # scale shape: (M, num_blocks, 1) 或 (num_blocks, 1, N)

        # 缩放输入
        scaled_x = x / scale

        # 处理符号
        sign = torch.sign(scaled_x)
        abs_x = torch.abs(scaled_x)

        # 饱和处理
        abs_x = torch.clamp(abs_x, 0, MAX_VAL)

        # 处理特殊情况：零值
        zero_mask = (abs_x == 0)

        # E5M2量化
        # 计算指数
        log_abs = torch.log2(torch.clamp(abs_x, min=1e-20))
        exp = torch.floor(log_abs).clamp(MIN_EXP, MAX_EXP)

        # 计算基数值
        base = torch.pow(2.0, exp)

        # 计算归一化的尾数 (范围 [1, 2))
        normalized = abs_x / base

        # 量化尾数到4个级别 (2 bits: 00, 01, 10, 11)
        # 对应值: 1.00, 1.25, 1.50, 1.75
        quantized_mant_idx = torch.round((normalized - 1.0) * 4).clamp(0, 3)
        quantized_mant = 1.0 + quantized_mant_idx / 4.0

        # 计算量化后的结果
        result = sign * base * quantized_mant

        # 处理零值
        result = torch.where(zero_mask, torch.zeros_like(result), result)

        return result

    # 批量处理A的行块
    def process_A_blocks(A_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量处理A的行块量化"""
        M, K = A_tensor.shape
        num_blocks = (K + block_size - 1) // block_size

        # 重塑为块结构
        padded_K = num_blocks * block_size
        if padded_K > K:
            A_padded = torch.zeros((M, padded_K), dtype=A_tensor.dtype, device=A_tensor.device)
            A_padded[:, :K] = A_tensor
        else:
            A_padded = A_tensor

        A_blocks = A_padded.view(M, num_blocks, block_size)

        # 计算每个块的缩放因子 (M, num_blocks, 1)
        block_max = torch.abs(A_blocks).max(dim=2, keepdim=True)[0]

        # 对于E5M2，缩放因子计算需要考虑更大的指数范围
        scales = torch.pow(2.0, torch.ceil(torch.log2(block_max / MAX_VAL)))
        scales = torch.where(block_max == 0, torch.ones_like(scales), scales)

        # 向量化量化
        A_quantized = vectorized_quantize_e5m2(A_blocks, scales)

        return A_quantized.view(M, padded_K)[:, :K], scales.squeeze(-1)

    # 批量处理B的列块
    def process_B_blocks(B_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量处理B的列块量化"""
        K, N = B_tensor.shape
        num_blocks = (K + block_size - 1) // block_size

        # 重塑为块结构
        padded_K = num_blocks * block_size
        if padded_K > K:
            B_padded = torch.zeros((padded_K, N), dtype=B_tensor.dtype, device=B_tensor.device)
            B_padded[:K, :] = B_tensor
        else:
            B_padded = B_tensor

        B_blocks = B_padded.view(num_blocks, block_size, N)

        # 计算每个块的缩放因子 (num_blocks, 1, N)
        block_max = torch.abs(B_blocks).max(dim=1, keepdim=True)[0]

        # 对于E5M2，缩放因子计算需要考虑更大的指数范围
        scales = torch.pow(2.0, torch.ceil(torch.log2(block_max / MAX_VAL)))
        scales = torch.where(block_max == 0, torch.ones_like(scales), scales)

        # 向量化量化
        B_quantized = vectorized_quantize_e5m2(B_blocks, scales)

        return B_quantized.view(padded_K, N)[:K, :], scales.squeeze(1)

    # 量化A和B
    A_quantized, A_scales = process_A_blocks(A)
    B_quantized, B_scales = process_B_blocks(B)

    # 分块矩阵乘法
    num_blocks = (K + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        start_k = block_idx * block_size
        end_k = min(start_k + block_size, K)

        # 提取当前块
        A_block = A_quantized[:, start_k:end_k]
        B_block = B_quantized[start_k:end_k, :]

        # 计算部分乘积
        partial = torch.matmul(A_block, B_block)

        # 应用缩放因子
        A_scale_block = A_scales[:, block_idx:block_idx+1]
        B_scale_block = B_scales[block_idx:block_idx+1, :]
        combined_scale = A_scale_block * B_scale_block

        # 累加到结果
        C += partial * combined_scale

    return C


if __name__ == "__main__":
    M, K, N = 128, 256, 64
    M, K, N = 1024, 256, 1024

    A = torch.rand((M, N), dtype=torch.bfloat16).cuda()
    B = torch.rand((N, K), dtype=torch.bfloat16).cuda()
    C = torch.matmul(A, B)

    C_opt = optimized_mxfp8_e4m3_matmul(A, B)
    mse_opt = torch.mean((C - C_opt) ** 2)
    max_err_opt = torch.max(torch.abs(C - C_opt))

    print(f"E4M3 OPT MSE: {mse_opt:.6f}")
    print(f"E4M3 OPT Max Error: {max_err_opt:.6f}")
    print(f"E4M3 OPT 相对误差: {mse_opt / torch.mean(C ** 2):.6f}")

    A = torch.rand((M, N), dtype=torch.bfloat16).cuda() * 1e-15 # 模拟梯度数值范围
    B = torch.rand((N, K), dtype=torch.bfloat16).cuda()

    C = torch.matmul(A, B)

    C_e5m2 = optimized_mxfp8_e5m2_matmul(A, B)
    mse_e5m2 = torch.mean((C - C_e5m2) ** 2)
    max_err_e5m2 = torch.max(torch.abs(C - C_e5m2))
    print(f"E5M2 OPT MSE: {mse_e5m2:.20f}")
    print(f"E5M2 OPT Max Error: {max_err_e5m2:.20f}")
    print(f"E5M2 OPT 相对误差: {mse_e5m2 / torch.mean(C ** 2):.20f}")
