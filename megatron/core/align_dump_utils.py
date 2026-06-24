"""
GLM4.5-Air 精度对齐工具函数 - PyTorch (Megatron) 侧

集中放置 PF/MG 跨框架对齐用的 tensor dump / md5 打印 / 梯度 hook 等辅助函数,
以便 megatron/core 内各模块统一从这里 import, 避免之前 sys.path hack 散落在各处。

环境变量 (默认全 0, 不影响原始训练):
  GLM_ALIGN_BIT_EXACT  逻辑级 bit-exact 对齐开关 (LNImpl / persist_layer_norm / use_transformer_engine
                        / rotary_percent / state-dict remap / cudnn deterministic / swiglu_eager 等)
  GLM_ALIGN_LOG        持续性插桩 (cp* tensor & grad info, weight grad 表)
  GLM_ALIGN_DUMP_DATA  一次性 dump (输入数据 md5/shape, 初始权重 md5/norm)

打印过滤控制(两个环境变量均为逗号分隔, 不设则全部保存):
  SAVE_TENSOR_SUBDIRS  只保存指定 subdir, 例如 "moe_layer"
  SAVE_TENSOR_NAMES    只保存名称含指定关键字的 checkpoint, 例如 "cp12b,cp12c"

迁移自:
  /root/paddlejob/share-storage/gpfs/system-public/zhanghonggeng/glm_45/save_tensor_torch.py
"""

import os
import hashlib

import numpy as np
import torch


# ==================== GLM 精度对齐三档总开关 ====================
def is_bit_exact() -> bool:
    """逻辑级 bit-exact 对齐开关 (默认关)"""
    return os.environ.get("GLM_ALIGN_BIT_EXACT", "0") == "1"


def is_log_enabled() -> bool:
    """持续性插桩开关 (cp* tensor/grad info, weight grad 表; 默认关)"""
    return os.environ.get("GLM_ALIGN_LOG", "0") == "1"


def is_dump_data_enabled() -> bool:
    """一次性 dump 开关 (输入数据/初始权重 md5/norm; 默认关)"""
    return os.environ.get("GLM_ALIGN_DUMP_DATA", "0") == "1"




def _is_enabled(name, subdir):
    """根据环境变量白名单判断是否需要保存"""
    allowed_subdirs = os.environ.get("SAVE_TENSOR_SUBDIRS", "")
    allowed_names = os.environ.get("SAVE_TENSOR_NAMES", "")
    if allowed_subdirs and subdir not in [s.strip() for s in allowed_subdirs.split(",")]:
        return False
    if allowed_names and not any(k.strip() in name for k in allowed_names.split(",")):
        return False
    return True


def _mg_tensor_info(name, tensor, layer_num=None, prefix="MG MoE"):
    """打印 MG 侧 tensor 信息(表格格式), 受 GLM_ALIGN_LOG 与 SAVE_TENSOR_NAMES 过滤控制"""
    if not is_log_enabled():
        return
    allowed_names = os.environ.get("SAVE_TENSOR_NAMES", "")
    if allowed_names and not any(k.strip() in name for k in allowed_names.split(",")):
        return
    layer_str = f"L{layer_num}:" if layer_num is not None else ""
    label = f"{prefix}:{layer_str}{name}"
    if tensor is None:
        print(f"| {label:<40s} | {'None':<16s} | {'N/A':<20s} | {'N/A':<20s} |")
        return
    data = tensor.detach().float().contiguous().cpu().numpy()
    # layout 对齐: MG [seq, batch, ...] -> [batch, seq, ...] 与 Paddle 一致
    if data.ndim >= 2 and data.shape[1] == 1:
        data = data.swapaxes(0, 1)
    md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    print(f"| {label:<40s} | {md5:<16s} | {shape_str:<20s} | {dtype_str:<20s} |")


def _mg_grad_info(name, layer_num=None, prefix="GRAD MG"):
    """
    返回一个 hook, backward 时打印梯度的 md5/shape/dtype/abs_mean/abs_max 信息。
    用法: tensor.register_hook(_mg_grad_info("some_name", layer_num=1))

    打印格式与 _pf_grad_info 一致, 方便两侧对比。
    受 GLM_ALIGN_LOG 控制: 关闭时返回的 hook 直接透传, 不做任何打印。
    """
    if not is_log_enabled():
        # 返回一个 no-op hook, 调用方仍可无脑 register
        def _noop_hook(grad):
            return grad
        return _noop_hook

    def hook(grad):
        if grad is None:
            return
        layer_str = f"L{layer_num}:" if layer_num is not None else ""
        label = f"{prefix}:{layer_str}{name}"
        data = grad.detach().float().contiguous().cpu().numpy()
        # layout 对齐: MG [seq, batch, ...] -> [batch, seq, ...] 与 Paddle 一致
        if data.ndim >= 2 and data.shape[1] == 1:
            data = data.swapaxes(0, 1)
        md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
        shape_str = str(list(grad.shape))
        dtype_str = str(grad.dtype)
        abs_mean = float(np.abs(data).mean())
        abs_max = float(np.abs(data).max())
        # 黄色高亮 GRAD 行
        print(
            f"\033[33m| {label:<40s} | md5={md5} | {shape_str:<20s} | {dtype_str:<14s} | "
            f"abs_mean={abs_mean:.6e} | abs_max={abs_max:.6e} |\033[0m"
        )
        return grad
    return hook


def _print_tensor_info(tensor, name, subdir, fname, tag="Saved"):
    """打印张量的详细信息"""
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    print(f"  📌 [MG][{subdir}] {tag}: {fname}  shape={shape}, dtype={dtype}")


def save_tensor(tensor, name, subdir, layer_idx=None):
    """[NO-SAVE] 仅打印 tensor 信息, 不再写 npy. 保留签名以兼容老调用方."""
    if tensor is None:
        return
    if not _is_enabled(name, subdir):
        return
    if layer_idx is not None:
        fname = f"{name}_{layer_idx}.npy"
    else:
        fname = f"{name}.npy"
    _print_tensor_info(tensor, name, subdir, fname, tag="Print(no-save)")


def save_tensor_grad(name, subdir, layer_idx=None):
    """[NO-SAVE] 返回 no-op hook (透传 grad), 不再写 npy. 保留签名以兼容老调用方."""
    def hook(grad):
        return grad
    return hook


def save_tensor_grad_step_indexed(name, subdir, layer_idx=None, base_dir=None):
    """[NO-SAVE] 返回 no-op hook, 不再写 npy. 保留签名以兼容老调用方."""
    def hook(grad):
        return grad
    return hook


# ==================== Router / Topk 表格打印 (与 PF _pf_ti 对齐) ====================
# 与 _mg_tensor_info 区别: 不做 [seq,batch,...] -> [batch,seq,...] layout swap,
# 因为 router/topk 的 tensor 通常是 [num_tokens, num_experts] 二维, 不需要 swap。
def _mg_router_info(name, tensor, layer_num=None, prefix="MG Router"):
    """打印 MG 侧 router tensor 信息(表格格式), 受 GLM_ALIGN_LOG 控制"""
    if not is_log_enabled():
        return
    layer_str = f"L{layer_num}:" if layer_num is not None else ""
    label = f"{prefix}:{layer_str}{name}"
    if tensor is None:
        print(f"| {label:<40s} | {'None':<16s} | {'N/A':<20s} | {'N/A':<20s} |")
        return
    data = tensor.detach().float().contiguous().cpu().numpy()
    md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    print(f"| {label:<40s} | {md5:<16s} | {shape_str:<20s} | {dtype_str:<20s} |")


# topk 内部 layer_number 全局变量(由 router.py 在 routing 调用前设置)
_mg_topk_layer_number = None


def set_mg_topk_layer_number(ln):
    """供 router.py 在 routing 前设置当前 layer_number, 让 _mg_topk_info/_mg_topk_grad_info 使用"""
    global _mg_topk_layer_number
    _mg_topk_layer_number = ln


def _mg_topk_info(name, tensor):
    """打印 MG 内部 topk 中间 tensor 信息(表格格式), layer 号取自 set_mg_topk_layer_number"""
    if not is_log_enabled():
        return
    ln = _mg_topk_layer_number
    layer_str = f"L{ln}:" if ln is not None else ""
    label = f"MG Router:{layer_str}{name}"
    if tensor is None:
        print(f"| {label:<40s} | {'None':<16s} | {'N/A':<20s} | {'N/A':<20s} |")
        return
    data = tensor.detach().float().contiguous().cpu().numpy()
    md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    print(f"| {label:<40s} | {md5:<16s} | {shape_str:<20s} | {dtype_str:<20s} |")


def _mg_topk_grad_info(name, prefix="GRAD MG Router"):
    """返回 backward hook, 打印 MG 内部 topk 中间梯度。GLM_ALIGN_LOG=0 时返回 no-op hook"""
    if not is_log_enabled():
        def _noop(grad):
            return grad
        return _noop
    ln_captured = _mg_topk_layer_number  # 注册时捕获 layer 号
    def hook(grad):
        if grad is None:
            return
        layer_str = f"L{ln_captured}:" if ln_captured is not None else ""
        label = f"{prefix}:{layer_str}{name}"
        data = grad.detach().float().contiguous().cpu().numpy()
        md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
        shape_str = str(list(grad.shape))
        dtype_str = str(grad.dtype)
        abs_mean = float(np.abs(data).mean())
        abs_max = float(np.abs(data).max())
        print(
            f"\033[33m| {label:<40s} | md5={md5} | {shape_str:<20s} | {dtype_str:<14s} | "
            f"abs_mean={abs_mean:.6e} | abs_max={abs_max:.6e} |\033[0m"
        )
        return grad
    return hook


# ==================== 跨框架 forward/grad dump (仅打印 md5/shape) ====================
_dump_fwd_counter = {}
_dump_grad_counter = {}


def _dump_to_np(t):
    if t is None:
        return None
    return t.detach().contiguous().cpu().float().numpy()


def mg_dump_forward(tag_layer, tensors, scalars=None):
    """forward 入参 dump (仅打印 md5/shape, 不写盘). tag_layer 形如 'auxloss_LMG'."""
    if not is_log_enabled():
        return 0
    key = ("mg", tag_layer)
    _dump_fwd_counter[key] = _dump_fwd_counter.get(key, 0) + 1
    step = _dump_fwd_counter[key]
    for name, t in tensors.items():
        arr = _dump_to_np(t)
        if arr is None:
            continue
        md5 = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        print(
            f"  [DUMP MG] forward step{step} {tag_layer}/{name} "
            f"md5={md5} shape={tuple(arr.shape)}",
            flush=True,
        )
    return step


def mg_dump_grad_hook(tag_layer, name):
    """返回 backward hook (仅打印 md5/shape, 不写盘). GLM_ALIGN_LOG=0 时返回 no-op hook."""
    if not is_log_enabled():
        def _noop(grad):
            return grad
        return _noop
    key = ("mg", tag_layer, name)

    def hook(grad):
        if grad is None:
            return grad
        _dump_grad_counter[key] = _dump_grad_counter.get(key, 0) + 1
        step = _dump_grad_counter[key]
        arr = _dump_to_np(grad)
        md5 = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        print(
            f"  [DUMP MG] grad {tag_layer}/{name} step{step} "
            f"md5={md5} shape={tuple(arr.shape)}",
            flush=True,
        )
        return grad

    return hook


# ==================== Optimizer state probe (GLM_ALIGN_OPTIM_PROBE) ====================
# 与 GLM_ALIGN_LOG 解耦, 单独控制 optimizer 前/中/后状态对齐 dump (仅打印, 不写盘)
#   phase ∈ {pre, mid, post}
#     pre  = optimizer.step() 前         (main_param fp32 + main_grad fp32 + adam state)
#     mid  = optimizer.step() 后, cast 前 (main_param fp32 已更新, model_param bf16 未更新)
#     post = _copy_main_params_to_model_params() 后 (model_param bf16 已更新 = 下一步 forward 用的权重)
_optim_probe_step = {"mg": 0}


def is_optim_probe_enabled() -> bool:
    """Optimizer 前后状态对齐 dump 开关 (默认关)"""
    return os.environ.get("GLM_ALIGN_OPTIM_PROBE", "0") == "1"


def _optim_probe_print_row(label, arr):
    if arr is None:
        print(f"\033[35m[OPTIM PROBE] {label:<60s} None\033[0m", flush=True)
        return
    md5 = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    af32 = arr.astype(np.float32)
    norm = float(np.linalg.norm(af32))
    am = float(np.abs(af32).mean())
    mx = float(np.abs(af32).max())
    print(
        f"\033[35m[OPTIM PROBE] {label:<60s} md5={md5} shape={tuple(arr.shape)} "
        f"norm={norm:.6f} abs_mean={am:.4e} abs_max={mx:.4e}\033[0m",
        flush=True,
    )

def mg_dump_optim_probe_print(phase, float16_groups, fp32_from_float16_groups, optimizer_state, param_groups=None):
    """
    只打印不落盘版: 在 optimizer.step 不同 phase 调用一次, 紫色 [OPTIM PROBE] 行输出
    md5/shape/norm/abs_mean/abs_max。
    Args:
        phase: "pre" | "mid" | "post"
        float16_groups: List[List[bf16 model param]]
        fp32_from_float16_groups: List[List[fp32 main param]] (与 float16_groups 一一对应)
        optimizer_state: torch optimizer.state dict (key=main_param)
    """
    if not is_optim_probe_enabled():
        return
    if phase == "pre":
        _optim_probe_step["mg"] += 1
    n = _optim_probe_step["mg"]

    for gi, (model_group, main_group) in enumerate(
        zip(float16_groups, fp32_from_float16_groups)
    ):
        for pi, (model_param, main_param) in enumerate(zip(model_group, main_group)):
            tag = f"g{gi}p{pi}"
            mp_np = model_param.detach().contiguous().cpu().float().numpy()
            _optim_probe_print_row(f"mg step{n} {phase} {tag} model_param", mp_np)
            ma_np = main_param.detach().contiguous().cpu().float().numpy()
            _optim_probe_print_row(f"mg step{n} {phase} {tag} main_param ", ma_np)
            if phase == "pre" and main_param.grad is not None:
                g_np = main_param.grad.detach().contiguous().cpu().float().numpy()
                _optim_probe_print_row(f"mg step{n} {phase} {tag} main_grad ", g_np)
            st = optimizer_state.get(main_param, {})
            for sk in ("exp_avg", "exp_avg_sq"):
                if sk in st and st[sk] is not None:
                    s_np = st[sk].detach().contiguous().cpu().float().numpy()
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} {sk:11s}", s_np)
            # --- 单步 Adam 验证: 用当前 state 手算一步, 对比 optimizer 输出 ---
            # 优先从 param_groups 取真实超参; 找不到时用默认值
            if param_groups is not None and len(param_groups) > 0:
                pg = param_groups[0]
                for cand in param_groups:
                    if any(main_param is p for p in cand.get('params', [])):
                        pg = cand
                        break
                beta1, beta2 = pg.get('betas', (0.9, 0.95))
                eps = pg.get('eps', 1e-8)
                wd  = pg.get('weight_decay', 0.1)
                lr  = pg.get('lr', 5e-5)
            else:
                beta1, beta2, eps, wd, lr = 0.9, 0.95, 1e-8, 0.1, 5e-5
            # pre 阶段: 用 pre 状态做单步 Adam 参考
            if phase == "pre" and main_param.grad is not None and "exp_avg" in st:
                try:
                    import torch
                    grad = main_param.grad.detach().clone()
                    m1 = st["exp_avg"].detach().clone()
                    m2 = st["exp_avg_sq"].detach().clone()
                    p_val = main_param.detach().clone()
                    step_t = float(st.get("step", n))
                    if torch.is_tensor(step_t):
                        step_t = float(step_t.item())
                    p_val.mul_(1.0 - lr * wd)
                    m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    m2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** step_t
                    bc2 = 1.0 - beta2 ** step_t
                    step_size = lr / bc1
                    denom = (m2.sqrt() / (bc2 ** 0.5)).add_(eps)
                    update = m1 / denom * step_size
                    p_val.addcdiv_(m1, denom, value=-step_size)
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} adam_ref  ", p_val.cpu().float().numpy())
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} denom     ", denom.cpu().float().numpy())
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} update    ", update.cpu().float().numpy())
                except Exception as e:
                    print(f"[OPTIM PROBE MG] {tag} adam_ref(pre) fail: {e}", flush=True)
            # mid 阶段: 用 mid 状态反推 update
            if phase == "mid" and "exp_avg" in st:
                try:
                    import torch
                    m1 = st["exp_avg"].detach().clone()
                    m2 = st["exp_avg_sq"].detach().clone()
                    step_t = float(st.get("step", n))
                    if torch.is_tensor(step_t):
                        step_t = float(step_t.item())
                    bc1 = 1.0 - beta1 ** step_t
                    bc2 = 1.0 - beta2 ** step_t
                    step_size = lr / bc1
                    denom = (m2.sqrt() / (bc2 ** 0.5)).add_(eps)
                    update = m1 / denom * step_size
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} denom     ", denom.cpu().float().numpy())
                    _optim_probe_print_row(f"mg step{n} {phase} {tag} update    ", update.cpu().float().numpy())
                except Exception as e:
                    print(f"[OPTIM PROBE MG] {tag} adam mid recompute fail: {e}", flush=True)
            # === expert 参数逐 expert 切片 ===
            NUM_LOCAL_EXPERTS = int(os.environ.get("GLM_NUM_LOCAL_EXPERTS", "16"))
            FFN_HIDDEN = int(os.environ.get("GLM_FFN_HIDDEN_SIZE", "1408"))
            USE_SWIGLU = os.environ.get("GLM_USE_SWIGLU", "1") == "1"
            fc1_per = (2 if USE_SWIGLU else 1) * FFN_HIDDEN
            total_per_expert = fc1_per + FFN_HIDDEN
            if (
                main_param.ndim == 2
                and main_param.shape[0] == NUM_LOCAL_EXPERTS * total_per_expert
            ):
                try:
                    fc1_total = NUM_LOCAL_EXPERTS * fc1_per
                    for ei in range(NUM_LOCAL_EXPERTS):
                        fc1_main = main_param[ei*fc1_per : (ei+1)*fc1_per]
                        fc2_main = main_param[fc1_total + ei*FFN_HIDDEN : fc1_total + (ei+1)*FFN_HIDDEN]
                        _optim_probe_print_row(
                            f"mg step{n} {phase} {tag} expert{ei}_fc1 main_param",
                            fc1_main.detach().contiguous().cpu().float().numpy()
                        )
                        _optim_probe_print_row(
                            f"mg step{n} {phase} {tag} expert{ei}_fc2 main_param",
                            fc2_main.detach().contiguous().cpu().float().numpy()
                        )
                        if phase == "pre" and main_param.grad is not None:
                            g_full = main_param.grad
                            fc1_grad = g_full[ei*fc1_per : (ei+1)*fc1_per]
                            fc2_grad = g_full[fc1_total + ei*FFN_HIDDEN : fc1_total + (ei+1)*FFN_HIDDEN]
                            _optim_probe_print_row(
                                f"mg step{n} pre {tag} expert{ei}_fc1 main_grad",
                                fc1_grad.detach().contiguous().cpu().float().numpy()
                            )
                            _optim_probe_print_row(
                                f"mg step{n} pre {tag} expert{ei}_fc2 main_grad",
                                fc2_grad.detach().contiguous().cpu().float().numpy()
                            )
                except Exception as e:
                    print(f"[OPTIM PROBE MG] {tag} expert slice fail: {e}", flush=True)
    print(f"  [OPTIM PROBE MG] phase={phase} step{n} printed", flush=True)


