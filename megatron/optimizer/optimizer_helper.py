import torch
from apex.optimizers import FusedAdam as Adam
# from torch.optim import AdamW as Adam


def rollback_optimizer_step(optimizer):
    try:
        return optimizer.step(rollback=True)
    except Exception:
        pass

    assert isinstance(optimizer, Adam), "Not supported optimizer type {}".format(type(optimizer))
    if optimizer.capturable:
        raise ValueError("Not supported")
    if not optimizer.adam_w_mode:
        raise ValueError("Not supported")
    loss = None

    for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
        if len(group['params']) == 0:
            continue

        bias_correction = 1 if group['bias_correction'] else 0
        beta1, beta2 = group['betas']

        # create lists for multi-tensor apply
        g_16, p_16, m_16, v_16 = [], [], [], []
        g_bf, p_bf, m_bf, v_bf = [], [], [], []
        g_32, p_32, m_32, v_32 = [], [], [], []
        p_16_master = []
        p_32_master = []

        for p, p_master in zip(group['params'], group_master['params']):
            if p.grad is None:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

            state = optimizer.state[p]
            # State initialization
            if len(state) == 0:
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data).float()
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data).float()

            if p.dtype == torch.float16:
                if optimizer.master_weights:
                    p_16_master.append(p_master.data)
                g_16.append(p.grad.data)
                p_16.append(p.data)
                m_16.append(state['exp_avg'])
                v_16.append(state['exp_avg_sq'])
            elif p.dtype == torch.bfloat16:
                g_bf.append(p.grad)
                p_bf.append(p)
                m_bf.append(state['exp_avg'])
                v_bf.append(state['exp_avg_sq'])
            elif p.dtype == torch.float32:
                if optimizer.master_weights:
                    p_32_master.append(p_master.data)
                g_32.append(p.grad.data)
                p_32.append(p.data)
                m_32.append(state['exp_avg'])
                v_32.append(state['exp_avg_sq'])
            else:
                raise RuntimeError('FusedAdam only support fp16 and fp32.')

        if len(g_16) > 0:
            multi_tensor_rollback_adamw(
                g_16, p_16, m_16, v_16,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                group['step'],
                bias_correction,
                group['weight_decay'])

        if len(g_bf) > 0:
            multi_tensor_rollback_adamw(
                g_bf, p_bf, m_bf, v_bf,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                group['step'],
                bias_correction,
                group['weight_decay'])

        if len(g_32) > 0:
            multi_tensor_rollback_adamw(
                g_32, p_32, m_32, v_32,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                group['step'],
                bias_correction,
                group['weight_decay'])
        group['step'] -= 1

    return loss


def multi_tensor_rollback_adamw(
    g_list, p_list, m_list, v_list,
    lr,
    beta1,
    beta2,
    eps,
    step,
    bias_correction,
    weight_decay,
):
    beta1_correction, beta2_correction = 1.0, 1.0
    if bias_correction == 1:
        beta1_correction = 1 - beta1 ** step
        beta2_correction = 1 - beta2 ** step
    for i, p in enumerate(p_list):
        rollback_adamw(
            g_list[i], p_list[i], m_list[i], v_list[i],
            lr,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        )


def rollback_adamw(
    g: torch.Tensor, p: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr,
    beta1,
    beta2,
    beta1_correction,
    beta2_correction,
    eps,
    decay,
):
    update = (m / beta1_correction) / ((v / beta2_correction).sqrt() + eps)
    update.mul_(lr)
    p.add_(update).div_(1 - lr * decay)
    v.addcmul_(g, g, value=beta2 - 1).div_(beta2)
    m.add_(g, alpha=beta1 - 1).div_(beta1)
