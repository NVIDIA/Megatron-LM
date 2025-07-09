import pytest

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
)
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer
from tests.unit_tests.test_utilities import Utils
import torch.nn.functional as F
import torch.profiler

@pytest.mark.parametrize(
    "tp_size,ep_size,cp_size",
    [
        (2, 1, 1),
        (1, 2, 1),
        # (1, 1, 2),  # 暂时注释掉不支持的配置
    ]
)
def test_aux_loss_parallel_trend(tp_size, ep_size, cp_size):
        # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，将在CPU上运行")
        device = "cpu"
    else:
        print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        device = "cuda"
    
    steps = 200
    lr = 2e-2
    batch_size = 128
    seq_len = 64
    num_classes = 8
    print(f"\n===== 测试配置: TP={tp_size}, EP={ep_size}, CP={cp_size} =====")
    container = MoEModelTestContainer(
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=1,
        cp_size=cp_size,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.5
    )
    
    # 打印并行配置信息
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        print(f"[Rank {rank}] 总专家数量: {container.config.num_moe_experts}")
        print(f"[Rank {rank}] TP并行大小: {tp_size}")
        print(f"[Rank {rank}] EP并行大小: {ep_size}")
        print(f"[Rank {rank}] 每张卡上的专家数量: {container.num_local_experts}")
        print(f"[Rank {rank}] 当前rank的专家索引: {container.local_expert_indices}")
        

    else:
        print(f"总专家数量: {container.config.num_moe_experts}")
        print(f"TP并行大小: {tp_size}")
        print(f"EP并行大小: {ep_size}")
        print(f"每张卡上的专家数量: {container.num_local_experts}")
        print(f"当前rank的专家索引: {container.local_expert_indices}")
    
    # 获取并行状态信息
    import megatron.core.parallel_state as parallel_state
    world_size = 1  # 默认值
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        print(f"[Rank {rank}] 分布式信息 - Rank: {rank}, World Size: {world_size}, EP Rank: {ep_rank}")
        
        # 收集所有rank的专家分布信息
        expert_info = {
            'rank': rank,
            'ep_rank': ep_rank,
            'local_experts': container.local_expert_indices,
            'num_local_experts': container.num_local_experts
        }
        
        # 使用all_gather收集所有rank的信息
        expert_info_tensor = torch.tensor([
            expert_info['rank'],
            expert_info['ep_rank'], 
            expert_info['num_local_experts']
        ], dtype=torch.int, device=device)
        
        all_expert_info = [torch.zeros_like(expert_info_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(all_expert_info, expert_info_tensor)
        
        # 每个rank都打印自己的信息
        print(f"\n[Rank {rank}] === Rank {rank} 的专家分布信息 ===")
        print(f"[Rank {rank}] 当前Rank: {rank}, EP Rank: {ep_rank}")
        print(f"[Rank {rank}] 本地专家数量: {container.num_local_experts}")
        print(f"[Rank {rank}] 本地专家索引: {container.local_expert_indices}")
        
        # 只在rank 0打印汇总信息
        if rank == 0:
            print("\n[Rank 0] === 全局EP并行专家分布汇总 ===")
            for i, info_tensor in enumerate(all_expert_info):
                r, ep_r, num_exp = info_tensor.cpu().numpy()
                print(f"[Rank 0] Rank {int(r)} (EP Rank {int(ep_r)}): {int(num_exp)} 个专家")
            print("[Rank 0] " + "=" * 30)
    else:
        print("未初始化分布式环境")
    
    moe_layer = container.moe_layer.to(device)
    
    # 打印TP并行信息
    if torch.distributed.is_initialized():
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        print(f"[Rank {rank}] TP Rank: {tp_rank}, TP World Size: {tp_world_size}")
        
        # 打印模型参数分布信息
        if hasattr(moe_layer, 'experts'):
            expert_params = list(moe_layer.experts.parameters())
            print(f"[Rank {rank}] 专家参数数量: {len(expert_params)}")
            for i, param in enumerate(expert_params):
                if hasattr(param, 'tensor_model_parallel'):
                    print(f"[Rank {rank}] 参数 {i}: tensor_parallel={param.tensor_model_parallel}, shape={param.shape}")
                else:
                    print(f"[Rank {rank}] 参数 {i}: shape={param.shape}")
    hidden_size = moe_layer.config.hidden_size
    aux_loss_list = []
    load_balance_ratio_list = []
    acc_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(moe_layer.parameters(), lr=lr)
    expert_token_history = []

    for step in range(steps):
        # 构造one-hot分类任务输入
        per_class = batch_size * seq_len // num_classes
        values = []
        labels = []
        for c in range(num_classes):
            v = torch.randint(c*10+1, (c+1)*10+1, (per_class,), device=device)
            values.append(v)
            labels.append(torch.full((per_class,), c, device=device, dtype=torch.long))
        values = torch.cat(values)[:batch_size*seq_len]
        labels = torch.cat(labels)[:batch_size*seq_len]
        idx = torch.randperm(batch_size*seq_len, device=device)
        values = values[idx]
        labels = labels[idx]
        label = labels.view(batch_size, seq_len)
        input_data = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        for i in range(batch_size):
            for j in range(seq_len):
                input_data[i, j, label[i, j] % hidden_size] = 1.0
        input_data = input_data + torch.randn_like(input_data) * 0.01
        input_data.requires_grad = True
        optimizer.zero_grad()
        clear_aux_losses_tracker()
        # MoE forward
        moe_out, _ = moe_layer(input_data)
        logits = moe_out[:, :, :num_classes]  # (batch, seq, num_classes)
        # 交叉熵损失
        loss = F.cross_entropy(logits.view(-1, num_classes), label.view(-1))
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss = tracker['load_balancing_loss']['values'][0]
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()
        aux_loss_list.append(float(aux_loss))
        # 统计准确率
        pred = logits.argmax(dim=-1)
        acc = (pred == label).float().mean().item()
        acc_list.append(acc)
        # 负载均衡分析
        with torch.no_grad():
            router = moe_layer.router
            scores, routing_map = router(input_data)
            tokens_per_expert = routing_map.sum(dim=0)
            load_balance_ratio = max(tokens_per_expert).item() / (min(tokens_per_expert).item() + 1)
            load_balance_ratio_list.append(float(load_balance_ratio))
            # 只记录当前rank负责的专家的token分配
            local_tokens = tokens_per_expert.cpu().numpy()[container.local_expert_indices]
            expert_token_history.append(local_tokens.tolist())
        if step % 20 == 0 or step == steps - 1:
            print(f"Step {step+1:03d}: acc={acc:.3f}, aux_loss={float(aux_loss):.4f}, load_balance_ratio={load_balance_ratio:.2f}")
            # 打印当前rank的专家分配情况
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                ep_rank = parallel_state.get_expert_model_parallel_rank()
                # 只显示当前rank负责的专家的token分配
                local_tokens = tokens_per_expert.cpu().numpy()[container.local_expert_indices]
                print(f"  [Rank {rank}] 专家 {container.local_expert_indices} 分配了 {local_tokens.tolist()} 个tokens")
            else:
                print(f"  专家分配情况: {tokens_per_expert.cpu().numpy().tolist()}")
     
    # 趋势图
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))
    # 1. aux_loss
    axs[0].plot(aux_loss_list, marker='o', label='aux_loss', color='blue')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Aux Loss')
    axs[0].set_title(f'Aux Loss Trend (TP={tp_size}, EP={ep_size}, CP={cp_size})')
    axs[0].legend()
    axs[0].grid(True)
    # 2. load_balance_ratio
    axs[1].plot(load_balance_ratio_list, marker='s', label='load_balance_ratio', color='red')
    axs[1].axhline(y=1.0, color='green', linestyle='--', label='ideal_ratio=1.0')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Load Balance Ratio')
    axs[1].set_title('Load Balance Ratio Trend')
    axs[1].legend()
    axs[1].grid(True)
    # 3. accuracy
    axs[2].plot(acc_list, marker='^', label='accuracy', color='purple')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Accuracy')
    axs[2].set_title('Accuracy Trend')
    axs[2].legend()
    axs[2].grid(True)
    # 4. 每个专家分配的token数
    expert_token_history_np = np.array(expert_token_history)  # shape: (steps, num_local_experts)
    for i in range(expert_token_history_np.shape[1]):
        expert_idx = container.local_expert_indices[i]
        axs[3].plot(expert_token_history_np[:, i], label=f'expert_{expert_idx}')
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Tokens per Expert')
    axs[3].set_title(f'Tokens Assigned to Local Experts {container.local_expert_indices} per Step')
    axs[3].legend()
    axs[3].grid(True)
    plt.tight_layout()
    plt.savefig(f'aux_loss_parallel_trend_tp{tp_size}_ep{ep_size}_cp{cp_size}_{int(time.time())}.png')
    plt.close()
    # 断言收敛性
    first10 = np.mean(aux_loss_list[:10])
    last10 = np.mean(aux_loss_list[-10:])
    print(f"Aux loss first10 mean: {first10:.4f}, last10 mean: {last10:.4f}")
    print(f"Load balance ratio last10 mean: {np.mean(load_balance_ratio_list[-10:]):.4f}")
    print(f"Accuracy last10 mean: {np.mean(acc_list[-10:]):.4f}")
    
    # 打印最终的专家分布统计
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        
        # 每个rank都打印自己的最终统计
        print(f"\n[Rank {rank}] === Rank {rank} 最终统计 (TP={tp_size}, EP={ep_size}, CP={cp_size}) ===")
        print(f"[Rank {rank}] 当前Rank: {rank}, EP Rank: {ep_rank}")
        print(f"[Rank {rank}] 本地专家数: {container.num_local_experts}")
        print(f"[Rank {rank}] 本地专家索引: {container.local_expert_indices}")
        
        # 只在rank 0打印全局汇总
        if rank == 0:
            print(f"\n[Rank 0] === 全局专家分布汇总 ===")
            print(f"[Rank 0] 总专家数: {container.config.num_moe_experts}")
            print(f"[Rank 0] EP并行大小: {ep_size}")
            print(f"[Rank 0] 每张卡专家数: {container.num_local_experts}")
            print(f"[Rank 0] 专家分布: {[f'Rank {i}: {container.num_local_experts} experts' for i in range(world_size)]}")
            print("[Rank 0] " + "=" * 50)
    
    # 断言aux loss收敛（可根据实际情况调整阈值）
    assert last10 < first10, "Aux loss 没有收敛"
    # 断言负载均衡比合理
    assert np.mean(load_balance_ratio_list[-10:]) < 1.7, "负载均衡比过高"
    # 断言准确率提升
    assert np.mean(acc_list[-10:]) > 0.7, "准确率未提升"
    Utils.destroy_model_parallel()