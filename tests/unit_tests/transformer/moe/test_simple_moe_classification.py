import torch
import time
import matplotlib.pyplot as plt
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
)
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer
from tests.unit_tests.test_utilities import Utils
import torch.nn.functional as F
import numpy as np


def test_moe_direct_classification():
    """直接使用MoE层输出进行分类，不使用额外的分类头"""
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，将在CPU上运行")
        device = "cpu"
    else:
        print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        device = "cuda"
    
    # 初始化单卡MoE容器
    container = MoEModelTestContainer(
        tp_size=1,
        ep_size=1,
        pp_size=1,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=0.3,
    )
    moe_layer = container.moe_layer.to(device)
    batch_size = 64
    seq_len = 16
    hidden_size = moe_layer.config.hidden_size
    steps = 300
    
    # 记录结果
    aux_loss_list = []
    load_balance_ratio_list = []
    acc_list = []
    expert_token_history = []
    main_loss_list = []
    total_loss_list = []
    
    # 只优化MoE层参数
    optimizer = torch.optim.SGD(list(moe_layer.parameters()), lr=3e-1)
    
    print("开始测试：直接使用MoE输出进行分类")
    
    for step in range(steps):
        # 构造均衡的输入和标签
        per_class = batch_size * seq_len // 8
        values = []
        labels = []
        for c in range(8):
            v = torch.randint(c*10+1, (c+1)*10+1, (per_class,), device=device)
            values.append(v)
            labels.append(torch.full((per_class,), c, device=device, dtype=torch.long))
        values = torch.cat(values)[:batch_size*seq_len]
        labels = torch.cat(labels)[:batch_size*seq_len]
        idx = torch.randperm(batch_size*seq_len, device=device)
        values = values[idx]
        labels = labels[idx]
        label = labels.view(batch_size, seq_len)

        # 构造输入：使用更复杂的特征表示
        input_data = torch.zeros(batch_size, seq_len, hidden_size, device=device)
        for i in range(batch_size):
            for j in range(seq_len):
                # 使用更丰富的特征表示
                class_id = label[i, j]
                # 在多个维度上设置特征（使用模运算避免越界）
                input_data[i, j, class_id % hidden_size] = 1.0
                input_data[i, j, (class_id + 8) % hidden_size] = 0.5
                input_data[i, j, (class_id + 16) % hidden_size] = 0.3
                # 添加一些噪声
                input_data[i, j, :] += torch.randn(hidden_size, device=device) * 0.1
        
        input_data.requires_grad = True
        
        clear_aux_losses_tracker()
        optimizer.zero_grad()
        
        # MoE forward
        moe_out, _ = moe_layer(input_data)  # (batch, seq, hidden_size)
        
        # 直接使用MoE输出进行分类
        # 方法1：使用MoE输出的前8个维度作为logits
        logits = moe_out[:, :, :8] # (batch, seq, 8)
        
        # 计算loss
        loss = F.cross_entropy(logits.view(-1, 8), label.view(-1))
        
        # 获取aux_loss
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss = tracker['load_balancing_loss']['values'][0]
        
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()
        
        # 统计准确率
        pred = logits.argmax(dim=-1)
        acc = (pred == label).float().mean().item()
        acc_list.append(acc)
        aux_loss_list.append(float(aux_loss))
        main_loss_list.append(float(loss))
        total_loss_list.append(float(total_loss))
        
        # 输出关键信息
        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step+1}: acc={acc:.3f}, loss={float(loss):.4f}, aux_loss={float(aux_loss):.4f}")
            print(f"预测分布: {pred.view(-1)[:10].tolist()}")
            print(f"真实标签: {label.view(-1)[:10].tolist()}")
            print(f"MoE输出: mean={moe_out.mean().item():.6f}, std={moe_out.std().item():.6f}")
            print(f"Logits: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
        
        # 负载均衡分析
        with torch.no_grad():
            router = moe_layer.router
            scores, routing_map = router(input_data)
            tokens_per_expert = routing_map.sum(dim=0)
            load_balance_ratio = max(tokens_per_expert).item() / (min(tokens_per_expert).item() + 1e-4)
            load_balance_ratio_list.append(float(load_balance_ratio))
            expert_token_history.append(tokens_per_expert.cpu().numpy().tolist())
        
      
    
    # 画趋势图
    fig, axs = plt.subplots(6, 1, figsize=(12, 30))
    
    # 单独绘制aux_loss
    axs[0].plot(aux_loss_list, marker='o', label='aux_loss', color='blue', linewidth=2)
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Auxiliary Loss')
    axs[0].set_title('Auxiliary Loss Trend (Direct MoE Classification)')
    axs[0].legend()
    axs[0].grid(True)
    
    # 单独绘制main_loss
    axs[1].plot(main_loss_list, marker='x', label='main_loss', color='orange', linewidth=2)
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Main Loss')
    axs[1].set_title('Main Loss Trend (Direct MoE Classification)')
    axs[1].legend()
    axs[1].grid(True)
    
    # 单独绘制total_loss
    axs[2].plot(total_loss_list, marker='s', label='total_loss', color='green', linewidth=2)
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Total Loss')
    axs[2].set_title('Total Loss Trend (Direct MoE Classification)')
    axs[2].legend()
    axs[2].grid(True)
    
    axs[3].plot(load_balance_ratio_list, marker='s', label='load_balance_ratio', color='red')
    axs[3].axhline(y=1.0, color='green', linestyle='--', label='ideal_ratio=1.0')
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('Load Balance Ratio')
    axs[3].set_title('Load Balance Ratio Trend (理想值=1.0)')
    axs[3].legend()
    axs[3].grid(True)
    
    axs[4].plot(acc_list, marker='^', label='accuracy', color='purple')
    axs[4].set_xlabel('Step')
    axs[4].set_ylabel('Accuracy')
    axs[4].set_title('Classification Accuracy Trend')
    axs[4].legend()
    axs[4].grid(True)
    
    # 专家分配token趋势
    expert_token_history = np.array(expert_token_history)
    for i in range(expert_token_history.shape[1]):
        axs[5].plot(expert_token_history[:, i], label=f'expert_{i}')
    axs[5].set_xlabel('Step')
    axs[5].set_ylabel('Tokens per Expert')
    axs[5].set_title('Tokens Assigned to Each Expert per Step')
    axs[5].legend()
    axs[5].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'moe_direct_classification_trend_{int(time.time())}.png')
    plt.close()


if __name__ == "__main__":
    print("="*50)
    print("测试: 直接使用MoE输出进行分类")
    print("="*50)
    test_moe_direct_classification()