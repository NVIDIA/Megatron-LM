import matplotlib.pyplot as plt
import re
import numpy as np
from tqdm import tqdm
import os
def parse_log_file(log_file_path):
    """
    解析日志文件，提取 iteration 和 lm loss
    """
    iterations = []
    losses = []

    with open(log_file_path, 'r') as file:
        for line in tqdm(file):
            # 使用正则表达式匹配所需的值
            
            match = re.search(r'\[.*?\]\s+iteration\s+(\d+)/\s*\d+\s*\|.*?lm loss:\s*([\d.E+-]+)', line)
            if match:
                # import pdb;pdb.set_trace()
                iteration = int(match.group(1))
                loss = float(match.group(2))
                if iteration > 300:
                    continue
                iterations.append(iteration)
                losses.append(loss)
                # print(f"Extracted: iteration={iteration}, loss={loss}")
            else:
                pass
                # print(f"Failed to match line: {line.strip()}")

    return iterations, losses

def plot_loss_curve(iterations, losses, labels, output_file='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    for i in range(len(iterations)):
        # import pdb ;pdb.set_trace()
        plt.plot(iterations[i], losses[i], label=labels[i], linewidth=0.1)
    plt.xlabel('Iteration')
    plt.ylabel('LM Loss')
    plt.title('LM Loss vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

def avg_rel_error(iterations,losses,max_len=20000):
    min_len = min(len(iteration) for iteration in iterations)
    idx = max(0,min_len-max_len)
    bf16   = np.array(losses[0][idx:min_len],dtype=float)
    res = []
    for i in range(1,len(iterations)):
        quant = np.array(losses[i][idx:min_len],dtype=float)
        rel = np.abs(bf16 - quant) / bf16
        res.append(float(rel.sum() / (min_len-idx)))
    return res


if __name__ == "__main__":
    LOG_PATH='/mtc_afs/charles/Megatron-LM/tensorboard_logs/llama3_8b_fp8'
    log_files_llama3_8b_pretrain=['training_wikipedia_bf16_25-08-05_03-14-01.log','training_wikipedia_25-08-03_22-08-24.log','training_wikipedia_fp8_25-08-03_22-07-19.log','training_wikipedia_fp4_25-08-03_22-05-47.log']
    quant_labels=['bf16','te_fp8','fp8','fp4']
    output_image_path = '/mtc_afs/charles/Megatron-LM/quant/curve/loss_curve_cmp_non_pretrain.png'
    # select mode
    log_files = log_files_llama3_8b_pretrain
    labels = quant_labels
    iterations,losses = [],[]
    for log_file in log_files:
        log_file_path = os.path.join(LOG_PATH,log_file)
        iteration, loss = parse_log_file(log_file_path)
        iterations.append(iteration)
        losses.append(loss)

    plot_loss_curve(iterations, losses, labels, output_image_path)
    loss_res = avg_rel_error(iterations,losses)
    for i in range(1,len(iterations)):
        print(f"{labels[i]}:{loss_res[i-1]}")
    loss_res = avg_rel_error(iterations,losses,500)
    for i in range(1,len(iterations)):
        print(f"{labels[i]} in last 500:{loss_res[i-1]}")
