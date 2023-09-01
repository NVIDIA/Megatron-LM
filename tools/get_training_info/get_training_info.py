import argparse
import json


def get_args():
    parser = argparse.ArgumentParser(
        description=
        'Offline theoretical analysis of MegatronLM GPTModel memory and communication'
    )

    parser.add_argument('--tensor-model-parallel-size',
                        type=int,
                        default=1,
                        help='Degree of tensor model parallelism.')
    parser.add_argument('--pipeline-model-parallel-size',
                        type=int,
                        default=1,
                        help='Degree of pipeline model parallelism.')
    parser.add_argument('--pipeline-model-parallel-split-rank',
                        type=int,
                        default=None,
                        help='Rank where encoder and decoder should be split.')
    parser.add_argument('--num-layers',
                        type=int,
                        default=None,
                        help='Number of transformer layers.')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=None,
                        help='Transformer hidden size.')
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=None,
                        help='Number of transformer attention heads.')
    parser.add_argument(
        '--micro-batch-size',
        type=int,
        default=None,
        help='Batch size per model instance (local batch size). '
        'Global batch size is local batch size times data '
        'parallel size times number of micro batches.')
    parser.add_argument(
        '--global-batch-size',
        type=int,
        default=None,
        help='Training batch size. If set, it should be a '
        'multiple of micro-batch-size times data-parallel-size. '
        'If this value is None, then '
        'use micro-batch-size * data-parallel-size as the '
        'global batch size. This choice will result in 1 for '
        'number of micro-batches.')
    parser.add_argument('--seq-length',
                        type=int,
                        default=None,
                        help='Maximum sequence length to process.')
    parser.add_argument('--max-position-embeddings',
                        type=int,
                        default=None,
                        help='Maximum number of position embeddings to use. '
                        'This is the size of position embedding.')
    parser.add_argument('--vocab-file',
                        type=str,
                        default=None,
                        help='Path to the vocab file.')
    parser.add_argument('--make_vocab_size_divisible_by',
                        type=int,
                        default=128,
                        help='make_vocab_size_divisible_by')
    parser.add_argument('--nproc_per_node',
                        type=int,
                        default=1,
                        help='nproc per node')
    parser.add_argument('--nnodes', type=int, default=1, help='nnodes')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--bf16',
                        action='store_true',
                        help='Run model in bfloat16 mode.')
    parser.add_argument('--use-flash-attn',
                        action='store_true',
                        help='use flash attention')
    parser.add_argument('--use-distributed-optimizer',
                        action='store_true',
                        help='use distributed optimizer')
    parser.add_argument(
        '--swiglu',
        action='store_true',
        help=
        'Use gated linear units and SiLU activation instead of default gelu')

    args, unknown = parser.parse_known_args()
    args.world_size = args.nproc_per_node * args.nnodes
    global nbytes
    nbytes = 2 if args.fp16 or args.bf16 else 4

    assert args.use_flash_attn, "Currently, only the enabling of use_flash_attn is supported."
    assert args.use_distributed_optimizer, "Currently, only the enabling of use_distributed_optimizer is supported."
    assert args.swiglu, "Currently, only the enabling of swiglu is supported."

    return args


def memory_giga_bytes(num, d=1):
    global nbytes
    return round(nbytes * num / pow(1024, 3), d)


def memory_mega_bytes(num=None, pad=15, suffix=""):
    global nbytes
    tmp = round(nbytes * num / pow(1024, 2), 1)
    return (str(tmp) + suffix).ljust(pad)


def space(t_str=None, pad=25):
    if t_str:
        return t_str.ljust(pad)
    else:
        return " " * pad


def get_shape_str(w=[], b=[], pad=25, prefix=True):
    w_str = ",".join(map(str, w))
    b_str = ",".join(map(str, b))
    shape_str = ""

    if w:
        shape_str += "w=[" if prefix else "["
        shape_str += w_str + "]"
    if b:
        shape_str += ",b=[" + b_str + "]"

    return shape_str.ljust(pad)


def get_dist_group_info(args):
    tp_groups, pp_groups, dp_groups = [], [], []

    num_tensor_model_parallel_groups: int = args.world_size // args.tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = args.world_size // args.pipeline_model_parallel_size

    def get_groups(ranks):
        nodes = [i // args.nproc_per_node for i in ranks]
        gpus = [i % args.nproc_per_node for i in ranks]
        groups = [f"n{n_idx}_g{g_idx}" for n_idx, g_idx in zip(nodes, gpus)]
        return groups

    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * args.tensor_model_parallel_size,
                      (i + 1) * args.tensor_model_parallel_size)
        tp_groups.append(get_groups(ranks))
        
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, args.world_size, num_pipeline_model_parallel_groups)
        pp_groups.append(get_groups(ranks))

    for i in range(args.pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(args.tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank,
                          args.tensor_model_parallel_size)
            dp_groups.append(get_groups(ranks))

    return tp_groups, pp_groups, dp_groups


def get_padded_vocab_size(args):
    vocab_size = len(json.load(open(args.vocab_file)))
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (vocab_size % multiple) != 0:
        vocab_size += 1
    return vocab_size


def get_ffn_hidden_size(args):
    return int((4 * args.hidden_size * 2 / 3) / 64) * 64


def get_megatron_info(disable_profile=False):
    args = get_args()
    h = args.hidden_size
    t = args.tensor_model_parallel_size
    n = args.num_layers
    p = args.pipeline_model_parallel_size
    d = args.world_size // p // t
    v = get_padded_vocab_size(args)
    s = args.seq_length
    b = args.micro_batch_size
    a = args.num_attention_heads
    ga = args.global_batch_size // args.micro_batch_size // d
    f = get_ffn_hidden_size(args)
    per_stage_layer_num = n // p

    # parameters
    total_parameters = v * h + s * h + (7 * h + 4 * h * h + 3 * f * h +
                                        2 * f) * n
    total_parameters_formatted = f'{total_parameters:,}'

    total_parameters_per_gpu = v * h / t + s * h + (
        7 * h + 4 * h * h / t + 3 * f * h / t + 2 * f) * per_stage_layer_num
    total_parameters_per_gpu_formatted = f'{int(total_parameters_per_gpu):,}'

    activations = n * (10 * s * h * b +
                       4 * s * b * f) + 1.5 * s * b * h + 4.5 * s * b * v
    activations_per_gpu = (per_stage_layer_num * (
        (5 * s * h * b + 4 * s * b * f) / t + 5 * s * h * b) +
                           1.5 * s * b * h) * p

    # communication
    pp_comm_count = 0 if p == 1 else args.world_size * ga * 4
    pp_comm_size = pp_comm_count * s * b * h

    tp_comm_count = 0 if t == 1 else args.world_size * 4 * per_stage_layer_num * ga
    tp_comm_size = tp_comm_count * s * b * h

    dp_comm_count = 0 if d == 1 else 2
    dp_comm_size = total_parameters_per_gpu * 4 if args.bf16 else total_parameters_per_gpu * 2

    total_comm_count = pp_comm_count + tp_comm_count + dp_comm_count
    total_comm_size = pp_comm_size + tp_comm_size + dp_comm_size

    # peak memory
    if args.bf16:
        loss_logits_mem = 5 * s * b * v / t if p == 1 else 0
        peak_mem = max(
            memory_giga_bytes(total_parameters_per_gpu * (1 + 2 + 2 / d + 2)),
            memory_giga_bytes(total_parameters_per_gpu * (1 + 2 + 6 / d) +
                              activations_per_gpu + loss_logits_mem))
        gradient = total_parameters * 2
        gradient_per_gpu = total_parameters_per_gpu * 2
        optimizer = total_parameters * 6
        optimizer_per_gpu = total_parameters_per_gpu * 6 / d
    elif args.fp16:
        loss_logits_mem = 2 * s * b * v / t if p == 1 else 0
        peak_mem = max(
            memory_giga_bytes(total_parameters_per_gpu * (1 + 1 + 2 / d + 2)),
            memory_giga_bytes(total_parameters_per_gpu * (1 + 1 + 8 / d) +
                              activations_per_gpu + loss_logits_mem))
        gradient = total_parameters
        gradient_per_gpu = total_parameters_per_gpu
        optimizer = total_parameters * 8
        optimizer_per_gpu = total_parameters_per_gpu * 8 / d
    else:  ##fp32
        loss_logits_mem = 2 * s * b * v / t if p == 1 else 0
        peak_mem = max(
            memory_giga_bytes(total_parameters_per_gpu * (1 + 1 + 1 / d + 1)),
            memory_giga_bytes(total_parameters_per_gpu * (1 + 1 + 2 / d) +
                              activations_per_gpu + loss_logits_mem))
        gradient = total_parameters
        gradient_per_gpu = total_parameters_per_gpu
        optimizer = total_parameters * 2
        optimizer_per_gpu = total_parameters_per_gpu * 2 / d

    print(f"""
***Full Model without Parallel***
===========================================================================================================
Layer                                      Param.(shape)           Param.(Mem. MB)  Act.(Mem. MB)        
----------------------------------------------------------------------------------------------------------
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                           {space()}\t{space(pad=15)}\t{memory_mega_bytes(1.5*s*b*h)}\t
│    │    └─word_embeddings                {get_shape_str([v,h])}\t{memory_mega_bytes(v*h)}\t  
│    │    └─position_embeddings            {get_shape_str([s,h])}\t{memory_mega_bytes(s*h)}\t
│    └─ParallelTransformer: X {n}(layer_num){space(pad=40)}\t{memory_mega_bytes(10*b*s*h+4*s*b*f,suffix="/layer")}\t
│    │    └─input_layernorm                {get_shape_str([h],[h])}\t{memory_mega_bytes(h+b)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─self_attention                 {space()}\t{space(pad=15)}\t{memory_mega_bytes(6*s*b*h)}\t
│    │    |     └─query_key_value          {get_shape_str([3*h,h],[h])}\t{memory_mega_bytes(3*h*h+h)}\t
│    │    |     └─rearrange                {space()}\t{space(pad=15)}\t{memory_mega_bytes(3*s*b*h)}\t
│    │    |     └─core_attention_flash     {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h)}\t
│    │    |     └─rearrange                {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h)}\t
│    │    |     └─dense                    {get_shape_str([h,h],[h])}\t{memory_mega_bytes(h*h+h)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─post_attention_layernorm       {get_shape_str([h],[h])}\t{memory_mega_bytes(h+h)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─mlp                            {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h+4*s*b*f)}\t
│    │    |     └─dense_h_to_4h            {get_shape_str([2*f,h],[2*f])}\t{memory_mega_bytes(2*f*h+2*f)}\t
│    │    |     └─bias_glue                {space()}\t{space(pad=15)}\t{memory_mega_bytes(4*s*b*f)}\t
│    │    |     └─dense_4h_to_h            {get_shape_str([h,f],[h])}\t{memory_mega_bytes(h*f+h)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─drop_add_fusion                {space()}\t{space(pad=15)}\t{memory_mega_bytes(1.5*s*b*h)}\t
-----------------------------------------------------------------------------------------------------------
Amount of Parameters: {total_parameters_formatted}  
Parameters: {memory_giga_bytes(total_parameters)}GB
Gradients: {memory_giga_bytes(gradient)}GB
Optimizers(Adam) States: {memory_giga_bytes(optimizer)}GB
Activations: {memory_giga_bytes(activations)}GB
Total memory demand: {memory_giga_bytes(total_parameters+gradient+optimizer+activations)}GB
==============================================================================================================
""")

    def transpose(matrix):
        return [[matrix[j][i] for j in range(len(matrix))]
                for i in range(len(matrix[0]))]

    def matrix_to_string(matrix, prefix=1):
        prefix_str = '│    ' * prefix + '└─['

        return '\n'.join(
            [prefix_str + ' '.join(map(str, row)) + ']' for row in matrix])

    tp_groups, pp_groups, dp_groups = get_dist_group_info(args)

    pp_mapping = transpose(pp_groups)

    def get_pp_left_str():
        ans = ""
        start_idx = per_stage_layer_num
        end_idx = per_stage_layer_num
        stage_idx = 1
        while start_idx < n:
            end_idx = start_idx + per_stage_layer_num - 1 if start_idx + per_stage_layer_num <= n else n - 1
            ans += f"""│    └─Stage{(stage_idx)}: ParallelTransformerLayer_Index{start_idx}-{end_idx}
│    │    └─stage_device_mappings 
│    │    │      └─[{' '.join(pp_mapping[stage_idx])}]
"""
            start_idx = end_idx + 1
            stage_idx += 1
        return ans

    print(f"""
***Cluster Communication Summary***
==============================
Pipeline Parallelism
│    └─frequency/iteration: {int(pp_comm_count)}
│    └─volume/iteration: {memory_giga_bytes(pp_comm_size)} GB
Data Parallelism
│    └─frequency/iteration: {int(dp_comm_count)}
│    └─volume/iteration: {memory_giga_bytes(dp_comm_size)} GB
Tensor Parallelism
│    └─frequency/iteration: {int(tp_comm_count)}
│    └─volume/iteration: {memory_giga_bytes(tp_comm_size)} GB
All Communication
│    └─frequency/iteration: {int(total_comm_count)}
│    └─volume/iteration: {memory_giga_bytes(total_comm_size)} GB
==============================
""")

    print(f"""
***Memory demand on each GPU in the cluster***
==============================
Amount of Parameters: {total_parameters_per_gpu_formatted}  
Parameters: {memory_giga_bytes(total_parameters_per_gpu)}GB
Gradients: {memory_giga_bytes(gradient_per_gpu)}GB
Optimizers(Adam) States: {memory_giga_bytes(optimizer_per_gpu)}GB
Activations: {memory_giga_bytes(activations_per_gpu)}GB
Memory Requirement: {peak_mem}GB
==============================
""")

    if p != 1:
        print(f"""
***Pipeline Parallel Communications***
========================================================================================
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                                              
│    │    └─word_embeddings                
│    │    └─position_embeddings            
│    └─Stage0: ParallelTransformerLayer_Index{0}-{per_stage_layer_num-1}
│    │    └─stage_device_mappings
│    │    │      └─[{' '.join(pp_mapping[0])}]
│    │    └─each single communication on each gpu
│    │    │    └─shape: {get_shape_str([b,s,h],prefix=False)}  
│    │    │    └─volume: {memory_mega_bytes(s*b*h,suffix="MB")}
│    │    │    └─func: isend, irecv
│    │    │    └─location: between stage in forward and backward process
│    │    └─each iteration communication on each gpu
│    │    │    └─frequency: {ga*4} (num_gradient_accumulation_steps * 4)
│    │    │    └─volume: {memory_mega_bytes(s*b*h*ga*4,suffix="MB")}
{get_pp_left_str()}
----------------------------------------------------------------------------------------
{len(pp_groups)} Pipeline Parallel Communication Groups:
{matrix_to_string(pp_groups)}
All Communication of Cluster in Pipeline Parallelism
│    └─frequency/iteration: {args.world_size*ga*4}
│    └─volume/iteration: {memory_giga_bytes(args.world_size*ga*4*s*b*h)}GB
========================================================================================
""")
    if d != 1:
        print(f"""
***Data Parallel Communications***
========================================================================================
GPTModel                                                         
├─each iteration                
│    └─synchronize_gradient                                         
│    │    └─{len(dp_groups)} Data Parallel Groups 
{matrix_to_string(dp_groups,3)}
│    │    └─communication 
│    │    │    └─volume: {memory_giga_bytes(dp_comm_size/2)}GB
│    │    │    └─func: reduce_scatter (using DistributedOptimizer) 
│    │    └─frequency/iteration: 1
│    │    └─location: after forward_and_backward_compute * {ga} times/iteration 
│    └─gather_model_param (using DistributedOptimizer)                                          
│    │    └─{len(dp_groups)} Data Parallel Groups 
{matrix_to_string(dp_groups,3)}
│    │    └─communication on each gpu
│    │    │    └─volume: {memory_giga_bytes(dp_comm_size/2)}GB
│    │    │    └─func: all_gather
│    │    └─frequency/iteration: 1
│    │    └─location: after optimizer.iteration
----------------------------------------------------------------------------------------
All Communication of Cluster in Data Parallelism
│    └─frequency/iteration: 2
│    └─volume/iteration: {memory_giga_bytes(dp_comm_size)}GB
========================================================================================
""")
    if t != 1:
        print(f"""
***Tensor Parallel Communications***
=================================================================================================================================================================================================================
Layer                                      Param(shape)           Param(Mem. MB)  Activations(Mem. MB)   TP_Fw.(Comm. Shape)  TP_Fw.(Comm. Mem. MB)   TP_Bw.(Comm. Shape)  TP_Bw.(Comm. Mem. MB)   TP(Comm. func)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
GPTModel                                                         
├─TransformerLanguageModel                 
│    └─Embedding                           {space()}\t{space(pad=15)}\t{memory_mega_bytes(1.5*s*b*h)}\t                    
│    │    └─word_embeddings                {get_shape_str([v//t],[h])}\t{memory_mega_bytes(v*h)}\t{space(pad=15)}\t{get_shape_str([b,s,h],prefix=False)}\t{memory_mega_bytes(b*s*h)}\t{space()}\t{space(pad=15)}\t{space("allreduce")}\t
│    │    └─position_embeddings            {get_shape_str([s],[h])}\t{memory_mega_bytes(s*h)}\t
│    └─ParallelTransformer: X {per_stage_layer_num}(layer_num){space(pad=40)}\t{memory_mega_bytes(10*b*s*h+4*s*b*f,suffix="/layer")}\t
│    │    └─input_layernorm                {get_shape_str([h],[h])}\t{memory_mega_bytes(h+b)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─self_attention                 {space()}\t{space(pad=15)}\t{memory_mega_bytes(6*s*b*h)}\t
│    │    |     └─query_key_value          {get_shape_str([3*h//t,h],[h])}\t{memory_mega_bytes(3*h*h/t+h)}\t{space(pad=15)}\t{space()}\t{space(pad=15)}\t{get_shape_str([b,s,h],prefix=False)}\t{memory_mega_bytes(b*s*h)}\t{space("allreduce")}\t
│    │    |     └─rearrange                {space()}\t{space(pad=15)}\t{memory_mega_bytes(3*s*b*h/t)}\t
│    │    |     └─core_attention_flash     {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h/t)}\t
│    │    |     └─rearrange                {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h/t)}\t
│    │    |     └─dense                    {get_shape_str([h//t,h],[h])}\t{memory_mega_bytes(h*h/t+h)}\t{memory_mega_bytes(s*b*h)}\t{get_shape_str([b,s,h],prefix=False)}\t{memory_mega_bytes(b*s*h)}\t{space()}\t{space(pad=15)}\t{space("allreduce")}\t
│    │    └─post_attention_layernorm       {get_shape_str([h],[h])}\t{memory_mega_bytes(h+h)}\t{memory_mega_bytes(s*b*h)}\t
│    │    └─mlp                            {space()}\t{space(pad=15)}\t{memory_mega_bytes(s*b*h+4*s*b*f)}\t
│    │    |     └─dense_h_to_4h            {get_shape_str([2*f//t,h],[2*f//t])}\t{memory_mega_bytes(2*f/t*h+2*f/t)}\t{space(pad=15)}\t{space()}\t{space(pad=15)}\t{get_shape_str([b,s,h],prefix=False)}\t{memory_mega_bytes(b*s*h)}\t{space("allreduce")}\t
│    │    |     └─bias_glue                {space()}\t{space(pad=15)}\t{memory_mega_bytes(4*s*b*f)}\t
│    │    |     └─dense_4h_to_h            {get_shape_str([h,f//t],[h])}\t{memory_mega_bytes(h*f+h)}\t{memory_mega_bytes(s*b*h)}\t{get_shape_str([b,s,h],prefix=False)}\t{memory_mega_bytes(b*s*h)}\t{space()}\t{space(pad=15)}\t{space("allreduce")}\t
│    │    └─drop_add_fusion                {space()}\t{space(pad=15)}\t{memory_mega_bytes(1.5*s*b*h)}\t
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
{len(tp_groups)} Tensor Parallel Communication Groups:
{matrix_to_string(tp_groups)}
Communication in Tensor Parallel
│    └─each gpu:
│    │    └─each micro_batch:
│    │    │    └─frequency: {4*per_stage_layer_num}
│    │    │    └─volume: {memory_giga_bytes(4*s*b*h*per_stage_layer_num)}GB
│    │    │    └─each transformer:
│    │    │    │    └─frequency: 2(forward)+2(backward)=4
│    │    │    │    └─volume: {memory_giga_bytes(4*s*b*h,2)}GB
│    │    └─each iteration:
│    │    │    └─frequency: {4*per_stage_layer_num*ga}
│    │    │    └─volume: {memory_giga_bytes(ga*4*s*b*h*per_stage_layer_num)}GB
│    └─cluster:
│    │    └─each micro_batch:
│    │    │    └─frequency: {args.world_size*4*per_stage_layer_num}
│    │    │    └─volume: {memory_giga_bytes(args.world_size*4*s*b*h*per_stage_layer_num)}GB
│    │    └─each iteration:
│    │    │    └─frequency: {args.world_size*4*per_stage_layer_num*ga}
│    │    │    └─volume: {memory_giga_bytes(args.world_size*ga*4*s*b*h*per_stage_layer_num)}GB
=======================================================================================================================================================================================================================
""")


get_megatron_info()
