import torch

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.parallel_state import initialize_model_parallel, get_expert_model_parallel_rank
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import MoESyncFreeElasticExpertDispatcher
from megatron.training.initialize import _set_random_seed

torch.distributed.init_process_group(backend="nccl")
initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=4,
    expert_tensor_parallel_size=1,
)

rank_id = torch.distributed.get_rank()
torch.cuda.set_device(rank_id)
model_comm_pgs = get_default_model_comm_pgs()

num_experts = 8
num_echo_experts = 8
HIDDEN_SIZE = 1024
MoE_FFN_HIDDEN_SIZE = 32
TOPK = 2

dispatch_config = TransformerConfig(
    num_layers=56,
    num_moe_experts=num_experts,
    moe_enable_echo=True,
    moe_num_echo_experts=num_echo_experts,
    # Hidden size
    hidden_size=HIDDEN_SIZE,
    moe_ffn_hidden_size=MoE_FFN_HIDDEN_SIZE,
    # Routing
    moe_router_topk=TOPK,
    moe_aux_loss_coeff=1e-2,
    moe_permute_fusion=False,
    # Training
    moe_token_dispatcher_type="flex",
    moe_enable_hybridep=True,
    # Attention
    num_attention_heads=48,
    num_query_groups=8,
    # Parallelism size
    expert_model_parallel_size=4,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    # Architecture specifics
    add_bias_linear=False,
    gated_linear_unit=True,  # swiglu in yaml
    moe_router_dtype='fp32',
    gradient_accumulation_fusion=False,
    moe_received_token_capacity=4.0,
    moe_router_pre_softmax=True,
)

def generate_fc1_expert_weights():
    num_local_experts = num_experts // model_comm_pgs.ep.size()
    ep_rank = get_expert_model_parallel_rank()
    expert_weights = []
    for expert_id in range(ep_rank*num_local_experts, (ep_rank + 1)*num_local_experts):
        expert_weight = torch.ones(MoE_FFN_HIDDEN_SIZE*2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")*(expert_id+1)*10000
        expert_weight = expert_weight + torch.arange(expert_weight.numel(), device="cuda").reshape(expert_weight.shape)
        expert_weights.append(expert_weight.ravel())
    expert_weights = torch.stack(expert_weights, dim=0).contiguous()
    return expert_weights

def generate_fc2_expert_weights():
    num_local_experts = num_experts // model_comm_pgs.ep.size()
    ep_rank = get_expert_model_parallel_rank()
    expert_weights = []
    for expert_id in range(ep_rank*num_local_experts, (ep_rank + 1)*num_local_experts):
        expert_weight = -torch.ones(HIDDEN_SIZE, MoE_FFN_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")*(expert_id+1)*10000
        expert_weight = expert_weight + torch.arange(expert_weight.numel(), device="cuda").reshape(expert_weight.shape)
        expert_weights.append(expert_weight.ravel())
    expert_weights = torch.stack(expert_weights, dim=0).contiguous()
    return expert_weights

def generate_expert_routing_map(num_home_experts, num_echo_experts):
    """
    Generate a routing map where each echo_expert is assigned to one home_expert.
    
    Args:
        num_home_experts: Number of home experts
        num_echo_experts: Number of echo experts
    
    Returns:
        routing_map: A tensor of shape (num_home_experts, num_echo_experts) where 
                     routing_map[i, j] = 1 if echo_expert j is assigned to home_expert i
    """
    # Initialize routing map with zeros
    routing_map = torch.zeros(num_home_experts, num_echo_experts)
    
    torch.manual_seed(42)
    for echo_idx in range(num_echo_experts):
        home_idx = torch.randint(0, num_home_experts, (1,)).item()
        routing_map[home_idx, echo_idx] = 1
    
    return routing_map.bool().cuda()

def check_results(expert_weights, metadata):
    pass

expert_dispatcher = MoESyncFreeElasticExpertDispatcher(
    config=dispatch_config,
    model_comm_pgs=model_comm_pgs,
)

routing_map = generate_expert_routing_map(num_experts, num_echo_experts)
if rank_id == 0:
    print(f"global routing_map: {routing_map.int()}")

fc1_expert_weights = generate_fc1_expert_weights()
fc2_expert_weights = generate_fc2_expert_weights()

metadata = expert_dispatcher.preprocess(routing_map)

assert metadata.routing_map.is_contiguous()

torch.distributed.barrier()
print("---------------------------dispatch fc1_expert_weights---------------------------")
dispatched_fc1_expert_weights = expert_dispatcher.expert_dispatch(fc1_expert_weights, metadata)

torch.distributed.barrier()
print("---------------------------dispatch fc2_expert_weights---------------------------")
# dispatched_fc2_expert_weights = expert_dispatcher.expert_dispatch(fc2_expert_weights, metadata)

ep_rank = get_expert_model_parallel_rank()
num_local_echo_experts = num_echo_experts // model_comm_pgs.ep.size()
for local_echo_expert_id in range(num_local_echo_experts):
    echo_expert_id = ep_rank*num_local_echo_experts + local_echo_expert_id
    home_expert_id = routing_map[:, echo_expert_id].int().argmax()
    print(f"echo_expert_id: {echo_expert_id}, home_expert_id: {home_expert_id}, dispatched_fc1_expert_weights: {dispatched_fc1_expert_weights[local_echo_expert_id][:10]}")
