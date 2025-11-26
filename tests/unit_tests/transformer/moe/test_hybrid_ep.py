import torch

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.parallel_state import initialize_model_parallel, get_expert_model_parallel_rank
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, MoESyncFreeElasticExpertDispatcher
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

total_num_tokens = 512
num_experts = 8
HIDDEN_SIZE = 1024
MoE_FFN_HIDDEN_SIZE = 1024
TOPK = 1

dispatch_config = TransformerConfig(
    num_layers=56,
    num_moe_experts=num_experts,
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
    moe_received_token_capacity=None,
    moe_router_pre_softmax=True,
)

global_tokens = torch.arange(total_num_tokens, device="cuda", dtype=torch.int32).unsqueeze(1).expand(-1, HIDDEN_SIZE).bfloat16()
num_tokens_per_rank = total_num_tokens // model_comm_pgs.ep.size()
local_tokens = global_tokens[num_tokens_per_rank*rank_id:num_tokens_per_rank*(rank_id+1)]

print(f"rank {rank_id} local_tokens: {local_tokens}")

global_routing_map = torch.zeros(total_num_tokens, num_experts, device="cuda")

num_tokens_per_expert = total_num_tokens // num_experts
for i in range(num_experts):
    if i != 3:
        global_routing_map[num_tokens_per_expert*i:num_tokens_per_expert*(i+1), num_experts-i-1] = 1

local_routing_map = global_routing_map[num_tokens_per_rank*rank_id:num_tokens_per_rank*(rank_id+1)].contiguous()

print(f"rank {rank_id} local_routing_map: {local_routing_map}")


num_local_experts = num_experts // model_comm_pgs.ep.size()
local_expert_indices = list(range(num_local_experts*rank_id, num_local_experts*(rank_id+1)))
expert_dispatcher = MoEFlexTokenDispatcher(
    num_local_experts=num_local_experts,
    local_expert_indices=local_expert_indices,
    config=dispatch_config,
    model_comm_pgs=model_comm_pgs,
)

metadata = expert_dispatcher.preprocess(local_routing_map.bool())

hidden_states, _ = expert_dispatcher.dispatch_preprocess(local_tokens, torch.randn_like(local_routing_map), metadata)

dispatched_tokens, dispatched_probs = expert_dispatcher.token_dispatch(hidden_states, None, metadata)
dispatched_tokens, tokens_per_expert, permuted_probs = expert_dispatcher.dispatch_postprocess(dispatched_tokens, dispatched_probs, metadata)

print(f"rank {rank_id} dispatched_tokens: {dispatched_tokens[:, 0]}")
print(f"rank {rank_id} tokens_per_expert: {tokens_per_expert}")
