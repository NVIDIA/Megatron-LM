from contextlib import nullcontext
import torch

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
# from batch_invariant_ops import set_batch_invariant_mode
from megatron.core.fp8_utils import get_fp8_context

"""
Test for MoE Layer with Echo Experts functionality.

This test compares the output and gradients of:
1. MoE Layer with echo experts enabled (using offloading planner)
2. Baseline MoE Layer without echo experts

The echo expert implementation redistributes overflow tokens from overloaded
experts to spare/echo experts for better load balancing.
"""


def run_echo_test(
    use_group_gemm=True,
    fp8=False,
    dtype=torch.bfloat16,
    ep_size=4,
    num_tokens=1024,
    hidden_size=4096,
    moe_ffn_hidden_size=2048,
    num_experts=8,
    num_echo_experts=8,
    topk=1,
    random_seed=1234,
    token_dispatcher_type="alltoall",
    moe_received_token_capacity=None,
    moe_echo_enable_random_offloading=False,
    moe_echo_expert_dispatcher_type="hybridep",
    verbose=True
):
    """
    Run echo experts test with configurable parameters.
    
    Args:
        ep_size: Expert model parallel size
        num_tokens: Number of tokens in input
        hidden_size: Hidden dimension size
        moe_ffn_hidden_size: MoE FFN hidden size
        num_experts: Total number of experts
        num_echo_experts: Number of echo experts
        topk: Top-k routing parameter
        use_group_gemm: Whether to use grouped GEMM
        fp8: Whether to use FP8 precision
        dtype: Data type for computations (torch.float32, torch.float64, torch.bfloat16)
        random_seed: Random seed for reproducibility
        token_dispatcher_type: Type of token dispatcher ("alltoall" or "flex")
        moe_received_token_capacity: Received token capacity multiplier
        verbose: Whether to print detailed output
    
    Returns:
        dict: Test results including outputs, gradients, and success flags
    """
    if token_dispatcher_type == "flex":
        assert dtype == torch.bfloat16, "Flex token dispatcher only supports bfloat16 precision"
    rank = torch.distributed.get_rank()
    
    num_local_experts = num_experts // ep_size
    num_local_echo_experts = num_echo_experts // ep_size
    
    if verbose and rank == 0:
        border = "=" * 80
        print("\n" + border)
        print(f"{'Echo Test Parameters':^80}")
        print("-" * 80)
        rows = [
            ("EP_SIZE", ep_size),
            ("NUM_TOKENS", num_tokens),
            ("HIDDEN_SIZE", hidden_size),
            ("NUM_EXPERTS", num_experts),
            ("NUM_ECHO_EXPERTS", num_echo_experts),
            ("TOPK", topk),
            ("USE_GROUP_GEMM", use_group_gemm),
            ("FP8", fp8),
            ("DTYPE", dtype),
            ("TOKEN_DISPATCHER", token_dispatcher_type),
            ("ECHO_EXPERT_DISPATCHER", moe_echo_expert_dispatcher_type),
            ("RANDOM_OFFLOADING", moe_echo_enable_random_offloading),
            ("CAPACITY", moe_received_token_capacity),
            ('RANDOM_OFFLOADING', moe_echo_enable_random_offloading),
        ]
        key_width = max(len(k) for k, _ in rows)
        for k, v in rows:
            print(f"  {k:<{key_width}} : {v}")
        print(border + "\n")
    
    echo_config = TransformerConfig(
        num_layers=56,
        num_moe_experts=num_experts,
        moe_enable_echo=True,
        moe_num_echo_experts=num_echo_experts,
        # Hidden size
        hidden_size=hidden_size,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        # Routing
        moe_router_topk=topk,
        moe_aux_loss_coeff=0,
        moe_permute_fusion=False,
        # Training
        moe_token_dispatcher_type=token_dispatcher_type,
        moe_flex_dispatcher_backend="hybridep",
        # Attention
        num_attention_heads=48,
        num_query_groups=8,
        # Parallelism size
        expert_model_parallel_size=ep_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        moe_echo_expert_dispatcher_type=moe_echo_expert_dispatcher_type,
        moe_echo_enable_random_offloading=moe_echo_enable_random_offloading,
        # Architecture specifics
        add_bias_linear=False,
        gated_linear_unit=True,  # swiglu in yaml
        moe_router_dtype='fp32',
        gradient_accumulation_fusion=False,
        moe_received_token_capacity=moe_received_token_capacity,
        moe_router_pre_softmax=True,
        fp8="e4m3" if fp8 else None,
        fp8_recipe="blockwise" if fp8 else None,
    )

    baseline_config = TransformerConfig(
        num_layers=56,
        num_moe_experts=num_experts,
        # Hidden size
        hidden_size=hidden_size,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        # Routing
        moe_router_topk=topk,
        moe_aux_loss_coeff=0,
        moe_permute_fusion=False,
        # Training
        moe_token_dispatcher_type=token_dispatcher_type,
        # Attention
        num_attention_heads=48,
        num_query_groups=8,
        # Parallelism size
        expert_model_parallel_size=ep_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        # Architecture specifics
        add_bias_linear=False,
        gated_linear_unit=True,  # swiglu in yaml
        moe_router_dtype='fp32',
        gradient_accumulation_fusion=False,
        moe_router_pre_softmax=True,
        fp8="e4m3" if fp8 else None,
        fp8_recipe="blockwise" if fp8 else None,
    )

    
    spec = get_moe_module_spec(
        use_te=use_group_gemm, num_experts=160, moe_grouped_gemm=use_group_gemm, moe_use_legacy_grouped_gemm=False
    )

    _set_random_seed(random_seed)
    moe_layer = MoELayer(echo_config, spec.submodules, layer_number=1).to(dtype)

    baseline_moe_layer = MoELayer(baseline_config, spec.submodules, layer_number=1).to(dtype)

    # Copy weights from echo layer to baseline layer for fair comparison
    baseline_moe_layer.router.load_state_dict(moe_layer.router.state_dict())
    if use_group_gemm:
        for i in range(num_local_experts):
            getattr(baseline_moe_layer.experts.linear_fc1, f"weight{i}").data.copy_(getattr(moe_layer.experts.linear_fc1, f"weight{i}").data)
            getattr(baseline_moe_layer.experts.linear_fc2, f"weight{i}").data.copy_(getattr(moe_layer.experts.linear_fc2, f"weight{i}").data)
    else:
        for i in range(num_local_experts):
            baseline_moe_layer.experts.local_experts[i].linear_fc1.weight.data.copy_(moe_layer.experts.local_experts[i].linear_fc1.weight.data)
            baseline_moe_layer.experts.local_experts[i].linear_fc2.weight.data.copy_(moe_layer.experts.local_experts[i].linear_fc2.weight.data)
    
    hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    output_grad = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)

    def echo_forward_backward(hidden_states):
        """Forward and backward pass using the MoELayer with echo experts enabled.

        The MoELayer now handles all the echo expert logic internally when
        moe_enable_echo=True is set in the configuration.
        """
        context = get_fp8_context(echo_config) if fp8 else nullcontext()
        with context:
            output, mlp_bias = moe_layer.forward(hidden_states)
            # Perform backward pass
            output.backward(gradient=output_grad)

        # Collect gradients for comparison
        if use_group_gemm:
            grads = (
                [x.grad for x in moe_layer.experts.linear_fc1.parameters()]
                + [x.grad for x in moe_layer.experts.linear_fc2.parameters()]
                + [x.grad for x in moe_layer.router.parameters()]
            )
        else:
            grads = (
                [x.grad for x in moe_layer.experts.local_experts.parameters()]
                + [x.grad for x in moe_layer.router.parameters()]
            )

        return output, grads


    def baseline_forward_backward(hidden_states):
        """Forward and backward pass using the baseline MoELayer without echo experts."""
        context = get_fp8_context(baseline_config) if fp8 else nullcontext()
        with context:
            output, _ = baseline_moe_layer.forward(hidden_states)
            output.backward(gradient=output_grad)
        if use_group_gemm:
            grads = (
                [x.grad for x in baseline_moe_layer.experts.linear_fc1.parameters()]
                + [x.grad for x in baseline_moe_layer.experts.linear_fc2.parameters()]
                + [x.grad for x in baseline_moe_layer.router.parameters()]
            )
        else:
            grads = (
                [x.grad for x in baseline_moe_layer.experts.local_experts.parameters()]
                + [x.grad for x in baseline_moe_layer.router.parameters()]
            )

        return output, grads

    
    # Run tests: Compare echo expert implementation vs baseline
    batch_invariant_flag = (dtype == torch.bfloat16 or dtype == torch.float32)
    # with set_batch_invariant_mode(batch_invariant_flag):
    baseline_output, baseline_grads = baseline_forward_backward(hidden_states)
    torch.cuda.synchronize()
    if verbose:
        print(f"[Rank {rank}] Baseline forward backward finished")
    output, grads = echo_forward_backward(hidden_states)
    torch.cuda.synchronize()
    if verbose:
        print(f"[Rank {rank}] Echo forward backward finished")

    # Verify that outputs and gradients match between echo and baseline implementations
    output_match = True
    grad_match = True
    error_msg = ""
    
    try:
        if topk == 1 and batch_invariant_flag:
            assert torch.equal(output, baseline_output), f"Output mismatch: num of mismatch: {torch.sum(output != baseline_output)}, max abs diff: {torch.max(torch.abs(output - baseline_output))}"
        else:
            torch.testing.assert_close(output, baseline_output)
        if verbose:
            print(f"✅ [Rank {rank}] Outputs match between echo and baseline implementations")
    except AssertionError as e:
        output_match = False
        error_msg += f"Output mismatch: {e}\n"
        if verbose:
            print(f"❌ [Rank {rank}] Output mismatch: {e}")

    try:
        for grad, baseline_grad in zip(grads, baseline_grads):
            torch.testing.assert_close(baseline_grad, grad)
        if verbose:
            print(f"✅ [Rank {rank}] Gradients match between echo and baseline implementations")
    except AssertionError as e:
        grad_match = False
        error_msg += f"Gradients mismatch: {e}\n"
        if verbose:
            print(f"❌ [Rank {rank}] Gradients mismatch: {e}")

    # Return test results
    return {
        "output": output,
        "baseline_output": baseline_output,
        "grads": grads,
        "baseline_grads": baseline_grads,
        "output_match": output_match,
        "grad_match": grad_match,
        "success": output_match and grad_match,
        "error_msg": error_msg if error_msg else None,
    }

def print_rank0(message):
    torch.distributed.barrier()
    torch.cuda.synchronize()
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(message)


if __name__ == "__main__":
    # Initialize distributed processing
    torch.distributed.init_process_group(backend="nccl")
    
    # Set up parallelism
    EP_SIZE = 4
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
    )
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    # Basic setting
    kargs = {
        "num_tokens": 1024,
        "hidden_size": 4096,
        "moe_ffn_hidden_size": 2048,
        "num_experts": 8,
        "num_echo_experts": 8,
        "ep_size": EP_SIZE,
    }
    
    # Test 1: Pure PyTorch implementation with fp64 precision
    print_rank0(f"\n🔥###############################Running Test 1: Pure PyTorch implementation with fp64 precision#####################\n")
    for topk in [1, 2]:
        run_echo_test(
            use_group_gemm=False,
            fp8=False,
            topk=topk,
            dtype=torch.float64,
            token_dispatcher_type="alltoall",
            moe_echo_expert_dispatcher_type="alltoall",
            moe_echo_enable_random_offloading=False,
            moe_received_token_capacity=None,
            **kargs,
        )

    print_rank0(f"\n🔥###############################Running Test 2: + Random Offloading ###############################\n")
    for topk in [1, 2]:
        run_echo_test(
            use_group_gemm=False,
            fp8=False,
            topk=topk,
            dtype=torch.float64,
            token_dispatcher_type="alltoall",
            moe_echo_expert_dispatcher_type="alltoall",
            moe_echo_enable_random_offloading=True,
            moe_received_token_capacity=None,
            **kargs,
        )

    # print_rank0(f"\n🔥###############################Running Test 3: + FP32 Precision ###############################\n")
    # for topk in [1, 2]:
    #     for random_offloading in [True, False]:
    #         run_echo_test(
    #             use_group_gemm=False,
    #             fp8=False,
    #             topk=topk,
    #             dtype=torch.float32,
    #             token_dispatcher_type="alltoall",
    #             moe_echo_expert_dispatcher_type="alltoall",
    #             moe_echo_enable_random_offloading=random_offloading,
    #             moe_received_token_capacity=None,
    #             **kargs,
    #         )
