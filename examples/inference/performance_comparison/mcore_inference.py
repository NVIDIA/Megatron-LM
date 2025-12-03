import io
import os
import sys
import torch
from functools import partial
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
import megatron
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.module import MegatronModule
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from megatron.training import get_args, get_model as _get_model, initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider
from gpt_builders import gpt_builder
from megatron.core.utils import configure_nvtx_profiling
import json
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider
from gpt_builders import gpt_builder
torch.serialization.add_safe_globals([io.BytesIO])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunState])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunDiagnostic])

def print_cuda_memory_usage(stage: str = ""):
    """Print CUDA memory usage statistics."""
    if not torch.cuda.is_available():
        return
    
    i = 0
    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
    max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB
    max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)    # GB
    
    print(f"[{stage}] GPU {i} Memory:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Max Reserved:  {max_reserved:.2f} GB")

def get_model() -> MegatronModule:
    """Initialize model and load checkpoint."""
    args = get_args()
    model_builder = gpt_builder
    # Build model.
    model = _get_model(
        partial(model_provider, model_builder),
        wrap_with_ddp=False
    )
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=False,
    )
    model = model[0]
    # Eval mode.
    model.eval()
    return model
@torch.inference_mode()
def main():
    # Initialize Megatron.
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )


    args = get_args()
    tokenizer = build_tokenizer(args)
    # Sampling params.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        skip_prompt_log_probs=False,
        return_log_probs=True,
        num_tokens_total=512,
        num_tokens_to_generate=None,
        termination_id=tokenizer.eod,
    )
    model = get_model()
    print_cuda_memory_usage("After Model Load")
    
    # Inference context.
    context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers // args.pipeline_model_parallel_size,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=512,
        num_cuda_graphs=16,
        block_size_tokens=args.inference_dynamic_batching_block_size,
        active_buffer_size_gb=10,
        #max_tokens=512, # Setting this throws an assertion errro
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        materialize_only_last_token_logits=False,
        cache_mla_latent=args.multi_latent_attention and args.cache_mla_latents,
        kv_lora_rank=args.kv_lora_rank if args.multi_latent_attention else None,
        qk_pos_emb_head_dim=args.qk_pos_emb_head_dim,
        use_cuda_graphs_for_non_decode_steps=not args.decode_only_cuda_graphs,
        use_flashinfer_fused_rope=None,
        unified_memory_level=args.inference_dynamic_batching_unified_memory_level
    )
    # Wrap model in inference wrapper.
    model = GPTInferenceWrapper(model, args, context)
    model.model_is_pipeline_parallel = False
    # Text generation controller.
    controller = TextGenerationController(model, tokenizer)
    # Inference engine.
    dynamic_engine = DynamicInferenceEngine(
        controller,
        context,
        enable_cuda_graph=True,
        random_seed=args.seed,
        track_paused_request_events=args.inference_dynamic_batching_track_paused_request_events,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        inference_logging_step_interval=args.inference_wandb_logging_step_interval,
    )
    print_cuda_memory_usage("After Engine Initialization")
    
    # Use the same prompt lengths as vLLM inference for fair comparison
    prompt_lengths = [143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 143, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]
    vocab_size = 50000
    max_prompt_len = max(prompt_lengths)
    prompt_tokens_tensor = torch.randint(low=0, high=vocab_size, size=(512, max_prompt_len))
    request_id = 0
    for p, prompt_len in zip(
        prompt_tokens_tensor, prompt_lengths, strict=True
    ):
        dynamic_engine.add_request(
            request_id,
            p[:prompt_len].cuda(),
            sampling_params=sampling_params,
        )
        request_id += 1
    # Start Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()
    
    configure_nvtx_profiling(True)
    
    # Reset peak memory stats before inference
    torch.cuda.reset_peak_memory_stats()
    print_cuda_memory_usage("Before Inference")
    
    results = []
    import time
    start_time = time.perf_counter()
    while dynamic_engine.has_unfinished_requests():
            result = dynamic_engine.step_modern(verbose=False)
            finished_request_records = result["finished_request_records"]
            for finished_request_record in finished_request_records:
                finished_request = finished_request_record.merge(dynamic_engine.controller.tokenizer)
                results.append(finished_request)
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()
    
    print_cuda_memory_usage("After Inference")
    
    # Calculate and print metrics matching vLLM script
    print("-" * 80)
    print(f"Total time: {latency:.2f} seconds")
    print(f"Throughput: {len(results) / latency:.2f} requests/sec")
    
    # Calculate token statistics
    total_prompt_tokens = sum(len(result.prompt_tokens) for result in results)
    total_generated_tokens = sum(len(result.generated_tokens) for result in results)
    total_tokens = total_prompt_tokens + total_generated_tokens

  
    
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Token throughput: {total_tokens / latency:.2f} tokens/sec")
    print(f"Generation throughput: {total_generated_tokens / latency:.2f} tokens/sec")
    
    # Print peak memory usage
    if torch.cuda.is_available():
        i = 0
        peak_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i} Peak Memory - Allocated: {peak_allocated:.2f} GB, Reserved: {peak_reserved:.2f} GB")
    
    print("-" * 80)

if __name__ == "__main__":
    main()
