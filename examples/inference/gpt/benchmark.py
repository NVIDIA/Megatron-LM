from megatron.core.inference.inference_client import InferenceClient
from examples.inference.gpt.utils import add_common_inference_args
import asyncio
import torch.distributed as dist
from examples.inference.gpt.gpt_dynamic_inference import get_model, get_inference_context, get_inference_controller, add_dynamic_inference_args
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.training import initialize_megatron
import torch
import os 
from megatron.training import get_args, get_tokenizer 
from megatron.core.inference.sampling_params import SamplingParams
from examples.inference.gpt.utils import build_requests, build_dynamic_engine_setup_prefix, Request
from megatron.core.inference.engines import DynamicInferenceEngine
import time
from tqdm import tqdm
from typing import List
import json
from megatron.training.arguments import parse_args
from megatron.core import parallel_state

if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp-8 optimizations 
    # check for it.
    with torch.inference_mode():
        initialize_megatron(
            #parsed_args=args
            extra_args_provider=add_dynamic_inference_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        args = get_args()
        tokenizer = get_tokenizer()

        # Sampling params.
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
        )

        # Requests, context, conroller.
        model = get_model()
        requests = build_requests(args, tokenizer) if dist.get_rank() == 0 else None

        
        context = get_inference_context(None, 
                                        None,
                                        calculate_max_sequence_length_from_requests=False)
        
        controller = get_inference_controller(model, context)

        # Inference engine.
        engine = DynamicInferenceEngine(
            controller,
            context,
            termination_id=tokenizer.eod,
            enable_cuda_graph=args.cuda_graph_impl == "local",
            random_seed=args.seed,
            enable_chunked_prefill=not args.disable_chunked_prefill
        )

        
        if dist.get_rank() == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        batch_size = args.inference_dynamic_batching_max_requests_override


        # Warmup
        for _ in range(5):
            context.initialize_attention_state(
                num_warmup_tokens=batch_size,
            )
            input_ids, position_ids = context.current_input_and_position_ids(
                        num_warmup_tokens=batch_size
            )

            # Forward pass -> logits.
            with torch.inference_mode():
                controller.inference_wrapped_model.run_one_forward_step(
                    {
                        "tokens": input_ids,
                        "position_ids": position_ids,
                        "attention_mask": None,
                    }
                )
                context.reset()

        TIMED_ITERS = 10
        st_events = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]
        en_events = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_ITERS)]

        for i in range(TIMED_ITERS):
            context.initialize_attention_state(
                num_warmup_tokens=batch_size,
            )
            input_ids, position_ids = context.current_input_and_position_ids(
                        num_warmup_tokens=batch_size
            )
            st_events[i].record()
            # Forward pass -> logits.
            with torch.inference_mode():
                controller.inference_wrapped_model.run_one_forward_step(
                    {
                        "tokens": input_ids,
                        "position_ids": position_ids,
                        "attention_mask": None,
                    }
                )
                context.reset()
            en_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [st_events[i].elapsed_time(en_events[i]) for i in range(TIMED_ITERS)]
        elapsed_time = sum(elapsed_times) / TIMED_ITERS
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print(f"Overlapped GEMM: = {args.tp_comm_overlap}")
            print(f"Inference Optimized Layers: {args.use_inference_optimized_layers}")
            print(f"Avg latency per forward pass: {elapsed_time:.2f} ms")
            with open("bench_tp.jsonl", "a") as f:
                json.dump({
                    "tp_comm_overlap": args.tp_comm_overlap,
                    "use_inference_optimized_layers": args.use_inference_optimized_layers,
                    "avg_latency_ms": elapsed_time,
                    "batch_size": batch_size,
                }, f)
                f.write("\n")

        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
            
 
        