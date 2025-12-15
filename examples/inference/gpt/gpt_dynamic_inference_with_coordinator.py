# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import json
import os
import time
import torch
import torch.distributed as dist
from collections import defaultdict
from tqdm import tqdm
from typing import List
import warnings
import logging

from examples.inference.gpt.gpt_dynamic_inference import (
    add_dynamic_inference_args,
    get_inference_context,
    get_inference_controller,
    get_model,
)
from examples.inference.gpt.utils import (
    Request, 
    build_dynamic_engine_setup_prefix, 
    build_requests,
    add_common_inference_args
)

from megatron.core import parallel_state
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_mamba_inference_state_config_from_model

from megatron.training import get_args, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_args

# pylint: disable=line-too-long

logging.basicConfig(level=logging.INFO, force=True)

async def main(
    engine: DynamicInferenceEngine,
    requests: List[Request],
    port: int,
    sampling_params: SamplingParams | None = None,
):
    if sampling_params is not None:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )
        
    # once you call engine.start_listening_to_data_parallel_coordinator,
    # the engine will start accepting requests from the data parallel coordinator.
    # and processing them in an asyncio coroutine.
    
    await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=port,
        launch_inference_coordinator=True,
    )

    args = get_args()

    # Test suspend/resume intervals.
    if dist.get_rank() == 0 and args.suspend_resume_interval is not None:
        # Since the client doesn't directly call engine.async_step here, we test
        # the suspend-resume system ~4 times.
        suspend_resume_interval = max(1, len(requests) // 4)
        suspend_idxs = set(range(
            suspend_resume_interval,
            len(requests) + 1,
            suspend_resume_interval,
        ))
        resume_idxs = set(
            min(len(requests), i + suspend_resume_interval // 2)
            for i in suspend_idxs
        )
    else:
        suspend_idxs = set()
        resume_idxs = set()

    # Create client and run example.
    if dist.get_rank() == 0:
        client = InferenceClient(port)  # submits requests to the inference coordinator
        await client.start()
        base_arrival_time = time.time_ns() / 10**9
        for request in requests:
            request.time_arrival = request.time_offset + base_arrival_time
        futures = []
        num_requests_total = len(requests)
        num_requests_added = 0
        # logging.info("Waiting for 20 seconds before starting to add requests. This is to mimic an RL style setup..")
        # time.sleep(20)
        while True:
            current_time = time.time_ns() / 10**9
            if args.incoming_requests_per_step is None:
                # Only add requests that have arrived at the current time.
                while num_requests_added < num_requests_total and requests[num_requests_added].time_arrival <= current_time:
                    request = requests[num_requests_added]
                    # These add-request calls will queue up the request on a zmq socket and return
                    # instantaneously. They will return an asyncio future which can be awaited for
                    # request completion.
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    # Test suspend/resume.
                    if num_requests_added in suspend_idxs:
                        client.suspend_engines()
                    if num_requests_added in resume_idxs:
                        client.resume_engines()

            else:
                # Add deterministic number of requests (generally used for debugging).
                for i in range(min(
                    args.incoming_requests_per_step,
                    num_requests_total - num_requests_added
                )):
                    # Change sampling parameters to force different generation lengths.
                    request = requests[num_requests_added]
                    n = request.sampling_params.num_tokens_to_generate
                    request.sampling_params.num_tokens_to_generate = n + i
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    # Test suspend/resume.
                    if num_requests_added in suspend_idxs:
                        client.suspend_engines()
                    if num_requests_added in resume_idxs:
                        client.resume_engines()

            if num_requests_added == num_requests_total:
                break
            # Relinquish control since there are no more requests to add at the moment. This allows the engine to run.
            await asyncio.sleep(0)
        
        # While we wait for the requests to complete, the engine runs in the background.
        results: List[DynamicInferenceRequestRecord] = await asyncio.gather(*futures)

    if dist.get_rank() == 0:
        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}
            throughputs = []

            for record in results:
                req = record.merge()
                result_dict = {
                    "input_prompt": req.prompt,
                    "generated_text": req.generated_text.replace("\n", "\\n"),
                    "generated_tokens": req.generated_tokens,
                    "latency": req.latency,  # InferenceClient populates this field in the returned future.
                }
                if req.sampling_params["return_log_probs"]:
                    result_dict["logprobs"] = req.prompt_log_probs + req.generated_log_probs
                throughput = len(req.generated_tokens) / req.latency
                throughputs.append(throughput)
                json_results[req.request_id] = result_dict
            throughput_dict = {"throughput": throughputs}
            if args.throughput_check_only:
                json_results = throughput_dict
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=4)
        else:
            print("Results:")
            unique_prompt_map = defaultdict(list)
            for record in results:
                req = record.merge()
                unique_prompt_map[req.prompt].append(req)
            for idx, (prompt_text, reqs) in enumerate(unique_prompt_map.items()):
                print(f"%d/%d. prompt '%s' ... [%d] output '%s'." % (
                    idx,
                    len(unique_prompt_map),
                    prompt_text.replace("\n", "\\n"),
                    len(reqs),
                    reqs[0].generated_text.replace("\n", "\\n"),
                ))

        # kill the engines and suspend the client
        # Right now, we can only call stop when all requests are done. 
        # Todo: Make this explicit in the Client class....
        await client.stop_engines()
        client.stop()

    # once the stop signal eventually makes its way to each GPU, the engines will stop.
    await asyncio.gather(engine.engine_loop_task)
    logging.info(f"Rank: {dist.get_rank()} stopped their engine instance successfully.")


if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp-8 optimizations
    # check for it.
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_dynamic_inference_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        args = get_args()
        tokenizer = get_tokenizer()

        # Sampling params.
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
            termination_id=(
                args.termination_id if args.termination_id is not None else tokenizer.eod
            ),
        )

        # Requests, context, conroller.
        model = get_model()
        mamba_inference_state_config = get_mamba_inference_state_config_from_model(model)
        requests = (
            build_requests(args, tokenizer, sampling_params) if dist.get_rank() == 0 else None
        )

        context = get_inference_context(
            None,
            None,
            calculate_max_sequence_length_from_requests=False,
            mamba_inference_state_config=mamba_inference_state_config,
        )

        controller = get_inference_controller(model, context)

        # Inference engine.
        engine = DynamicInferenceEngine(
            controller,
            context,
            enable_cuda_graph=args.cuda_graph_impl == "local",
            random_seed=args.seed,
            enable_chunked_prefill=not args.disable_chunked_prefill,
            inference_logging_step_interval=args.inference_logging_step_interval,
        )

        if dist.get_rank() == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        asyncio.run(
            main(
                engine,
                requests,
                args.inference_coordinator_port,
            )
        )

        # Stop Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
