# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import json
import os 
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import List

from examples.inference.gpt.gpt_dynamic_inference import (
    get_model,
    get_inference_context,
    get_inference_controller,
    add_dynamic_inference_args,
)
from examples.inference.gpt.utils import (
    add_common_inference_args,
    build_requests,
    build_dynamic_engine_setup_prefix,
    Request,
)
from megatron.core import parallel_state
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.training import get_args, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_args

async def main(engine: DynamicInferenceEngine, requests: List[Request], sampling_params: SamplingParams, port: int):
    # once you call engine.start_listening_to_data_parallel_coordinator,
    # the engine will start accepting requests from the data parallel coordinator.
    # and processing them in an asyncio coroutine. 
    await engine.start_listening_to_data_parallel_coordinator(sampling_params, 
                                                inference_coordinator_port=port,
                                                launch_inference_coordinator=True)   
    # if you want to use your own inference coordinator - 
    # 1. set launch_inference_coordinator to False
    # 2. setup a router socket at tcp://MASTER_ADDR:PORT
    # 3. wait for data parallel groups to establish connection (BasicInferenceCoordinator.__init__)
    # 4. look at InferenceCoordinator.start() to see how we can route requests from users <-> data parallel groups
    #   based on headers. 
    # 5. look at InferenceClient to see how we create requests with headers. 
    # >>>
    from lutil import pax
    # <<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    args = get_args()
    # test_suspend_resume = args.suspend_resume_interval is not None
    # if test_suspend_resume:
    if args.suspend_resume_interval is not None:
        # Since the client doesn't directly call engine.async_step here, we test
        # the suspend-resume system ~4 times.
        suspend_resume_interval = max(1, len(requests) // 4)
        # suspend_idxs = set(range(
        #     suspend_resume_interval,
        #     len(requests) + 1,
        #     suspend_resume_interval,
        # ))
        # resume_idxs = set(
        #     min(len(requests), i + suspend_resume_interval // 2)
        #     for i in suspend_idxs
        # )
        # pax("suspend_idxs, resume_idxs")
    else:
        suspend_resume_interval = None
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # >>>
    # n_iters = 0
    # pax("test_suspend_resume")
    # pax("requests", {
    #     "num_tokens_to_generate" : args.num_tokens_to_generate,
    #     "suspend_resume_interval" : args.suspend_resume_interval,
    # })
    # <<<
    if dist.get_rank() == 0: 
        client = InferenceClient(port) # submits requests to the inference coordinator
        await client.start()
        base_arrival_time = time.time_ns() / 10**9
        for request in requests:
            request.time_arrival = request.time_offset + base_arrival_time
        futures = []
        num_requests_total = len(requests)
        num_requests_added = 0
        #tbar = tqdm(total=num_requests_total)
        while True:
            # >>>
            # n_iters += 1
            # <<<
            current_time = time.time_ns() / 10**9
            # Only add requests that have arrived at the current time.
            while num_requests_added < num_requests_total and requests[num_requests_added].time_arrival <= current_time:
                request = requests[num_requests_added]
                # These add-request calls will queue up the request on a zmq socket and return
                # instantaneously. They will return an asyncio future which can be awaited for
                # request completion.
                futures.append(client.add_request(request.prompt_text, 
                                                        sampling_params))
                num_requests_added += 1
                # >>>
                # if num_requests_added % args.suspend_resume_interval == 0:
                #     pax({"suspend_resume_interval": args.suspend_resume_interval}, "num_requests_added")

                if suspend_resume_interval is not None:
                    if num_requests_added % suspend_resume_interval == 0:
                        print("++++++ suspend ... r %d / %d, s %d." % (
                            num_requests_added,
                            num_requests_total,
                            suspend_resume_interval,
                        ))
                        client.pause_engines()
                    # if (num_requests_added - suspend_resume_interval // 2) % suspend_resume_interval == 0:
                    if (
                        (
                            num_requests_added >= suspend_resume_interval
                            and
                            (num_requests_added - suspend_resume_interval // 2) % suspend_resume_interval == 0
                        )
                        or
                        num_requests_added == num_requests_total
                    ):
                        print("------ resume ... r %d / %d, s %d." % (
                            num_requests_added,
                            num_requests_total,
                            suspend_resume_interval,
                        ))
                        client.unpause_engines()
                # <<<
                #tbar.update(1)
            if num_requests_added == num_requests_total:
                break
            # Relinquish control since there are no more requests to add at the moment. This allows the engine to run. 
            await asyncio.sleep(0)
        # While we wait for the requests to complete, the engine runs in the background.
        results: List[DynamicInferenceRequest] = await asyncio.gather(*futures)

    # >>>
    # pax("n_iters")
    # <<<

    if dist.get_rank() == 0:
        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}

            for req in results:
                result_dict = {
                    "input_prompt": req.prompt,
                    "generated_text": req.generated_text.replace("\n", "\\n"),
                    "generated_tokens": req.generated_tokens,
                    "latency": req.latency, #InferenceClient populates this field in the returned future.
                }
                if sampling_params.return_log_probs:
                    result_dict["logprobs"] = req.prompt_log_probs + req.generated_log_probs
                json_results[req.request_id] = result_dict
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=4)
        else:
            print("Results:")
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # for req in results:
            #     print(f"rid: {req.request_id}\nprompt: {req.prompt!r}\noutput: {req.generated_text!r}\n\n")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            from collections import defaultdict
            unique_prompt_map = defaultdict(list)
            for req in results:
                unique_prompt_map[req.prompt].append(req)
            for idx, (prompt_text, reqs) in enumerate(unique_prompt_map.items()):
                print(f"%d/%d. prompt '%s' ... [%d] output '%s'." % (
                    idx,
                    len(unique_prompt_map),
                    prompt_text,
                    len(reqs),
                    reqs[0].generated_text.replace("\n", "\\n"),
                ))
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 
        # kill the engines and suspend the client
        client.stop_engines()
        client.stop()
        
    # once the stop signal eventually makes its way to each GPU, the engines will stop.
    await asyncio.gather(engine.engine_loop_task)

if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp-8 optimizations 
    # check for it.
    with torch.inference_mode():
        initialize_megatron(
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
        
        asyncio.run(main(engine, 
                        requests,
                        sampling_params,
                        args.inference_coordinator_port))

