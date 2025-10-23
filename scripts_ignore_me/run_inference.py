# lawrence mcafee

# ~~~~~~~~ command ~~~~~~~~
# # For debugging.
# RUN apt-get update && apt-get install -y gdb strace && apt-get install -y vim

# torchrun --nproc_per_node=1 -m ...
# python -m scripts_ignore_me.run_inference -e mcore-dynamic -m 357m -c

# ~~~~~~~~ import ~~~~~~~~
import argparse
import datetime
import os
import subprocess
import sys
import time
import torch
from tqdm import tqdm

try:
    from vllm import EngineArgs, LLM, LLMEngine, SamplingParams
    from examples.inference.gpt.utils import (
        build_requests as _build_requests,
        get_curr_time,
    )
except Exception as e:
    pass

# USER_DIR = "/lustre/fs11/portfolios/adlr/users/lmcafee"
USER_DIR = "/lustre/fsw/portfolios/adlr/users/lmcafee"

from lutil import pax

get_curr_time = time.time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_requests_for_vllm(args, tokenizer):

    # ~~~~~~~~ args ~~~~~~~~
    args.seed = 42
    args.prompts = os.getenv("PROMPTS", None)
    if not args.prompts:
        args.num_tokens_to_prompt = list(map(int, os.environ["NUM_TOKENS_TO_PROMPT"].split()))
        args.num_tokens_to_generate = int(os.environ["NUM_TOKENS_TO_GENERATE"])
        args.incoming_requests_duration = float(os.environ["INCOMING_REQUESTS_DURATION"])
        args.incoming_requests_per_sec = float(os.environ["INCOMING_REQUESTS_PER_SEC"])

    requests = _build_requests(args, tokenizer)

    return requests

def run_vllm_static(args):

    assert args.model == "12b"
    model_key = "mistralai/Mistral-Nemo-Instruct-2407"

    llm = LLM(model=model_key)

    # >>>
    # pax("llm", {"tokenizer": llm.llm_engine.tokenizer})
    # <<<

    requests = build_requests_for_vllm(args, llm.llm_engine.tokenizer.tokenizer)
    prompts = [ r.prompt_text for r in requests ]
    sampling_params = SamplingParams(max_tokens=args.num_tokens_to_generate)

    t = get_curr_time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = get_curr_time() - t

    for request, output in zip(requests, outputs):
        request.output_text = output.outputs[0].text.replace("\n", "\\n")
        request.output_tokens = output.outputs[0].token_ids

    # pax("prompts, outputs", {
    #     "outputs / 0" : outputs[0],
    #     "outputs / 0 / outputs" : outputs[0].outputs,
    #     "outputs / 0 / outputs / 0" : outputs[0].outputs[0],
    #     "requests / 0" : requests[0],
    #     "requests / -1" : requests[-1],
    # })

    return requests, {
        "step_time" : total_time,
        "add_time" : 0.,
        "output_time" : 0.,
        "total_time" : total_time,
    }

def run_vllm_dynamic(args):

    # ~~~~~~~~ engine ~~~~~~~~
    assert args.model == "12b"
    model_key = "mistralai/Mistral-Nemo-Instruct-2407"
    # model_key = "facebook/opt-125m"

    engine_args = EngineArgs()
    engine_args.model = model_key
    engine_args.tokenizer = model_key

    engine = LLMEngine.from_engine_args(engine_args)

    # >>>
    pax("engine", {
        "vllm_config" : engine.vllm_config,
        "compilation_config" : engine.vllm_config.compilation_config,
        "cudagraph_capture_sizes" : engine.vllm_config.compilation_config.cudagraph_capture_sizes,
    })
    # <<<

    # ~~~~~~~~ requests, sampling params ~~~~~~~~
    requests = build_requests_for_vllm(args, engine.tokenizer.tokenizer)

    sampling_params = SamplingParams(max_tokens=args.num_tokens_to_generate)
    
    start_time = get_curr_time()
    for r in requests:
        r.time_arrival = start_time + r.time_offset

    # ~~~~~~~~ generation loop ~~~~~~~~
    num_requests_total = len(requests)
    num_requests_added = 0
    total_add_time = 0
    total_step_time = 0
    total_output_time = 0
    test_time_start = get_curr_time()
    tbar = tqdm(total=num_requests_total, desc="vllm.add_request()")
    while True:
        curr_time = get_curr_time()

        # ~~~~~~~~ add request ~~~~~~~~
        t = get_curr_time()
        while num_requests_added < num_requests_total and \
              curr_time >= requests[num_requests_added].time_arrival:
            engine.add_request(str(num_requests_added),
                               requests[num_requests_added].prompt_text,
                               sampling_params)
            requests[num_requests_added].start_time = get_curr_time()
            requests[num_requests_added].state = "started"
            num_requests_added += 1
            tbar.update(1)
        total_add_time += get_curr_time() - t

        # ~~~~~~~~ step ~~~~~~~~
        t = get_curr_time()
        request_outputs = engine.step()
        total_step_time += get_curr_time() - t

        # ~~~~~~~~ finished requests ~~~~~~~~
        t = get_curr_time()
        for request_output in request_outputs:
            if request_output.finished:
                req_id  = int(request_output.request_id)
                req = requests[req_id]
                req.time_end = get_curr_time()
                req.output_tokens = request_output.outputs[0].token_ids
                req.output_text = request_output.outputs[0].text
                req.state = "finished"
        total_output_time += get_curr_time() - t

        # ~~~~~~~~ break ~~~~~~~~
        if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
            break

    total_time = get_curr_time() - test_time_start

    return requests, {
        "step_time" : total_step_time,
        "add_time" : total_add_time,
        "output_time" : total_output_time,
        "total_time" : total_time,
    }

def run_vllm(args):

    # ~~~~~~~~ start nsight ~~~~~~~~
    if os.environ.get("NSIGHT_PREFIX"):
        raise Exception("hi.")
        torch.cuda.cudart().cudaProfilerStart()

    requests, result = globals()[f"run_{args.engine.replace('-', '_')}"](args)

    # ~~~~~~~~ outputs ~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if torch.distributed.get_rank() == 0:
        from collections import defaultdict
        unique_prompt_map = defaultdict(list)
        for request_idx, request in enumerate(requests):
            unique_prompt_map[request.prompt_text].append(request_idx)

        for unique_idx, (prompt, request_idxs) in enumerate(unique_prompt_map.items()):
            print(f"{unique_idx}/{len(unique_prompt_map)} [{len(request_idxs)}]. {prompt} ... %s" % requests[request_idxs[0]].output_text.replace("\n", "\\n"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # ~~~~~~~~ timing ~~~~~~~~
    print("~~~")
    print("%s | %s | <auto prompts, (%s), %d, %.1e, %.1e, n%d> ... time: step %.3f, add %.3f, out %.3f, total %.3f" % (
        args.model,
        args.engine,
        ",".join(map(str, args.num_tokens_to_prompt)),
        args.num_tokens_to_generate,
        args.incoming_requests_duration,
        args.incoming_requests_per_sec,
        len(requests),
        result["step_time"],
        result["add_time"],
        result["output_time"],
        result["total_time"],
    ))
    print("~~~")

    # ~~~~~~~~ stop nsight ~~~~~~~~
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":

    # ~~~~~~~~ args ~~~~~~~~
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True,
                        # choices=["357m", "12b"])
                        choices=["357m", "3b", "12b"])
    parser.add_argument("--engine", "-e", required=True,
                        choices=[
                            # "mcore-static",
                            "mcore-dynamic",
                            # "vllm-static",
                            # "vllm-dynamic",
                        ])
    # parser.add_argument("--repo", "-r", default="lazy-uvm-compile")
    parser.add_argument("-r", "--repo")
    # parser.add_argument("-n", "--nsight", action="store_true")
    # parser.add_argument("-b", "--launch-block", action="store_true")
    parser.add_argument("-g", "--num-graphs", type=int, default=0)
    parser.add_argument("-u", "--unified", type=int, default=0)
    args = parser.parse_args()

    if args.repo is None:
        REPO_DIR = os.getcwd()
        args.repo = os.path.basename(REPO_DIR)
        # pax("REPO_DIR", {"args.repo": args.repo})
    else:
        REPO_DIR = f"{USER_DIR}/inference/megatrons/{args.repo}"

    # ~~~~~~~~ model ~~~~~~~~
    BASE_CKPT_DIR = f"{USER_DIR}/checkpoints"
    # >>>
    # if args.model == "357m":
    if args.model in ("357m", "3b"):
    # <<<
        os.environ["CHECKPOINT_DIR"] = f"{BASE_CKPT_DIR}/357m/core-local-tp1-pp1"
        os.environ["VOCAB_FILE"] = f"{BASE_CKPT_DIR}/357m/vocab/gpt2-vocab.json"
        os.environ["MERGE_FILE"] = f"{BASE_CKPT_DIR}/357m/vocab/gpt2-merges.txt"

    elif args.model == "12b":
        os.environ["CHECKPOINT_DIR"] = f"{BASE_CKPT_DIR}/12b/core-local-tp1-pp1"
        os.environ["TOKENIZER_MODEL"] = f"{BASE_CKPT_DIR}/12b/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"

    else:
        raise Exception(f"specialize for model '{args.model}'.")

    # ~~~~~~~~ env ~~~~~~~~
    if 0:
        os.environ["NUM_TOKENS_TO_PROMPT"] = "4 7" # "4 32"
        os.environ["NUM_TOKENS_TO_GENERATE"] = "16"
        os.environ["INCOMING_REQUESTS_DURATION"] = str(args.duration)
        os.environ["INCOMING_REQUESTS_PER_SEC"] = "100." # 100
    else:
        os.environ["PROMPTS"] = " ".join(f'"{p}"' for p in (
            "Lawrence would like to",
            "NVIDIA is best at",
            "The inventor of the GPU is",
            "Michigan is best known for",
            "All I want for Christmas is",
        ))
        os.environ["NUM_TOKENS_TO_GENERATE"] = "64" # *16

    os.environ["BUFFER_SIZE_GB"] = "2." # *40
    os.environ["BUFFER_OVERFLOW_FACTOR"] = "1." # *1.
    os.environ["BUFFER_GUARANTEED_FRACTION"] = "0.1" # *0.05, 0.1
    os.environ["ENGINE"] = args.engine

    # >>>
    os.environ["EXTRA_ARGS"] = ""
    # os.environ["ENABLE_CUDA_GRAPHS"] = str(int(args.num_graphs != 0))
    # if args.launch_block:
    #     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # <<<

    # ~~~~~~~~ nsight ~~~~~~~~
    # if args.nsight:
    #     date_str = datetime.date.today().strftime("%Y%m%d")
    #     os.environ["NSIGHT_PREFIX"] = f"{REPO_DIR}/scripts/nsight/{date_str}/{args.engine}-{args.model}"
    #     os.makedirs(os.path.dirname(os.environ["NSIGHT_PREFIX"]), exist_ok=True)

    # ~~~~~~~~ run ~~~~~~~~
    if "mcore" in args.engine:

        if "dynamic" in args.engine:
            # >>>
            # pax({"env": {k:v for k,v in dict(os.environ).items() if "NCCL" in k}})
            del os.environ["NCCL_DEBUG"]
            # <<<
            os.environ["EXTRA_ARGS"] += " --inference-ckpt-non-strict"
            os.environ["EXTRA_ARGS"] += f" --inference-dynamic-batching-unified-memory-level {args.unified}"
            os.environ["NUM_CUDA_GRAPHS"] = str(args.num_graphs)
        if "static" in args.engine:
            os.environ["EXTRA_ARGS"] = "--inference-max-requests 64"

        os.environ["ENGINE"] = args.engine.replace("mcore-", "")
        subprocess.run([
            "bash",
            f"examples/inference/gpt/gpt_dynamic_inference_{args.model}.sh",
        ], cwd=REPO_DIR)

    elif "vllm" in args.engine:

        if args.nsight:
            sub_argv = [ a for a in sys.argv if a not in ("-n", "--nsight") ]
            sub_argv = [
                "nsys",
                "profile",
                "-s",
                "none",
                "-t",
                "nvtx,cuda",
                "--cudabacktrace=all",
                "--cuda-graph-trace=node",
                "--python-backtrace=cuda",
                "--wait",
                "all",
                "--force-overwrite",
                "true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "-o", os.environ["NSIGHT_PREFIX"],
                "python",
                *sub_argv,
            ]
            subprocess.run(sub_argv)
        else:
            run_vllm(args)

    else:
        raise Exception("specialize for engine '%s'." % args.engine)

# eof
