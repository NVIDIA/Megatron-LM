# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import itertools
import random
import time
import torch
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from typing import Any, List, Optional

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.transformer.module import MegatronModule

from megatron.core.inference.sampling_params import SamplingParams



def add_common_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Common inference arguments."""

    group = parser.add_argument_group(title='Common inference')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--num-tokens-to-prompt",
        type=int,
        nargs="+",
        default=[64, 1024],
        help='Number of tokens to use for simulated prompts. This should be a '
        'space-separated pair of integers, and the generated prompt lengths will '
        'be uniformly sampled within this range.',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--top-n-logprobs",
        type=int,
        default=0,
        help='Return the top n logprobs for the generated tokens and their corresponding token as a dictionary',
    )
    group.add_argument(
        "--incoming-requests-per-step",
        type=int, default=None,
        help="Add a deterministic number of requests per step. This arg is "
        "prioritized over `--incoming-requests-per-sec` below (which is non-"
        "deterministic). Note that the number of requests added per step is "
        "additionally limited by the inference context's `max_requests`, "
        "`max_tokens`, and KV buffer size.",
    )
    group.add_argument(
        "--incoming-requests-per-sec",
        type=float,
        default=100.0,
        help="Simulated number of requests per second. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--incoming-requests-duration",
        type=float,
        default=10.0,
        help="Total amount of time to simulate that requests are "
        "arriving. Multiply this value with "
        "`--incoming-requests-per-sec` to get the approximate "
        "total number of requests. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--model-provider",
        choices=["mamba", "gpt"],
        default="gpt",
        help="Model provider",
    )
    group.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save generations as JSON",
    )
    group.add_argument(
        "--output-every-n-results",
        type=int,
        default=1,
        help="To minimize the output file size of larger runs, only write the "
        "results of every `n` requests.",
    )
    group.add_argument(
        "--prompt-file",
        help='Jsonl file containing input prompts, where each item (i.e., line) '
        'contains the field \'text\' where the value is the prompt. All other '
        'fields within each item are ignored, and may be customized for each '
        'application.',
    )
    group.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        help='Number of samples to use from the loaded prompt file (see '
        '`--prompt-file` above). The first `--prompt-file-num-truncate` samples '
        'will be used, in order.',
    )
    group.add_argument(
        "--inference-coordinator-port",
        type=int,
        help="This port will be used to setup the inference co-ordinator on node-0",
        default=12346
    )
    group.add_argument(
        "--use-flashinfer-fused-rope",
        action='store_true',
        default=False,
        help='Use flashinfer fused rope implementation.',
    )

    return parser


def get_default_sampling_params(termination_id: int = None):
    return SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        return_log_probs=False,
        num_tokens_to_generate=30,
        termination_id = termination_id,
    )

def get_curr_time() -> float:
    """Get synchronized time across ranks."""
    curr_time = torch.cuda.LongTensor([time.time_ns()])
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(curr_time, src=0)
    return curr_time.item() / 10**9


class Request:
    """Class to hold attributes for a single request.

    A request is initialized with its prompt text. As it is added, processed,
    and completed through the inference engine, the request is populated with its
    start time, end time, and output tokens.

    Args:
        prompt_text (str): Prompt text.
        time_offset (float): Artificial time offset for simulating incoming
            requests. This value is later added to the `base_arrival_time` to
            simulate the requests arrival time.
        tokenizer (Any): Tokenizer for tokenizing the prompt.
    """

    def __init__(self, prompt_text: str, time_offset: float, tokenizer: Any, sampling_params: SamplingParams = None):
        self.prompt_text = prompt_text
        self.prompt_tokens = tokenizer.tokenize(prompt_text)
        self.output_text = None
        self.output_tokens = []
        self.time_offset = time_offset
        self.time_arrival = None
        self.time_start = None
        self.time_end = None
        self.state = "not-started"
        self.sampling_params: SamplingParams = sampling_params if sampling_params is not None else get_default_sampling_params(tokenizer.eod)

    def __str__(self) -> str:
        return "state '%s'; toffset %.1e; prompt len %d; output len %d; '%s'" % (
            self.state,
            self.time_offset,
            len(self.prompt_tokens),
            len(self.output_tokens),
            self.prompt_text,
        )


def get_time_offsets(
    seed: int | None,
    incoming_requests_per_step: int,
    incoming_requests_per_sec: float,
    num_requests: int,
) -> list[float]:
    """Get example time offsets."""

    # Time offsets to add all requests at once.
    if incoming_requests_per_step is not None or incoming_requests_per_sec <= 0:
        return [-1] * num_requests

    # if num_requests is not None:
    incoming_requests_duration = num_requests / incoming_requests_per_sec
    incoming_requests_duration *= 2 # extra margin, to accomodate time sampling

    random.seed(seed)
    
    import simpy  # Guard against this import in test case

    # Generate random time offsets.
    def arrival(r):
        while True:
            yield env.timeout(random.expovariate(r))
            time_offsets.append(env.now)

    time_offsets = []
    env = simpy.Environment()
    env.process(arrival(incoming_requests_per_sec))
    env.run(incoming_requests_duration)
    
    # Ensure at least a single request.
    if len(time_offsets) == 0:
        time_offsets = [0.0]

    # Ensure first time is 0.
    time_offsets = [to - time_offsets[0] for to in time_offsets]
    
    # Truncate to num_requests.
    assert len(time_offsets) >= num_requests
    time_offsets = time_offsets[:num_requests]

    return time_offsets


def get_cli_requests(
        args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:

    # Get time offsets.
    t_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        len(args.prompts),
    )

    # Init requests.
    requests = [Request(p, t, tokenizer, sampling_params) for p,t in zip(args.prompts, t_offsets)]
    return requests


def get_synthetic_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get example requests."""

    # Get time offsets.
    time_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        int(args.incoming_requests_per_sec * args.incoming_requests_duration),
    )

    # Init requests.
    requests = [
        Request("hi " * random.randint(*args.num_tokens_to_prompt), t, tokenizer, sampling_params)
        for t in time_offsets
    ]

    return requests


def get_requests_from_file(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get requests from a file."""
    if not args.prompt_file:
        raise ValueError("Prompt file is required to read requests from a file.")

    # Load prompts.
    n_prompts = sum(1 for _ in open(args.prompt_file))
    prompts = []
    with open(args.prompt_file) as f:
        for line in tqdm(f.readlines(), "read prompt file", total=n_prompts):
            prompts.append(json.loads(line)["text"])
            if len(prompts) == args.prompt_file_num_truncate:
                break

    # Get time offsets.
    time_offsets: list[float] = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        len(prompts),
    )

    # Init requests.
    requests = [
        Request(p, t, tokenizer, sampling_params)
        for p, t in tqdm(zip(prompts, time_offsets), "init requests", total=len(prompts))
    ]

    return requests


def build_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    # Check if we have any prompts (from command line or JSONL)
    if args.prompts:
        if args.prompt_file:
            raise ValueError("Cannot use both --prompts and --prompt-file")
        return get_cli_requests(args, tokenizer, sampling_params)
    elif args.prompt_file:
        return get_requests_from_file(args, tokenizer, sampling_params)
    else:
        return get_synthetic_requests(args, tokenizer, sampling_params)


def get_model_size_str(model):
    n = sum(p.numel() for p in model.parameters())
    for exp, suffix in ((12, "t"), (9, "b"), (6, "m"), (3, "k"), (0, "")):
        nquery = int(10**exp)
        if n > nquery:
            return "%d%s" % (n // nquery, suffix)
    raise Exception("something went wrong.")


def build_dynamic_engine_setup_prefix(
    args: Namespace,
    model: MegatronModule,
    context: DynamicInferenceContext,
    requests: list[DynamicInferenceRequest],
):
    """
    Returns a compact, pipe-separated summary of the dynamic-batching setup.

    Example output:

    `dynamic | cg True | prompts: synth(16 256), n 1024, g 512, t 1.0e+02 5.0e-01 | bf 4, 1.2 [r 1024, t 8192] | gtd 0.50 [r 512] | reqs 100` # pylint: disable=line-too-long

    Args:
        args (Namespace): Command-line arguments for this run.
        context (DynamicInferenceContext): Stores limits such as `max_requests`,
            `max_tokens`, and `gtd_request_count`.
        requests (List[DynamicInferenceRequest]): List of inference requests.

    Returns:
        A configuration string for logging.
    """
    # CUDA graph config
    if args.cuda_graph_impl == "local":
        cg_str = (
            f"graphs {context.cuda_graph_token_counts[0]}:"
            f"{context.cuda_graph_token_counts[-1]}"
        )
    else:
        cg_str = "--"

    # Unified memory (UVM).
    uvm_str = f"uvm {int(context.unified_memory_level)}"

    # Prompt description
    prompt_src_str = (
        "cli" if args.prompts else
        "file" if args.prompt_file else
        f"synth({', '.join(map(str, args.num_tokens_to_prompt))})"
    )
    request_str = (
        f"requests: {prompt_src_str}, "
        f"n {len(requests):d}, g {args.num_tokens_to_generate:d}, "
    )
    request_str += (
        f"dur {args.incoming_requests_duration:.1e} "
        f"r/sec {args.incoming_requests_per_sec:.1e}"
        if args.incoming_requests_per_step is None else
        f"r/step {args.incoming_requests_per_step}"
    )

    # Buffer limits config
    flw = args.inference_dynamic_batching_buffer_overflow_factor
    flw_str = "no overflow" if flw is None else f"{flw:.1f}"
    buffer_limits_str = (
        f"bf {args.inference_dynamic_batching_buffer_size_gb:.0f}, {flw_str} "
        f"[r {context.max_requests}, t {context.max_tokens}]"
    )

    # Guaranteed request config
    guaranteed_fraction_str = (
        f"gtd {args.inference_dynamic_batching_buffer_guaranteed_fraction:.2f} "
        f"[r {context.gtd_request_count}]"
    )

    parts = [
        get_model_size_str(model),
        "dynamic",
        cg_str,
        uvm_str,
        request_str,
        buffer_limits_str,
        guaranteed_fraction_str,
    ]

    return " | ".join(parts)
