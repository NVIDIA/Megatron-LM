#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/1 19:16
# @Author : Ethan-yt
import pandas as pd
from megatron import get_args, get_num_microbatches, initialize_megatron


def print_notations():
    print()
    print(
        pd.DataFrame(
            [
                ("a", "Number of microbatches / gradient accumulation steps", a),
                ("b", "Global batchsize", b),
                ("s", "Sequence length", s),
                ("h", "Hidden size", h),
                ("i", "Intermediate size / FFN hidden size", i),
                ("l", "Number of layers", l),
                ("v", "Vocab size", v),
                ("nq", "Number of query attention heads", nq),
                ("nkv", "Number of key/value attention heads", nkv),
                ("d", "Data parallel size", d),
                ("t", "Tensor parallel size", t),
                ("p", "Pipeline parallel size", p),
            ],
            columns=["notation", "description", "value"],
        ).to_string(index=False)
    )


def calculate_model_param():
    print()
    print("Model parameters:")
    df_params = pd.DataFrame(
        [
            # 2 = Wq + Wo
            # 2 * nkv / n = Wk + Wv
            ("Attention", (2 + 2 * (nkv / nq)) * l * h**2),
            ("FFN", 3 * l * h * i),
            # 3 = (h->i) + (h->i) + (i->h)
            ("Layernorm", (2 * l + 1) * h),
            ("Embedding & LM Head", 2 * h * v),
        ],
        columns=["name", "param"],
    )
    total_params = df_params["param"].sum()
    df_params = pd.concat([df_params, pd.DataFrame([("Total", total_params)], columns=["name", "param"])])
    print(df_params.to_string(index=False, float_format=lambda x: f"{int(x):,d}"))
    return total_params


def calculate_flops_per_iteration():
    print()
    print("Total FLOPs per iteration")
    # see https://arxiv.org/pdf/2104.04473.pdf APPENDIX
    df_flops = pd.DataFrame(
        [
            # 2 = Wq + Wo
            # 2 * nkv / n = Wk + Wv
            ("Attention Q, K, V, O Transformations", 6 * (2 + 2 * (nkv / nq)) * b * s * l * h**2),
            # 2 = attention matrix computation + attention over values
            ("Attention Score, Values", 6 * 2 * b * s**2 * l * h),
            # 3 = (h->i) + (h->i) + (i->h)
            ("FFN", 6 * 3 * b * s * l * h * i),
            ("LM Head", 6 * b * s * h * v),
        ],
        columns=["name", "flops"],
    )
    total_flops_per_iteration = df_flops["flops"].sum()
    df_flops = pd.concat([df_flops, pd.DataFrame([("Total", total_flops_per_iteration)], columns=["name", "flops"])])
    print(df_flops.to_string(index=False, float_format=lambda x: f"{int(x):,d}"))
    return total_flops_per_iteration


def extra_args_provider(parser):
    group = parser.add_argument_group(title="calculate_speed")
    group.add_argument(
        "--max-tflops",
        type=float,
        default=430,
        help="Maximum FLOPs per iteration, used to compute the forward/backward time."
        "can be obtained by start a single gpu training and see the log.",
    )

    group.add_argument("--dp-bandwidth", type=float, default=21.25, help="Data parallel bandwidth in GB/s.")

    group.add_argument("--tp-bandwidth", type=float, default=340, help="Tensor parallel bandwidth in GB/s.")

    group.add_argument("--pp-bandwidth", type=float, default=340, help="Pipeline parallel bandwidth in GB/s.")

    return parser


def print_communication_summary():
    print()
    print("Communication per GPU:")
    df_communication_summary = pd.DataFrame(
        [
            ("TP", tp_comm_size / 10**9, tp_comm_count, tp_comm_time),
            ("DP", dp_comm_size / 10**9, dp_comm_count, dp_comm_time),
            ("PP", pp_comm_size / 10**9, pp_comm_count, pp_comm_time),
        ],
        columns=["parallelism", "size(GB)", "count", "time(seconds)"],
    )
    print(df_communication_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_time_breakdown():
    print()
    print("Time breakdown:")
    df_time_breakdown = pd.DataFrame(
        [
            ("Forward / Backward", time_forward_backward),
            ("PP Bubble", time_pp_bubble),
            ("PP Communication", pp_comm_time),
            ("TP Communication", tp_comm_time),
            ("DP Communication", dp_comm_time),
        ],
        columns=["name", "time(seconds)"],
    )
    total_time = df_time_breakdown["time(seconds)"].sum()
    df_time_breakdown = pd.concat(
        [df_time_breakdown, pd.DataFrame([("Total", total_time)], columns=["name", "time(seconds)"])]
    )
    print(df_time_breakdown.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    estimated_flops = total_flops_per_iteration / (t * p * d * total_time)
    print()
    print(f"Estimated FLOPs per second: {estimated_flops / 10**12:.3f} TFLOPs")


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=extra_args_provider)
    args = get_args()
    a = get_num_microbatches()
    b = args.global_batch_size
    d = args.data_parallel_size
    s = args.seq_length
    h = args.hidden_size
    i = args.ffn_hidden_size
    l = args.num_layers
    v = args.padded_vocab_size
    if args.group_query_attention:
        nkv = args.num_query_groups
    else:
        nkv = args.num_attention_heads
    nq = args.num_attention_heads
    t = args.tensor_model_parallel_size
    p = args.pipeline_model_parallel_size
    flops_max = args.max_tflops * 1e12
    dp_bandwidth = args.dp_bandwidth * 1e9
    tp_bandwidth = args.tp_bandwidth * 1e9
    pp_bandwidth = args.pp_bandwidth * 1e9
    print_notations()
    params = calculate_model_param()
    total_flops_per_iteration = calculate_flops_per_iteration()
    time_forward_backward = total_flops_per_iteration / (t * p * d * flops_max)
    time_pp_bubble = time_forward_backward * (p - 1) / a

    # communication per gpu per iteration
    pp_comm_size = 2 * b / d * s * h * int(p > 1)
    # 4 = 2(send, recv) * 2(forward, backward)
    pp_comm_count = 4 * int(p > 1)
    pp_comm_time = pp_comm_size * pp_comm_count / pp_bandwidth

    # 4 = 2(bf16 bytes) * 2(t-1)/t(all-reduce)
    tp_comm_size = 4 * (t - 1) / t * b / d * s * h
    # 4 = 2(attn, mlp) * 2(forward, backward)
    tp_comm_count = 4 * (l / p) * int(t > 1)
    tp_comm_time = tp_comm_size * tp_comm_count / tp_bandwidth

    # 4 = 2(bf16 bytes) * 2(t-1)/t(all-reduce)
    dp_comm_size = 4 * (d - 1) / d * params / (p * t)
    dp_comm_count = int(d > 1)
    dp_comm_time = dp_comm_size * dp_comm_count / dp_bandwidth
    print_communication_summary()
    print_time_breakdown()
