# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-LM entry point for the financial time-series example (runnable MLP model).

Run from the repository root:
    torchrun --nproc_per_node=1 -m examples.financial_timeseries.pretrain_ts \
        --ts-data-path <tensors> ...
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from examples.financial_timeseries.ts_dataset import get_batch, train_valid_test_datasets_provider
from examples.financial_timeseries.ts_mlp_model import TSMlpModel
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain, print_rank_0
from megatron.training.argument_utils import (
    core_transformer_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.global_vars import set_run_config


# --------------------------------------------------------------------------- #
# Custom CLI args                                                              #
# --------------------------------------------------------------------------- #
def add_ts_args(parser):
    """Register the example's CLI flags on Megatron's argument parser.

    Args:
        parser: The ``argparse.ArgumentParser`` Megatron passes to ``extra_args_provider``.

    Returns:
        The same parser, with a "financial time-series" argument group added.
    """
    g = parser.add_argument_group(title="financial time-series")
    g.add_argument(
        "--ts-data-path",
        type=str,
        required=True,
        help="folder holding train.pt / val.pt / test.pt / meta.pt from preprocessing.py",
    )
    g.add_argument(
        "--ts-no-mmap",
        action="store_true",
        help="load each .pt fully into RAM instead of memory-mapping it",
    )
    g.add_argument(
        "--loss-lambda-volume",
        type=float,
        default=1.0,
        help="lambda_v: weight on the volume-share loss term relative to return",
    )
    g.add_argument(
        "--loss-lambda-l1",
        type=float,
        default=0.0,
        help="lambda_l1: L1 fraction: (1-lambda_l1)*MSE + lambda_l1*MAE",
    )
    g.add_argument(
        "--ts-horizon-weights",
        type=float,
        nargs="+",
        default=None,
        help="optional per-horizon loss weights; length must equal len(horizons)",
    )
    return parser


# --------------------------------------------------------------------------- #
# Model provider                                                              #
# --------------------------------------------------------------------------- #
def model_provider(pre_process: bool = True, post_process: bool = True, **kwargs):
    """Build the MLP. meta.pt carries the data sizes so embeddings/heads auto-fit the data.

    --hidden-size and --num-layers flow in via TransformerConfig and size the MLP. To scale this
    into a real (TP/PP-sharded) model later, replace TSMlpModel with one built from mcore parallel
    layers, still honouring pre_process / post_process.

    Some Megatron-LM builds construct the TransformerConfig themselves and pass it as config=...
    (and may pass extras like vp_stage). We accept **kwargs to tolerate that across versions: use
    the framework's config when given, otherwise build our own from the CLI args.
    """
    args = get_args()
    meta = torch.load(Path(args.ts_data_path) / "meta.pt", map_location="cpu")
    print_rank_0(
        f"[ts-model] meta: n_instruments={meta['n_instruments']} "
        f"F={len(meta['features'])} P={meta['patch_len']} "
        f"L={meta['context_len']} H={len(meta['horizons'])} "
        f"patches_per_day={meta['patches_per_day']}"
    )

    config = kwargs.get("config")
    if config is None:
        config = core_transformer_config_from_args(args)
    model = TSMlpModel(
        config,
        n_instruments=meta["n_instruments"],
        n_features=len(meta["features"]),
        patch_len=meta["patch_len"],
        context_len=meta["context_len"],
        n_horizons=len(meta["horizons"]),
        n_tod=meta.get("num_tod_buckets", meta["patches_per_day"]),
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=kwargs.get("pg_collection"),
        vp_stage=kwargs.get("vp_stage"),
    )
    return model


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #
def loss_func(batch, output_tensor):
    """Masked MSE over the return and volume-share targets at every position/horizon.

    Args:
        batch: Dict with ``target_return``, ``target_volshare`` and ``target_mask`` tensors,
            each of shape ``(B, L, H)``.
        output_tensor: Model output dict with ``"return"`` and ``"volshare"`` tensors of shape
            ``(B, L, H)``.

    Returns:
        Tuple ``(loss_sum, num_supervised, report)`` following Megatron's loss contract, where
        ``report`` maps metric names to ``[value_sum, count]`` pairs.
    """
    args = get_args()
    pred_ret = output_tensor["return"]
    pred_share = output_tensor["volshare"]
    mask = batch["target_mask"].to(pred_ret.dtype)  # (B, L, H), false across masked closes

    if args.ts_horizon_weights is not None:  # optional per-horizon emphasis
        w = torch.as_tensor(args.ts_horizon_weights, device=mask.device, dtype=mask.dtype)
        mask = mask * w  # broadcasts over (B, L, H)

    se_ret = ((pred_ret - batch["target_return"]) ** 2) * mask
    se_share = ((pred_share - batch["target_volshare"]) ** 2) * mask
    loss_sum = se_ret.sum() + args.loss_lambda_volume * se_share.sum()  # SUM over supervised elems
    num = mask.sum().clamp_min(1.0).to(torch.int)  # supervised-element count

    # Megatron's loss contract: (summed loss, number of supervised elements, report); do NOT
    # pre-divide. With calculate_per_token_loss (set unconditionally in __main__) Megatron keeps
    # the sum and scales the gradient by the GLOBAL element count over all micro-batches and
    # data-parallel ranks, i.e. a mean over every supervised element in the global batch.
    # Without it, each micro-batch would be divided by its own count and those local means
    # averaged, which over-weights elements from micro-batches where intraday masking removed
    # many targets. Each report entry is a [value_sum, count] pair; the logger reduces them as
    # sum/count over the global batch, so the logged loss matches the optimised one.
    def pair(x):
        return torch.cat([x.detach().view(1), num.view(1)])

    report = {
        "loss": pair(loss_sum),
        "return mse": pair(se_ret.sum()),
        "volume mse": pair(se_share.sum()),
    }
    return loss_sum, num, report


# --------------------------------------------------------------------------- #
# Forward step (Megatron calls this each micro-batch)                          #
# --------------------------------------------------------------------------- #
def forward_step(data_iterator, model):
    """Pull a micro-batch via get_batch, run the model, bind targets to the loss closure.

    Written for PP=1 (the debug path), where the batch holds inputs AND targets. For PP>1, gate
    the model inputs on the first stage and let mcore thread activations; middle stages get
    batch=None.

    Args:
        data_iterator: Iterator over the split's ``DataLoader``.
        model: The (possibly wrapped) ``TSMlpModel``.

    Returns:
        Tuple ``(output, loss_closure)`` as expected by Megatron's forward-backward step.
    """
    patches, inst_id, patch_tod, targets, target_mask = get_batch(data_iterator)
    # positional: Megatron's Float16Module casts positional float inputs to bf16/fp16, not kwargs
    output = model(patches, inst_id, patch_tod)
    batch = {
        "target_return": targets["return"],
        "target_volshare": targets["volume"],
        "target_mask": target_mask,
    }
    return output, partial(loss_func, batch)


# --------------------------------------------------------------------------- #
# Launch                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True  # each rank builds its own datasets
    # NullTokenizer with vocab_size 1: there is no token vocabulary. There is no --vocab-size CLI
    # flag; args_defaults sets the attribute that gpt_config_from_args / build_tokenizer read.
    args = parse_and_validate_args(
        extra_args_provider=add_ts_args,
        args_defaults={"tokenizer_type": "NullTokenizer", "vocab_size": 1},
    )
    # Normalise the loss by the GLOBAL number of supervised elements (what Megatron requires for
    # sequence packing, for the same reason): intraday masking makes that count differ between
    # micro-batches and data-parallel ranks, and the default would average per-micro-batch
    # means instead. Set here rather than via args_defaults, which only fills attributes that
    # parsed as None; the store_true flag parses as False. See loss_func.
    args.calculate_per_token_loss = True
    model_cfg = gpt_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    set_run_config(full_config)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=model_provider,
    )
