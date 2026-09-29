# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Score a GPT checkpoint on its validation sets: loss, next-token Brier score and accuracy.

Run like pretrain_gpt.py with --skip-train, --load <checkpoint dir> and the validation-set
flags; every validation set is evaluated once and printed as

    validation-<name> loss at iteration <it> on validation set | lm loss value: ... |
        brier value: ... | acc value: ... |

For each scored token (loss_mask 1) with next token y and predicted distribution p over the
vocabulary: loss = -log p_y, Brier = sum_k (p_k - 1[k = y])^2 = 1 - 2 p_y + sum_k p_k^2, and
accuracy = 1[argmax p = y]. All three are reduced like the validation loss: token-weighted for
pretraining-style data, per sample and then averaged over samples under --sft.

Supports tensor and pipeline parallel size 1 only (full logits on one rank).
"""

from functools import partial

import torch

import pretrain_gpt
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, inprocess_restart, pretrain
from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args


def score_func(loss_mask, labels, logits):
    """Masked sums of loss, Brier score and accuracy, reported as [sum, scored tokens]."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Masked positions may hold ignore labels (-100); any in-range index will do there.
    targets = labels.clamp(min=0).unsqueeze(-1)
    log_p_target = log_probs.gather(-1, targets).squeeze(-1)
    probs = log_probs.exp()
    brier = 1.0 - 2.0 * log_p_target.exp() + probs.square().sum(dim=-1)
    correct = (log_probs.argmax(dim=-1) == targets.squeeze(-1)).float()

    mask = loss_mask.float()
    num_tokens = mask.sum().to(torch.int)
    loss = -(log_p_target * mask).sum()

    def report(values):
        return torch.cat([values.view(1), num_tokens.view(1).float()])

    return (
        loss,
        num_tokens,
        {
            'lm loss': report(loss.detach()),
            'brier': report((brier * mask).sum()),
            'acc': report((correct * mask).sum()),
        },
    )


def forward_step(data_iterator, model, return_schedule_plan=False):
    """Forward pass returning the full logits, scored by score_func."""
    args = get_args()
    assert args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1
    timers = get_timers()
    timers('batch-generator', log_level=2).start()
    (_, cu_seqlens, _, _, labels, _, loss_mask, _, position_ids, tokens) = pretrain_gpt.get_batch(
        data_iterator
    )
    timers('batch-generator').stop()
    assert cu_seqlens is None, 'score_gpt.py expects unpacked batches (use --sft-cross-document-attention)'
    logits = model(tokens, position_ids, None)  # no labels: [b, s, vocab]
    return logits, partial(score_func, loss_mask, labels)


if __name__ == "__main__":
    setattr(pretrain_gpt.train_valid_test_datasets_provider, "is_distributed", True)
    pretrain_fn, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    assert args.skip_train, 'score_gpt.py only evaluates; pass --skip-train'
    full_config = pretrain_cfg_container_from_args(args, gpt_config_from_args(args))
    pretrain_fn(
        full_config,
        pretrain_gpt.train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=pretrain_gpt.get_embedding_ranks,
    )
