# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-level full recompute when every microbatch runs through the same model.

Split out of ``test_schedule_chunk_1f1b.py`` so that each CI bucket of the chunk-schedule
parity matrix fits its time budget (the original file had grown to 189 cases, about an
hour on one H100 bucket).
"""

import pytest
import torch

from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_chunk_1f1b import (
    _A2AOverlapTestBase,
    capture_same_model_grads,
)
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    compare_captures,
    get_valid_dispatcher_configs,
    get_valid_fp8_flags,
)


class TestA2AOverlap(_A2AOverlapTestBase):
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize(
        "mtp_layers,recompute_method,recompute_num_layers",
        [(0, "uniform", 2), (0, "block", 1), (1, "uniform", 1)],
    )
    def test_full_recompute_same_model_steady_state(
        self,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        mtp_layers,
        recompute_method,
        recompute_num_layers,
    ):
        """Full recompute holds when every microbatch runs through the same layer modules.

        ``combined_1f1b_schedule_for_no_pipelining`` passes one model as both ``f_model`` and
        ``b_model``, so microbatch N+1's forward - and then N+2's - interleaves with microbatch
        N's backward and segment replay on the very same modules. The two-chunk parity cases pair
        two different models, so none of them can see a replay that reads state another
        microbatch's forward has since overwritten. Four microbatches with distinct inputs run
        through one model in that order; the accumulated grads are compared bitwise against an
        independently built model with identical weights running ``GPTModel.forward`` on the
        same inputs. Segmentation: ``uniform``/2 (a multi-layer segment), ``block``/1 (a
        recomputed -> eager hand-off) and ``uniform``/1 with MTP, which also runs with an
        ``mtp_input_mask`` against the eager overlap path.
        """
        extra_kwargs = {
            "recompute_granularity": "full",
            "recompute_method": recompute_method,
            "recompute_num_layers": recompute_num_layers,
        }
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if mtp_layers > 0:
            extra_kwargs["mtp_num_layers"] = mtp_layers
            extra_kwargs["mtp_loss_scaling_factor"] = 1.1

        # Against the reference without a mask: with one, the overlap schedule and
        # GPTModel.forward already disagree on MTP grads when recompute is off.
        reference = capture_same_model_grads(3, extra_kwargs, overlap=False)
        recomputed = capture_same_model_grads(3, extra_kwargs)
        comp_res = compare_captures(reference, recomputed, True, True)
        assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

        if mtp_layers > 0:
            # The mask's asymmetric holes only matter once MTP rolls it together with input_ids,
            # and its embedding rows are arbitrated overlap against overlap: the reference path
            # does not agree with the overlap one there even without recompute.
            eager_kwargs = {k: v for k, v in extra_kwargs.items() if not k.startswith("recompute_")}
            eager = capture_same_model_grads(3, eager_kwargs, use_mtp_input_mask=True)
            recomputed = capture_same_model_grads(3, extra_kwargs, use_mtp_input_mask=True)
            assert "embedding.word_embeddings.weight" in eager, "embedding grad not captured"
            comp_res = compare_captures(eager, recomputed, True, False)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
