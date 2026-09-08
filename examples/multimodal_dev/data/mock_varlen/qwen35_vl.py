# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed mock image-text data for Qwen3.5-VL training.

The torch layer of the packed-document multimodal varlen mock: whole
documents (image atoms at the prefix, one text run, terminal EOD) are
packed into variable-length windows planned by
:class:`PackedDocumentPlanGenerator`, with full-sequence shifted-target
supervision (every target that is ordinary text or the EOD; image
  placeholders are never LM targets); physical padding to the fixed
  runtime target is the THD packer's job, not the dataset's.

Zero config resolves the context-scaled practical default for the final
seq_length (the kernel's ``context_scaled_default``); an explicit
config must be COMPLETE — partial configs are rejected, never merged.

This module is a token/pixel adapter over those plans; window-level
statistics (documents per window, image counts, vision share) are emergent
from the document layer and are measured, not configured.

The generic text-only ``MockVarlenDataset`` cannot transport ragged vision
payloads through the core packing scheduler, so this provider keeps the
raw per-sample contract and leaves multimodal packing to
``multimodal_dev.forward_step.pack_or_pad_batch``. Fixed-shape single-image
scenarios are served by ``--dataset-provider mock`` instead.
"""

import json
import numbers
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.data.mock_varlen.packed_document import (
    PackedDocumentPlanGenerator,
    context_scaled_default,
)
from examples.multimodal_dev.forward_step import per_segment_alignment
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    VISION_KWARGS,
)
from megatron.training.datasets.utils import load_json_arg

# Auto plan-pool sizing target: >= 2^26 (64Mi) plan tokens regardless of
# seq_length, so rare mixture components stay statistically represented in
# the layout pool (see the context-scaled default provenance in
# packed_document.py).
AUTO_PLAN_POOL_TOKENS = 1 << 26
# Ceiling on seq_length x plan_pool_windows (the auto rule lands 16-64x
# below it, closest at long context; this only bounds explicit profiles).
_MAX_PLAN_TOKENS = 1 << 32

# Per-sample content-stream ids for _seed_sequence. Content and layout share
# ONE SeedSequence namespace, not two: numpy zero-pads the entropy pool, so
# [seed, idx, stream] and [seed, idx, stream, 0] yield byte-identical state,
# and plan_seed and --seed both default to 1234. Other coordinates (item
# ordinal, differing seeds) usually separate the two anyway; ids disjoint from
# the plan kernel's per-document layout streams (21-23 in packed_document.py)
# are what still separates them when the other coordinates coincide, as they
# do by default. A new content stream must never reuse a layout id.
_PDOC_TEXT_TOKEN_STREAM = 30
_PDOC_PIXEL_VALUE_STREAM = 31

# Processor geometry, taken from the model's own vision config so the mock
# payload shape can never drift from the tower that consumes it.
QWEN35_VL_PATCH_SIZE = VISION_KWARGS["patch_size"]
QWEN35_VL_TEMPORAL_PATCH_SIZE = VISION_KWARGS["temporal_patch_size"]
QWEN35_VL_SPATIAL_MERGE_SIZE = VISION_KWARGS["spatial_merge_size"]

# Fixed mock EOD id: the terminal EOD is STRUCTURE, not content (a real
# tokenizer supplies a constant id), so the mock uses a constant ordinary
# text id and masking tests can address it literally. It is never an
# input position — it appears only as the last input position's TARGET.
MOCK_EOD_TOKEN_ID = 13


def _seed_sequence(seed: int, idx: int, stream: int, item: int = 0) -> np.random.SeedSequence:
    """Return an access-order-independent RNG namespace for one sample stream."""
    return np.random.SeedSequence([int(seed), int(idx), int(stream), int(item)])


def _print_rank0(message: str) -> None:
    """Print a startup-scan artifact once per job instead of once per rank.

    Only the PRINTING is gated on rank 0; every guard raise stays on every
    rank. Without initialized torch.distributed (unit tests, single-process
    tooling) the print always happens.
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        print(message, flush=True)


def _parse_bucket_table(
    image_size_config: dict[str, Any] | None,
    *,
    patch_size: int = QWEN35_VL_PATCH_SIZE,
    spatial_merge_size: int = QWEN35_VL_SPATIAL_MERGE_SIZE,
) -> tuple[list[tuple[int, int, int]], list[int], list[int], list]:
    """Validate an ``image_sizes`` bucket table and derive its geometry.

    Single structural authority for the dataset; returns
    ``(grids, merged_tokens, raw_patches, weights)``. Numeric weight
    validity (finite / non-negative / positive sum) stays with the plan
    kernel, the single authority for what is actually drawn.
    """
    if (
        not isinstance(image_size_config, dict)
        or not image_size_config.get("resolutions")
        or not set(image_size_config) <= {"resolutions", "weights"}
    ):
        raise ValueError(
            "The packed-document multimodal varlen mock requires image_size_config = "
            '{"resolutions": [[H, W], ...]} with optional "weights" (no '
            f"other keys); got {image_size_config!r}."
        )

    if patch_size <= 0 or spatial_merge_size <= 0:
        raise ValueError("patch_size and spatial_merge_size must be positive.")
    block = patch_size * spatial_merge_size
    grids: list[tuple[int, int, int]] = []
    merged_tokens: list[int] = []
    raw_patches: list[int] = []
    for index, resolution in enumerate(image_size_config["resolutions"]):
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or not all(
                isinstance(side, numbers.Integral) and not isinstance(side, bool)
                for side in resolution
            )
            or not all(int(side) > 0 for side in resolution)
        ):
            raise ValueError(
                f"Bucket resolution at index {index} must be exactly two positive "
                f"integers [height, width]; got {resolution!r}."
            )
        height, width = int(resolution[0]), int(resolution[1])
        if height % block or width % block:
            raise ValueError(
                f"Bucket resolution {height}x{width} must be divisible by "
                f"patch_size*spatial_merge_size={block}."
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        grids.append((1, grid_h, grid_w))
        merged_tokens.append((grid_h // spatial_merge_size) * (grid_w // spatial_merge_size))
        raw_patches.append(grid_h * grid_w)
    if "weights" not in image_size_config:
        weights = [1.0] * len(grids)
    else:
        # Present-but-invalid must fail loudly: `[]`, null, or wrong types
        # must never silently degrade to uniform weights. Zero entries are
        # legal (they disable a bucket); the kernel remains the single
        # authority for finite / non-negative / positive-sum.
        weights = image_size_config["weights"]
        if (
            not isinstance(weights, (list, tuple))
            or not weights
            or any(isinstance(w, bool) or not isinstance(w, numbers.Real) for w in weights)
        ):
            raise ValueError(
                "Bucket 'weights', when present, must be a non-empty list of "
                f"numbers (zeros allowed; the sum must be positive); got {weights!r}."
            )
    if len(weights) != len(grids):
        raise ValueError(
            f"Bucket 'weights' must match 'resolutions' in length; got "
            f"{len(weights)} weights for {len(grids)} resolutions."
        )
    return grids, merged_tokens, raw_patches, weights


def _validate_multimodal_token_ids(special_ids: set[int], vocab_size: int) -> int:
    """Reject unusable special-id tables; return the first safe text id.

    Token ID 0 is reserved for packing padding; a special ID of 0 could
    be miscounted as an image placeholder after collate padding.
    """
    if any(not 0 < token_id < vocab_size for token_id in special_ids):
        raise ValueError(
            f"All multimodal token IDs must be in [1, vocab_size={vocab_size}); "
            f"got {sorted(special_ids)}."
        )
    safe_text_token_id = next(
        (token_id for token_id in range(1, vocab_size) if token_id not in special_ids), None
    )
    if safe_text_token_id is None:
        raise ValueError("vocab_size does not contain a usable non-special text token ID.")
    return safe_text_token_id


def _pop_plan_pool_and_seed(
    window_config: dict[str, Any], *, seq_length: int, num_samples: int
) -> tuple[dict[str, Any], int, int]:
    """Split the plan-pool/seed knobs off a window config.

    Returns ``(kernel_config, plan_pool_windows, plan_seed)``. The plan pool
    bounds construction time and memory independently of the virtual dataset
    length Megatron requests for the full training schedule: indices wrap
    onto the pool for the window LAYOUT while token/pixel content stays
    keyed by the virtual index. A zero-sample split reports a zero pool;
    no plan is built for it. The window LAYOUT is seeded by plan_seed
    (profile default 1234, the calibration-snapshot seed), independently of
    the training seed: the workload shape — segment structure, image
    placement, per-window payloads — is a constant of the profile, while
    --seed varies token/pixel CONTENT only. Finite pools realize
    heavy-tailed statistics with visible seed-to-seed variance, so a
    floating layout would make throughput/memory numbers incomparable
    across seeds.
    """
    window_config = dict(window_config)
    pool_windows = window_config.pop("plan_pool_windows", "auto")
    if pool_windows == "auto":
        # >= AUTO_PLAN_POOL_TOKENS plan tokens at any S: enough expected
        # rare-component documents in the pool that realized pool statistics
        # stay near nominal instead of drifting with the pool seed; the
        # floor keeps long-context pools at the proven 2048-window cost
        # bound.
        if seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {seq_length}.")
        pool_windows = max(2048, -(-AUTO_PLAN_POOL_TOKENS // seq_length))
    elif isinstance(pool_windows, bool) or not isinstance(pool_windows, int):
        raise ValueError(
            "plan_pool_windows must be the string 'auto' or a positive "
            f"integer, got {pool_windows!r}."
        )
    if pool_windows <= 0:
        raise ValueError(f"plan_pool_windows must be positive, got {pool_windows}.")
    # Plan construction costs O(seq_length x plan_pool_windows). Bounding
    # seq_length alone leaves an explicit pool free to turn startup into an
    # hours-long silent build, so the product carries the ceiling.
    if pool_windows * seq_length > _MAX_PLAN_TOKENS:
        raise ValueError(
            f"plan_pool_windows {pool_windows} x seq_length {seq_length} exceeds the "
            f"{_MAX_PLAN_TOKENS} planned-token ceiling; lower either one."
        )
    plan_seed = window_config.pop("plan_seed", 1234)
    if isinstance(plan_seed, bool) or not isinstance(plan_seed, int) or plan_seed < 0:
        raise ValueError(f"plan_seed must be a non-negative integer, got {plan_seed!r}.")
    if num_samples is None:
        return window_config, pool_windows, int(plan_seed)
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}.")
    return window_config, min(pool_windows, num_samples), int(plan_seed)


class PackedDocumentQwen35VLDataset(Dataset):
    """Whole documents packed into variable-length windows.

    One item is one window of WHOLE documents planned by
    :class:`~examples.multimodal_dev.data.mock_varlen.packed_document.PackedDocumentPlanGenerator`.
    Tensors are ``logical_tokens`` long (T <= window capacity; physical
    padding to the fixed runtime target is the THD packer's job, never the
    dataset's) and ``seq_lens`` holds whole per-document input lengths with
    ``seq_lens.sum() == T``. Labels are full-sequence shifted targets with
    the pure-text mock's document semantics: within each document the
    unshifted stream is ``input tokens + EOD`` and ``labels = stream[1:]``,
    so position ``t`` targets ``t+1``, the last input position targets the
    (never-input) EOD, and positions whose target is an image placeholder
    or vision_start (plan-span truth, not token values) are ``-100``. No
    position ever targets across a document boundary.
    ``loss_mask == (labels != -100)`` by construction.

    NOT a standalone public entry point: the supported construction path
    is the dataset provider, whose deterministic startup scan gates the
    FULL plan pool against --max-vision-patches-per-microbatch before the
    DataLoader starts (the runtime packer's microbatch guard remains as
    defense in depth). Direct construction skips that scan.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        seq_length: int,
        resolved_profile: dict[str, Any],
        segment_alignment: int = 1,
        seed: int = 1234,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        patch_size: int = QWEN35_VL_PATCH_SIZE,
        temporal_patch_size: int = QWEN35_VL_TEMPORAL_PATCH_SIZE,
        spatial_merge_size: int = QWEN35_VL_SPATIAL_MERGE_SIZE,
    ) -> None:
        if num_samples is None:
            # training.py sets eval_samples=None under --full-validation; this
            # dataset has a finite plan pool and a fixed __len__, so it cannot
            # represent an unbounded split.
            raise ValueError(
                "The packed-document multimodal varlen mock cannot serve an "
                "unbounded split (--full-validation); set a finite --eval-iters."
            )
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        special_ids = {image_token_id, video_token_id, vision_start_token_id}
        # The EOD marker is a fixed ordinary text id: it must exist in the
        # vocabulary and never alias an image placeholder.
        if not 0 < MOCK_EOD_TOKEN_ID < vocab_size or MOCK_EOD_TOKEN_ID in special_ids:
            raise ValueError(
                f"packed_document fixed EOD token id {MOCK_EOD_TOKEN_ID} must be in "
                f"[1, vocab_size={vocab_size}) and distinct from the multimodal special "
                f"ids {sorted(special_ids)}."
            )
        # Structural ids are the vision special ids AND the EOD marker: a
        # random text draw that hits any of them would fake a structural
        # marker inside a text run, so text is scrubbed against the whole
        # set and the safe replacement id is chosen outside it too.
        self.structural_ids = special_ids | {MOCK_EOD_TOKEN_ID}
        self.safe_text_token_id = _validate_multimodal_token_ids(self.structural_ids, vocab_size)

        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.image_token_id = int(image_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
        # Plan construction goes through the shared helper, which owns the
        # auto-pool rule and the bucket geometry. An empty split validates
        # its profile through the helper's one-window plan but keeps none:
        # it serves no data, so it advertises no pool and no maxima.
        plan, geometry = build_packed_document_plan(
            resolved_profile,
            seq_length=seq_length,
            num_samples=num_samples,
            segment_alignment=segment_alignment,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
        )
        self.plan = plan if num_samples > 0 else None
        self.plan_pool_windows = geometry["plan_pool_windows"]
        self.plan_seed = geometry["plan_seed"]
        self.grids = geometry["grids"]
        self.bucket_weights = [float(weight) for weight in geometry["weights"]]

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, idx: int, stream: int, item: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        seed_state = _seed_sequence(self.seed, idx, stream, item).generate_state(1, dtype=np.uint64)
        generator.manual_seed(int(seed_state[0]))
        return generator

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.num_samples == 0:
            raise IndexError("Cannot index an empty PackedDocumentQwen35VLDataset.")
        # Standard sequence bounds (negative = from the end): silently
        # wrapping out-of-range indices would alias samples and make plain
        # iteration endless. Virtual POOL wrap below is layout-only.
        idx = int(idx)
        if idx < 0:
            idx += self.num_samples
        if not 0 <= idx < self.num_samples:
            raise IndexError(f"index {idx} out of range for {self.num_samples} samples.")
        window = self.plan.window(idx % self.plan_pool_windows)
        total_raw_patches = window.raw_patches
        total_tokens = window.logical_tokens
        input_ids = torch.randint(
            1,
            self.vocab_size,
            (total_tokens,),
            dtype=torch.long,
            generator=self._generator(idx, stream=_PDOC_TEXT_TOKEN_STREAM),
        )
        # Scrub ALL structural ids (vision + EOD) from the random text:
        # the structural spans below re-write them at exactly the planner's
        # positions, so any other occurrence would be a fake marker.
        for structural_id in self.structural_ids:
            input_ids[input_ids == structural_id] = self.safe_text_token_id

        # Walk the plan's span layout: image atoms overwrite the random
        # text; target masking comes from the plan's span-type truth, never
        # from token values.
        labels = torch.full((total_tokens,), -100, dtype=torch.long)
        is_image_position = torch.zeros(total_tokens, dtype=torch.bool)
        position = 0
        for document in window.documents:
            doc_start = position
            for span in document.spans:
                start, end = position, position + span.length
                if span.span_type == "image":
                    input_ids[start] = self.vision_start_token_id
                    input_ids[start + 1 : end] = self.image_token_id
                    is_image_position[start:end] = True
                position = end
            doc_end = position
            # Pure-text-mock document semantics: the unshifted stream is
            # ``input tokens + EOD``, labels = stream[1:]. Every in-document
            # next-token target is kept unless the TARGET is an image
            # placeholder/vision_start; the last input position targets the
            # EOD (supervised); nothing targets across the boundary.
            if doc_end - doc_start > 1:
                targets = input_ids[doc_start + 1 : doc_end]
                kept = ~is_image_position[doc_start + 1 : doc_end]
                labels[doc_start : doc_end - 1] = torch.where(
                    kept, targets, torch.full_like(targets, -100)
                )
            labels[doc_end - 1] = MOCK_EOD_TOKEN_ID
        assert position == total_tokens, "span walk must cover the whole window"
        loss_mask = (labels != -100).to(torch.float32)

        # Pixel/grid rows follow the plan's window.images order, which is by
        # construction the token placeholder order (document order ->
        # in-document image order).
        if window.images:
            image_grid_thw = torch.tensor(
                [self.grids[image.bucket_index] for image in window.images], dtype=torch.long
            )
        else:
            image_grid_thw = torch.empty((0, 3), dtype=torch.long)
        if window.images:
            # One preallocated buffer, filled per image (host peak = payload).
            pixel_values = torch.empty((total_raw_patches, self.pixel_dim), dtype=torch.float32)
            row = 0
            for ordinal, image in enumerate(window.images):
                pixel_values[row : row + image.raw_patches].normal_(
                    generator=self._generator(idx, stream=_PDOC_PIXEL_VALUE_STREAM, item=ordinal)
                )
                row += image.raw_patches
        else:
            pixel_values = torch.empty((0, self.pixel_dim), dtype=torch.float32)

        seq_lens = torch.tensor(window.seq_lens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "image_grid_thw": image_grid_thw,
            "seq_lens": seq_lens,
            "pixel_values": pixel_values,
        }


def resolve_varlen_config(spec: str | None, *, seq_length: int) -> dict[str, Any]:
    """Resolve the profile: context-scaled default or a COMPLETE user config.

    ``spec`` is the raw --multimodal-varlen-mock-dataset-config-json value
    (inline JSON or a JSON-file path). Omitted or ``{}`` resolves the
    kernel-owned ``context_scaled_default(seq_length)``. A non-empty
    config must be COMPLETE (all four top-level keys: components,
    image_sizes, plan_pool_windows, plan_seed):
    partial configs are never merged onto the scaled default — a partial
    override of an S-dependent baseline is ambiguous, so it fails loudly
    and the error carries the fully resolved default JSON to copy and
    edit (including for a plan_seed-only tweak).
    """
    try:
        user = load_json_arg(spec)
    except ValueError as error:
        raise ValueError(
            "--multimodal-varlen-mock-dataset-config-json is not valid JSON "
            f"(inline or a readable JSON-file path): {error}"
        ) from error
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ValueError(
            "--multimodal-varlen-mock-dataset-config-json must be a JSON "
            f"object; got {type(user).__name__}."
        )
    if not user:
        return context_scaled_default(seq_length)

    def _default_hint() -> str:
        try:
            return "start from the resolved default: " + json.dumps(
                context_scaled_default(seq_length), sort_keys=True
            )
        except ValueError:
            return (
                "the context-scaled default is unavailable at this seq_length, so a "
                "full explicit profile is the only option"
            )

    profile_keys = {"components", "image_sizes", "plan_pool_windows", "plan_seed"}
    unknown = set(user) - profile_keys
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in "
            "--multimodal-varlen-mock-dataset-config-json; allowed top-level "
            f"keys: {sorted(profile_keys)} — {_default_hint()}"
        )
    missing = profile_keys - set(user)
    if missing:
        raise ValueError(
            f"Missing key(s) {sorted(missing)} in "
            "--multimodal-varlen-mock-dataset-config-json: an explicit profile "
            "must be COMPLETE (partial configs are not merged onto the "
            "context-scaled default — the baseline depends on seq_length, so a "
            f"partial override is ambiguous). To edit, {_default_hint()}"
        )
    # components / image_sizes structure is validated once, by the plan
    # kernel and dataset constructor (the single authorities).
    return dict(user)


def build_packed_document_plan(
    resolved_profile: dict[str, Any],
    *,
    seq_length: int,
    num_samples: int | None,
    segment_alignment: int,
    patch_size: int = QWEN35_VL_PATCH_SIZE,
    spatial_merge_size: int = QWEN35_VL_SPATIAL_MERGE_SIZE,
) -> tuple["PackedDocumentPlanGenerator", dict[str, Any]]:
    """Single plan-construction entry shared by every consumer.

    Owns the auto-pool rule, bucket-geometry derivation, kernel-config
    extraction, and generator construction so no consumer re-derives any
    of them. ``num_samples`` clamps the pool (None = unclamped). The
    generator is always constructed so its validation always runs; a
    zero-sample split gets a one-window plan its caller discards. Returns
    ``(generator, geometry)`` where geometry carries the bucket
    grids/weights and the resolved pool/seed.
    """
    grids, merged_tokens, raw_patches, weights = _parse_bucket_table(
        resolved_profile.get("image_sizes"),
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )
    window_config, pool_windows, plan_seed = _pop_plan_pool_and_seed(
        {key: value for key, value in resolved_profile.items() if key != "image_sizes"},
        seq_length=seq_length,
        num_samples=num_samples,
    )
    # An empty split still constructs a one-window plan: the kernel is the
    # only authority for component/weight/feasibility validation, so skipping
    # it would let an illegal profile through unnoticed. The caller decides
    # whether to keep it (see PackedDocumentQwen35VLDataset).
    generator = PackedDocumentPlanGenerator(
        seq_length=seq_length,
        num_windows=max(pool_windows, 1),
        seed=plan_seed,
        config=window_config,
        bucket_merged_tokens=merged_tokens,
        bucket_raw_patches=raw_patches,
        bucket_weights=weights,
        segment_alignment=segment_alignment,
    )
    geometry = {
        "grids": grids,
        "weights": weights,
        "plan_pool_windows": pool_windows,
        "plan_seed": plan_seed,
    }
    return generator, geometry


def train_valid_test_varlen_datasets_provider(
    train_val_test_num_samples,
) -> tuple[Dataset, Dataset, Dataset]:
    """Provide packed mock train, validation, and test datasets.

    Serves packed_document variable-length windows of whole documents;
    requires the fixed-physical-target runtime packing contract
    (--pad-packed-seq-alignment max, --max-seqlen-per-dp-cp-rank,
    dummy-THD-tail padding) in addition to the shared packed-THD flags.
    """
    from megatron.training import get_args

    args = get_args()
    for flag, active in (
        ("--use-varlen-dataset", getattr(args, "use_varlen_dataset", False)),
        (
            "--sequence-packing-scheduler",
            getattr(args, "sequence_packing_scheduler", None) is not None,
        ),
    ):
        if active:
            raise ValueError(
                f"The multimodal mock_varlen provider is incompatible with {flag}; "
                "vision payloads are packed by multimodal_dev.forward_step instead."
            )
    if not getattr(args, "use_packed_sequence", False):
        raise ValueError(
            "The multimodal mock_varlen packed_document provider requires "
            "--use-packed-sequence: windows carry multiple document segments "
            "(seq_lens) and the padded BSHD layout has no segment representation."
        )
    uses_hybridep = (
        getattr(args, "moe_token_dispatcher_type", None) == "flex"
        and getattr(args, "moe_flex_dispatcher_backend", None) == "hybridep"
    )
    if uses_hybridep and not getattr(args, "moe_hybridep_pad_variable_tokens", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires "
            "--moe-hybridep-pad-variable-tokens with packed THD + HybridEP; "
            "locally packed token counts can differ across the HybridEP group."
        )
    if not getattr(args, "use_vanilla_collate_fn", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires --use-vanilla-collate-fn "
            "so variable-length samples remain a list until multimodal packing."
        )

    model_seq_length = getattr(args, "seq_length", None)
    if model_seq_length is None:
        raise ValueError("The multimodal mock_varlen provider requires --seq-length.")
    # packed_document windows pack against a physical capacity equal to
    # seq_length by construction; --seq-length is the sole authority
    # (the legacy --total-seq-length knob belongs to the fixed-shape
    # providers and is ignored here).
    total_seq_length = int(model_seq_length)
    # Plan construction is O(seq_length x pool windows): an unbounded S
    # with an explicit profile turns startup into an hours-long silent
    # build. 2^21 is a generous engineering bound (16x the qualified 128K
    # domain top); the zero-config default is bounded to [4096, 131072]
    # by the resolver regardless.
    if not 1 <= total_seq_length <= 2_097_152:
        raise ValueError(
            f"--seq-length {total_seq_length} is outside the supported range "
            "[1, 2097152] for the mock_varlen provider (plan construction cost "
            "scales with seq_length x plan_pool_windows)."
        )

    if getattr(args, "varlen_mock_dataset_config_json", None) is not None:
        raise ValueError(
            "The multimodal mock_varlen provider no longer reads the core "
            "--varlen-mock-dataset-config-json flag (it belongs to the "
            "text-side --use-varlen-dataset datasets). Move the config to "
            "--multimodal-varlen-mock-dataset-config-json (omit it entirely "
            "for the context-scaled default)."
        )
    config = resolve_varlen_config(
        getattr(args, "multimodal_varlen_mock_dataset_config_json", None),
        seq_length=total_seq_length,
    )

    micro_batch_size = int(getattr(args, "micro_batch_size", 1) or 1)
    if micro_batch_size != 1:
        raise ValueError(
            "The packed-document multimodal varlen mock requires micro_batch_size == 1 for "
            f"training and evaluation: one item already is a full {total_seq_length}-"
            f"token-capacity window; got micro_batch_size={micro_batch_size}."
        )

    # Variable-length whole-document windows only stay THD-shape-
    # static through the runtime packer's fixed physical target; the
    # planner must pack against that exact target, so all three knobs
    # of the contract are startup requirements.
    pad_alignment = getattr(args, "pad_packed_seq_alignment", None)
    if pad_alignment != "max":
        raise ValueError(
            "The packed-document multimodal varlen mock requires "
            "--pad-packed-seq-alignment max (the fixed physical THD target the plan "
            "packs against); got "
            f"{pad_alignment!r}."
        )
    max_seqlen_per_dp_cp_rank = getattr(args, "max_seqlen_per_dp_cp_rank", None)
    if max_seqlen_per_dp_cp_rank is None:
        raise ValueError(
            "The packed-document multimodal varlen mock requires "
            "--max-seqlen-per-dp-cp-rank: it defines the CP-local physical window "
            "capacity the planner packs whole "
            "documents into."
        )
    if not getattr(args, "pad_packed_seq_by_appending_dummy_seq", True):
        raise ValueError(
            "The packed-document multimodal varlen mock requires the dummy THD "
            "tail: the physical tail up to the fixed target is represented as one "
            "ordinary dummy THD "
            "sequence. This is an implementation invariant (the core default); "
            "disabling it — e.g. via the auto-generated "
            "--no-pad-packed-seq-by-appending-dummy-seq switch — is not "
            "supported."
        )
    cp = int(getattr(args, "context_parallel_size", 1) or 1)
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    sequence_parallel = bool(getattr(args, "sequence_parallel", False))
    local_target = int(max_seqlen_per_dp_cp_rank)
    # Public contract: the physical window capacity IS the model sequence
    # length (T <= S, physically padded to exactly S by the packer).
    if local_target * cp != total_seq_length:
        raise ValueError(
            "The packed-document multimodal varlen mock requires "
            "--max-seqlen-per-dp-cp-rank * CP to equal "
            f"--seq-length: {local_target} * {cp} = {local_target * cp} must equal "
            f"seq_length {total_seq_length}."
        )
    # The runtime packer owns this rule; importing it (rather than
    # restating it) is what makes aligned plan costs equal physical costs.
    segment_alignment = per_segment_alignment(cp, tp, sequence_parallel)

    kwargs = dict(
        seq_length=total_seq_length,
        resolved_profile=config,
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        segment_alignment=segment_alignment,
    )
    seed = int(getattr(args, "seed", 1234))
    datasets = tuple(
        PackedDocumentQwen35VLDataset(
            num_samples=train_val_test_num_samples[split], seed=seed + split, **kwargs
        )
        for split in range(3)
    )

    # Deterministic startup contract: the plan pool is fully built at
    # construction, so guard verdicts are startup facts. Scan every split's
    # pool, fail BEFORE the DataLoader starts, and log the resolved profile
    # plus the pool maxima as the launch artifact. The scan applies to
    # explicit configs exactly as to the default; changing plan_seed or the
    # distribution re-verifies automatically. One line, sorted keys: runs
    # group by profile with `sort | uniq` over this line.
    resolved_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # Printed BEFORE any guard verdict so a failing launch still leaves a
    # reproducible artifact, not just the offending maxima.
    _print_rank0(f"[mock_varlen] resolved profile: {resolved_json}")
    patch_budget = getattr(args, "max_vision_patches_per_microbatch", None)
    for split_name, dataset in zip(("train", "valid", "test"), datasets):
        plan = dataset.plan
        if plan is None:
            # Empty split: no plan was built, so there is no pool to report
            # or gate (the profile is validated by whichever split has
            # samples, or by the dry construct above when none do).
            continue
        _print_rank0(
            f"[mock_varlen] {split_name}: "
            f"plan_seed={dataset.plan_seed} pool_windows={dataset.plan_pool_windows} "
            f"max_raw_patches_per_window={plan.pool_max_raw_patches} "
            f"max_raw_patches_per_image={plan.pool_max_image_raw_patches} "
            f"max_images_per_window={plan.pool_max_images} "
            f"max_logical_tokens={plan.pool_max_logical_tokens} "
            f"padding_fraction={plan.total_padding_fraction:.4f} "
            f"count_conditioning="
            f"{plan.image_count_conditioning_events / max(plan.total_documents, 1):.4f} "
            f"geometry_conditioning="
            f"{plan.image_conditioning_events / max(plan.total_documents, 1):.4f}"
        )
        if patch_budget is not None and plan.pool_max_raw_patches > int(patch_budget):
            raise ValueError(
                f"startup plan scan ({split_name}): the plan pool's heaviest window "
                f"carries {plan.pool_max_raw_patches} raw vision patches, exceeding "
                f"--max-vision-patches-per-microbatch={int(patch_budget)}. The full "
                "pool is deterministic at startup, so this run would fail mid-epoch "
                "by construction. Shrink the image profile, change plan_seed, or "
                f"raise the budget. Resolved profile: {resolved_json}"
            )

    if getattr(args, "max_vision_patches_per_image", None) is None:
        # The per-image guard is a true invariant, not a tunable: atom sizes
        # come from the same (now fully validated) bucket table, so its exact
        # upper bound is the largest drawable (weight > 0) bucket's raw-patch
        # count. Resolve it once so the packer-side check (forward_step) sees
        # a concrete bound even when no explicit cap was configured.
        train = datasets[0]
        args.max_vision_patches_per_image = max(
            grid_t * grid_h * grid_w
            for (grid_t, grid_h, grid_w), weight in zip(train.grids, train.bucket_weights)
            if weight > 0  # zero-weight buckets are disabled and never lift the bound
        )
    return datasets
