# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-numpy plan generator for the packed-document varlen mock.

Single source of truth for the document -> window token plan
behind the packed-document multimodal varlen mock (sample-atomic packing),
shared with the CPU calibration simulator. This module plans token
layouts only: no torch, no pixels.

Generative model (full-sequence causal-LM): each document first draws a
mixture COMPONENT (document-count weights), then its FINAL logical
length, then the images that live inside that length:

    L_target ~ component.length      (truncated lognormal, max <= the
                                      aligned window budget — == seq_length
                                      at every shipped call site, where
                                      window_capacity == seq_length)
    k        ~ feasible image counts under THIS L_target (renormalized)
    text     = L_target - sum_j(1 + V_j)   (>= 1 by construction)

so the realized document-length distribution IS the configured one —
image atoms are part of the length, not an addition to it. The stream is

    unshifted_stream = image_atom_1 .. image_atom_k + text + EOD

where ``image_atom_j = vision_start + V_j merged IMG tokens``. The
terminal EOD is the last input position's TARGET, never an input
position itself, exactly like the pure-text mock (input = stream[:-1],
labels = stream[1:], shift inside the document), so

    len(input_ids) = logical_document_length = L_target      (no +1).

L_target excludes raw vision patches, the EOD, segment-alignment
padding, the dummy THD tail, and all CP/SP physical padding. Documents
are indivisible packing atoms, packed next_fit in original order; a
document may legally reach L_target == S and occupy a whole window.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from examples.multimodal_dev.data.mock_varlen.distributions import (
    Categorical,
    TruncatedLognormal,
    draw_from_cdf,
    require_exact_dict,
    require_integer,
    require_number,
    seed_stream_rng,
)

# Per-document layout stream ids (disjoint from the dataset's content
# streams in mock_varlen/qwen35_vl.py: layout and content live in different seed
# namespaces, but keeping the ids globally unique makes collisions
# impossible if the namespaces ever merge).
_STREAM_PDOC_IMAGES = 21
_STREAM_PDOC_LENGTH = 22
_STREAM_PDOC_COMPONENT = 23

# The kernel consumes only "components"; the profile's other keys
# (image_sizes, plan_pool_windows, plan_seed) are extracted by the build
# helper before the kernel is constructed.
_COMPONENT_KEYS = frozenset({"name", "weight", "length", "images_per_document"})


# CONTEXT-SCALED PRACTICAL DEFAULT — STRUCTURAL/DERIVED, NOT FITTED.
#
# One default, active whenever no explicit profile is given: a
# reference-shaped practical default for document-oriented multimodal
# training, informed by (not fitted to) a customer hint of "95% short
# documents around 1-2K tokens, 5% long documents up to the context
# window". Lengths are FINAL logical sample lengths (image atoms
# included). Two components:
#
#   short (weight 95): the S-independent everyday mixture. length in
#     [512, min(4096, S)], mean 1536, sigma 0.30 — the 512 floor is a
#     structural support bound (mass below 512 was <0.02%; raised from 32
#     as a declared-range honesty fix, statistically a no-op). Bulk around
#     1-2K with
#     thin tails both ways; 0 or 1 image (45/55 — tuned by the official
#     snapshot gate: an all-short 128K window is the image-density worst
#     case, and denser rows (30/55/15 -> 89K, 45/45/10 -> 81K raw) put
#     the pool maximum past the 65,536 raw-patch guard
#     (--max-vision-patches-per-microbatch), driven by one seed-fixed
#     count+bucket clustering window; multi-image documents live in the
#     long component. Revisit if the guard budget is ever re-derived).
#   long (weight 5): scales with S. length in [2048, S], mean
#     round(0.7*S), sigma 0.8 — continuous support from 2K all the way
#     to S (no 2K-64K vacuum; a draw at S fills a whole window), with
#     meaningful mass in the top quarter of the context. 0/1/2/4/8
#     images (24/35/25/12/4) — max_count 8 is a practical structural
#     default for document-oriented multimodal SFT, NOT a model or
#     runtime limit (explicit profiles may configure more, gated only by
#     the deterministic startup scan).
#
# DERIVED (empirically calibrated, not re-chosen per release):
#   image_sizes: a 12-bucket subset of the Qwen3.5-VL processor ladder
#     from an empirical parity calibration, excluding the three largest
#     rows (1184x960, 960x1184, 1184x1184), so the default max image is
#     1120x896 = 3,920 raw patches.
#   plan_pool_windows "auto" = max(2048, ceil(2^26 / seq_length))
#   plan_seed 1234 = the calibration-snapshot convention (layout keyed
#     independently of the training --seed)
#
# Domain is [4096, 131072] inclusive — out-of-range seq_length raises
# (provide a full explicit profile instead of trusting extrapolation).
# Component weights are document-count proportions; realized shares,
# window composition, padding (~14-22% pool padding across the default
# domain per the official five-S snapshots is the known
# price of sample atomicity + next_fit under a heavy-tailed mixture) and
# guard margins are measured by the simulator snapshots, never asserted.
DEFAULT_IMAGE_SIZES: dict[str, Any] = {
    "resolutions": [
        [224, 224],
        [448, 224],
        [224, 448],
        [352, 448],
        [448, 352],
        [448, 448],
        [672, 448],
        [448, 672],
        [896, 672],
        [672, 896],
        [1120, 896],
        [896, 1120],
    ],
    "weights": [33, 9, 9, 7, 7, 13, 6.5, 6.5, 5.5, 5.5, 3.5, 3.5],
}


def context_scaled_default(seq_length: int) -> dict[str, Any]:
    """Resolve the context-scaled practical default profile for one S.

    The single implementation of the zero-config default (external
    consumers such as simulators must call this, never re-derive the
    formulas; there is exactly ONE default). Returns a COMPLETE
    profile dict — the same schema a full explicit config must carry.
    """
    s = require_integer(seq_length, what="seq_length")
    if not 4096 <= s <= 131072:
        raise ValueError(
            f"context_scaled_default supports seq_length in [4096, 131072]; got {s}. "
            "Outside this range there is no default hypothesis to scale — provide a "
            "full explicit profile via --multimodal-varlen-mock-dataset-config-json."
        )
    return {
        "components": [
            {
                "name": "short",
                "weight": 95.0,
                "length": {"min": 512, "max": min(4096, s), "mean": 1536, "sigma": 0.30},
                "images_per_document": {"counts": [0, 1], "weights": [45, 55]},
            },
            {
                "name": "long",
                "weight": 5.0,
                "length": {"min": 2048, "max": s, "mean": round(0.7 * s), "sigma": 0.8},
                "images_per_document": {"counts": [0, 1, 2, 4, 8], "weights": [24, 35, 25, 12, 4]},
            },
        ],
        "image_sizes": {
            "resolutions": [list(resolution) for resolution in DEFAULT_IMAGE_SIZES["resolutions"]],
            "weights": list(DEFAULT_IMAGE_SIZES["weights"]),
        },
        "plan_pool_windows": "auto",
        "plan_seed": 1234,
    }


def _align_up(value: int, alignment: int) -> int:
    return -(-int(value) // int(alignment)) * int(alignment)


def _parse_length_model(spec: Any, *, what: str) -> TruncatedLognormal:
    """Truncated-lognormal length model with the strict field rules."""
    spec = require_exact_dict(spec, {"min", "max", "mean", "sigma"}, what=what)
    return TruncatedLognormal(
        mean=require_number(spec["mean"], what=f"{what}.mean"),
        sigma=require_number(spec["sigma"], what=f"{what}.sigma"),
        minimum=require_integer(spec["min"], what=f"{what}.min"),
        maximum=require_integer(spec["max"], what=f"{what}.max"),
    )


@dataclass(frozen=True)
class _Component:
    """One parsed mixture component (internal)."""

    name: str
    weight: float
    length: TruncatedLognormal
    # Positive-weight image counts only (weight-0 entries are disabled at
    # parse time; a component disables images by configuring counts [0]).
    image_counts: tuple[int, ...]
    image_weights: tuple[float, ...]


@dataclass(frozen=True)
class DocumentSpan:
    """One contiguous span of a document's INPUT-token layout.

    Only two span types exist: ``image`` atoms (vision_start + merged
    placeholders, carrying their bucket geometry) and the single ``text``
    run. The terminal EOD is not a span: it is never an input position
    (it is the last input position's target, dataset-side).
    """

    span_type: str  # image | text
    length: int
    bucket_index: int = -1
    merged_tokens: int = 0
    raw_patches: int = 0


@dataclass(frozen=True)
class PackedDocument:
    """One whole document (never split across windows).

    ``component_index`` identifies the mixture component the document was
    drawn from (per-component statistics address components by name via
    the generator's ``components`` tuple). ``image_count_conditioned`` is
    True when the drawn L_target made some positive-weight counts
    infeasible and the count categorical was renormalized over the
    feasible subset; ``image_geometry_substitutions`` counts the atoms
    substituted down to the smallest drawable bucket so they fit inside
    L_target (geometry-conditioned iff > 0). The image COUNT is never
    changed after it is drawn. ``supervised_tokens`` counts input positions whose shifted
    TARGET is ordinary text or the terminal EOD (all of them,
    full-sequence supervision): T + 1 with images, T without.
    """

    component_index: int
    spans: tuple[DocumentSpan, ...]
    logical_length: int
    supervised_tokens: int
    image_count_conditioned: bool
    image_geometry_substitutions: int


@dataclass(frozen=True)
class PackedDocumentWindow:
    """One packed window; the accounting identity holds per window:

    ``window_capacity == logical_tokens + alignment_padding + tail_padding``
    """

    documents: tuple[PackedDocument, ...]
    seq_lens: tuple[int, ...]  # logical per-document lengths, stream order
    logical_tokens: int
    alignment_padding: int  # sum of align_up(L, A) - L over documents
    tail_padding: int  # capacity left after the last aligned document
    supervised_tokens: int
    images: tuple[DocumentSpan, ...]
    vision_tokens: int  # sum of image span lengths (vision_start + merged)
    raw_patches: int


class PackedDocumentPlanGenerator:
    """Deterministic document -> window plan generator.

    Unlike a pretraining-style token stream sliced at window lines,
    documents here are indivisible: each draws a mixture component, its
    FINAL logical length L_target, and the images living inside that
    length, then is packed next_fit in original order — the document
    that overflows window N closes it and opens window N+1, so
    ``plan(N)`` is a strict prefix of ``plan(M)`` for ``N < M``. Every
    component's length.max is validated against the aligned budget
    ``floor(window_capacity / A) * A``, so every document fits some
    window by construction (L_target <= the aligned window budget —
    == seq_length at every shipped call site, where window_capacity ==
    seq_length).

    Args:
        seq_length: window size S; the physical window capacity IS S.
        num_windows: number of CLOSED windows to plan.
        seed: the profile's plan_seed; all layout randomness derives from
            ``(seed, document ordinal)`` via SeedSequence-keyed numpy
            streams.
        config: the packed_document profile WITHOUT plan_pool_windows /
            plan_seed (strict: unknown keys rejected, bools rejected where
            numbers are expected).
        bucket_merged_tokens / bucket_raw_patches / bucket_weights: image
            bucket support and categorical weights (kernel authority; the
            config's image_sizes table never reaches the kernel).
        segment_alignment: A, provider-derived — e.g.
            ``(2*CP*(TP if SP else 1)) if CP > 1 else (TP if SP else 1)``.
    """

    def __init__(
        self,
        *,
        seq_length: int,
        num_windows: int,
        seed: int,
        config: dict[str, Any],
        bucket_merged_tokens: list[int],
        bucket_raw_patches: list[int],
        bucket_weights: list[float],
        segment_alignment: int = 1,
    ) -> None:
        seq_length = require_integer(seq_length, what="seq_length")
        num_windows = require_integer(num_windows, what="num_windows")
        seed = require_integer(seed, what="seed")
        if seq_length <= 0 or num_windows <= 0:
            raise ValueError("seq_length and num_windows must be positive.")
        if not (len(bucket_merged_tokens) == len(bucket_raw_patches) == len(bucket_weights) > 0):
            raise ValueError("Bucket arrays must be non-empty and of equal length.")
        # Element validity, not just shape: a negative merged count would make
        # an image span shorter than its vision_start marker (silently writing
        # no tokens while still emitting a grid row), and a negative raw count
        # would let a pool slip under the vision patch-budget guard.
        for index, merged in enumerate(bucket_merged_tokens):
            if require_integer(merged, what=f"bucket_merged_tokens[{index}]") < 1:
                raise ValueError(f"bucket_merged_tokens[{index}] must be >= 1, got {merged}.")
        for index, raw in enumerate(bucket_raw_patches):
            if require_integer(raw, what=f"bucket_raw_patches[{index}]") < 1:
                raise ValueError(f"bucket_raw_patches[{index}] must be >= 1, got {raw}.")
        weight_array = np.asarray(bucket_weights, dtype=np.float64)
        if (
            not np.all(np.isfinite(weight_array))
            or np.any(weight_array < 0)
            # A finite-SUM check too: individually finite weights can still
            # overflow the normalizer to inf and corrupt every draw.
            or not np.isfinite(weight_array.sum())
            or weight_array.sum() <= 0
        ):
            raise ValueError(
                "Bucket weights must be finite and non-negative with a positive finite sum."
            )

        unknown = set(config) - {"components"}
        if unknown:
            raise ValueError(
                f"packed_document config has unknown key(s) {sorted(unknown)}; "
                "allowed: ['components']."
            )
        if "components" not in config:
            raise ValueError("packed_document config is missing the required key 'components'.")

        self.seq_length = int(seq_length)
        self.num_windows = int(num_windows)
        self.seed = int(seed)
        self.segment_alignment = require_integer(segment_alignment, what="segment_alignment")
        if self.segment_alignment < 1:
            raise ValueError(f"segment_alignment must be >= 1, got {self.segment_alignment}.")
        # The physical window capacity IS the model sequence length; the
        # provider rejects any topology where max_seqlen_per_dp_cp_rank*CP
        # differs from it.
        self.window_capacity = self.seq_length
        if self.window_capacity % self.segment_alignment != 0:
            # Aligned document costs are multiples of A, so with a
            # non-multiple capacity the tail_padding would not be a multiple
            # of the segment alignment — the runtime packer (whose every pad
            # chunk is a multiple of A) could not realize the physical target.
            raise ValueError(
                f"seq_length {self.seq_length} must be a multiple of the CP/SP "
                f"segment alignment {self.segment_alignment} "
                "(= (2*CP*(TP if SP else 1)) if CP > 1 else (TP if SP else 1)); "
                "pick --max-seqlen-per-dp-cp-rank so that its product with CP is."
            )
        self.bucket_merged = tuple(int(v) for v in bucket_merged_tokens)
        self.bucket_raw = tuple(int(v) for v in bucket_raw_patches)
        self.bucket_probs = weight_array / weight_array.sum()

        # Largest logical length whose ALIGNED cost still fits the capacity;
        # component length maxima and image feasibility all target this.
        self._logical_budget = (self.window_capacity // self.segment_alignment) * (
            self.segment_alignment
        )
        drawable_bucket_indices = [index for index, weight in enumerate(weight_array) if weight > 0]
        self._smallest_drawable_bucket = min(
            drawable_bucket_indices, key=lambda index: (self.bucket_merged[index], index)
        )
        self._smallest_atom = 1 + self.bucket_merged[self._smallest_drawable_bucket]

        self.components = self._parse_components(config["components"])
        component_weights = np.asarray(
            [component.weight for component in self.components], dtype=np.float64
        )
        total_component_weight = float(component_weights.sum())
        if not np.isfinite(total_component_weight):
            # Individually finite weights whose SUM overflows would zero the
            # normalized CDF and silently pin every draw to the last
            # component — only the weight ratios matter, so scale them down.
            raise ValueError(
                "component weights sum to a non-finite value "
                f"({[component.weight for component in self.components]}); scale the "
                "weights down (only their ratios matter)."
            )
        self._component_cdf = np.cumsum(component_weights / total_component_weight)

        self._windows: list[PackedDocumentWindow] = []
        self._build()

    def _parse_components(self, spec: Any) -> tuple[_Component, ...]:
        """Strict parse + startup feasibility of the mixture components.

        Image-count feasibility is TWO-layered (the count set depends on
        the drawn L_target, so a static filter would be wrong — e.g. a
        short component with length.min 512 must still draw images for its
        1-2K documents):

        1. at length.min at least one positive-weight count must fit
           (usually count 0), else no document at the floor is
           constructible;
        2. at length.max EVERY positive-weight count must fit, else that
           count could never be drawn at any length — a dead config that
           fails loudly instead of silently skewing the distribution.

        The per-document restriction to the counts feasible under the
        DRAWN L_target happens at sampling time (renormalized), and is
        reported via ``image_count_conditioning_events``.
        """
        if not isinstance(spec, (list, tuple)) or not spec:
            raise ValueError("components must be a non-empty list of component objects.")
        parsed: list[_Component] = []
        names: set[str] = set()
        for index, entry in enumerate(spec):
            what = f"components[{index}]"
            entry = require_exact_dict(entry, _COMPONENT_KEYS, what=what)
            name = entry["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"{what}.name must be a non-empty string.")
            if name in names:
                raise ValueError(f"{what}.name {name!r} is not unique.")
            names.add(name)
            weight = require_number(entry["weight"], what=f"{what}.weight")
            if weight <= 0:
                # Disable a component by deleting it, not by zeroing it.
                raise ValueError(f"{what}.weight must be > 0, got {weight}.")
            length = _parse_length_model(entry["length"], what=f"{what}.length")
            if length.maximum > self._logical_budget:
                raise ValueError(
                    f"{what} ({name!r}): length.max {length.maximum} exceeds the aligned "
                    f"window budget {self._logical_budget} (capacity {self.window_capacity}, "
                    f"alignment {self.segment_alignment}); a drawn document could never fit "
                    "any window."
                )
            categorical = Categorical(
                entry["images_per_document"], what=f"{what}.images_per_document"
            )
            positive = [
                (count, float(weight_value))
                for count, weight_value in zip(categorical.counts, categorical.weights)
                if weight_value > 0
            ]
            if not any(count * self._smallest_atom + 1 <= length.minimum for count, _ in positive):
                raise ValueError(
                    f"{what} ({name!r}): no positive-weight images_per_document count fits a "
                    f"length.min={length.minimum} document even with the smallest drawable "
                    f"atom ({self._smallest_atom} tokens); include count 0 or raise "
                    "length.min."
                )
            for count, _ in positive:
                if count * self._smallest_atom + 1 > length.maximum:
                    raise ValueError(
                        f"{what} ({name!r}): images_per_document count {count} cannot fit "
                        f"even a length.max={length.maximum} document with the smallest "
                        f"drawable atom ({self._smallest_atom} tokens) — a dead entry that "
                        "could never be drawn; remove it or raise length.max."
                    )
            parsed.append(
                _Component(
                    name=name,
                    weight=float(weight),
                    length=length,
                    image_counts=tuple(count for count, _ in positive),
                    image_weights=tuple(weight_value for _, weight_value in positive),
                )
            )
        return tuple(parsed)

    # ------------------------------------------------------------------
    # Per-document generation
    # ------------------------------------------------------------------

    def _rng(self, doc_id: int, stream: int) -> np.random.Generator:
        return seed_stream_rng(self.seed, doc_id, stream)

    def _draw_document(self, doc_id: int) -> PackedDocument:
        component_rng = self._rng(doc_id, _STREAM_PDOC_COMPONENT)
        component_index = draw_from_cdf(component_rng, self._component_cdf)
        component = self.components[component_index]

        length_target = component.length.sample(self._rng(doc_id, _STREAM_PDOC_LENGTH))

        # Image count k comes from the categorical restricted to the counts
        # feasible under THIS document's drawn length (constructor-verified:
        # the restriction is never empty — count 0 or a small count fits at
        # length.min, and every positive count fits at length.max). k is a
        # contract once drawn: if the drawn geometries overshoot L_target,
        # the LARGEST atom (ties -> earliest) is substituted with the
        # smallest drawable bucket until the atoms fit — images are never
        # dropped, and text = L_target - atoms >= 1 by construction.
        image_rng = self._rng(doc_id, _STREAM_PDOC_IMAGES)
        feasible = [
            slot
            for slot, count in enumerate(component.image_counts)
            if count * self._smallest_atom + 1 <= length_target
        ]
        count_conditioned = len(feasible) < len(component.image_counts)
        weights = np.asarray([component.image_weights[slot] for slot in feasible], dtype=np.float64)
        count = component.image_counts[
            feasible[draw_from_cdf(image_rng, np.cumsum(weights / weights.sum()))]
        ]
        bucket_list = [
            int(bucket)
            for bucket in image_rng.choice(len(self.bucket_probs), size=count, p=self.bucket_probs)
        ]
        substitutions = 0
        atom_tokens = sum(1 + self.bucket_merged[bucket] for bucket in bucket_list)
        # Each pass demotes one image to the smallest drawable bucket, so
        # `count` passes exhaust every substitution available. Startup
        # feasibility guarantees convergence before that; the bound turns a
        # broken invariant into a loud error instead of a silent hang.
        while atom_tokens + 1 > length_target:
            if substitutions >= count:
                raise RuntimeError(
                    f"geometry substitution failed to fit {count} image(s) into a "
                    f"{length_target}-token document even at the smallest drawable "
                    "bucket; startup feasibility validation should have rejected this."
                )
            largest = max(
                range(len(bucket_list)), key=lambda i: (self.bucket_merged[bucket_list[i]], -i)
            )
            atom_tokens -= (
                self.bucket_merged[bucket_list[largest]]
                - self.bucket_merged[self._smallest_drawable_bucket]
            )
            bucket_list[largest] = self._smallest_drawable_bucket
            substitutions += 1
        buckets = tuple(bucket_list)

        text_tokens = length_target - atom_tokens
        if text_tokens < 1:
            raise RuntimeError(
                f"document layout produced {text_tokens} text tokens; feasibility "
                "validation guarantees at least one."
            )

        spans: list[DocumentSpan] = [
            DocumentSpan(
                "image",
                1 + self.bucket_merged[bucket],
                bucket_index=bucket,
                merged_tokens=self.bucket_merged[bucket],
                raw_patches=self.bucket_raw[bucket],
            )
            for bucket in buckets
        ]
        spans.append(DocumentSpan("text", text_tokens))

        # Full-sequence supervision: every input position whose shifted
        # target is text or the terminal EOD. With images the last atom
        # position targets the first text token, so T + 1; image-internal
        # targets are masked; without images, T (matching the pure-text
        # mock exactly).
        supervised = text_tokens + (1 if count else 0)
        return PackedDocument(
            component_index=component_index,
            spans=tuple(spans),
            logical_length=length_target,
            supervised_tokens=supervised,
            image_count_conditioned=count_conditioned,
            image_geometry_substitutions=substitutions,
        )

    # ------------------------------------------------------------------
    # next_fit packing with lookahead
    # ------------------------------------------------------------------

    def _build(self) -> None:
        capacity = self.window_capacity
        windows: list[PackedDocumentWindow] = []
        pending: list[PackedDocument] = []
        pending_cost = 0
        doc_id = 0
        while len(windows) < self.num_windows:
            document = self._draw_document(doc_id)
            doc_id += 1
            cost = _align_up(document.logical_length, self.segment_alignment)
            # next_fit with lookahead: the overflowing document CLOSES the
            # current window and opens the next one, so plan(N) is a strict
            # prefix of plan(M) for N < M — including the last window, which
            # only closes once its overflowing successor has been read.
            if pending and pending_cost + cost > capacity:
                windows.append(self._close_window(pending, pending_cost))
                pending = [document]
                pending_cost = cost
            else:
                pending.append(document)
                pending_cost += cost
        self._windows = windows

    def _close_window(
        self, documents: list[PackedDocument], physical_cost: int
    ) -> PackedDocumentWindow:
        seq_lens = tuple(document.logical_length for document in documents)
        logical_tokens = int(sum(seq_lens))
        alignment_padding = physical_cost - logical_tokens
        tail_padding = self.window_capacity - physical_cost
        if tail_padding < 0:
            raise RuntimeError(
                f"window physical cost {physical_cost} exceeds capacity "
                f"{self.window_capacity}; the next_fit closure rule must never allow this."
            )
        # Image atoms in window order (document order -> in-document).
        images = tuple(
            span for document in documents for span in document.spans if span.span_type == "image"
        )
        window = PackedDocumentWindow(
            documents=tuple(documents),
            seq_lens=seq_lens,
            logical_tokens=logical_tokens,
            alignment_padding=alignment_padding,
            tail_padding=tail_padding,
            supervised_tokens=sum(document.supervised_tokens for document in documents),
            images=images,
            vision_tokens=sum(image.length for image in images),
            raw_patches=sum(image.raw_patches for image in images),
        )
        return window

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_windows

    def window(self, idx: int) -> PackedDocumentWindow:
        """Plan for window *idx* (identity: capacity == logical + padding)."""
        return self._windows[int(idx)]

    # Pool aggregates derived on demand: the pool is fixed at
    # construction, so no aggregate needs incremental maintenance.

    @property
    def total_documents(self) -> int:
        return sum(len(w.documents) for w in self._windows)

    @property
    def image_count_conditioning_events(self) -> int:
        return sum(d.image_count_conditioned for w in self._windows for d in w.documents)

    @property
    def image_conditioning_events(self) -> int:
        return sum(d.image_geometry_substitutions > 0 for w in self._windows for d in w.documents)

    @property
    def pool_max_raw_patches(self) -> int:
        return max(w.raw_patches for w in self._windows)

    @property
    def pool_max_image_raw_patches(self) -> int:
        # default=0 belongs here and nowhere else: a text-only profile draws
        # no images at all, while the window sequence is never empty
        # (num_windows >= 1 is enforced at construction).
        return max((i.raw_patches for w in self._windows for i in w.images), default=0)

    @property
    def pool_max_images(self) -> int:
        return max(len(w.images) for w in self._windows)

    @property
    def pool_max_logical_tokens(self) -> int:
        return max(w.logical_tokens for w in self._windows)

    @property
    def total_padding_fraction(self) -> float:
        """Pool-aggregate (alignment + tail) padding fraction of capacity."""
        return sum(w.alignment_padding + w.tail_padding for w in self._windows) / float(
            self.num_windows * self.window_capacity
        )
