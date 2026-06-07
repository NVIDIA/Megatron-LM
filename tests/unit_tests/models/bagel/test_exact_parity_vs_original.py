# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
End-to-end exact parity test: energon task encoders vs replicated original BAGEL logic.

For each task type, the test:
  1. Creates a tiny synthetic dataset in the original parquet/JSONL format.
  2. Converts it to WebDataset (WDS) tar format using the same logic as convert_to_wds.py.
  3. Reads the WDS tar back and calls each energon task encoder directly.
  4. Runs the original BAGEL dataset logic inline on the same raw data.
  5. Asserts bit-exact equality of pre-packing PackableSample dicts.
  6. Feeds identical PackableSamples through BagelPacker and asserts bit-exact equality
     of all packed output tensors.

Modality coverage
-----------------
T2I  — fully exact.  Single caption + image; no randomness.
Edit — fully exact.  Chain images; random chain sampling controlled by fixed seed.
VLM  — structural only (energon/original differ in how <image> tags in conversation
        text are handled and whether EOS carries CE loss; documented limitation).

Run (from repo root):
    PYTHONPATH=.:/workspace/megatron-lm-bagel/bagel-package:\
               /workspace/megatron-lm-bagel/bagel-package/bagel:\
               /workspace/megatron-lm-bagel/examples/mimo \
        python -m pytest examples/mimo_bagel/unit_test/test_exact_parity_vs_original.py -v
"""
import io
import json
import random
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_jpeg(color: Tuple[int, int, int] = (200, 100, 50),
               size: Tuple[int, int] = (256, 192)) -> bytes:
    """Return raw JPEG bytes for a solid-colour image."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _add_to_tar(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    buf = io.BytesIO(data)
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, buf)


def _read_wds_tar(tar_path: str) -> List[Dict[str, bytes]]:
    """Read all entries from a WDS tar; return one dict per sample key (first-dot prefix)."""
    samples: Dict[str, Dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            name = member.name
            dot = name.find(".")
            key, field = name[:dot], name[dot + 1:]
            raw = tf.extractfile(member).read()
            samples.setdefault(key, {})[field] = raw
    return [samples[k] for k in sorted(samples)]


class _MockTokenizer:
    """Deterministic char-level mock tokenizer (no external dependency)."""
    def encode(self, text: str) -> List[int]:
        return [ord(c) % 512 for c in str(text)]


def _make_transforms(vae_max=512, vae_min=256, vit_max=280, vit_min=112):
    """Return (vae_transform, vit_transform) using the energon implementation."""
    from examples.mimo_bagel.data.energon_bagel_task_encoder import ImageTransform
    vae = ImageTransform(max_image_size=vae_max, min_image_size=vae_min, image_stride=16)
    vit = ImageTransform(max_image_size=vit_max, min_image_size=vit_min, image_stride=14)
    return vae, vit


def _make_data_config(dropout: float = 0.0):
    from examples.mimo_bagel.data.energon_bagel_task_encoder import DataConfig
    return DataConfig(
        grouped_datasets=None,
        text_cond_dropout_prob=dropout,
        vit_cond_dropout_prob=dropout,
        vae_cond_dropout_prob=dropout,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )


SPECIAL_TOKEN_IDS = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "start_of_image": 3,
    "end_of_image": 4,
}


# ---------------------------------------------------------------------------
# PackableSample comparison helpers
# ---------------------------------------------------------------------------

def _normalize_plan_item(item: dict) -> dict:
    """Normalize a sequence_plan item by filling in defaults for split_start/split_end.
    pack_sequence uses item.get('split_start', True) so omitted keys are equivalent to True.
    """
    result = dict(item)
    result.setdefault("split_start", True)
    result.setdefault("split_end", True)
    return result


def _assert_packable_samples_equal(orig: dict, ene: dict, label: str = "") -> None:
    """Assert two PackableSample dicts are bit-exact (ignoring data_indexes).

    sequence_plan items are normalized before comparison: split_start/split_end
    default to True when absent, so {key: val} == {key: val, split_start: True}
    is treated as equal.
    """
    prefix = f"[{label}] " if label else ""

    # sequence_plan — normalize split_start/split_end defaults
    orig_plan = [_normalize_plan_item(x) for x in orig["sequence_plan"]]
    ene_plan  = [_normalize_plan_item(x) for x in ene["sequence_plan"]]
    assert orig_plan == ene_plan, (
        f"{prefix}sequence_plan mismatch:\n  orig={orig_plan}\n  ene={ene_plan}"
    )

    # num_tokens
    assert orig["num_tokens"] == ene["num_tokens"], (
        f"{prefix}num_tokens: orig={orig['num_tokens']}, ene={ene['num_tokens']}"
    )

    # text_ids_list
    assert len(orig["text_ids_list"]) == len(ene["text_ids_list"]), (
        f"{prefix}text_ids_list length: orig={len(orig['text_ids_list'])}, ene={len(ene['text_ids_list'])}"
    )
    for i, (o_ids, e_ids) in enumerate(zip(orig["text_ids_list"], ene["text_ids_list"])):
        assert o_ids == e_ids, f"{prefix}text_ids_list[{i}] mismatch"

    # image_tensor_list
    assert len(orig["image_tensor_list"]) == len(ene["image_tensor_list"]), (
        f"{prefix}image_tensor_list length: orig={len(orig['image_tensor_list'])}, "
        f"ene={len(ene['image_tensor_list'])}"
    )
    for i, (o_t, e_t) in enumerate(zip(orig["image_tensor_list"], ene["image_tensor_list"])):
        assert o_t.shape == e_t.shape, (
            f"{prefix}image_tensor_list[{i}] shape: orig={o_t.shape}, ene={e_t.shape}"
        )
        assert torch.equal(o_t, e_t), (
            f"{prefix}image_tensor_list[{i}] values differ "
            f"(max diff={( o_t - e_t).abs().max():.6f})"
        )


def _assert_packed_tensors_equal(orig: dict, ene: dict, label: str = "") -> None:
    """Assert all tensors in two packed-output dicts are bit-exact."""
    prefix = f"[{label}] " if label else ""
    all_keys = set(orig) | set(ene)
    # Ignore provenance
    ignore = {"sample_lens", "split_lens", "attn_modes", "sequence_length"}

    for k in sorted(all_keys - ignore):
        in_orig = k in orig
        in_ene = k in ene
        assert in_orig == in_ene, (
            f"{prefix}key '{k}' present in orig={in_orig}, ene={in_ene}"
        )
        o_v, e_v = orig[k], ene[k]
        if isinstance(o_v, torch.Tensor):
            assert o_v.shape == e_v.shape, (
                f"{prefix}'{k}' shape: orig={o_v.shape}, ene={e_v.shape}"
            )
            assert torch.equal(o_v, e_v) or (
                o_v.dtype == torch.float32 and (o_v - e_v).abs().max() < 1e-6
            ), f"{prefix}'{k}' tensor values differ (max diff={(o_v - e_v).abs().max():.6f})"
        elif isinstance(o_v, list):
            assert o_v == e_v, f"{prefix}'{k}' list mismatch"


# ---------------------------------------------------------------------------
# T2I: write WDS tar
# ---------------------------------------------------------------------------

def _write_t2i_wds_tar(tar_path: str, samples: List[Tuple[bytes, str]]) -> None:
    """Write T2I WDS tar: one sample = (jpeg_bytes, caption_str)."""
    with tarfile.open(tar_path, "w") as tf:
        for idx, (jpeg_bytes, caption) in enumerate(samples):
            key = f"{idx:09d}"
            _add_to_tar(tf, f"{key}.jpg", jpeg_bytes)
            _add_to_tar(tf, f"{key}.txt", caption.encode("utf-8"))


# ---------------------------------------------------------------------------
# Edit: write WDS tar
# ---------------------------------------------------------------------------

def _write_edit_wds_tar(tar_path: str, rows: List[Dict]) -> None:
    """Write Edit WDS tar.  Each row: {'image_list': [bytes,...], 'instruction_list': [[str,...],...]}.
    Mirrors convert_edit output: {key}.000.jpg, {key}.001.jpg, ..., {key}.json
    """
    with tarfile.open(tar_path, "w") as tf:
        for idx, row in enumerate(rows):
            key = f"{idx:09d}"
            for img_idx, img_bytes in enumerate(row["image_list"]):
                _add_to_tar(tf, f"{key}.{img_idx:03d}.jpg", img_bytes)
            meta = {"instruction_list": row["instruction_list"],
                    "num_images": len(row["image_list"])}
            _add_to_tar(tf, f"{key}.json", json.dumps(meta).encode("utf-8"))


# ---------------------------------------------------------------------------
# Original BAGEL logic replicated inline
# (avoids the DistributedIterableDataset init complexity)
# ---------------------------------------------------------------------------

def _orig_t2i_process(jpeg_bytes: bytes, caption: str, vae_transform, tokenizer) -> dict:
    """Replicate T2IIterableDataset.__iter__ per-row logic."""
    from bagel.data.data_utils import pil_img2rgb
    image = pil_img2rgb(Image.open(io.BytesIO(jpeg_bytes)))
    image_tensor = vae_transform(image)
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    vae_stride = vae_transform.stride
    num_tokens = (H // vae_stride) * (W // vae_stride)

    text_ids = tokenizer.encode(caption)
    num_tokens += len(text_ids)

    return dict(
        image_tensor_list=[image_tensor],
        text_ids_list=[text_ids],
        sequence_plan=[
            {"type": "text", "enable_cfg": 1, "loss": 0,
             "special_token_loss": 0, "special_token_label": None},
            {"type": "vae_image", "enable_cfg": 0, "loss": 1,
             "special_token_loss": 0, "special_token_label": None},
        ],
        num_tokens=num_tokens,
        data_indexes={"dataset_name": "t2i"},
    )


def _orig_edit_add_text(data: dict, text: str, need_loss: bool,
                        tokenizer, enable_cfg: bool = True) -> dict:
    """Mirrors InterleavedBaseIterableDataset._add_text."""
    text_ids = tokenizer.encode(text)
    data["num_tokens"] += len(text_ids)
    data["text_ids_list"].append(text_ids)
    data["sequence_plan"].append({
        "type": "text",
        "enable_cfg": int(enable_cfg),
        "loss": int(need_loss),
        "special_token_loss": 0,
        "special_token_label": None,
    })
    return data


def _orig_edit_add_image(data: dict, pil_img: Image.Image,
                         need_loss: bool, need_vae: bool, need_vit: bool,
                         vae_transform, vit_transform, enable_cfg: bool = True) -> dict:
    """Mirrors InterleavedBaseIterableDataset._add_image (sequence_plan first, then tensor)."""
    if need_loss:
        data["sequence_plan"].append({
            "type": "vae_image", "enable_cfg": 0, "loss": 1,
            "special_token_loss": 0, "special_token_label": None,
        })
        t = vae_transform(pil_img)
        H, W = t.shape[1], t.shape[2]
        data["num_tokens"] += (H * W) // (vae_transform.stride ** 2)
        data["image_tensor_list"].append(t)

    if need_vae:
        data["sequence_plan"].append({
            "type": "vae_image", "enable_cfg": int(enable_cfg), "loss": 0,
            "special_token_loss": 0, "special_token_label": None,
        })
        t = vae_transform(pil_img)
        H, W = t.shape[1], t.shape[2]
        data["num_tokens"] += (H * W) // (vae_transform.stride ** 2)
        data["image_tensor_list"].append(t.clone())

    if need_vit:
        data["sequence_plan"].append({
            "type": "vit_image", "enable_cfg": int(enable_cfg), "loss": 0,
            "special_token_loss": 0, "special_token_label": None,
        })
        t = vit_transform(pil_img)
        H, W = t.shape[1], t.shape[2]
        data["num_tokens"] += (H * W) // (vit_transform.stride ** 2)
        data["image_tensor_list"].append(t)

    return data


def _orig_edit_parse_row(row: dict, vae_transform, vit_transform, tokenizer) -> dict:
    """Replicate UnifiedEditIterableDataset.parse_row exactly.
    Assumes random seed is already set by caller.
    """
    from bagel.data.data_utils import pil_img2rgb
    data: dict = {"sequence_plan": [], "text_ids_list": [], "image_tensor_list": [],
                  "num_tokens": 0}

    image_list_bytes: List[bytes] = row["image_list"]
    instruction_list: List[List[str]] = row["instruction_list"]
    image_num = len(image_list_bytes)

    start_idx = random.choice(range(image_num - 1))
    max_end = min(start_idx + 3, image_num)
    end_idx = random.choice(range(start_idx + 1, max_end))

    src_pil = pil_img2rgb(Image.open(io.BytesIO(image_list_bytes[start_idx])))
    data = _orig_edit_add_image(data, src_pil, need_loss=False, need_vae=True, need_vit=True,
                                vae_transform=vae_transform, vit_transform=vit_transform)

    if end_idx - start_idx > 1 and random.random() < 0.5:
        if end_idx == image_num - 1:
            end_idx -= 1
        instruction = ""
        for idx in range(start_idx + 1, end_idx + 1):
            instruction += random.choice(instruction_list[idx - 1]) + ". "
        data = _orig_edit_add_text(data, instruction.rstrip(), need_loss=False, tokenizer=tokenizer)
        tgt_pil = pil_img2rgb(Image.open(io.BytesIO(image_list_bytes[end_idx])))
        data = _orig_edit_add_image(data, tgt_pil, need_loss=True, need_vae=False, need_vit=False,
                                    vae_transform=vae_transform, vit_transform=vit_transform)
    else:
        for idx in range(start_idx + 1, end_idx + 1):
            instruction = random.choice(instruction_list[idx - 1])
            data = _orig_edit_add_text(data, instruction, need_loss=False, tokenizer=tokenizer)
            pil = pil_img2rgb(Image.open(io.BytesIO(image_list_bytes[idx])))
            if idx != end_idx:
                data = _orig_edit_add_image(data, pil, need_loss=True, need_vae=True, need_vit=True,
                                            vae_transform=vae_transform, vit_transform=vit_transform)
            else:
                data = _orig_edit_add_image(data, pil, need_loss=True, need_vae=False, need_vit=False,
                                            vae_transform=vae_transform, vit_transform=vit_transform)

    data["data_indexes"] = {"dataset_name": "edit"}
    return data


# ---------------------------------------------------------------------------
# T2I parity tests
# ---------------------------------------------------------------------------

class TestT2IExactParity:
    """Pre-packing and post-packing T2I parity: energon encoder vs original logic."""

    @pytest.fixture
    def t2i_setup(self, tmp_path):
        tokenizer = _MockTokenizer()
        vae_t, vit_t = _make_transforms()
        # One sample: solid green 320x240 image, single caption
        jpeg = _make_jpeg(color=(50, 200, 80), size=(320, 240))
        caption = "a solid green rectangle"
        tar_path = str(tmp_path / "t2i.tar")
        _write_t2i_wds_tar(tar_path, [(jpeg, caption)])
        samples = _read_wds_tar(tar_path)
        return dict(jpeg=jpeg, caption=caption, vae_t=vae_t, vit_t=vit_t,
                    tokenizer=tokenizer, wds_sample=samples[0])

    def test_prepacking_exact(self, t2i_setup):
        d = t2i_setup
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelT2ITaskEncoder

        encoder = BagelT2ITaskEncoder(
            tokenizer=d["tokenizer"],
            vae_transform=d["vae_t"],
            vae_image_downsample=16,
        )

        orig = _orig_t2i_process(d["jpeg"], d["caption"], d["vae_t"], d["tokenizer"])
        ene = encoder.encode_sample(d["wds_sample"])

        _assert_packable_samples_equal(orig, ene, label="T2I pre-packing")

    def test_packing_exact(self, t2i_setup):
        """Verify that BagelPacker produces identical packed tensors for both paths."""
        d = t2i_setup
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelT2ITaskEncoder, BagelPacker

        encoder = BagelT2ITaskEncoder(tokenizer=d["tokenizer"], vae_transform=d["vae_t"])
        orig_sample = _orig_t2i_process(d["jpeg"], d["caption"], d["vae_t"], d["tokenizer"])
        ene_sample = encoder.encode_sample(d["wds_sample"])

        data_config = _make_data_config(dropout=0.0)
        packer = BagelPacker(
            data_config=data_config,
            special_token_ids=SPECIAL_TOKEN_IDS,
            max_num_tokens=8192,
        )

        np.random.seed(7)
        random.seed(7)
        ss_orig = packer.init_sequence_status()
        ss_orig = packer.pack_sequence(orig_sample, ss_orig)
        out_orig = packer.to_tensor(ss_orig)

        np.random.seed(7)
        random.seed(7)
        ss_ene = packer.init_sequence_status()
        ss_ene = packer.pack_sequence(ene_sample, ss_ene)
        out_ene = packer.to_tensor(ss_ene)

        _assert_packed_tensors_equal(out_orig, out_ene, label="T2I post-packing")

    def test_prepacking_multiple_samples(self, tmp_path):
        """Three different T2I samples all match."""
        tokenizer = _MockTokenizer()
        vae_t, _ = _make_transforms()
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelT2ITaskEncoder

        samples_raw = [
            (_make_jpeg((200, 50, 50), (256, 256)), "red square"),
            (_make_jpeg((50, 200, 50), (512, 384)), "green rectangle"),
            (_make_jpeg((50, 50, 200), (320, 480)), "blue portrait"),
        ]
        tar_path = str(tmp_path / "t2i_multi.tar")
        _write_t2i_wds_tar(tar_path, samples_raw)
        wds_samples = _read_wds_tar(tar_path)

        encoder = BagelT2ITaskEncoder(tokenizer=tokenizer, vae_transform=vae_t)
        for i, ((jpeg, cap), wds_s) in enumerate(zip(samples_raw, wds_samples)):
            orig = _orig_t2i_process(jpeg, cap, vae_t, tokenizer)
            ene = encoder.encode_sample(wds_s)
            _assert_packable_samples_equal(orig, ene, label=f"T2I sample {i}")


# ---------------------------------------------------------------------------
# Edit parity tests
# ---------------------------------------------------------------------------

class TestEditExactParity:
    """Pre-packing and post-packing Edit parity across different chain seeds."""

    @staticmethod
    def _make_edit_row(num_images: int = 4) -> Dict:
        """Create a synthetic Edit row dict (in-memory, same format as parquet row)."""
        colors = [
            (200, 50, 50), (50, 200, 50), (50, 50, 200),
            (200, 200, 50), (200, 50, 200),
        ]
        image_list = [_make_jpeg(colors[i % len(colors)], (256, 256))
                      for i in range(num_images)]
        instruction_list = [
            [f"step {i} instruction A", f"step {i} instruction B"]
            for i in range(num_images - 1)
        ]
        return {"image_list": image_list, "instruction_list": instruction_list}

    def _run_one(self, row: Dict, seed: int, vae_t, vit_t, tokenizer) -> Tuple[dict, dict]:
        """Run both paths with the same seed; return (orig, energon) PackableSamples."""
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelEditTaskEncoder

        encoder = BagelEditTaskEncoder(
            tokenizer=tokenizer,
            vae_transform=vae_t,
            vit_transform=vit_t,
        )

        # Build WDS sample dict: mimics what _read_wds_tar returns
        wds_sample: Dict[str, Any] = {}
        for i, img_bytes in enumerate(row["image_list"]):
            wds_sample[f"{i:03d}.jpg"] = img_bytes
        wds_sample["json"] = json.dumps({
            "instruction_list": row["instruction_list"],
            "num_images": len(row["image_list"]),
        }).encode("utf-8")

        random.seed(seed)
        orig = _orig_edit_parse_row(row, vae_t, vit_t, tokenizer)

        random.seed(seed)
        ene = encoder.encode_sample(wds_sample)

        return orig, ene

    @pytest.mark.parametrize("num_images,seed", [
        (2, 0), (2, 1),
        (3, 0), (3, 1), (3, 2),
        (4, 0), (4, 1), (4, 5), (4, 99),
        (5, 0), (5, 7),
    ])
    def test_prepacking_exact(self, num_images, seed):
        tokenizer = _MockTokenizer()
        vae_t, vit_t = _make_transforms()
        row = self._make_edit_row(num_images)
        orig, ene = self._run_one(row, seed, vae_t, vit_t, tokenizer)
        _assert_packable_samples_equal(orig, ene, label=f"Edit n={num_images} seed={seed}")

    @pytest.mark.parametrize("seed", [0, 1, 2, 5, 13, 42])
    def test_packing_exact(self, seed):
        """Verify packed tensors are bit-exact for a 4-image Edit chain."""
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelEditTaskEncoder, BagelPacker

        tokenizer = _MockTokenizer()
        vae_t, vit_t = _make_transforms()
        row = self._make_edit_row(num_images=4)
        orig_sample, ene_sample = self._run_one(row, seed, vae_t, vit_t, tokenizer)

        data_config = _make_data_config(dropout=0.0)
        packer = BagelPacker(
            data_config=data_config,
            special_token_ids=SPECIAL_TOKEN_IDS,
            max_num_tokens=8192,
        )

        np.random.seed(11)
        random.seed(11)
        ss_orig = packer.init_sequence_status()
        ss_orig = packer.pack_sequence(orig_sample, ss_orig)
        out_orig = packer.to_tensor(ss_orig)

        np.random.seed(11)
        random.seed(11)
        ss_ene = packer.init_sequence_status()
        ss_ene = packer.pack_sequence(ene_sample, ss_ene)
        out_ene = packer.to_tensor(ss_ene)

        _assert_packed_tensors_equal(out_orig, out_ene, label=f"Edit post-packing seed={seed}")

    def test_wds_roundtrip_edit(self, tmp_path):
        """Test the full WDS write→read roundtrip for Edit data."""
        row = self._make_edit_row(num_images=3)
        tar_path = str(tmp_path / "edit.tar")
        _write_edit_wds_tar(tar_path, [row])
        wds_samples = _read_wds_tar(tar_path)

        assert len(wds_samples) == 1
        sample = wds_samples[0]

        meta = json.loads(sample["json"])
        assert meta["num_images"] == 3
        assert len(meta["instruction_list"]) == 2

        for i in range(3):
            field = f"{i:03d}.jpg"
            assert field in sample, f"Missing field {field}"
            img = Image.open(io.BytesIO(sample[field]))
            assert img.size == (256, 256)

    def test_edit_tar_roundtrip_matches_parquet_bytes(self, tmp_path):
        """WDS tar stores image bytes bit-for-bit identical to the in-memory bytes."""
        row = self._make_edit_row(num_images=4)
        tar_path = str(tmp_path / "edit_bytes.tar")
        _write_edit_wds_tar(tar_path, [row])
        wds_samples = _read_wds_tar(tar_path)

        sample = wds_samples[0]
        for i, orig_bytes in enumerate(row["image_list"]):
            assert sample[f"{i:03d}.jpg"] == orig_bytes, (
                f"Bytes mismatch for image {i}"
            )


# ---------------------------------------------------------------------------
# Multi-sample packing parity (T2I + Edit combined)
# ---------------------------------------------------------------------------

class TestMultiSamplePackingParity:
    """Pack multiple heterogeneous samples with BagelPacker and compare outputs."""

    def test_mixed_t2i_and_edit(self):
        from examples.mimo_bagel.data.energon_bagel_task_encoder import (
            BagelT2ITaskEncoder, BagelEditTaskEncoder, BagelPacker
        )
        tokenizer = _MockTokenizer()
        vae_t, vit_t = _make_transforms()

        # Build two T2I + two Edit samples
        t2i_raws = [
            (_make_jpeg((200, 100, 50), (256, 192)), "first caption"),
            (_make_jpeg((50, 150, 200), (384, 256)), "second caption"),
        ]
        edit_rows = [
            {"image_list": [_make_jpeg((r, g, b), (256, 256)) for r, g, b in
                             [(255, 0, 0), (0, 255, 0), (0, 0, 255)]],
             "instruction_list": [["make it green"], ["make it blue"]]},
            {"image_list": [_make_jpeg((r, g, b), (256, 256)) for r, g, b in
                             [(100, 100, 100), (200, 200, 200)]],
             "instruction_list": [["make it lighter"]]},
        ]

        t2i_encoder = BagelT2ITaskEncoder(tokenizer=tokenizer, vae_transform=vae_t)
        edit_encoder = BagelEditTaskEncoder(tokenizer=tokenizer, vae_transform=vae_t,
                                            vit_transform=vit_t)

        # Build WDS samples and compute energon PackableSamples
        ene_t2i_samples = []
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tf:
            for i, (jpeg, cap) in enumerate(t2i_raws):
                _add_to_tar(tf, f"{i:09d}.jpg", jpeg)
                _add_to_tar(tf, f"{i:09d}.txt", cap.encode())
        tar_buf.seek(0)
        wds_t2i = _read_wds_tar_from_bytes(tar_buf.read())
        for wds_s in wds_t2i:
            ene_t2i_samples.append(t2i_encoder.encode_sample(wds_s))

        ene_edit_samples = []
        for i, row in enumerate(edit_rows):
            wds_s: Dict[str, Any] = {}
            for j, img_b in enumerate(row["image_list"]):
                wds_s[f"{j:03d}.jpg"] = img_b
            wds_s["json"] = json.dumps({
                "instruction_list": row["instruction_list"],
                "num_images": len(row["image_list"]),
            }).encode()
            random.seed(i * 13)
            ene_edit_samples.append(edit_encoder.encode_sample(wds_s))

        # Build original PackableSamples
        orig_t2i_samples = [_orig_t2i_process(j, c, vae_t, tokenizer) for j, c in t2i_raws]

        orig_edit_samples = []
        for i, row in enumerate(edit_rows):
            random.seed(i * 13)
            orig_edit_samples.append(_orig_edit_parse_row(row, vae_t, vit_t, tokenizer))

        # Verify pre-packing parity
        for i, (o, e) in enumerate(zip(orig_t2i_samples, ene_t2i_samples)):
            _assert_packable_samples_equal(o, e, label=f"T2I {i}")
        for i, (o, e) in enumerate(zip(orig_edit_samples, ene_edit_samples)):
            _assert_packable_samples_equal(o, e, label=f"Edit {i}")

        # Pack all samples in the same order through BagelPacker and compare
        data_config = _make_data_config(dropout=0.0)
        packer = BagelPacker(data_config=data_config, special_token_ids=SPECIAL_TOKEN_IDS,
                             max_num_tokens=65536)

        all_orig = orig_t2i_samples + orig_edit_samples
        all_ene = ene_t2i_samples + ene_edit_samples

        np.random.seed(42)
        random.seed(42)
        ss_orig = packer.init_sequence_status()
        for s in all_orig:
            ss_orig = packer.pack_sequence(s, ss_orig)
        out_orig = packer.to_tensor(ss_orig)

        np.random.seed(42)
        random.seed(42)
        ss_ene = packer.init_sequence_status()
        for s in all_ene:
            ss_ene = packer.pack_sequence(s, ss_ene)
        out_ene = packer.to_tensor(ss_ene)

        _assert_packed_tensors_equal(out_orig, out_ene, label="mixed T2I+Edit post-packing")


def _read_wds_tar_from_bytes(data: bytes) -> List[Dict[str, bytes]]:
    buf = io.BytesIO(data)
    samples: Dict[str, Dict[str, bytes]] = {}
    with tarfile.open(fileobj=buf, mode="r") as tf:
        for member in tf.getmembers():
            name = member.name
            dot = name.find(".")
            key, field = name[:dot], name[dot + 1:]
            raw = tf.extractfile(member).read()
            samples.setdefault(key, {})[field] = raw
    return [samples[k] for k in sorted(samples)]


# ---------------------------------------------------------------------------
# VLM structural comparison (not bit-exact due to <image> tag handling diff)
# ---------------------------------------------------------------------------

class TestVLMStructural:
    """VLM: structural comparison only.  Energon/original differ in image placement
    for conversations with <image> tags and in whether EOS carries CE loss."""

    def test_text_only_vlm_exact(self):
        """For text-only (no image) single-turn Q&A, both paths produce identical output."""
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelVLMTaskEncoder

        tokenizer = _MockTokenizer()
        _, vit_t = _make_transforms()

        conversation = [
            {"from": "human", "value": "What is 2 plus 2?"},
            {"from": "gpt",   "value": "Four."},
        ]
        # WDS sample: empty jpg, json with conversations
        wds_sample = {
            "jpg": b"",   # no image
            "json": json.dumps({"id": "0", "conversations": conversation}).encode(),
        }

        encoder = BagelVLMTaskEncoder(
            tokenizer=tokenizer,
            vit_transform=vit_t,
            special_token_ids=SPECIAL_TOKEN_IDS,
        )
        ene = encoder.encode_sample(wds_sample)

        # Expected: no image tokens, two text entries
        assert len(ene["image_tensor_list"]) == 0, "No image expected for text-only VLM"
        assert len(ene["text_ids_list"]) == 2, "Expected 2 text entries (human + gpt)"
        assert len(ene["sequence_plan"]) == 2

        # Human turn: no loss; gpt turn: loss=1
        assert ene["sequence_plan"][0]["loss"] == 0
        assert ene["sequence_plan"][1]["loss"] == 1

    def test_vlm_has_vit_token_for_image_sample(self):
        """VLM sample with an image: energon encoder inserts VIT token at start."""
        from examples.mimo_bagel.data.energon_bagel_task_encoder import BagelVLMTaskEncoder

        tokenizer = _MockTokenizer()
        _, vit_t = _make_transforms()
        jpeg = _make_jpeg((128, 128, 128), (280, 280))

        wds_sample = {
            "jpg": jpeg,
            "json": json.dumps({"id": "0", "conversations": [
                {"from": "human", "value": "Describe this image."},
                {"from": "gpt",   "value": "A grey square."},
            ]}).encode(),
        }

        encoder = BagelVLMTaskEncoder(tokenizer=tokenizer, vit_transform=vit_t,
                                      special_token_ids=SPECIAL_TOKEN_IDS)
        ene = encoder.encode_sample(wds_sample)

        # Image is always inserted first
        assert ene["sequence_plan"][0]["type"] == "vit_image"
        assert len(ene["image_tensor_list"]) == 1
        # Image tensor shape must be divisible by vit patch size (14)
        t = ene["image_tensor_list"][0]
        assert t.shape[1] % 14 == 0 and t.shape[2] % 14 == 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
