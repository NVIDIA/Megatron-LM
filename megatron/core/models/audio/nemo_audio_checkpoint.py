# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Load NeMo ``.nemo`` archives for the transformer audio encoder.

A ``.nemo`` file is the standard NeMo ``SaveRestoreConnector`` artifact: an
uncompressed tar containing ``model_config.yaml`` (OmegaConf) and
``model_weights.ckpt`` (``torch.save`` of the full ``EncDecRNNTBPEModel``
state dict). This module supports only that format -- generic ``.pt`` /
``.bin`` checkpoint files are intentionally not handled.

Public API:
- ``extract_nemo_archive(path)`` -> ``(model_cfg, full_state_dict)``
- ``nemo_audio_configs_from_archive(path)`` -> ``(encoder_cfg, preproc_cfg, encoder_state)``
- ``load_nemo_transformer_audio_weights(audio_module, ckpt_path, *, strict=False)``
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
import tarfile
import tempfile

import torch

from .audio_feature_config import NemoAudioFeatureConfig
from .nemo_transformer_audio_model import NemoTransformerAudioConfig

MODEL_CONFIG_NAME = "model_config.yaml"
MODEL_WEIGHTS_NAME = "model_weights.ckpt"
CHECKPOINT_NEMO_TRANSFORMER_AUDIO_CONFIG_NAME = "nemo_transformer_audio_config.json"
CHECKPOINT_NEMO_AUDIO_PREPROCESSOR_CONFIG_NAME = "nemo_audio_preprocessor_config.json"

# Encoder _target_s we know map onto the vendored ``transformer_encoder.TransformerEncoder``.
# ``transformer_encoder_flex`` is parameter-compatible with the non-flex variant for
# ``attn_mode == "full"`` (FlexAttention only adds attention-mask plumbing, not new params).
_KNOWN_ENCODER_TARGETS = {
    "nemo.collections.asr.modules.TransformerEncoder",
    "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
    # ``transformer_encoder_flex`` is parameter-compatible with the non-flex
    # variant; FlexAttention adds attention-mask plumbing only (see header).
    "nemo.collections.asr.modules.transformer_encoder_flex.TransformerEncoder",
}

_KNOWN_PREPROC_TARGETS = {
    "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
    "nemo.collections.asr.modules.audio_preprocessing.AudioToMelSpectrogramPreprocessor",
}


def _load_omegaconf(path: str):
    """Lazy import of OmegaConf -- only paid for when we actually open a .nemo."""
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "omegaconf is required to read .nemo configs. "
            "Install with `pip install omegaconf` (it ships with nemo_toolkit)."
        ) from exc
    return OmegaConf.load(path)


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_preprocessor_config(path: str | Path | None) -> "NemoAudioFeatureConfig":
    if path:
        return NemoAudioFeatureConfig.from_dict(_load_json(path))
    return NemoAudioFeatureConfig()


def nemo_audio_config_paths_from_checkpoint_dir(
    checkpoint_dir: str | Path,
) -> Tuple[Path, Path]:
    """Return checkpoint-local NeMo audio config paths for an iteration dir."""
    checkpoint_dir = Path(checkpoint_dir)
    return (
        checkpoint_dir / CHECKPOINT_NEMO_TRANSFORMER_AUDIO_CONFIG_NAME,
        checkpoint_dir / CHECKPOINT_NEMO_AUDIO_PREPROCESSOR_CONFIG_NAME,
    )


def has_nemo_audio_configs_in_checkpoint_dir(checkpoint_dir: str | Path) -> bool:
    """True when both checkpoint-local NeMo audio config JSON files are present."""
    encoder_path, preproc_path = nemo_audio_config_paths_from_checkpoint_dir(checkpoint_dir)
    return encoder_path.is_file() and preproc_path.is_file()


def nemo_audio_configs_from_json_paths(
    encoder_config_path: str | Path,
    preprocessor_config_path: str | Path | None = None,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig"]:
    """Load NeMo audio encoder/preprocessor configs from JSON files."""
    encoder_cfg = NemoTransformerAudioConfig.from_dict(_load_json(encoder_config_path))
    preproc_cfg = _load_preprocessor_config(preprocessor_config_path)
    return encoder_cfg, preproc_cfg


def nemo_audio_configs_from_checkpoint_dir(
    checkpoint_dir: str | Path,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig"]:
    """Load NeMo audio configs persisted next to a Megatron checkpoint iteration."""
    encoder_path, preproc_path = nemo_audio_config_paths_from_checkpoint_dir(checkpoint_dir)
    if not encoder_path.is_file() or not preproc_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint-local NeMo audio config files in {checkpoint_dir}: "
            f"{encoder_path.name}, {preproc_path.name}"
        )
    return nemo_audio_configs_from_json_paths(encoder_path, preproc_path)


def resolve_nemo_audio_configs_from_args(
    args,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig"]:
    """Resolve NeMo audio configs from training/runtime args.

    Source precedence is:
    1. ``--load-audio-from`` when it points to a ``.nemo`` archive.
    2. ``--nemo-transformer-audio-config`` / ``--nemo-audio-preprocessor-config`` JSON.
    3. Dataclass defaults.

    ``--nemo-transformer-audio-attn-impl`` is a runtime backend override and is
    materialized into the returned encoder config so checkpoint-local artifacts
    reproduce the model that was actually instantiated.
    """
    nemo_path = getattr(args, "load_audio_from", None)
    if nemo_path and str(nemo_path).endswith(".nemo"):
        encoder_cfg, preproc_cfg = nemo_audio_configs_from_path(nemo_path)
    else:
        encoder_path = getattr(args, "nemo_transformer_audio_config", None)
        preproc_path = getattr(args, "nemo_audio_preprocessor_config", None)
        if encoder_path:
            encoder_cfg, preproc_cfg = nemo_audio_configs_from_json_paths(
                encoder_path, preproc_path
            )
        else:
            encoder_cfg = NemoTransformerAudioConfig()
            preproc_cfg = _load_preprocessor_config(preproc_path)

    attn_impl = getattr(args, "nemo_transformer_audio_attn_impl", None)
    if attn_impl:
        encoder_cfg.attn_impl = attn_impl
    left_context = getattr(args, "nemo_transformer_audio_left_context", None)
    if left_context is not None:
        encoder_cfg.left_context = None if left_context < 0 else left_context
    encoder_cfg.recompute_layers = bool(getattr(args, "recompute_audio", False))
    return encoder_cfg, preproc_cfg


def write_nemo_audio_configs_to_checkpoint_dir(
    checkpoint_dir: str | Path,
    encoder_cfg: NemoTransformerAudioConfig,
    preproc_cfg: "NemoAudioFeatureConfig",
) -> Tuple[Path, Path]:
    """Persist resolved NeMo audio configs as JSON under a checkpoint iteration dir."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    encoder_path, preproc_path = nemo_audio_config_paths_from_checkpoint_dir(checkpoint_dir)
    _write_json(encoder_path, asdict(encoder_cfg))
    _write_json(preproc_path, asdict(preproc_cfg))
    return encoder_path, preproc_path


def write_nemo_audio_configs_from_args_to_checkpoint_dir(
    args,
    checkpoint_dir: str | Path,
) -> Tuple[Path, Path] | None:
    """Persist resolved NeMo audio configs for ``args`` when audio is enabled."""
    if (getattr(args, "audio_model_type", "") or "") != "nemo_transformer":
        return None
    encoder_cfg, preproc_cfg = resolve_nemo_audio_configs_from_args(args)
    return write_nemo_audio_configs_to_checkpoint_dir(checkpoint_dir, encoder_cfg, preproc_cfg)


def _validate_archive(nemo_path: Path) -> None:
    if not nemo_path.is_file():
        raise FileNotFoundError(f"No such file: {nemo_path}")
    if not tarfile.is_tarfile(nemo_path):
        raise ValueError(
            f"Expected a .nemo (tar) archive, got {nemo_path}. "
            ".pt/.bin/.ckpt are not supported."
        )


def _extract_member(tar: tarfile.TarFile, member_name: str, out_dir: str) -> str:
    """Extract a single archive member by basename. NeMo writes ``./model_config.yaml``.

    Iterates lazily via ``tar.next()`` and stops at the first match. We deliberately
    avoid ``tar.getmembers()`` here: it walks the entire archive to EOF, which trips
    on ``.nemo`` files whose trailing zero-block region is malformed or missing
    (a known NeMo ``SaveRestoreConnector`` quirk that surfaces as
    ``tarfile.ReadError: unexpected end of data``). For NeMo's standard layout
    (``model_config.yaml`` precedes ``model_weights.ckpt``), two sequential calls
    on the same handle never need to walk past ``model_weights.ckpt`` and never
    hit the bad tail.
    """
    seen: list[str] = []
    try:
        while True:
            member = tar.next()
            if member is None:
                break
            seen.append(member.name)
            if os.path.basename(member.name) == member_name:
                try:
                    tar.extract(member, out_dir)
                except tarfile.ReadError as e:
                    raise ValueError(
                        f"Archive ended unexpectedly while extracting {member.name} "
                        f"from {tar.name}. The archive may be truncated or the member "
                        f"size metadata may not match the stored payload."
                    ) from e
                return os.path.join(out_dir, member.name)
    except tarfile.ReadError as e:
        raise ValueError(
            f"Archive ended unexpectedly while searching for {member_name}; "
            f"members seen so far: {seen[:8]}. The archive may be truncated."
        ) from e
    raise ValueError(f"Archive missing {member_name}; members: {seen[:8]}")


def read_nemo_config(nemo_path: str | Path) -> Dict[str, Any]:
    """Read just the ``model_config.yaml`` from a ``.nemo`` archive.

    Cheap (only ~kB of data extracted), so safe to call on every rank during
    model construction.
    """
    nemo_path = Path(nemo_path)
    _validate_archive(nemo_path)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(nemo_path, "r:") as tar:
            cfg_path = _extract_member(tar, MODEL_CONFIG_NAME, tmp)
        from omegaconf import OmegaConf

        cfg = _load_omegaconf(cfg_path)
        if "model" in cfg:
            cfg = cfg.model
        return OmegaConf.to_container(cfg, resolve=True)


def extract_nemo_archive(
    nemo_path: str | Path,
    out_dir: str | Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Extract a ``.nemo`` archive and return ``(model_cfg_dict, full_state_dict)``.

    Args:
        nemo_path: Path to the ``.nemo`` file.
        out_dir: Optional directory to extract into. If ``None``, a temp directory
            is used and cleaned up automatically.

    Returns:
        A tuple ``(cfg, state)`` where ``cfg`` is the resolved OmegaConf ``model``
        block as a plain dict, and ``state`` is the full flat ``torch.save``'d
        state dict (preprocessor + encoder + decoder + joint + ...).
    """
    nemo_path = Path(nemo_path)
    _validate_archive(nemo_path)

    if out_dir is None:
        cm = tempfile.TemporaryDirectory()
        tmp_dir = cm.name
    else:
        cm = None
        tmp_dir = str(out_dir)
        os.makedirs(tmp_dir, exist_ok=True)

    try:
        with tarfile.open(nemo_path, "r:") as tar:
            cfg_path = _extract_member(tar, MODEL_CONFIG_NAME, tmp_dir)
            wts_path = _extract_member(tar, MODEL_WEIGHTS_NAME, tmp_dir)

        from omegaconf import OmegaConf

        cfg = _load_omegaconf(cfg_path)
        if "model" in cfg:
            cfg = cfg.model
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        state = _torch_load(wts_path)
        if not isinstance(state, dict):
            raise ValueError(
                f"{MODEL_WEIGHTS_NAME} in {nemo_path} did not deserialize to a dict, "
                f"got {type(state).__name__}"
            )
    finally:
        if cm is not None:
            cm.cleanup()

    return cfg_dict, state


def _strip_target(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _split_audio_configs(
    cfg: Dict[str, Any],
    nemo_path: str | Path,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig"]:
    """Validate ``_target_``s and convert the model config dict into our dataclasses."""
    if "encoder" not in cfg:
        raise ValueError(f"{nemo_path}: model_config.yaml has no 'encoder' block")
    if "preprocessor" not in cfg:
        raise ValueError(f"{nemo_path}: model_config.yaml has no 'preprocessor' block")

    enc_cfg = dict(cfg["encoder"])
    pre_cfg = dict(cfg["preprocessor"])
    if "causal_mask" not in enc_cfg and enc_cfg.get("attn_mode") == "causal":
        enc_cfg["causal_mask"] = True

    enc_target = enc_cfg.get("_target_", "")
    pre_target = pre_cfg.get("_target_", "")
    if enc_target and enc_target not in _KNOWN_ENCODER_TARGETS:
        raise ValueError(
            f"{nemo_path}: encoder._target_={enc_target!r} is not a known transformer "
            f"encoder. Supported: {sorted(_KNOWN_ENCODER_TARGETS)}"
        )
    if pre_target and pre_target not in _KNOWN_PREPROC_TARGETS:
        raise ValueError(
            f"{nemo_path}: preprocessor._target_={pre_target!r} is not a known mel "
            f"preprocessor. Supported: {sorted(_KNOWN_PREPROC_TARGETS)}"
        )

    encoder_cfg = NemoTransformerAudioConfig.from_dict(_strip_target(enc_cfg))
    preproc_cfg = NemoAudioFeatureConfig.from_dict(_strip_target(pre_cfg))
    return encoder_cfg, preproc_cfg


def nemo_audio_configs_from_path(
    nemo_path: str | Path,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig"]:
    """Cheap config-only read of a ``.nemo`` archive.

    Use this when you only need the encoder/preprocessor hyperparameters (e.g.
    on every rank during model construction). Does not touch the state dict.
    """
    cfg = read_nemo_config(nemo_path)
    return _split_audio_configs(cfg, nemo_path)


def nemo_audio_configs_from_archive(
    nemo_path: str | Path,
) -> Tuple[NemoTransformerAudioConfig, "NemoAudioFeatureConfig", Dict[str, torch.Tensor]]:
    """Parse a ``.nemo`` archive into the encoder/preprocessor configs + encoder state.

    The encoder state dict has the leading ``encoder.`` prefix stripped so it can
    be loaded directly into ``NemoTransformerAudioModel.encoder``.

    Raises ``ValueError`` if the archive does not look like a NeMo ASR model that
    pairs ``AudioToMelSpectrogramPreprocessor`` with one of the supported
    transformer encoder ``_target_``s.
    """
    cfg, full_state = extract_nemo_archive(nemo_path)
    encoder_cfg, preproc_cfg = _split_audio_configs(cfg, nemo_path)

    encoder_state = {
        k[len("encoder."):]: v
        for k, v in full_state.items()
        if k.startswith("encoder.") and isinstance(v, torch.Tensor)
    }
    if not encoder_state:
        raise ValueError(
            f"{nemo_path}: no tensors with 'encoder.' prefix found in {MODEL_WEIGHTS_NAME}. "
            f"Top-level prefixes were: "
            f"{sorted({k.split('.', 1)[0] for k in full_state.keys()})}"
        )

    return encoder_cfg, preproc_cfg, encoder_state


def load_nemo_transformer_audio_weights(
    audio_module: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load encoder weights from a ``.nemo`` archive into ``NemoTransformerAudioModel``.

    Validates that the archive's ``model.encoder`` config matches the audio
    module's config (``n_mels``, ``d_model``, ``n_heads``, ``n_layers``,
    ``pre_encode``, ``subsampling_factor``, ``qk_norm``) before loading.

    Returns:
        ``(missing_keys, unexpected_keys)`` from ``load_state_dict(strict=False)``,
        with TransformerEngine ``_extra_state`` entries removed.
    """
    from .nemo_transformer_audio_model import NemoTransformerAudioModel

    if not isinstance(audio_module, NemoTransformerAudioModel):
        raise TypeError(f"Expected NemoTransformerAudioModel, got {type(audio_module)}")

    enc_cfg, _, encoder_state = nemo_audio_configs_from_archive(ckpt_path)
    _validate_encoder_cfg(audio_module.config, enc_cfg, ckpt_path)

    missing, unexpected = audio_module.encoder.load_state_dict(encoder_state, strict=False)
    missing = [k for k in missing if "_extra_state" not in k]
    unexpected = [k for k in unexpected if "_extra_state" not in k]

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"strict load failed for {ckpt_path}: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )

    return missing, unexpected


def _validate_encoder_cfg(
    model_cfg: NemoTransformerAudioConfig,
    ckpt_cfg: NemoTransformerAudioConfig,
    ckpt_path: str | Path,
) -> None:
    """Raise if a structural field disagrees between model and checkpoint."""
    structural = (
        "n_mels",
        "d_model",
        "n_heads",
        "n_layers",
        "pre_encode",
        "subsampling_factor",
        "qk_norm",
        "qkv_bias",
    )
    mismatches = []
    for field in structural:
        m = getattr(model_cfg, field)
        c = getattr(ckpt_cfg, field)
        if m != c:
            mismatches.append(f"{field}: model={m!r} ckpt={c!r}")
    if mismatches:
        raise ValueError(
            f"Encoder config mismatch when loading {ckpt_path}:\n  "
            + "\n  ".join(mismatches)
            + "\nRebuild the audio model from the .nemo config (use "
            "nemo_audio_configs_from_archive) or supply a matching "
            "--nemo-transformer-audio-config JSON."
        )
