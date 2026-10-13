# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Save/load parity for the fused-MLA and MTP ``fsdp_dtensor`` checkpoint handlers.

``tests/unit_tests/transformer/test_fsdp_dtensor_checkpoint.py`` covers the pure key
mapping, and monkeypatches ``split_fused_fsdp_param`` so it can run without GPUs. This
module covers what that cannot: the DTensor slice arithmetic inside
``split_fused_fsdp_param``, driven through real Megatron-FSDP shards and a real Torch DCP
checkpoint.

The model is small but reproduces the three structural features the handlers key on, all
of which DeepSeek-V3 has:

* a fused ``linear_qkv_down_proj`` holding the row concatenation of the unfused
  ``linear_q_down_proj`` (``q_lora_rank`` rows) and ``linear_kv_down_proj``
  (``kv_lora_rank + qk_pos_emb_head_dim`` rows), with asymmetric section sizes so an
  off-by-one or swapped split cannot pass by coincidence;
* an ``input_layernorm`` absorbed into the fused module as ``layer_norm_weight``;
* an MTP layer whose inner module is ``mtp_model_layer`` in the model and
  ``transformer_layer`` on disk, and which itself holds a fused attention, so the
  MLA-before-MTP handler ordering is covered.

q and kv reach the loss through separate downstream weights, so a split at the wrong
offset moves the loss and the gradients instead of cancelling out.

Tensors are compared as each rank's local shard and never gathered. Megatron-FSDP shards
are uneven and a rank can legitimately own zero elements of a section, which the DTensor
gather paths do not handle; the save path never gathers either, since DCP writes shards
from chunk metadata. Per-rank local shards agreeing bit for bit is also a stronger
statement than gathered tensors agreeing.
"""

import os
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_model
from tests.unit_tests.test_utilities import Utils

# Section sizes are asymmetric and neither divides the other.
HIDDEN = 128
Q_LORA_RANK = 96
KV_LORA_RANK = 48
QK_POS_EMB_HEAD_DIM = 16
KV_SECTION = KV_LORA_RANK + QK_POS_EMB_HEAD_DIM  # 64
FUSED_ROWS = Q_LORA_RANK + KV_SECTION  # 160
NUM_DECODER_LAYERS = 2

# All ranks must agree on the checkpoint directory without exchanging it.
SHARED_CKPT_DIR = "/tmp/pytest-shared-tmp/mfsdp_fused_mla_checkpoint"

# get_mla_fused_down_proj_splits() reads exactly these three attributes off mod.config.
MLA_CONFIG = SimpleNamespace(
    q_lora_rank=Q_LORA_RANK, kv_lora_rank=KV_LORA_RANK, qk_pos_emb_head_dim=QK_POS_EMB_HEAD_DIM
)


def _layernorm(x, weight):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=None)


class UnfusedDownProj(nn.Module):
    """One of the two separate down projections, as stored on disk."""

    def __init__(self, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, HIDDEN))

    def forward(self, x):
        return x @ self.weight.T


class FusedDownProj(nn.Module):
    """The fused down projection, which also absorbs the layer's input layernorm."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(FUSED_ROWS, HIDDEN))
        self.layer_norm_weight = nn.Parameter(torch.empty(HIDDEN))

    def forward(self, x):
        return _layernorm(x, self.layer_norm_weight) @ self.weight.T


class Attention(nn.Module):
    """MLA-like attention in either the fused or the unfused layout."""

    def __init__(self, fused):
        super().__init__()
        self.config = MLA_CONFIG
        self.fused = fused
        if fused:
            # hasattr(mod, 'linear_qkv_down_proj') is what marks a module as fused.
            self.linear_qkv_down_proj = FusedDownProj()
        else:
            self.linear_q_down_proj = UnfusedDownProj(Q_LORA_RANK)
            self.linear_kv_down_proj = UnfusedDownProj(KV_SECTION)

    def forward(self, x, input_layernorm_weight):
        if self.fused:
            fused = self.linear_qkv_down_proj(x)
            return fused[..., :Q_LORA_RANK], fused[..., Q_LORA_RANK:]
        normed = _layernorm(x, input_layernorm_weight)
        return self.linear_q_down_proj(normed), self.linear_kv_down_proj(normed)


class Layer(nn.Module):
    """Transformer-like layer. The input layernorm lives here when unfused."""

    def __init__(self, fused):
        super().__init__()
        self.fused = fused
        if not fused:
            self.input_layernorm = nn.Module()
            self.input_layernorm.weight = nn.Parameter(torch.empty(HIDDEN))
        self.self_attention = Attention(fused)
        # q and kv reach the loss through different weights.
        self.out_q = nn.Linear(Q_LORA_RANK, HIDDEN, bias=False)
        self.out_kv = nn.Linear(KV_SECTION, HIDDEN, bias=False)

    def forward(self, x):
        ln_weight = None if self.fused else self.input_layernorm.weight
        q, kv = self.self_attention(x, ln_weight)
        # The 0.1 keeps activations O(1) across layers, so parity is read at a sane scale.
        return x + 0.1 * (self.out_q(q) + self.out_kv(kv))


class MTPLayer(nn.Module):
    """get_mtp_inner_layer_paths() matches on the inner module's name."""

    def __init__(self, fused):
        super().__init__()
        self.mtp_model_layer = Layer(fused)

    def forward(self, x):
        return self.mtp_model_layer(x)


class Decoder(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.layers = nn.ModuleList([Layer(fused) for _ in range(NUM_DECODER_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MTP(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.layers = nn.ModuleList([MTPLayer(fused)])

    def forward(self, x):
        return self.layers[0](x)


class TinyMLAModel(nn.Module):
    """Module paths mirror the real model: ``decoder.layers.N.self_attention`` and
    ``mtp.layers.0.mtp_model_layer.self_attention``."""

    def __init__(self, fused):
        super().__init__()
        self.decoder = Decoder(fused)
        self.mtp = MTP(fused)
        self.head = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return self.head(self.mtp(self.decoder(x)))


class Float16ModuleLike(nn.Module):
    """Stands in for mcore's Float16Module, whose state_dict() delegates to the inner
    module. That is what makes state dict keys carry one ``module.`` level while
    named_parameters() carries two, the relationship
    preprocess_fsdp_dtensor_state_dict relies on when it resolves
    ``model.get_parameter('module.' + key)``."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)


def build_model(fused, seed, device="cuda"):
    """Build a model whose weights are reproducible but are not any framework default.
    Iterating in sorted-name order attaches values to parameter names rather than to
    module construction order, so the fused and unfused models are independently seeded
    rather than accidentally equal."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = TinyMLAModel(fused=fused)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            sample = torch.empty(param.shape).normal_(generator=generator)
            if name.endswith(("layer_norm_weight", "input_layernorm.weight")):
                param.copy_(1.0 + 0.1 * sample)
            else:
                param.copy_(sample * (param.shape[-1] ** -0.5) + 0.01)
    return Float16ModuleLike(model).to(device)


def raw_bits(tensor):
    """Exact bit pattern of a tensor's values, for bit-parity comparison."""
    return tensor.detach().to("cpu").contiguous().numpy().tobytes()


def snapshot(tensors):
    """Record, per key, the global shape and this rank's local shard copied to CPU, so a
    later load cannot mutate it and no collective is needed to compare."""
    from torch.distributed.tensor import DTensor

    out = {}
    for key, value in tensors.items():
        local = value.to_local() if isinstance(value, DTensor) else value
        out[key] = (tuple(value.shape), local.detach().to("cpu").clone())
    return out


def local_mismatches(left, right):
    """Differences between two snapshots on this rank. Pure, so it is unit-testable."""
    problems = []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            problems.append(f"present on only one side: {key}")
            continue
        (left_shape, left_local), (right_shape, right_local) = left[key], right[key]
        if left_shape != right_shape:
            problems.append(f"global shape {left_shape} != {right_shape}: {key}")
        elif left_local.shape != right_local.shape:
            problems.append(
                f"local shard shape {tuple(left_local.shape)} != "
                f"{tuple(right_local.shape)}: {key}"
            )
        elif raw_bits(left_local) != raw_bits(right_local):
            delta = (
                (left_local.float() - right_local.float()).abs().max().item()
                if left_local.numel()
                else 0.0
            )
            problems.append(f"local shard bits differ (max|delta|={delta:.3e}): {key}")
    return problems


def assert_bit_identical(label, left, right):
    """Fail on every rank if any rank sees a mismatch, so the assertion is collective."""
    problems = local_mismatches(left, right)
    total = torch.tensor([len(problems)], dtype=torch.long, device="cuda")
    dist.all_reduce(total)
    assert int(total.item()) == 0, (
        f"{label}: {int(total.item())} mismatch(es) across "
        f"{dist.get_world_size()} rank(s); this rank saw {problems[:6]}"
    )


def shard_model(model, mesh):
    return fully_shard_model(
        model,
        device_mesh=mesh,
        dp_shard_dim="dp_shard",
        tp_dim="tp",
        # Layer is this model's transformer-layer equivalent, and is what real runs pass
        # here as TransformerLayer. Under optim_grads_params sharding Megatron-FSDP sizes
        # its communication units by dividing by the number of these, so it must be set.
        fsdp_unit_modules=[Layer],
        zero_dp_strategy=3,
        device=torch.device("cuda", torch.cuda.current_device()),
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )


def model_state_dict_for_preprocess(sharded):
    """The state dict the real save/load paths hand to
    preprocess_fsdp_dtensor_state_dict, asserting the key/parameter relationship that
    function depends on."""
    state_dict = sharded.state_dict()
    probe = next(iter(state_dict))
    try:
        sharded.get_parameter(f"module.{probe}")
    except AttributeError as exc:
        raise AssertionError(
            f"preprocess_fsdp_dtensor_state_dict resolves parameters as "
            f"model.get_parameter('module.' + key), which fails for key {probe!r}: {exc}"
        ) from exc
    return state_dict


def preprocess(sharded, state_dict):
    from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict

    args = SimpleNamespace(swiglu=False, num_experts=None)
    return preprocess_fsdp_dtensor_state_dict(args, state_dict, sharded)


def save_checkpoint(sharded, path):
    """Mirrors the fsdp_dtensor save path in megatron/training/checkpointing.py."""
    import torch.distributed.checkpoint as dcp

    state_dict = preprocess(sharded, {"model": model_state_dict_for_preprocess(sharded)})
    dcp.save(state_dict, checkpoint_id=path)
    dist.barrier()
    return snapshot(state_dict["model"])


def load_checkpoint(sharded, path, strict="assume_ok_unexpected"):
    """Mirrors the fsdp_dtensor load path, including the validation step."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import default_planner

    from megatron.core.transformer.fsdp_dtensor_checkpoint import validate_fsdp_dtensor_model_load

    state_dict = {"model": model_state_dict_for_preprocess(sharded)}
    # The raw, un-preprocessed model dict, kept before the handlers rewrite the keys,
    # exactly as _load_base_checkpoint does.
    raw_model_state_dict = state_dict["model"].copy()
    state_dict = preprocess(sharded, state_dict)
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata().state_dict_metadata

    unexpected = validate_fsdp_dtensor_model_load(metadata, state_dict, path, strict=strict)

    dcp.load_state_dict(
        state_dict=state_dict,
        storage_reader=reader,
        planner=default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )
    dist.barrier()
    loaded = snapshot(state_dict["model"])
    state_dict["model"] = raw_model_state_dict

    # DCP writes into the fp32 main weights. Megatron-FSDP only copies those into the
    # weights the forward pass uses from a load_state_dict post hook, which a DCP load
    # does not trigger on its own, so the real load path calls load_state_dict here too.
    # It falls back to strict=False the same way, because the state dict carries one
    # 'module.' level while the module nesting expects two.
    try:
        sharded.load_state_dict(state_dict["model"], strict=True)
    except Exception:
        sharded.load_state_dict(state_dict["model"], strict=False)

    return loaded, unexpected


def forward_backward(sharded, seed=1234):
    """One forward/backward through the real Megatron-FSDP model. Every rank uses the
    same input, so the loss is directly comparable between models."""
    torch.manual_seed(seed)
    x = torch.randn(8, HIDDEN, device="cuda")
    loss = sharded(x).square().mean()
    loss.backward()
    return loss.detach().to("cpu")


def grad_snapshot(sharded):
    """Local gradient shards, keyed by parameter name."""
    grads = {}
    for name, param in sharded.named_parameters():
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is not None:
            grads[name] = grad
    return snapshot(grads)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.4.0'),
    reason="Requires DTensor and DeviceMesh support in (approximately) PyTorch 2.4.0 or later.",
)
class TestFusedMlaMtpFsdpDtensorCheckpoint:
    """Round trips a fused-MLA + MTP model through a real Megatron-FSDP DCP checkpoint.

    FIXME(@cspades): Megatron-FSDP leaves behind corrupted NCCL state that affects other
    tests, which is why these live in the Megatron-FSDP bucket.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()
        from torch.distributed.device_mesh import init_device_mesh

        cls.mesh = init_device_mesh(
            "cuda", (Utils.world_size, 1), mesh_dim_names=("dp_shard", "tp")
        )
        if dist.get_rank() == 0:
            shutil.rmtree(SHARED_CKPT_DIR, ignore_errors=True)
            os.makedirs(SHARED_CKPT_DIR, exist_ok=True)
        dist.barrier()

    @classmethod
    def teardown_class(cls):
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(SHARED_CKPT_DIR, ignore_errors=True)
        del cls.mesh
        Utils.destroy_model_parallel()

    def test_fused_parameter_is_split_across_shard_boundaries(self):
        """The point of running this on GPUs: assert that at least one rank owns parts of
        both the q and the kv section, so the split does partial-overlap slice arithmetic
        rather than whole-shard copies. If this ever stops holding, the parity tests below
        keep passing while covering much less."""
        from megatron.core.transformer.fsdp_dtensor_checkpoint import _intersect_slice

        sharded = shard_model(build_model(fused=True, seed=1234), self.mesh)
        fused_slice = sharded.get_parameter(
            "module.module.decoder.layers.0.self_attention.linear_qkv_down_proj.weight"
        ).megatron_fsdp_slice

        boundary = Q_LORA_RANK * HIDDEN
        sections = {"q": slice(0, boundary), "kv": slice(boundary, FUSED_ROWS * HIDDEN)}
        owned = {}
        for name, section in sections.items():
            overlap = _intersect_slice(fused_slice, section)
            owned[name] = max(0, overlap.stop - overlap.start)

        straddles = torch.tensor(
            [1 if owned["q"] and owned["kv"] else 0], dtype=torch.long, device="cuda"
        )
        dist.all_reduce(straddles)
        assert int(straddles.item()) > 0, (
            "no rank owns parts of both the q and kv sections, so the fused split is not "
            f"exercising partial-overlap arithmetic; this rank owns "
            f"[{fused_slice.start}:{fused_slice.stop}) of {FUSED_ROWS * HIDDEN} elements "
            f"with the q|kv boundary at {boundary}"
        )

    def test_unfused_checkpoint_loads_into_fused_model(self):
        """The case that motivated the fix: an existing unfused checkpoint read back by a
        model built with mla_down_proj_fusion=True."""
        saver = shard_model(build_model(fused=False, seed=1234), self.mesh)
        path = os.path.join(SHARED_CKPT_DIR, "unfused_ckpt")
        on_disk = save_checkpoint(saver, path)

        # The unfused model is what defines the on-disk layout, so the split has a
        # reference to be wrong against.
        assert any("linear_q_down_proj.weight" in key for key in on_disk)
        assert any("linear_kv_down_proj.weight" in key for key in on_disk)
        assert not any("linear_qkv_down_proj" in key for key in on_disk)

        loader = shard_model(build_model(fused=True, seed=4321), self.mesh)
        loaded, unexpected = load_checkpoint(loader, path)

        assert not unexpected, f"validator reported absent model keys: {sorted(unexpected)}"
        # The decisive check: after the handlers rewrite the fused model's state dict it
        # exposes q and kv separately, and those must match what the unfused model wrote,
        # which holds only if the fused parameter was split at the right offsets and the
        # absorbed layernorm was moved to the right key.
        assert_bit_identical("values loaded by the fused model", on_disk, loaded)

        # One fused GEMM and two separate GEMMs only agree to float32 precision, so this
        # direction is checked relatively; the round trip below is the bit-exact one.
        loss_saver = forward_backward(saver)
        loss_loader = forward_backward(loader)
        assert torch.allclose(loss_saver, loss_loader, rtol=1e-6, atol=1e-6), (
            f"fused model computes a different function than the unfused model that saved "
            f"the checkpoint: {loss_saver.item():.17g} vs {loss_loader.item():.17g}"
        )

    def test_fused_round_trip_preserves_loss_and_grads(self):
        """Bit parity of weights, loss and gradients for the model this change supports,
        against a non-trivial checkpoint."""
        src = shard_model(build_model(fused=True, seed=2468), self.mesh)
        path = os.path.join(SHARED_CKPT_DIR, "fused_ckpt")
        on_disk = save_checkpoint(src, path)

        # A fused model still writes the unfused layout, which is what makes the
        # checkpoint readable by both variants.
        assert not any("linear_qkv_down_proj" in key for key in on_disk)

        dst = shard_model(build_model(fused=True, seed=8642), self.mesh)
        loaded, unexpected = load_checkpoint(dst, path)

        assert not unexpected, f"validator reported absent model keys: {sorted(unexpected)}"
        assert_bit_identical("round-tripped state dict", on_disk, loaded)
        # The state dict exposes the fp32 main weights; loss and gradient parity depend on
        # the weights the forward pass actually reads.
        assert_bit_identical(
            "weights the forward pass reads",
            snapshot(dict(src.named_parameters())),
            snapshot(dict(dst.named_parameters())),
        )

        loss_src = forward_backward(src)
        loss_dst = forward_backward(dst)
        assert raw_bits(loss_src) == raw_bits(loss_dst), (
            f"loss is not bit-identical after the round trip: "
            f"{loss_src.item():.17g} vs {loss_dst.item():.17g}"
        )

        grads_src, grads_dst = grad_snapshot(src), grad_snapshot(dst)
        assert grads_src, "no gradients were reachable, so gradient parity is untested"
        assert_bit_identical("gradients", grads_src, grads_dst)

    def test_absent_fused_keys_raise_instead_of_partial_load(self):
        """Negative control. With the MLA handler disabled the fused model asks for keys
        the checkpoint does not have, which used to be skipped silently and left those
        weights at their initialized values."""
        import megatron.training.checkpointing as ckpt_mod
        from megatron.core.dist_checkpointing.core import CheckpointingException

        path = os.path.join(SHARED_CKPT_DIR, "negative_control_ckpt")
        save_checkpoint(shard_model(build_model(fused=True, seed=2468), self.mesh), path)

        original = ckpt_mod.handle_mla_down_proj_in_state_dict
        ckpt_mod.handle_mla_down_proj_in_state_dict = lambda model, msd, osd: (msd, osd)
        try:
            broken = shard_model(build_model(fused=True, seed=1357), self.mesh)
            with pytest.raises(CheckpointingException, match="linear_qkv_down_proj"):
                load_checkpoint(broken, path)

            # With the check switched off the load is accepted and quietly leaves the
            # fused parameters at their initial values, which is what broke convergence.
            quiet = shard_model(build_model(fused=True, seed=1357), self.mesh)
            before = snapshot(
                preprocess(quiet, {"model": model_state_dict_for_preprocess(quiet)})["model"]
            )
            after, _ = load_checkpoint(quiet, path, strict="ignore_all")
            stale = [
                key
                for key in after
                if "linear_qkv_down_proj" in key
                and key in before
                and raw_bits(before[key][1]) == raw_bits(after[key][1])
            ]
            assert stale, (
                "expected ignore_all to leave the fused parameters untouched, which is the "
                "silent partial load the validator exists to catch"
            )
        finally:
            ckpt_mod.handle_mla_down_proj_in_state_dict = original
