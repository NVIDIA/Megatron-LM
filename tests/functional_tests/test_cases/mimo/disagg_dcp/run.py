# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Small disaggregated MIMO fixture through the real native pretrain entry.

Launch with eight ranks (E4/D4, GBS8), or twenty (E4/D16, GBS4) for a real
CP8 multi-image pack alongside two CP4 packs. All training and checkpoint handling
remain in ``examples.mimo.pretrain_mimo.main``. This synthetic fixture tests the
integration; it is not a Qwen convergence or quality benchmark.

Use --reference-transport for an independent identity-based feature/gradient
oracle in the same native model layout. Its broadcasts are test-only.
"""

import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from examples.mimo import pretrain_mimo
from examples.mimo.data.source_batch import SourceBatchIterator
from examples.mimo.model_providers import MimoProvider
from examples.mimo.training import builder
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import get_run_config, get_train_state


def language_spec(args, pg_collection, grid):
    """Two real MoE GPT layers plus a MoE MTP layer, with explicit role groups."""
    config = core_transformer_config_from_args(args)
    config.use_cpu_initialization = True
    experts = 16 if dist.get_world_size() == 20 else 4
    assert (config.num_layers, config.hidden_size, config.num_moe_experts) == (2, 64, experts)
    assert config.mtp_num_layers == 1
    assert args.max_position_embeddings >= (1024 if experts == 16 else 512)
    layer = get_gpt_layer_with_transformer_engine_spec(num_experts=experts, moe_grouped_gemm=True)
    return ModuleSpec(
        GPTModel,
        params=dict(
            config=config,
            transformer_layer_spec=layer,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer, True, pp_rank=0),
            vocab_size=256,
            max_sequence_length=args.max_position_embeddings,
            position_embedding_type="rope",
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=False,
            pg_collection=pg_collection,
        ),
    )


def encoder_spec(args, pg_collection, grid):
    """One real CLIP layer; each 16x16 image produces five feature rows."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0,
        attention_dropout=0,
    )
    return ModuleSpec(
        VisionModalitySubmodules,
        params={"pg_collection": pg_collection},
        submodules={
            "encoders": {
                "clip": ModuleSpec(
                    CLIPViTModel,
                    params=dict(
                        transformer_config=config,
                        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                        img_h=16,
                        img_w=16,
                        patch_dim=8,
                        pg_collection=pg_collection,
                    ),
                )
            }
        },
    )


class TinySourceDataset:
    """Deterministic multi-image samples, with every seventh step text-only."""

    def __init__(self, seed):
        self.seed = seed

    def build_global_batch(self, step, global_batch_size):
        complex_pack = dist.get_world_size() == 20
        assert global_batch_size == (4 if complex_pack else 8)
        lengths = (768, 256, 512, 512) if complex_pack else (192, 128, 96, 64, 64, 32, 32, 32)
        image_starts = (
            ((48, 94, 144), (20, 30), (), ())
            if complex_pack
            else tuple(tuple(2 + 32 * i for i in range(n)) for n in (3, 2, 1, 1, 0, 1, 0, 0))
        )
        samples, media = {}, []
        for sid, length in enumerate(lengths):
            logical = length - 3
            tokens = (torch.arange(length) + 11 * sid + step % 3) % 127 + 1
            labels = (tokens % 127) + 1
            starts = image_starts[sid] if step % 7 != 6 else ()
            ids = list(range(len(media), len(media) + len(starts)))
            tokens[logical:] = labels[logical:] = 0
            mask = torch.zeros(length)
            mask[2:logical] = 1
            for start in starts:
                tokens[start : start + 5] = 255
                mask[start : start + 5] = 0
            samples[sid] = dict(
                tokens=tokens,
                labels=labels,
                loss_mask=mask,
                position_ids=torch.arange(logical),
                original_seq_len=logical,
                padded_seq_len=length,
                media_ids=ids,
            )
            media.extend(
                dict(
                    image_id=mid,
                    pixels=torch.randn(
                        (3, 16, 16), generator=torch.Generator().manual_seed(self.seed + mid)
                    ),
                )
                for mid in ids
            )
        return samples, media


def data_provider(args, topology):
    """Native consumed-sample state determines the source batch after restart."""
    config, state = get_run_config(), get_train_state()
    train = SourceBatchIterator(
        TinySourceDataset(config.rng.seed),
        config.train.global_batch_size,
        state.consumed_train_samples,
    )
    valid = (
        SourceBatchIterator(
            TinySourceDataset(config.rng.seed + 10000),
            config.validation.eval_global_batch_size,
            0 if config.validation.skip_train else state.consumed_valid_samples,
        )
        if config.validation.eval_iters
        else None
    )
    return train, valid, None


def prepare_source(source, *, args, topology):
    """Only producer ranks materialize their assigned image tensors on CUDA."""
    grid = topology.grids["images"]
    producers = tuple(range(grid.rank_offset, grid.rank_offset + grid.size))
    tasks = [MediaTask(item["image_id"], 5) for item in source["media"]]
    assignments, slices = assign_media_to_producers(tasks, producers)
    pixels = {item["image_id"]: item["pixels"] for item in source["media"]}
    local = [pixels[task.media_id] for task in assignments.get(dist.get_rank(), ())]
    inputs = {"images": {"clip": {"x": torch.stack(local).cuda().bfloat16()}}} if local else {}
    return source["samples"], slices, inputs


def provider(_args):
    return MimoProvider(
        encoder_module_names=("images",),
        language_spec=language_spec,
        encoder_specs={"images": encoder_spec},
        special_token_ids=lambda args: {"images": 255},
        build_communicator=lambda args, topology: None,
        train_valid_test_data_provider=data_provider,
        prepare_source=prepare_source,
    )


class IdentityFeatureReference:
    """Tiny-fixture oracle independent of production feature slices and routes."""

    def __init__(self, adapter):
        assert dist.get_world_size() in (8, 20)
        assert adapter.decoder_ranks == tuple(range(4, dist.get_world_size()))
        self.adapter = adapter
        prepare = adapter.prepare_source

        def remember_source(source):
            self.samples = source["samples"]
            self.num_images = len(source["media"])
            assert self.num_images in (0, 5 if dist.get_world_size() == 20 else 8)
            assert [image["image_id"] for image in source["media"]] == list(range(self.num_images))
            return prepare(source)

        adapter.prepare_source = remember_source

    @torch.no_grad()
    def forward(self, local_features, _plans, _cp_group, *, forward_only=False):
        rank, hidden = dist.get_rank(), local_features.shape[-1]
        table = local_features.new_zeros((8, 5, hidden))
        # Equal-cost fixture media are assigned in ID order: producer=ID%4,
        # producer-local row offset=(ID//4)*5. No production route is consulted.
        for producer in range(4):
            count = len(range(producer, self.num_images, 4))
            if count:
                payload = (
                    local_features.detach().reshape(count, 5, hidden).contiguous()
                    if rank == producer
                    else local_features.new_empty((count, 5, hidden))
                )
                dist.broadcast(payload, src=producer)
                table[producer : self.num_images : 4].copy_(payload)
        media_ids, features = (), None
        if self.adapter.is_decoder:
            item = self.adapter.plan.rounds[self.adapter.runtime.completed_rounds]
            media_ids = tuple(
                mid for sid in item.sample_ids for mid in self.samples[sid]["media_ids"]
            )
            features = table[list(media_ids)].reshape(-1, hidden).requires_grad_(not forward_only)
        return SimpleNamespace(
            features=features,
            local_features=local_features,
            media_ids=media_ids,
            num_images=self.num_images,
            forward_only=forward_only,
        )

    @torch.no_grad()
    def backward(self, transfer):
        assert not transfer.forward_only
        hidden = transfer.local_features.shape[-1]
        gradient = transfer.local_features.new_zeros((8, 5, hidden), dtype=torch.float32)
        if transfer.features is not None and transfer.features.grad is not None:
            rows = transfer.features.grad.reshape(-1, 5, hidden)
            for image, contribution in zip(transfer.media_ids, rows):
                gradient[image].add_(contribution.float())
        dist.all_reduce(gradient)
        if dist.get_rank() < 4 and transfer.num_images:
            return (
                gradient[dist.get_rank() : transfer.num_images : 4].reshape(-1, hidden).contiguous()
            )
        return torch.zeros_like(transfer.local_features, dtype=torch.float32)


if __name__ == "__main__":
    # Replace only example wiring in this test process; use the unmodified native loop.
    pretrain_mimo.resolve_provider = builder.resolve_provider = provider
    reference_transport = "--reference-transport" in sys.argv
    if reference_transport:
        sys.argv.remove("--reference-transport")
    native_build = builder.MimoModelBuilder.build_distributed_models

    def build_fixture(self, *args, **kwargs):
        models = native_build(self, *args, **kwargs)
        assert len(models) == 1
        adapter = models[0].mimo_packed_step
        if reference_transport:
            adapter.runtime.bridge = IdentityFeatureReference(adapter)
        if dist.get_world_size() == 20:
            prepare = adapter.prepare

            def prepare_complex(*args, **kwargs):
                result = prepare(*args, **kwargs)
                if adapter.is_decoder and adapter.model.config.dynamic_context_parallel:
                    assert len(adapter.plan.rounds) == 1
                    item = adapter.plan.rounds[0]
                    assert [plan.cp_ranks for plan in item.feature_plans] == [
                        tuple(range(4, 12)),
                        tuple(range(12, 16)),
                        tuple(range(16, 20)),
                    ]
                    rank = dist.get_rank()
                    assert item.sample_ids == ((0, 1) if rank < 12 else (2,) if rank < 16 else (3,))
                return result

            models[0].config.sequence_packing_data_adapter = prepare_complex
        return models

    builder.MimoModelBuilder.build_distributed_models = build_fixture
    pretrain_mimo.main()
