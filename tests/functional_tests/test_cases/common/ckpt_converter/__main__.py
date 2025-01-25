# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import shutil
import subprocess
import sys
import time
import types
import typing as T
from collections import namedtuple

import numpy as np
import torch

from megatron.core import parallel_state
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_tokenizer
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint
from megatron.training.global_vars import set_global_variables, unset_global_variables
from megatron.training.training import get_model
from pretrain_gpt import model_provider
from tests.unit_tests.test_utilities import Utils

CHECKPOINTS_DIR = "/tmp/ckpt-converter-tests"
FORWARD_ITERS = 1  # *3
SKIP_CONVERSION = False


class TempSharedDir:
    """Context that makes & removes a directory to hold the checkpoints."""

    def __enter__(self):
        """Make checkpoint directory."""
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(CHECKPOINTS_DIR, ignore_errors=True)
            os.mkdir(CHECKPOINTS_DIR)
        torch.distributed.barrier()

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Remove checkpoint directory."""
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            shutil.rmtree(CHECKPOINTS_DIR, ignore_errors=True)
        torch.distributed.barrier()


_ModelParallelState = namedtuple("_ModelParallelState", "tp pp ep")


class ModelParallelState(_ModelParallelState):
    """Parallel state struct, that contains TP, PP, and EP."""

    def __new__(cls, tp=1, pp=1, ep=1):
        return super(ModelParallelState, cls).__new__(cls, tp, pp, ep)


class ModelMeta:
    """Basic information about a model.

    Args:
        format (str): 'core', 'legacy', 'meta', or 'hf'.
        mp (ModelParallelState): Defines TP, PP, EP.
        transformer_impl (str): 'transformer_engine' or 'local'.
    """

    def __init__(self, format: str, mp: ModelParallelState, transformer_impl: str = None):

        if isinstance(mp, tuple):
            mp = ModelParallelState(*mp)
        if transformer_impl is None:
            transformer_impl = "transformer_engine" if format == "core" else "local"

        assert format in ("core", "legacy", "meta", "hf")
        assert isinstance(mp, ModelParallelState)
        assert transformer_impl in ("transformer_engine", "local")

        self.format = format
        self.mp = mp
        self.transformer_impl = transformer_impl


class Pipeline:
    """A pipeline manages a single conversion and validation.

    The pipeline consists of the following steps:
    - Initialize model & inference pass.
    - Save model.
    - Convert model.
    - Load model & inference pass.
    - Validate before/after output tensors.

    Args:
        src (ModelMeta): Model meta for loading.
        dst (ModelMeta): Model meta for storing.
    """

    def __init__(self, src: ModelMeta, dst: ModelMeta):
        """Source & destination metas."""
        assert isinstance(src, ModelMeta)
        assert isinstance(dst, ModelMeta)
        self.src = src
        self.dst = dst

    def get_model_argv(self):
        """Get argv list for customizing initialization."""
        raise NotImplementedError(self.__class__.__name__ + ".get_model_argv()")

    def get_converter_model_type(self):
        """Get converter type: 'GPT' or 'Bert'."""
        raise NotImplementedError(self.__class__.__name__ + ".get_converter_model_type()")

    def get_meta(self, key):
        """Get meta from key, which must be either 'src' or 'dst'."""
        assert key in ("src", "dst")
        return getattr(self, f"{key}")

    def init_args_and_model(self, key):
        """Initialize Megatron and build model."""

        meta = self.get_meta(key)

        # Destroy & initialize new parallel state.
        unset_global_variables()
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=meta.mp.tp,
            pipeline_model_parallel_size=meta.mp.pp,
            expert_model_parallel_size=meta.mp.ep,
        )

        # Environment vars.
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

        # Command line args.
        sys.argv = [
            "[script]",
            *self.get_model_argv(),
            "--tensor-model-parallel-size",
            str(meta.mp.tp),
            "--pipeline-model-parallel-size",
            str(meta.mp.pp),
            "--expert-model-parallel-size",
            str(meta.mp.ep),
            "--save-interval",
            "2",
            "--save",
            os.path.join(CHECKPOINTS_DIR, "src"),
            "--load",
            os.path.join(CHECKPOINTS_DIR, "dst" if not SKIP_CONVERSION else "src"),
            "--ckpt-format",
            "torch",
            "--use-checkpoint-args",
            "--no-save-optim",
            "--no-save-rng",
            "--no-load-optim",
            "--no-load-rng",
            "--bf16",
            "--use-cpu-initialization",
            "--no-one-logger",
            "--transformer-impl",
            meta.transformer_impl,
        ]

        # Fail on missing checkpoint.
        if key == "dst":
            sys.argv.append("--exit-on-missing-checkpoint")

        # Use legacy.
        if meta.format == "legacy":
            sys.argv.append("--use-legacy-models")

        # Parse args.
        args = parse_args()
        validate_args(args)

        # Set global args, build tokenizer.
        unset_global_variables()
        set_global_variables(args)

        # Random seed.
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Model.
        models = get_model(
            model_provider_func=model_provider, model_type=ModelType.encoder_or_decoder
        )
        [m.eval() for m in models]

        return args, models

    @classmethod
    def is_model_parallel_rank_0(cls):
        return (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_pipeline_model_parallel_rank() == 0
        )

    @classmethod
    def get_input_ids(cls):
        """Randomly initialize input token IDs."""
        if cls.is_model_parallel_rank_0():
            # Generate different data on each DP rank.
            args = get_args()

            orig_numpy_seed = np.random.get_state()[1][0]
            temp_numpy_seed = orig_numpy_seed + torch.distributed.get_rank()

            np.random.seed(temp_numpy_seed)
            numpy_input_ids = np.random.randint(
                low=0, high=args.vocab_size, size=(args.seq_length,), dtype=np.int64
            )
            np.random.seed(orig_numpy_seed)

            torch_input_ids = torch.from_numpy(numpy_input_ids).to("cuda")

            return torch_input_ids
        else:
            return None

    @classmethod
    def _broadcast(cls, item):
        """Broadcast data from TP rank 0 to other ranks."""
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    @classmethod
    def get_batch(cls, input_ids):
        """Get batch of data, from input token IDs."""

        args = get_args()

        # TP rank 0, PP rank 0.
        # (Note: mimics megatron/training/utils.py:get_batch_on_this_tp_rank().)
        if cls.is_model_parallel_rank_0():

            tokenizer = get_tokenizer()

            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                data=input_ids,
                eod_token=tokenizer.eod,
                reset_position_ids=args.reset_position_ids,
                reset_attention_mask=args.reset_attention_mask,
                eod_mask_loss=args.eod_mask_loss,
                create_attention_mask=args.create_attention_mask_in_dataloader,
            )
            input_ids = input_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        # Other TP ranks on PP rank 0.
        elif parallel_state.is_pipeline_first_stage():
            input_ids = torch.empty(
                (args.micro_batch_size, args.seq_length),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            position_ids = torch.empty(
                (args.micro_batch_size, args.seq_length),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            if args.create_attention_mask_in_dataloader:
                attention_mask = torch.empty(
                    (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                )
            else:
                attention_mask = None

        # Other PP ranks.
        # (Note: mimics pretrain_gpt.py:get_batch().)
        else:
            input_ids = None
            position_ids = None
            attention_mask = None

        # Broadcast.
        if parallel_state.is_pipeline_first_stage():
            cls._broadcast(input_ids)
            cls._broadcast(attention_mask)
            cls._broadcast(position_ids)

        return input_ids, position_ids, attention_mask

    @classmethod
    def forward_step(cls, orig_input_ids: T.Iterator, model: torch.nn.Module):
        """Forward step.

        Args:
            orig_input_ids (T.Iterator): Input token IDs.
            model (GPTModel): The GPT Model.
        """

        # Unpack input ids.
        orig_input_ids = list(orig_input_ids)[0]

        # Get batch.
        input_ids, position_ids, attention_mask = cls.get_batch(orig_input_ids)

        # Forward pass test data (multi iters for JIT warm-up).
        for _ in range(FORWARD_ITERS):
            output_tensor = model(input_ids, position_ids, attention_mask)

        # Aggregate data, for validation.
        data = {
            "orig_input_ids": orig_input_ids,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "output_tensor": output_tensor,
        }

        return output_tensor, lambda _, non_loss_data: data

    @classmethod
    def forward_model(cls, models, orig_input_ids):
        """Forward pass data, and gather parallel output tensors."""

        args = get_args()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        data = forward_backward_func(
            forward_step_func=cls.forward_step,
            data_iterator=iter([orig_input_ids]),
            model=models,
            num_microbatches=1,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )
        if parallel_state.is_pipeline_last_stage():
            output_tensor = data[0]["output_tensor"]
        else:
            output_tensor = None

        # All-gather across the partitions.
        if parallel_state.is_pipeline_last_stage():
            output_tensor_gathered = gather_from_tensor_model_parallel_region(output_tensor)
        else:
            output_tensor_gathered = None

        return output_tensor_gathered

    def rand_init_model_params(self, key, models):
        """Randomly initialize model params."""

        meta = self.get_meta(key)

        with torch.no_grad():

            # Randomly initialize all params.
            for m in models:
                for p in m.parameters():
                    p.normal_(0, 0.1)

            # Synchronize embeddings.
            if meta.mp.pp != 1 and parallel_state.is_rank_in_embedding_group():
                if parallel_state.is_pipeline_first_stage():
                    emb = models[0].module.module.shared_embedding_or_output_weight()
                elif parallel_state.is_pipeline_last_stage():
                    emb = models[-1].module.module.shared_embedding_or_output_weight()
                else:
                    raise Exception("should be either first/last pipeline rank.")
                torch.distributed.all_reduce(emb, group=parallel_state.get_embedding_group())

    def save_checkpoint(self):
        """Initialize params, forward pass data, and save checkpoint."""

        args, models = self.init_args_and_model("src")

        # Init params.
        self.rand_init_model_params("src", models)

        # Test data.
        orig_input_ids = self.get_input_ids()
        output_tensor = self.forward_model(models, orig_input_ids)

        # Save checkpoint.
        _save_checkpoint(
            iteration=2,
            model=models,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=None,
        )

        return output_tensor, orig_input_ids

    def load_checkpoint(self, orig_input_ids):
        """Load checkpoint, and forward pass data."""

        args, models = self.init_args_and_model("dst")

        # Load checkpoint.
        args.iteration, args.num_floating_point_operations_so_far = _load_checkpoint(
            models, optimizer=None, opt_param_scheduler=None
        )

        # Test data.
        output_tensor_real = self.forward_model(models, orig_input_ids)

        # Random output tensor.
        # Note: need two random initializations to differ from `save_checkpoint()` above.
        self.rand_init_model_params("dst", models)
        self.rand_init_model_params("dst", models)
        output_tensor_fake = self.forward_model(models, orig_input_ids)

        return output_tensor_real, output_tensor_fake

    def convert_checkpoint(self):
        """Convert checkpoint"""

        args = get_args()

        torch.distributed.barrier()

        # Convert.
        if torch.distributed.get_rank() == 0:

            cmd = [
                "python",
                "tools/checkpoint/convert.py",
                "--model-type",
                self.get_converter_model_type(),
                "--loader",
                self.src.format,
                "--load-dir",
                args.save,
                "--loader-transformer-impl",
                self.src.transformer_impl,
                "--saver",
                self.dst.format,
                "--save-dir",
                args.load,
                "--saver-transformer-impl",
                self.dst.transformer_impl,
                "--target-tensor-parallel-size",
                str(self.dst.mp.tp),
                "--target-pipeline-parallel-size",
                str(self.dst.mp.pp),
                "--megatron-path",
                os.getcwd(),
            ]
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("convert checkpoint cmd: %s" % " ".join(cmd))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            result = subprocess.run(cmd)

            assert result.returncode == 0, "checkpoint conversion failed."

        torch.distributed.barrier()

    def run(self):
        """Run pipeline.

        Running a pipeline consists of:

        - Save checkpoint (includes initializing params & forward passing data).
        - Convert checkpoint.
        - Load checkpoint (includes forward passing data).
        - Validate before/after output tensors.
        """

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.src.mp.tp,
            pipeline_model_parallel_size=self.src.mp.pp,
            expert_model_parallel_size=self.src.mp.ep,
        )
        with TempSharedDir():

            # Save checkpoint.
            src_output_tensor, input_ids = self.save_checkpoint()

            # Convert checkpoint.
            if not SKIP_CONVERSION:
                self.convert_checkpoint()

            # Load checkpoint.
            dst_output_tensor_real, dst_output_tensor_fake = self.load_checkpoint(input_ids)

            # Validate output tensor.
            torch.distributed.barrier()
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if rank == world_size - 1:
                args = get_args()
                get_mse = lambda dst_output_tensor: torch.nn.MSELoss()(
                    src_output_tensor[:, :, : args.vocab_size],
                    dst_output_tensor[:, :, : args.vocab_size],
                ).item()
                mse_real = get_mse(dst_output_tensor_real)
                mse_fake = get_mse(dst_output_tensor_fake)
                assert mse_real < 0.01 * mse_fake, "mse_real (%e) >= 0.01 mse_fake (%e)." % (
                    mse_real,
                    mse_fake,
                )
            torch.distributed.barrier()

            # Teardown.
            unset_global_variables()
            Utils.destroy_model_parallel()

            # Broadcast MSE's.
            mses = torch.zeros((2,), dtype=torch.float, device="cuda")
            if rank == world_size - 1:
                mses[0] = mse_real
                mses[1] = mse_fake
            torch.distributed.broadcast(mses, world_size - 1)

            return mses.tolist()


class GPTPipeline(Pipeline):
    """GPT-specific pipeline customizations.

    Args:
        src (Union[ModelMeta, Tuple]): Model meta for loading.
        dst (Union[ModelMeta, Tuple]): Model meta for storing.
        num_moe_experts (Optional[int]): Number of MoE experts.
    """

    def __init__(self, src: ModelMeta, dst: ModelMeta, num_moe_experts: T.Optional[int] = None):
        super().__init__(ModelMeta(*src), ModelMeta(*dst))
        assert isinstance(num_moe_experts, (int, types.NoneType))
        self.num_moe_experts = num_moe_experts

    def get_model_argv(self):
        """GPT model args."""
        args = [
            "--num-layers",
            "8",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "8",
            "--seq-length",
            "16",
            "--max-position-embeddings",
            "16",
            "--micro-batch-size",
            "1",  # single sample generated.
            "--tokenizer-type",
            "NullTokenizer",
            "--vocab-size",
            "127",  # ... NullTokenizer adds +1 EOD token.
            "--make-vocab-size-divisible-by",
            "1",
        ]
        if self.num_moe_experts is not None and self.num_moe_experts > 1:
            args.extend(["--num-experts", str(self.num_moe_experts or 1), "--sequence-parallel"])
        return args

    def get_converter_model_type(self):
        return "GPT"


def get_gpt_pipelines():
    """Get GPT (non-MoE) pipelines."""
    return [
        GPTPipeline(("core", (8, 1)), ("core", (1, 8))),
        GPTPipeline(("core", (4, 2)), ("core", (2, 4))),
        GPTPipeline(("core", (2, 4)), ("core", (4, 2))),
        GPTPipeline(("core", (1, 8)), ("core", (8, 1))),
        GPTPipeline(("core", (4, 2)), ("core", (2, 4), "local")),
        GPTPipeline(("legacy", (4, 2)), ("core", (2, 4))),
        GPTPipeline(("core", (4, 2), "local"), ("core", (2, 4), "local")),
        GPTPipeline(("core", (4, 2), "local"), ("core", (2, 4))),
        # [todo] GPTPipeline(("legacy", (4, 2)), ("legacy", (2, 4))),
        # [todo] GPTPipeline(("legacy", (4, 2), "te"), ("legacy", (2, 4), "te")),
        # [todo] GPTPipeline("meta", "core", None, (8, 1)),
        # [todo] GPTPipeline("hf", "core", None, (8, 1)),
    ]


def get_moe_pipelines():
    """Get MoE pipelines."""
    return [
        GPTPipeline(("core", (2, 1, 2)), ("core", (1, 4, 1)), num_moe_experts=8),
        GPTPipeline(("core", (1, 4, 1)), ("core", (2, 1, 2)), num_moe_experts=4),
    ]


def test_all_pipelines():
    """Run all pipelines."""

    # Collect pipelines.
    pipelines = [
        *get_gpt_pipelines(),
        # [todo] *get_moe_pipelines(), # todo: MoE support in loader_core.py.
        # [todo] *get_bert_pipelines(),
        # [todo] *get_t5_pipelines(),
    ]

    # Run pipelines.
    results = []
    for pipeline in pipelines:
        t = time.time()
        mses = pipeline.run()
        elapsed_time = time.time() - t
        results.append((elapsed_time, *mses))

    # Print results.
    if int(os.environ["RANK"]) == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("checkpoint converter results:")
        [print("  t %.1f sec ... mse %.1e, %.1e." % (t, r, f)) for t, r, f in results]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    test_all_pipelines()
