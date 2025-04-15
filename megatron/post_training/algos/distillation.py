# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Distillation loss function(s)."""

import logging
import re
import types
from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.modules.loss import _Loss

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_and_context_parallel_rank,
    get_tensor_model_parallel_group,
    get_virtual_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.transformer import MegatronModule, TransformerConfig, TransformerLayer
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)

logger = logging.getLogger(__name__)


def load_distillation_config(
    config_path: Optional[str], student_cfg: TransformerConfig, teacher_cfg: TransformerConfig
) -> Dict[str, Any]:
    """Read the distillation yaml config file specified by ``args.export_kd_cfg``.

    Args:
        config_path: Path to user-defined distillation settings yaml file.
            If `None`, uses default logits-only distillation mode for GPT models.
        student_cfg: Model config for student model.
        teacher_cfg: Model config for teacher model.

    WARNING: Assumes intermediate hidden sizes are always that found in the model config's ``hidden_size`` attribute.
    """
    if not config_path:
        logger.warning("Distillation config not provided. Using default.")
        cfg = {
            "logit_layers": ["output_layer", "output_layer"],
            "intermediate_layer_pairs": [],
            "skip_lm_loss": True,
            "kd_loss_scale": 1.0,
        }
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    intermediate_pairs = cfg["intermediate_layer_pairs"]
    logit_pair = cfg["logit_layers"]
    skip_lm_loss = cfg["skip_lm_loss"]
    loss_scale = cfg["kd_loss_scale"]

    criterion = {}
    if student_cfg.pipeline_model_parallel_size == 1 or is_pipeline_last_stage():
        criterion[tuple(logit_pair)] = LogitsKLLoss(student_cfg)
        # NOTE: Projection layer shared among intermediate layer pairs.
        projection_layer = ProjectionLayer(student_cfg, teacher_cfg)

        for student_layer, teacher_layer in intermediate_pairs:
            if get_tensor_and_context_parallel_rank() == 0:
                print(
                    "Distillation: Adding intermediate loss between"
                    f" `{student_layer}` of student (hidden size {student_cfg.hidden_size}) and"
                    f" `{teacher_layer}` of teacher (hidden size {teacher_cfg.hidden_size})."
                )
            student_layer = _adjust_layer_index_for_pp(student_layer, student_cfg)
            teacher_layer = _adjust_layer_index_for_pp(teacher_layer, teacher_cfg)
            criterion[(student_layer, teacher_layer)] = HiddenStateCosineLoss(
                student_cfg, projection_layer=projection_layer
            )

    loss_balancer = LogitsAndIntermediatesLossBalancer(
        kd_loss_scale=loss_scale, skip_original_loss=skip_lm_loss
    )

    cfg["criterion"] = criterion
    cfg["loss_balancer"] = loss_balancer

    return cfg


def _adjust_layer_index_for_pp(submodule_name, model_cfg):
    """Adjust any sequence-based layer indices found in a submodule name for Pipeline Parallelism."""

    match = re.search(r'(?<=\.)\d+(?=\.)', submodule_name)
    if not match:
        return submodule_name

    offset = TransformerLayer._get_layer_offset(model_cfg)
    new_layer_idx = int(match.group(0)) - offset
    if new_layer_idx < 0:
        raise ValueError(f"Layer {submodule_name} does not fall on final PP rank.")

    new_submodule_name = submodule_name.replace(match.group(0), str(new_layer_idx))
    if get_tensor_and_context_parallel_rank() == 0:
        print(
            f'Distillation: Renamed layer "{submodule_name}" on final PP rank to "{new_submodule_name}"'
        )
    return new_submodule_name


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(
        self, model_config: TransformerConfig, projection_layer: Optional[nn.Module] = None
    ):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__()
        self._config = model_config
        self._projection = projection_layer

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        if self._projection is not None:
            predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor, tp_reduce: bool = False, is_sequence_parallel: bool = False) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce, is_sequence_parallel)


class HiddenStateCosineLoss(BaseLoss):
    """
    Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, so extra restrictions are in place.
    """

    def __init__(
        self, model_config: TransformerConfig, projection_layer: Optional[nn.Module] = None
    ):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            projection_layer: Module which projects student activations to teacher's hidden dim.
        """
        super().__init__(model_config, projection_layer=projection_layer)

        if self._config.tensor_model_parallel_size > 1 and not self._config.sequence_parallel:
            logger.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement or use `--sequence_parallel` if tensor parallel is enabled."
            )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            Cosine loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        loss = F.cosine_embedding_loss(
            predictions.view(-1, predictions.size(-1)),
            targets.view(-1, targets.size(-1)),
            targets.new_ones(1),
            reduction="none",
        )
        loss = loss.view(*predictions.shape[:2])

        # NOTE: Tensor sequence length is still split among TP ranks.
        return self.post_forward(loss, is_sequence_parallel=self._config.sequence_parallel)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(
        self, model_config: TransformerConfig, temperature: float = 1.0, reverse: bool = False
    ):
        """
        Constructor.

        Args:
            model_config: MCore transformer config.
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__(model_config)
        self._temperature = temperature
        self._reverse = reverse

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            KLD loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # Division by temp should happen prior to finding max for both student and teacher.
        # Currently we don't use temperature in any of ours runs (temp=1.0)
        output_teacher = targets.float() / self._temperature
        output_student = predictions.float() / self._temperature

        # Compute local softmax, and the reweight to compute global softmax.
        if self._config.tensor_model_parallel_size > 1:

            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use standard reduction function here since the computation
            # that follows it isn't identical across TP ranks.
            denom_teacher = all_reduce_autograd(
                denom_teacher, group=get_tensor_model_parallel_group()
            )

            # Maximum value along vocab dimension across all GPUs.
            student_logits_max, _ = torch.max(output_student, dim=-1)
            torch.distributed.all_reduce(
                student_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_student = output_student - student_logits_max.unsqueeze(dim=-1).detach()

            denom_student = torch.sum(torch.exp(output_student), dim=-1)
            denom_student = all_reduce_autograd(
                denom_student, group=get_tensor_model_parallel_group()
            )

            slen, bsz, sharded_vocab_size = output_student.shape
            student_log_prob = output_student - torch.log(denom_student).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )
            teacher_log_prob = output_teacher - torch.log(denom_teacher).view(slen, bsz, 1).expand(
                slen, bsz, sharded_vocab_size
            )

            if self._reverse:
                loss = torch.sum(
                    F.kl_div(teacher_log_prob, student_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
                    dim=-1,
                )

        else:
            if self._reverse:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_teacher, dim=-1),
                        F.softmax(output_student, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )
            else:
                loss = torch.sum(
                    F.kl_div(
                        F.log_softmax(output_student, dim=-1),
                        F.softmax(output_teacher, dim=-1),
                        reduction="none",
                    ),
                    dim=-1,
                )

        return self.post_forward(loss, tp_reduce=True)


########################################################


class LogitsAndIntermediatesLossBalancer(mtd.DistillationLossBalancer):
    """
    LossBalancer implementation for Logit and Intermediate losses.

    Dynamically weighs distillation and original losses to balance during training.
    """

    def __init__(self, kd_loss_scale: float = 1.0, skip_original_loss: bool = False):
        """Constructor.

        Args:
            kd_loss_scale: Multiply distillation losses by this before weighing.
                (Not used when `skip_original_loss` is True.)
            skip_original_loss: Used to signal whether the original loss should be used, regardless
                of whether it was passed into ``mtd.DistillationModel.compute_kd_loss()`` or not.
        """
        super().__init__()
        self._kd_loss_scale = kd_loss_scale
        self._skip_original_loss = skip_original_loss

    def forward(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        """Forward function.

        Args:
            loss_dict: All individual scalar losses, passed in during ``mtd.DistillationModel.compute_kd_loss()``

        Returns:
            Aggregate total scalar loss.
        """
        original_loss = loss_dict.pop(mtd.loss_balancers.STUDENT_LOSS_KEY)
        for _key in loss_dict:
            if _key.startswith(LogitsKLLoss.__name__):
                logits_key = _key  # should only be one
        logits_loss = loss_dict.pop(logits_key)
        intermediate_loss = sum(loss_dict.values()) / max(len(loss_dict), 1)

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.item() / intermediate_loss.item()
            intermediate_loss_scaled = intermediate_loss * dynamic_scale
            kd_loss_scale = self._kd_loss_scale / 2.0
        else:
            kd_loss_scale = self._kd_loss_scale
            intermediate_loss = logits_loss.new_tensor(intermediate_loss)
            intermediate_loss_scaled = intermediate_loss

        if self._skip_original_loss:
            total_loss = logits_loss + intermediate_loss_scaled
        else:
            kd_loss = (logits_loss + intermediate_loss_scaled) * kd_loss_scale
            dynamic_scale = original_loss.item() / kd_loss.item()
            total_loss = original_loss + kd_loss * dynamic_scale

        out_dict = {
            "kd_loss": total_loss,
            "logits_loss": logits_loss,
            "intermediate_loss": intermediate_loss,
        }
        return out_dict


########################################################


class ProjectionLayer(MegatronModule):
    """Module to project student layer activations to teacher's size."""

    def __init__(self, student_config: TransformerConfig, teacher_config: TransformerConfig):
        """
        Constructor.

        Args:
            student_config: Student's MCore transformer config.
            teacher_config: Teacher's MCore transformer config.
        """
        super().__init__(config=student_config)
        if student_config.hidden_size == teacher_config.hidden_size:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(student_config.hidden_size, teacher_config.hidden_size)
            self.apply(self._init_weights)
            # Attribute below needed to reduce gradients during backward properly.
            setattr(self._fit.weight, "sequence_parallel", self.config.sequence_parallel)
            setattr(self._fit.bias, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, student_tensor: Tensor):
        """
        Forward function.

        Args:
            student_tensor: Tensor to be fit to teacher size.
        """
        return self._fit(student_tensor)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(
    tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD
):
    """Custom all-reduce function.

    Needed instead of other all-reduce functions available when the computation following
    the all-reduce call differs per rank. In KL loss, this corresponds to the different numerators.
    """
    return _AllReduce.apply(op, group, tensor)


########################################################


def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Get rid of ModelOpt Distillation state
    # NOTE: If re-placed, above losses need modifcation as `TransformerConfig` has non-pickleable elements.
    mto.ModeloptStateManager(model)._state.pop()

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        with self.hide_teacher_model():
            return type(self).sharded_state_dict(self, *args, **kwargs)

    model.sharded_state_dict = types.MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if distill_cfg["skip_lm_loss"] and self.training:
            return torch.zeros_like(labels)
        return type(self).compute_language_model_loss(self, labels, logits)

    model.compute_language_model_loss = types.MethodType(_compute_language_model_loss, model)

    # HACK: Skip `lm_loss` always for teacher.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        return torch.zeros_like(labels)

    model.teacher_model.compute_language_model_loss = types.MethodType(
        _compute_language_model_loss, model.teacher_model
    )

    # HACK: Pipeline-parallel Distillation requires splitting input tensor into student and teacher parts.
    def _set_student_input_tensor_shape(self, shapes: List[Tuple[int]]):
        self._tensor_split_idx = shapes[0][-1]

    def _set_input_tensor(self, input_tensors: List[Tensor]):
        teacher_inputs = [t[..., self._tensor_split_idx:] if t is not None else t for t in input_tensors]
        student_inputs = [t[..., :self._tensor_split_idx] if t is not None else t for t in input_tensors]
        type(self).set_input_tensor(self.teacher_model, teacher_inputs)
        type(self).set_input_tensor(self, student_inputs)

    model.set_student_input_tensor_shape = types.MethodType(_set_student_input_tensor_shape, model)
    model.set_input_tensor = types.MethodType(_set_input_tensor, model)

    # HACK: Concatenate output tensors when PP>1 so they can be passed between ranks.
    def _forward(self, *args, **kwargs):
        if not self.training:
            with self.only_student_forward():
                return type(self).forward(self, *args, **kwargs)

        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(*args, **kwargs)
        with self.only_student_forward():
            student_output = type(self).forward(self, *args, **kwargs)

        if not is_pipeline_last_stage():
            return torch.cat([student_output, teacher_output], dim=-1)
        else:
            return student_output

    model.forward = types.MethodType(_forward, model)


def get_tensor_shapes_adjust_fn_for_distillation(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
) -> Union[Callable, None]:
    if (
        forward_only
        or get_pipeline_model_parallel_world_size() == 1
        or get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        return None
    # Unwrap
    if isinstance(model, list):
        model = model[0]
    while hasattr(model, "module"):
        model = model.module
    if not isinstance(model, mtd.DistillationModel):
        return None

    def adjust_tensor_shapes(recv_tensor_shapes: List[Tuple[int, ...]], send_tensor_shapes: List[Tuple[int, ...]]):
        rank = get_pipeline_model_parallel_rank()
        teacher_config = get_model_config(model.teacher_model)
        teacher_model_type = get_model_type(model.teacher_model)
        teacher_encoder_decoder_xattn = get_model_xattn(model.teacher_model)

        teacher_recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=teacher_model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            encoder_decoder_xattn=teacher_encoder_decoder_xattn,
        )
        teacher_send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=teacher_model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=teacher_config,
            encoder_decoder_xattn=teacher_encoder_decoder_xattn,
        )
        model.set_student_input_tensor_shape(recv_tensor_shapes)

        for i, shape in enumerate(recv_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_recv_tensor_shapes[0][-1]
            recv_tensor_shapes[i] = tuple(shape)
        for i, shape in enumerate(send_tensor_shapes):
            shape = list(shape)
            shape[-1] += teacher_send_tensor_shapes[0][-1]
            send_tensor_shapes[i] = tuple(shape)

        return recv_tensor_shapes, send_tensor_shapes

    return adjust_tensor_shapes
