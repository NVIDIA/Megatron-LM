# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Distillation loss function(s)."""

import logging
import types
from abc import ABCMeta
from typing import Any, Dict, Optional, Tuple

import modelopt.torch.distill as mtd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn.modules.loss import _Loss

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args, print_rank_0

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

    hidden_size_student = student_cfg.hidden_size
    hidden_size_teacher = teacher_cfg.hidden_size

    criterion = {tuple(logit_pair): LogitsKLLoss()}
    for layer_names in intermediate_pairs:
        print_rank_0(
            "Distillation: Adding intermediate loss between"
            f" `{layer_names[0]}` of student (hidden size {hidden_size_student}) and"
            f" `{layer_names[1]}` of teacher (hidden size {hidden_size_teacher})."
        )
        criterion[tuple(layer_names)] = HiddenStateCosineLoss(
            hidden_size_student, hidden_size_teacher
        )

    loss_balancer = LogitsAndIntermediatesLossBalancer(
        kd_loss_scale=loss_scale, skip_original_loss=skip_lm_loss
    )

    cfg["criterion"] = criterion
    cfg["loss_balancer"] = loss_balancer

    return cfg


########################################################


class BaseLoss(_Loss, metaclass=ABCMeta):
    """Abstract base class for Megatron distillation losses."""

    def __init__(
        self, hidden_size_student: Optional[int] = None, hidden_size_teacher: Optional[int] = None
    ):
        """
        Constructor.

        Args:
            hidden_size_student: Size of the student's hidden dimension.
            hidden_size_teacher: Size of the teacher's hidden dimension.
        """
        super().__init__()
        self._projection = ProjectionLayer(hidden_size_student, hidden_size_teacher)
        args = get_args()
        self._tensor_parallel = args.tensor_model_parallel_size > 1
        self._sequence_parallel = args.sequence_parallel

    def pre_forward(self, predictions: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs projection of student tensor to match teacher's size if necessary."""
        if isinstance(predictions, tuple):
            # `ColumnParallelLinear` returns bias too
            predictions, targets = predictions[0], targets[0]

        predictions = self._projection(predictions)
        targets = targets.detach()

        return predictions, targets

    def post_forward(self, loss: Tensor, tp_reduce: bool = False) -> Tensor:
        """Reshapes tensor from [s, b] to [b, s] for upcoming loss masking."""
        loss = loss.transpose(0, 1).contiguous()
        return (loss, tp_reduce)


class MSELoss(BaseLoss):
    """Calculates Mean Squared Error loss between two tensors without reducing the sequence dim."""

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            predictions: Student model tensors (size [s, b, h])
            targets: Teacher model tensors (size [s, b, h])

        Returns:
            MSE loss of tensors (size [b, s])
        """
        predictions, targets = self.pre_forward(predictions, targets)

        # TP irrelevant since MSE loss gradients are per-input element.
        loss = F.mse_loss(predictions, targets, reduction="none")
        loss = loss.sum(dim=-1)

        return self.post_forward(loss)


class HiddenStateCosineLoss(BaseLoss):
    """
    Calculates Cosine loss between two tensors without reducing the sequence dim.

    The tensors are assumed to be intermediate activations, so extra restrictions are in place.
    """

    def __init__(
        self, hidden_size_student: Optional[int] = None, hidden_size_teacher: Optional[int] = None
    ):
        """
        Constructor.

        Args:
            hidden_size_student: Size of the student's hidden dimension.
            hidden_size_teacher: Size of the teacher's hidden dimension.
        """
        super().__init__(hidden_size_student, hidden_size_teacher)

        if self._tensor_parallel and not self._sequence_parallel:
            logger.warning(
                "``HiddenStateCosineLoss`` only works with tensors with full hidden dim. Ensure the "
                "tensor inputs meet this requirement or use `--sequence_parallel` if tensor parallel is enabled."
            )
        if hidden_size_student is None or hidden_size_teacher is None:
            logger.warning(
                "Hidden sizes of teacher and student not provided. This assumes "
                "they are the same shape, which may be a mistake."
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

        if self._sequence_parallel:
            # Can efficiently gather size [s, b] tensor now for loss-masking purposes.
            # TODO(aanoosheh) Reconsider for memory savings by splitting loss mask instead.
            loss = gather_from_sequence_parallel_region(loss)

        return self.post_forward(loss)


class LogitsKLLoss(BaseLoss):
    """Calculates KL-Divergence loss between two logits tensors without reducing the sequence dim."""

    def __init__(self, temperature: float = 1.0, reverse: bool = False):
        """
        Constructor.

        Args:
            temperature: Divide tensors by this value prior to calculating loss.
            reverse: Whether to reverse the loss as KLD(teacher, student) instead of KLD(student, teacher)
        """
        super().__init__()
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
        if self._tensor_parallel:

            # Maximum value along vocab dimension across all GPUs.
            teacher_logits_max, _ = torch.max(output_teacher, dim=-1)
            torch.distributed.all_reduce(
                teacher_logits_max,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tensor_model_parallel_group(),
            )
            output_teacher = output_teacher - teacher_logits_max.unsqueeze(dim=-1)

            denom_teacher = torch.sum(torch.exp(output_teacher), dim=-1)
            # We can't use `gather_from_tensor_model_parallel_region` here since it discards
            # gradients from other ranks - we need to all_reduce the gradients as well.
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
        for _key, _loss in loss_dict.items():
            if _key.startswith(LogitsKLLoss.__name__):
                logits_loss = _loss  # should only be one
        intermediate_loss = sum(loss_dict.values())

        if intermediate_loss > 0:
            dynamic_scale = logits_loss.item() / intermediate_loss.item()
            intermediate_loss *= dynamic_scale
            kd_loss_scale = self._kd_loss_scale / 2.0
        else:
            kd_loss_scale = self._kd_loss_scale

        if self._skip_original_loss:
            kd_loss = logits_loss + intermediate_loss
            total_loss = kd_loss
        else:
            kd_loss = (logits_loss + intermediate_loss) * kd_loss_scale
            dynamic_scale = original_loss.item() / kd_loss.item()
            total_loss = original_loss + kd_loss * dynamic_scale

        return total_loss


########################################################


class ProjectionLayer(nn.Module):
    """Module to project student layer activations to teacher's size."""

    def __init__(self, hidden_size_student: int, hidden_size_teacher: int):
        """
        Constructor.

        Args:
            hidden_size_student: Size of the student's hidden dimension.
            hidden_size_teacher: Size of the teacher's hidden dimension.
        """
        super().__init__()
        if hidden_size_student == hidden_size_teacher:
            self._fit = nn.Identity()
        else:
            self._fit = nn.Linear(hidden_size_student, hidden_size_teacher)
            self.apply(self._init_weights)
            setattr(self._fit.weight, 'sequence_parallel', get_args().sequence_parallel)
            setattr(self._fit.bias, 'sequence_parallel', get_args().sequence_parallel)

    def forward(self, student_tensor: Tensor):
        """
        Forward function.

        Args:
            student_tensor: Tensor to be fit to teacher size.
        """
        return self._fit(student_tensor)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
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
    return _AllReduce.apply(op, group, tensor)


########################################################


def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):
    """Extra modifcations to ``mtd.DistillationModel`` requried for Megatron-Core."""

    # HACK: Hide teacher during `sharded_state_dict` method.
    def _sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        with self.hide_teacher_model():
            return self._sharded_state_dict(*args, **kwargs)

    model._sharded_state_dict = model.sharded_state_dict
    model.sharded_state_dict = types.MethodType(_sharded_state_dict, model)

    # HACK: Skip `lm_loss` bypassing it when training if not needed for backprop.
    def _compute_language_model_loss(self, labels, logits) -> Tensor:
        if self.training:
            return torch.zeros_like(labels)
        return self._compute_language_model_loss(labels, logits)

    if distill_cfg["skip_lm_loss"]:
        model._compute_language_model_loss = model.compute_language_model_loss
        model.compute_language_model_loss = types.MethodType(_compute_language_model_loss, model)
