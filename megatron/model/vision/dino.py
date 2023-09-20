# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/facebookresearch/dino/blob/main/main_dino.py
# reworked/refactored some parts to make it run in Megatron.
import math
import apex
import einops
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from megatron import get_args, print_rank_0
from megatron.model.utils import get_linear_layer
from megatron.model.vision.vit_backbone import VitBackbone
from megatron.model.module import MegatronModule
from megatron.model.vision.mit_backbone import mit_b5_avg
from megatron.model.vision.esvit_swin_backbone import get_swin


class DINOLoss(torch.nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output, iteration):
        """
        Cross-entropy between softmax outputs of the teacher
        and student network.
        """
        args = get_args()
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        epoch = iteration // args.iter_per_epoch

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)

        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        torch.distributed.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * torch.distributed.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, nlayers=3):
        super().__init__()
        args = get_args()
        hidden_dim = args.dino_head_hidden_size
        bottleneck_dim = args.dino_bottleneck_size
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = torch.nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [torch.nn.Linear(in_dim, hidden_dim)]
            layers.append(torch.nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(MegatronModule):

    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        #backbone.fc, backbone.head = torch.nn.Identity(), torch.nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        if self.training:
            return self.head(output)
        else:
            return output


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = \
                np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) \
        * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_student_backbone_and_num_features(config, pre_process=True, post_process=True):
    args = get_args()

    if args.vision_backbone_type == 'vit':
        student = VitBackbone(config,
                              pre_process=pre_process,
                              post_process=post_process,
                              drop_path_rate=0.1,
                              single_token_output=True)
        num_features = args.hidden_size
    elif args.vision_backbone_type == 'mit':
        student = mit_b5_avg(drop_path_rate=0.1)
        num_features = 512
    elif args.vision_backbone_type == 'swin':
        student = get_swin()
        num_features = student.num_features
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))

    return student, num_features

def get_teacher_backbone_and_num_features(config, pre_process=True, post_process=True):
    args = get_args()

    if args.vision_backbone_type == 'vit':
        teacher = VitBackbone(config,
                              pre_process=pre_process,
                              post_process=post_process,
                              single_token_output=True)
        num_features = args.hidden_size
    elif args.vision_backbone_type == 'mit':
        teacher = mit_b5_avg(drop_path_rate=0.0)
        num_features = 512
    elif args.vision_backbone_type == 'swin':
        teacher = get_swin(is_teacher=True)
        num_features = teacher.num_features
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))
    return teacher, num_features


class DINOPretrainModel(MegatronModule):
    def __init__(self, config, pre_process=True, post_process=True):
        super(DINOPretrainModel, self).__init__()
        args = get_args()
        self.config = config
        self.out_dim = 65536

        self.dino_loss = DINOLoss(
            self.out_dim,
            args.dino_local_crops_number + 2,
            args.dino_warmup_teacher_temp,
            args.dino_teacher_temp,
            args.dino_warmup_teacher_temp_epochs,
            300,
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.momentum_teacher = 0.996

        student_backbone, num_features = \
            get_student_backbone_and_num_features(config, pre_process, post_process)

        self.student = MultiCropWrapper(
            student_backbone,
            DINOHead(num_features, self.out_dim,
                     norm_last_layer=args.dino_norm_last_layer)
        )

        self.momentum_schedule = cosine_scheduler(
            self.momentum_teacher, 1,
            args.train_iters // args.iter_per_epoch,
            args.iter_per_epoch
        )

        teacher_backbone, num_features = \
            get_teacher_backbone_and_num_features(config, pre_process, post_process)
        self.teacher = MultiCropWrapper(
            teacher_backbone,
            DINOHead(num_features, self.out_dim)
        )
        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            if hasattr(p, "requires_grad") and p.requires_grad is not None:
                p.requires_grad = False

    def set_input_tensor(self, tensor):
        pass

    def forward(self, input):
        student_output = None
        if self.training:
            student_output = self.student(input)
            teacher_output = self.teacher(input[:2])
        else:
            teacher_output = self.teacher(input)
        return student_output, teacher_output

    def cancel_gradients_last_layer(self, iteration):
        args = get_args()
        epoch = iteration // args.iter_per_epoch
        if epoch < args.dino_freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None

    def update_momentum(self, iteration):
        with torch.no_grad():
            m = self.momentum_schedule[iteration]
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

