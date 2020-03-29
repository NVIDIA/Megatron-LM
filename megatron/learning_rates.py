# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Learning rate decay functions."""

import math

from megatron import print_rank_0


class AnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(self, optimizer, start_lr,
                 warmup_iter, total_iters,
                 decay_style, last_iter, min_lr=0.0,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):

        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, 'both override and '\
                'use-checkpoint are set.'
        # Set the learning rate
        self.step(self.num_iters)

        print_rank_0('> learning rate decay style: {}'.format(self.decay_style))


    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        num_iters_ = min(self.num_iters, self.end_iter - self.warmup_iter)
        # Warmup.
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * num_iters_ / self.warmup_iter

        num_iters_ = num_iters_ - self.warmup_iter
        if self.decay_style == 'linear':
            lr = self.start_lr * (self.end_iter - num_iters_) / self.end_iter
        elif self.decay_style == 'cosine':
            lr = self.start_lr / 2.0 * (math.cos(
                math.pi * num_iters_ / self.end_iter) + 1)
        elif self.decay_style == 'exponential':
            # exp(-0.693) = 1/2
            lr = self.start_lr * math.exp(-0.693 * num_iters_ / self.end_iter)
        else:
            lr = self.start_lr
        return max(lr, self.min_lr)


    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr


    def state_dict(self):
        state_dict = {
            'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            'num_iters': self.num_iters,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter,
            'min_lr': self.min_lr
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, 'AnnealingLR: class input value' \
                'and checkpoint values for {} do not match'.format(name)
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        self.start_lr = self._check_and_set(self.start_lr, sd['start_lr'],
                                            'learning rate')
        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')
        self.warmup_iter = self._check_and_set(self.warmup_iter,
                                               sd['warmup_iter'],
                                               'warmup iterations')
        self.end_iter = self._check_and_set(self.end_iter, sd['end_iter'],
                                            'total number of iterations')
        self.decay_style = self._check_and_set(self.decay_style,
                                               sd['decay_style'],
                                               'decay style')

        self.num_iters = sd['num_iters']
        self.step(self.num_iters)
