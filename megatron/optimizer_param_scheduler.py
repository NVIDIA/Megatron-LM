# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""

import math

from megatron import print_rank_0, get_args

class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay"""

    def __init__(self, optimizer, max_lr, min_lr,
                 lr_warmup_steps, lr_decay_steps, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False):
        args = get_args()
        # Class values.
        self.optimizer = optimizer

        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_tokens = args.lr_decay_tokens
        self.num_tokens = 0
        self.lr_warmup_tokens = args.lr_warmup_tokens

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        # Set the learning rate
        self.step(0)
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == 'constant':
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Use linear warmup for the initial part.
        if self.lr_warmup_tokens is None:
            if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
                if self.num_steps == self.lr_warmup_steps and \
                    self.lr_decay_tokens is not None:
                    # The case of step/sample-wise warmup + token-wise decay
                    self.lr_warmup_tokens = self.num_tokens
                return self.max_lr * float(self.num_steps) / \
                    float(self.lr_warmup_steps)
        else:
            if self.lr_warmup_tokens > 0 and self.num_tokens <= self.lr_warmup_tokens:
                return self.max_lr * float(self.num_tokens) / \
                    float(self.lr_warmup_tokens)

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return self.max_lr

        # For any steps larger than `self.lr_decay_steps`, use `self.min_lr`.
        if self.lr_decay_tokens is None:
            if self.num_steps > self.lr_decay_steps:
                return self.min_lr
        else:
            if self.num_tokens > self.lr_decay_tokens:
                return self.min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            if self.lr_warmup_tokens is None:
                warmup_steps = max(self.lr_warmup_steps, 1)
                num_steps = max(self.num_steps, 1)
                lr = self.max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
            else:
                warmup_tokens = max(self.lr_warmup_tokens, 1)
                num_tokens = max(self.num_tokens, 1)
                lr = self.max_lr * warmup_tokens ** 0.5 / (num_tokens ** 0.5)
            return max(self.min_lr, lr)

        if self.lr_decay_tokens is None:
            num_steps_ = self.num_steps - self.lr_warmup_steps
            decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
            decay_ratio = float(num_steps_) / float(decay_steps_)
        else:
            num_tokens_ = self.num_tokens - self.lr_warmup_tokens
            decay_tokens_ = self.lr_decay_tokens - self.lr_warmup_tokens
            decay_ratio = float(num_tokens_) / float(decay_tokens_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr

        if self.lr_decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.lr_decay_style))

        return self.min_lr + coeff * delta_lr


    def step(self, increment, token_num=None):
        """Set lr for all parameters groups."""
        if token_num is None:
            args = get_args()
            token_num = args.consumed_train_tokens
        self.num_tokens = token_num
        self.num_steps += increment
        new_lr = self.get_lr()
        new_wd = self.get_wd()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr * group.get('lr_mult', 1.0)
            group['weight_decay'] = new_wd * group.get('wd_mult', 1.0)


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'lr_warmup_tokens': self.lr_warmup_tokens,
            'num_steps': self.num_steps,
            'num_tokens': self.num_tokens,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'lr_decay_tokens': self.lr_decay_tokens,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, \
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')
        
        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            lr_warmup_steps_ = sd['warmup_iter']
        elif 'warmup_steps' in sd:
            lr_warmup_steps_ = sd['warmup_steps']
        else:
            lr_warmup_steps_ = sd['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps,
                                                lr_warmup_steps_,
                                                'warmup iterations')
        if 'warmup_tokens' in sd:
            lr_warmup_tokens_ = sd['warmup_tokens']
        else:
            lr_warmup_tokens_ = sd['lr_warmup_tokens']
        self.lr_warmup_tokens = self._check_and_set(self.lr_warmup_tokens,
                                                lr_warmup_tokens_,
                                                'warmup tokens')

        if 'end_iter' in sd:
            lr_decay_steps_ = sd['end_iter']
        elif 'decay_steps' in sd:
            lr_decay_steps_  = sd['decay_steps']
        else:
            lr_decay_steps_ = sd['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, lr_decay_steps_,
                                               'total number of iterations')
        if 'decay_tokens' in sd:
            lr_decay_tokens_ = sd['decay_tokens']
        else:
            lr_decay_tokens_ = sd['lr_decay_tokens']
        self.lr_decay_tokens = self._check_and_set(self.lr_decay_tokens,
                                                lr_decay_tokens_,
                                                'decay tokens')

        if 'decay_style' in sd:
            lr_decay_style_ = sd['decay_style']
        else:
            lr_decay_style_ = sd['lr_decay_style']
        self.lr_decay_style = self._check_and_set(self.lr_decay_style,
                                               lr_decay_style_,
                                               'learning rate decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        if 'num_tokens' in sd:
            self.num_tokens = sd['num_tokens']
        self.step(increment=num_steps, token_num=self.num_tokens)


        if 'start_wd' in sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                sd['start_wd'],
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                                sd['end_wd'],
                                                "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                sd['wd_incr_steps'],
                                                "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                sd['wd_incr_style'],
                                                "weight decay incr style")
            







