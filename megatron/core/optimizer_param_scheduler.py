# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""
import logging
import math
from typing import Optional

from megatron.core.optimizer import MegatronOptimizer
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


class OptimizerParamScheduler:
    """Anneals learning rate and weight decay

    Args:
        optimizer (MegatronOptimizer): the optimizer to be used
        init_lr (float): initial learning rate
        max_lr (float): maximum learning rate
        min_lr (float): minimum learning rate
        lr_warmup_steps (int): number of warmup steps
        lr_decay_steps (int): number of decay steps
        lr_decay_style (str): decay style for learning rate
        start_wd (float): initial weight decay
        end_wd (float): final weight decay
        wd_incr_steps (int): number of weight decay increment steps
        wd_incr_style (str): weight decay increment style
        use_checkpoint_opt_param_scheduler (bool, optional): whether to use the checkpoint values
            for the optimizer param scheduler
        override_opt_param_scheduler (bool, optional): whether to override the optimizer param
            scheduler values with the class values
        wsd_decay_steps (int, optional): number of weight decay decay steps
        lr_wsd_decay_style (str, optional): decay style for learning rate during weight decay decay
            steps

    """

    def __init__(
        self,
        optimizer: MegatronOptimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        use_checkpoint_opt_param_scheduler: Optional[bool] = True,
        override_opt_param_scheduler: Optional[bool] = False,
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: Optional[str] = None,
    ) -> None:

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, (
                'both override and ' 'use-checkpoint are set.'
            )

        # Set the learning rate
        self.step(0)
        log_single_rank(logger, logging.INFO, f"> learning rate decay style: {self.lr_decay_style}")

    def get_wd(self) -> float:
        """Weight decay incr functions"""
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
            raise Exception(f'{self.wd_incr_style} weight decay increment style is not supported.')

        return self.start_wd + coeff * delta_wd

    def get_lr(self, param_group: dict) -> float:
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4

        Args:
            param_group (dict): parameter group from the optimizer.
        """

        max_lr = param_group.get('max_lr', self.max_lr)
        min_lr = param_group.get('min_lr', self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == 'linear':
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == 'WSD':
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.num_steps <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.num_steps - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
        else:
            raise Exception(f'{self.lr_decay_style} decay style is not supported.')

        return min_lr + coeff * delta_lr

    def step(self, increment: int) -> None:
        """Set lr for all parameters groups.

        Args:
            increment (int): number of steps to increment
        """
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)

    def state_dict(self) -> dict:
        """Return the state dict."""
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps,
        }
        return state_dict

    def _check_and_set(self, cls_value: float, sd_value: float, name: str) -> float:
        """Auxiliary function for checking the values in the checkpoint and
        setting them.

        Args:
            cls_value (float): class value
            sd_value (float): checkpoint value
            name (str): name of the parameter
        """

        if self.override_opt_param_scheduler:
            log_single_rank(logger, logging.INFO, f" > overriding {name} value to {cls_value}")
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, (
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint'
                f'value {sd_value} for {name} do not match'
            )

        log_single_rank(logger, logging.INFO, f" > using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict.

        Args:
            state_dict (dict): state dict to be load
        """

        if 'start_lr' in state_dict:
            max_lr_ = state_dict['start_lr']
        else:
            max_lr_ = state_dict['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, 'learning rate')

        self.min_lr = self._check_and_set(
            self.min_lr, state_dict['min_lr'], 'minimum learning rate'
        )

        if 'warmup_iter' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_iter']
        elif 'warmup_steps' in state_dict:
            lr_warmup_steps_ = state_dict['warmup_steps']
        else:
            lr_warmup_steps_ = state_dict['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, 'warmup iterations'
        )

        if 'end_iter' in state_dict:
            lr_decay_steps_ = state_dict['end_iter']
        elif 'decay_steps' in state_dict:
            lr_decay_steps_ = state_dict['decay_steps']
        else:
            lr_decay_steps_ = state_dict['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, 'total number of iterations'
        )

        if 'decay_style' in state_dict:
            lr_decay_style_ = state_dict['decay_style']
        else:
            lr_decay_style_ = state_dict['lr_decay_style']
        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style, lr_decay_style_, 'learning rate decay style'
        )

        if 'num_iters' in state_dict:
            num_steps = state_dict['num_iters']
        else:
            num_steps = state_dict['num_steps']
        self.step(increment=num_steps)

        if 'start_wd' in state_dict:
            self.start_wd = self._check_and_set(
                self.start_wd, state_dict['start_wd'], "start weight decay"
            )
            self.end_wd = self._check_and_set(self.end_wd, state_dict['end_wd'], "end weight decay")
            self.wd_incr_steps = self._check_and_set(
                self.wd_incr_steps,
                state_dict['wd_incr_steps'],
                "total number of weight decay iterations",
            )
            self.wd_incr_style = self._check_and_set(
                self.wd_incr_style, state_dict['wd_incr_style'], "weight decay incr style"
            )
