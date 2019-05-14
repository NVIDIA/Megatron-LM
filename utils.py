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

"""Utilities for logging and serialization"""

import os
import random
import time
import numpy as np
import torch


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0/ normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print(string, flush=True)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_cached()/ mega_bytes)
    print(string, flush=True)


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """Load a model checkpoint."""

    checkpoint_path = args.load
    model_path = checkpoint_path
    model_sd = torch.load(model_path, map_location='cpu')
    total_iters = model_sd['total_iters']
    epoch = model_sd['epoch']
    i = model_sd['mid_epoch_iters']
    model.load_state_dict(model_sd['sd'])

    checkpoint_path = os.path.dirname(checkpoint_path)
    if args.load_optim:
        optim_path = os.path.join(checkpoint_path, 'optim.pt')
        optim_sd, lr_sd = torch.load(optim_path, map_location='cpu')
        optimizer.load_state_dict(optim_sd)
        lr_scheduler.load_state_dict(lr_sd)
    elif args.fp16:
        optimizer._model_params_to_master_params()

    rng_path = None
    if args.load_rng:
        rng_path = os.path.join(checkpoint_path, 'rng.pt')
    if args.load_all_rng:
        rng_path = os.path.join(checkpoint_path,
                                'rng.%d.pt'%(torch.distributed.get_rank()))
    if rng_path is not None:
        rng_state = torch.load(rng_path)
        torch.cuda.set_rng_state(rng_state[0])
        torch.set_rng_state(rng_state[1])
        np.random.set_state(rng_state[2])
        random.setstate(rng_state[3])

    return epoch, i, total_iters


def save_checkpoint(model_suffix, epoch, i, model, optimizer, lr_scheduler, args):
    """Save a model checkpoint."""

    model_path = os.path.join(args.save, model_suffix)
    checkpoint_dir = os.path.dirname(model_path)
    rng_state = (torch.cuda.get_rng_state(),
                 torch.get_rng_state(),
                 np.random.get_state(),
                 random.getstate())
    if not (torch.distributed.is_initialized() and \
            torch.distributed.get_rank() > 0):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        total_iters = args.train_iters * (epoch-1) + i
        sd = {'sd': model.state_dict()}
        sd['total_iters'] = total_iters
        sd['epoch'] = epoch
        sd['mid_epoch_iters'] = i
        torch.save(sd, model_path)
        print('saved', model_path)

        if args.save_optim:
            optim_path = os.path.join(checkpoint_dir, 'optim.pt')
            torch.save((optimizer.state_dict(),
                        lr_scheduler.state_dict()), optim_path)
            print('saved', optim_path)

        if args.save_rng:
            rng_path = os.path.join(checkpoint_dir, 'rng.pt')
            torch.save(rng_state, rng_path)
            print('saved', rng_path)
    else:
        while not os.path.exists(checkpoint_dir):
            time.sleep(1)
    if args.save_all_rng:
        rng_path = os.path.join(checkpoint_dir,
                                'rng.%d.pt'%(torch.distributed.get_rank()))
        torch.save(rng_state, rng_path)
        print('saved', rng_path)
