# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from itertools import product
import unittest
import copy

import torch
from torch import nn
from torch.testing._internal.common_device_type import largeTensorTest
import transformer_engine.pytorch as te
from megatron.core.optimizer.hybrid_adam import HybridAdam

class TestFusedOptimizer(unittest.TestCase):
    def setUp(self, iters=7):
        self.iters = iters
        torch.manual_seed(9876)

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, options, tst_options=None):

        # Adding this to make backward compatible with existing tests. Just in
        # case "tst_options" are not provided, it gets a copy of options
        # which contains the parameters for the reference optimizer
        if tst_options == None:
            tst_options = options

        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        ref_optim = self.ref_optim(ref_param, **options)
        tst_optim = self.hybrid_optim(tst_param, **tst_options)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def gen_mixed_grad(self, ref_param, tst_param, scale=1.0):
        half_grads = []
        for p_ref, p_tst in zip(ref_param, tst_param):
            half_grads.append(torch.rand_like(p_ref).half())
            p_ref.grad = half_grads[-1].float() / scale
        return half_grads

    def gen_single_type_test(
        self, param_type=torch.float, device="cuda", *, skip_assert: bool = False
    ):
        nelem = 278011

        # Some ref and test optimizers may require different set of options.
        # This is a quick workaround to add that functionality while making
        # minimum changes in existing code.
        # If there is no "tst_options" field provided, safe to initialize
        # the test optimizer with the parameters of reference optimizer.
        if not hasattr(self, "tst_options"):
            self.tst_options = self.options

        tensor = torch.rand(nelem, dtype=param_type, device=device)

        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
            [tensor], self.options, self.tst_options
        )

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()
            if skip_assert:
                return
            torch.testing.assert_close(ref_param, tst_param)

class TestHybridAdam(TestFusedOptimizer):

    def setUp(self):
        super().setUp()
        self.options = {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False,
        }
        self.ref_optim = te.optimizers.FusedAdam
        self.hybrid_optim = HybridAdam

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    # NOTE(mkozuki): Current threshold values look too small for BFloat16.
    # TODO(mkozuki): Refactor `TestFusedOptimizer`
    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16, skip_assert=True)

    def test_bfloat16(self):
        self.gen_single_type_test(param_type=torch.bfloat16, skip_assert=True)

    @unittest.skipIf(torch.cuda.device_count() < 2, "more than 1 GPU required")
    def test_multi_device(self):
        devices = ("cuda:0", "cuda:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.cuda.device(current_dev):
                self.gen_single_type_test(param_type=torch.float, device=tensor_dev)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]

        tensors = []
        for size in sizes:
            tensors.append(torch.rand(size, dtype=torch.float, device="cuda"))
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(tensors, self.options)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def test_adam_option(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

    def test_frozen_model(self):
        nelem = 1
        adam_option = {
            "lr": 0.01,
            "betas": (0.6, 0.9),
            "eps": 3e-06,
            "weight_decay": 0,
            "amsgrad": False,
        }

        tensor = torch.rand(nelem, dtype=torch.float, device="cuda")
        ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim([tensor], adam_option)

        # Add an empty param group which may occur for pipeline parallel p-tuning
        tst_optim.add_param_group({"params": []})

        for i in range(self.iters):
            self.gen_grad(ref_param, tst_param)
            ref_optim.step()
            tst_optim.step()

            torch.testing.assert_close(ref_param, tst_param)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class AdamTest(unittest.TestCase):
    def setUp(self, seed=0):
        super().setUp()
        torch.manual_seed(seed)

        self.model = Model().cuda()
        self.model_ = Model().cuda()
        self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

        self.lr = 0.00001
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def testGradScaler(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = HybridAdam(params_, lr=self.lr, capturable=False)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def testNative(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = te.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            y = self.model(x)
            loss = ((gt - y) ** 2).mean()

            loss.backward()
            self.optimizer.step()

            # DUT
            y = self.model_(x)
            loss_ = ((gt_ - y) ** 2).mean()

            loss_.backward()
            optimizer_.step()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(
                        m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )
                    torch.testing.assert_close(
                        m.weight.grad,
                        m_.weight.grad,
                        atol=1e-3,
                        rtol=1e-3,
                        equal_nan=True,
                    )

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    @largeTensorTest("60GB", "cuda")
    def testLargeTensor(self):
        t = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        t2 = torch.zeros(2359332864, dtype=torch.half, device="cuda")
        grad = torch.randn_like(t)
        t.grad = grad
        t2.grad = grad
        params = [t]
        params2 = [t2]
        optimizer = HybridAdam(params, lr=self.lr)
        optimizer.step()
        optimizer2 = torch.optim.Adam(params2, lr=self.lr)
        torch.testing.assert_close(t, t2)
        torch.cuda.synchronize()