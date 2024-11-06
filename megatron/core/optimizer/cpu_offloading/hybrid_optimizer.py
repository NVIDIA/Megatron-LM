import torch
from collections import defaultdict
from typing import Dict

class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params, offload_ratio=0.5, cpu_optimizer_cls=None, gpu_optimizer_cls=None, pin_cpu_grads: bool=True, **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(params, defaults={})
        self.params = params.copy()
        self.offload_ratio = offload_ratio

        (
            self.cpu_params,
            self.gpu_params,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
        ) = self._split_parameters_updated_on_the_cpu_and_gpu()
        self.pin_cpu_grads = pin_cpu_grads
        self.cpu_optimizer = cpu_optimizer_cls(self.cpu_params, **kwargs)
        self.gpu_optimizer = gpu_optimizer_cls(self.gpu_params, **kwargs)
        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._data_stream = torch.cuda.Stream()
        self._step_stream = torch.cuda.Stream()

        self.register_grad_cpu_copy_hook()
        self.register_param_copy_back_gpu_hook()

    def register_grad_cpu_copy_hook(self):
        def grad_cpu_copy_hook_closure():
            def grad_cpu_copy_hook(optimizer, args, kwargs):
                for gpu_param, cpu_copy in self.gpu_params_map_cpu_copy.items():
                    if cpu_copy not in self.cpu_copy_map_grad:
                        self.cpu_copy_map_grad[cpu_copy] = torch.empty(
                            gpu_param.grad.shape,
                            dtype=gpu_param.grad.dtype,
                            pin_memory=self.pin_cpu_grads
                        )
                self._data_stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(self._data_stream):
                    for gpu_param, cpu_copy in self.gpu_params_map_cpu_copy.items():
                        self.cpu_copy_map_grad[cpu_copy].data.copy_(gpu_param.grad, non_blocking=True)
                        cpu_copy.grad = self.cpu_copy_map_grad[cpu_copy]
            return grad_cpu_copy_hook

        self.register_step_pre_hook(grad_cpu_copy_hook_closure())

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._data_stream.wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(self._data_stream):      
                    for cpu_copy, gpu_param in self.cpu_copys_map_gpu_param.items():
                        gpu_param.data.copy_(cpu_copy.data, non_blocking=True)
                # NOTE: ensure all H2D/D2H Transfer and update operations finish
                torch.cuda.synchronize()

            return param_copy_back_gpu_hook

        self.register_step_post_hook(param_copy_back_gpu_hook_closure())

    def state_dict(self):
        cpu_state_dict = self.cpu_optimizer.state_dict()
        gpu_state_dict = self.gpu_optimizer.state_dict()

        state = cpu_state_dict["state"].copy()
        state.update(
            {k + len(cpu_state_dict["state"]): v for k, v in gpu_state_dict["state"].items()}
        )

        param_groups = cpu_state_dict["param_groups"].copy()
        cpu_params_num = sum([len(pg['params']) for pg in cpu_state_dict["param_groups"]])
        for param_group in gpu_state_dict["param_groups"]:
            pg_copy = param_group.copy()
            pg_copy["params"] = [cpu_params_num + i for i in pg_copy["params"]]
            param_groups.append(pg_copy)

        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        cpu_state_dict = {
            "state": state_dict["state"][: len(self.cpu_params)],
            "param_groups": state_dict["param_groups"][: len(self.cpu_params)],
        }
        gpu_state_dict = {
            "state": state_dict["state"][len(self.cpu_params) :],
            "param_groups": state_dict["param_groups"][len(self.cpu_params) :],
        }
        self.cpu_optimizer.load_state_dict(cpu_state_dict)
        self.gpu_optimizer.load_state_dict(gpu_state_dict)

    def param_groups(self):
        return self.cpu_optimizer.param_groups + self.gpu_optimizer.param_groups

    def step(self, closure=None):
        self._step_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(self._step_stream):
            self.gpu_optimizer.step(closure)
        self._data_stream.synchronize()
        self.cpu_optimizer.step(closure)

    def _split_parameters_updated_on_the_cpu_and_gpu(self):
        total_params_numel = sum([p.numel() for p in self.params])
        offload_threshold = total_params_numel * self.offload_ratio

        cpu_params = []
        gpu_params = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        offloaded_params_numel = 0
        for param in self.params:
            if offloaded_params_numel < offload_threshold:
                assert param.is_cuda
                param_cpu_copy = param.detach().cpu().pin_memory()
                param_cpu_copy.requires_grad = True
                gpu_params_map_cpu_copy[param] = param_cpu_copy
                cpu_copys_map_gpu_param[param_cpu_copy] = param
                cpu_params.append(param_cpu_copy)
            else:
                gpu_params.append(param)

            offloaded_params_numel += param.numel()

        return cpu_params, gpu_params, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param

