import torch
from typing import Any, Dict, Iterable, Union, TypeAlias


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params: ParamsT, offload_fraction=0.5, cpu_optimizer_cls=None, gpu_optimizer_cls=None, **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(params, defaults={
            "cpu_optimizer_cls": cpu_optimizer_cls,
            "gpu_optimizer_cls": gpu_optimizer_cls,
            "offload_fraction": offload_fraction,
            **kwargs,
        })

        (
            self.cpu_params,
            self.gpu_params,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
        ) = self._split_parameters_updated_on_the_cpu_and_gpu(params, offload_fraction)

        if len(self.cpu_params) > 0:
            self.cpu_optimizer = cpu_optimizer_cls(self.cpu_params, **kwargs)
        else:
            self.cpu_optimizer = None
        if len(self.gpu_params) > 0:
            self.gpu_optimizer = gpu_optimizer_cls(self.gpu_params, **kwargs)
        else:
            self.gpu_optimizer = None

        self.register_grad_cpu_copy_hook()
        self.register_param_copy_back_gpu_hook()

    def register_grad_cpu_copy_hook(self):
        def grad_cpu_copy_hook_closure():
            def grad_cpu_copy_hook(optimizer, args, kwargs):
                for group in self.cpu_optimizer.param_groups:
                    for param in group["params"]:
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        if hasattr(gpu_param, "grad"):
                            param.grad = gpu_param.grad.cpu()
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

            return grad_cpu_copy_hook

        self.register_step_pre_hook(grad_cpu_copy_hook_closure())

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                for cpu_copy, gpu_param in self.cpu_copys_map_gpu_param.items():
                    gpu_param.data.copy_(cpu_copy.data)

            return param_copy_back_gpu_hook

        if self.cpu_optimizer:
            self.cpu_optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())

    def state_dict(self):
        if self.cpu_optimizer:
            cpu_state_dict = self.cpu_optimizer.state_dict()
        else:
            cpu_state_dict = {"state": {}, "param_groups": []}
        if self.gpu_optimizer:
            gpu_state_dict = self.gpu_optimizer.state_dict()
        else:
            gpu_state_dict = {"state": {}, "param_groups": []}

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
        if self.cpu_optimizer:
            cpu_state_dict = {
                "state": {k: v for k, v in state_dict["state"].items() if k < len(self.cpu_params)},
                "param_groups": state_dict["param_groups"][: len(self.cpu_params)],
            }
            self.cpu_optimizer.load_state_dict(cpu_state_dict)
        if self.gpu_optimizer:
            gpu_state_dict = {
                "state": {k - len(self.cpu_params): v for k, v in state_dict["state"].items() if k >= len(self.cpu_params)},
                "param_groups": state_dict["param_groups"][len(self.cpu_params) :],
            }
            self.gpu_optimizer.load_state_dict(gpu_state_dict)

    def step(self, closure=None):
        if self.cpu_optimizer:
            self.cpu_optimizer.step(closure)
        if self.gpu_optimizer:
            self.gpu_optimizer.step(closure)

    def _split_parameters_updated_on_the_cpu_and_gpu(self, params: ParamsT, offload_fraction: float):
        if len(params) == 0:
            return [], [], {}, {}
        
        if not isinstance(params[0], torch.Tensor):
            param_groups = params
            params = []
            for group in param_groups:
                params.extend(group["params"])
        else:
            param_groups = None

        total_params_numel = sum([param.numel() for param in params])
        offload_threshold = total_params_numel * offload_fraction

        cpu_params = []
        gpu_params = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        offloaded_params_numel = 0
        for param in params:
            if offloaded_params_numel < offload_threshold:
                assert param.is_cuda
                param_cpu_copy = param.detach().cpu()
                param_cpu_copy.requires_grad = True
                gpu_params_map_cpu_copy[param] = param_cpu_copy
                cpu_copys_map_gpu_param[param_cpu_copy] = param
                cpu_params.append(param_cpu_copy)
            else:
                gpu_params.append(param)

            offloaded_params_numel += param.numel()

        if param_groups:
            cpu_param_groups = []
            gpu_param_groups = []
            for group in param_groups:
                group_defaults = group.copy()
                del group_defaults["params"]
                _cpu_params = []
                _gpu_params = []
                for param in group["params"]:
                    if param in gpu_params_map_cpu_copy:
                        _cpu_params.append(gpu_params_map_cpu_copy[param])
                    else:
                        _gpu_params.append(param)
                if len(_cpu_params) > 0:
                    cpu_param_groups.append({"params": _cpu_params, **group_defaults})
                if len(_gpu_params) > 0:
                    gpu_param_groups.append({"params": _gpu_params, **group_defaults})

            return cpu_param_groups, gpu_param_groups, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param

        return cpu_params, gpu_params, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param
