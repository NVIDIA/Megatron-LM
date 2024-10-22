import torch


class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(self, params, offload_ratio=0.5, cpu_optimizer_cls=None, gpu_optimizer_cls=None, **kwargs):
        self.params = params.copy()
        self.offload_ratio = offload_ratio

        self.cpu_update_params, self.gpu_update_params = self._separate_parameters_updated_on_the_cpu_and_gpu()
        self.cpu_optimizer = cpu_optimizer_cls(self.cpu_update_params, **kwargs)
        self.gpu_optimizer = gpu_optimizer_cls(self.gpu_update_params, **kwargs)

    def state_dict(self):
        cpu_state_dict = self.cpu_optimizer.state_dict()
        gpu_state_dict = self.gpu_optimizer.state_dict()
        return {
            "state": cpu_state_dict["state"] + gpu_state_dict["state"],
            "param_groups": cpu_state_dict["param_groups"] + gpu_state_dict["param_groups"],
        }
    
    def load_state_dict(self, state_dict):
        cpu_state_dict = {"state": state_dict["state"][:len(self.cpu_update_params)], "param_groups": state_dict["param_groups"][:len(self.cpu_update_params)]}
        gpu_state_dict = {"state": state_dict["state"][len(self.cpu_update_params):], "param_groups": state_dict["param_groups"][len(self.cpu_update_params):]}
        self.cpu_optimizer.load_state_dict(cpu_state_dict)
        self.gpu_optimizer.load_state_dict(gpu_state_dict)
    
    def param_groups(self):
        return self.cpu_optimizer.param_groups + self.gpu_optimizer.param_groups

    def step(self, closure=None):
        self.cpu_optimizer.step(closure)
        self.gpu_optimizer.step(closure)

    def _separate_parameters_updated_on_the_cpu_and_gpu(self):
        total_params_numel = sum([p.numel() for p in self.params])
        offload_threshold = total_params_numel * self.offload_ratio

        cpu_update_params = []
        gpu_update_params = []
        offloaded_params_numel = 0
        for param in self.params:
            if offloaded_params_numel < offload_threshold:
                cpu_update_params.append(param)
            else:
                gpu_update_params.append(param)

            offloaded_params_numel += param.numel()

        return cpu_update_params, gpu_update_params
