# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI. All rights reserved.
from collections import defaultdict
from typing import Dict

import torch


def _param_generator(cpu_optimizer):
    for group in cpu_optimizer.param_groups:
        for param in group["params"]:
            yield param


class HybridDeviceOptimizer(torch.optim.Optimizer):
    """
    HybridDeviceOptimizer is a custom optimizer designed to facilitate
    hybrid parameter updates across GPU and CPU. This optimizer allows
    users to adjust the fraction of parameters updated on the CPU and
    GPU through the `offload_fraction` parameter.

    It supports bf16 mixed-precision training. Additionally, the optimizer
    implements overlapping operations for improved performance, including
    gradient transfer from device to host (D2H) and parameter transfer
    from host to device (H2D).

    Example:
        from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
        from torch.optim import AdamW as CPUAdam
        optimizer = HybridDeviceOptimizer(
            param_groups,
            cpu_optimizer_cls=CPUAdam,
            gpu_optimizer_cls=GPUAdam,
            offload_fraction=0.5,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=True,
        )
        optimizer.step()

    Note:
        This optimizer is particularly useful in scenarios where memory
        constraints are present or when leveraging both CPU and GPU resources
        can lead to performance improvements.
    """

    def __init__(
        self,
        params,
        offload_fraction=0.5,
        cpu_optimizer_cls=None,
        gpu_optimizer_cls=None,
        param_update_in_fp32: bool = False,
        pin_cpu_grads: bool = True,
        pin_cpu_params: bool = True,
        overlap_cpu_optimizer_d2h_h2d: bool = True,
        **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(
            params,
            defaults={
                "offload_fraction": offload_fraction,
                "cpu_optimizer_cls": cpu_optimizer_cls,
                "gpu_optimizer_cls": gpu_optimizer_cls,
                "param_update_in_fp32": param_update_in_fp32,
                "pin_cpu_grads": pin_cpu_grads,
                "pin_cpu_params": pin_cpu_params,
                "overlap_cpu_optimizer_d2h_h2d": overlap_cpu_optimizer_d2h_h2d,
                **kwargs,
            },
        )

        self.offload_fraction = offload_fraction
        self.cpu_optimizer_cls = cpu_optimizer_cls
        self.gpu_optimizer_cls = gpu_optimizer_cls
        self.pin_cpu_grads = pin_cpu_grads
        self.pin_cpu_params = pin_cpu_params
        self.overlap_cpu_optimizer_d2h_h2d = overlap_cpu_optimizer_d2h_h2d
        self.param_update_in_fp32 = param_update_in_fp32
        self.sub_optimizer_kwargs = kwargs

        self._init_sub_optimizers()
        self._register_load_state_dict_hooks()

    def _set_sub_optimizer_grads(self):
        if self.param_update_in_fp32:
            for param in self.param_to_fp32_param:
                if param in self.gpu_params_map_cpu_copy:
                    # Skip if the param is offloaded to CPU, it should be handled
                    # in the following part.
                    continue
                fp32_param = self.param_to_fp32_param[param]
                grad = getattr(param, "decoupled_grad", param.grad)
                if grad is not None:
                    fp32_param.grad = grad.to(fp32_param.dtype)
                    fp32_param.requires_grad = True
                else:
                    fp32_param.requires_grad = False

        # Sync the grads from GPU to CPU.
        for optimizer in self.cpu_optimizers:
            for param in _param_generator(optimizer):
                gpu_param = self.cpu_copys_map_gpu_param[param]
                grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad)
                if grad is None:
                    param.requires_grad = False
                    continue

                param.requires_grad = False
                if param not in self.cpu_copy_map_grad:
                    self.cpu_copy_map_grad[param] = torch.empty(
                        param.shape, dtype=param.dtype, pin_memory=self.pin_cpu_grads, device="cpu"
                    )
                    param.grad = self.cpu_copy_map_grad[param]

                self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
            self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()

    def _register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._h2d_stream):
                    for param in _param_generator(optimizer):
                        gpu_param = self.cpu_copys_map_gpu_param[param]
                        gpu_param.data.copy_(param.data, non_blocking=True)
                self._d2h_stream.record_event().wait(torch.cuda.current_stream())

            return param_copy_back_gpu_hook

        def fp32_param_copy_back_gpu_hook_closure():
            def fp32_param_copy_back_gpu_hook(optimizer, args, kwargs):
                for group in self.param_groups:
                    for param in group["params"]:
                        if param in self.gpu_params_map_cpu_copy:
                            # Skip if the param is offloaded to GPU, it has been
                            # copied back in the previous hook.
                            continue

                        if param in self.param_to_fp32_param:
                            fp32_param = self.param_to_fp32_param[param]
                            param.data.copy_(fp32_param.data)

            return fp32_param_copy_back_gpu_hook

        for optimizer in self.sub_optimizers:
            if optimizer is not self.gpu_optimizer:
                optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
            elif self.param_update_in_fp32:
                optimizer.register_step_post_hook(fp32_param_copy_back_gpu_hook_closure())

    def step(self, closure=None):
        """
        Override the step method to perform the following operations:
            1. Sync the HDO param_groups to sub-optimizers.
            2. Sync the grads from GPU to CPU.
            3. Step the sub-optimizers.
            4. Sync the sub-optimizers state to HDO.
        """
        # Sync param_groups to sub-optimizers before each step to make sure
        # the lr, wd, etc. are up-to-date.
        self._sync_hdo_param_groups_to_sub_optimizers()

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()

        # Step the sub-optimizers.
        if self.gpu_optimizer:
            self.gpu_optimizer.step(closure)

        for cpu_optimizer in self.cpu_optimizers:
            d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step(closure)

        # Sync state and param_groups to HDO after each step.
        # NOTE: It is possible for the optimizer to change the properties
        #   in param_groups.
        self._sync_sub_optimizers_state_to_hdo()

    def _init_sub_optimizers(self):
        (
            self.cpu_param_groups,
            self.gpu_param_groups,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
            self.param_to_fp32_param,
        ) = self._get_sub_optimizer_param_groups(self.offload_fraction)
        self.param_to_inner_param = {}
        self.inner_param_to_orig_param = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param in self.param_to_fp32_param:
                    inner_param = self.param_to_fp32_param[param]
                elif param in self.gpu_params_map_cpu_copy:
                    inner_param = self.gpu_params_map_cpu_copy[param]
                else:
                    inner_param = param
                self.param_to_inner_param[param] = inner_param
                self.inner_param_to_orig_param[inner_param] = param
        self.fp32_param_to_orig_param = {v: k for k, v in self.param_to_fp32_param.items()}

        self.cpu_optimizers = []
        if self.overlap_cpu_optimizer_d2h_h2d:
            self.cpu_optimizers = self.build_cpu_optimizer_list(
                self.cpu_optimizer_cls, self.cpu_param_groups
            )
        elif len(self.cpu_param_groups) > 0:
            self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]

        if len(self.gpu_param_groups) > 0:
            self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._d2h_stream = torch.cuda.current_stream()
        self._h2d_stream = torch.cuda.current_stream()
        if self.overlap_cpu_optimizer_d2h_h2d:
            self._d2h_stream = torch.cuda.Stream()
            self._h2d_stream = torch.cuda.Stream()
        self._cpu_optimizer_map_data_event = dict()

        self._register_param_copy_back_gpu_hook()

    @staticmethod
    def build_cpu_optimizer_list(cpu_optimizer_cls, cpu_param_groups):
        """Build several cpu optimizers to enable overlap. Currently we naively
        assign each parameter to an individual optimizer.

        Args:
            cpu_optimizer_cls (Type[torch.optim.Optimizer]): A torch optimizer class
            cpu_param_groups (List[Dict[str, Any]]): The CPU parameter groups
        """
        cpu_optimizers = []

        if len(cpu_param_groups) == 0:
            return cpu_optimizers

        for group in cpu_param_groups:
            group_defaults = group.copy()
            params = group_defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            for param in params:
                _cpu_param_group = group_defaults.copy()
                _cpu_param_group["params"] = [param]
                cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
        return cpu_optimizers

    def _get_sub_optimizer_param_groups(self, offload_fraction: float):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        params_total_numel = sum([param.numel() for param in params])
        gpu_params_total_numel = sum([param.numel() for param in params if param.is_cuda])
        cpu_params_total_numel = params_total_numel - gpu_params_total_numel
        offload_threshold = gpu_params_total_numel * offload_fraction
        offload_params_numel = 0
        cpu_param_groups = []
        gpu_param_groups = []
        gpu_params_map_cpu_copy = {}
        cpu_copys_map_gpu_param = {}
        param_to_fp32_param = {}
        for group in self.param_groups:
            gpu_group = group.copy()
            cpu_group = group.copy()
            gpu_group["params"] = []
            cpu_group["params"] = []
            for param in group["params"]:
                orig_param = param
                cpu_copy = False
                if offload_params_numel < offload_threshold and param.is_cuda:
                    param = param.detach().clone().cpu().pin_memory()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    param_to_fp32_param[orig_param] = param

                if cpu_copy:
                    gpu_params_map_cpu_copy[orig_param] = param
                    cpu_copys_map_gpu_param[param] = orig_param

                if param.is_cuda:
                    gpu_group["params"].append(param)
                else:
                    cpu_group["params"].append(param)
            if len(gpu_group["params"]) != 0:
                gpu_param_groups.append(gpu_group)
            if len(cpu_group["params"]) != 0:
                cpu_param_groups.append(cpu_group)

        return (
            cpu_param_groups,
            gpu_param_groups,
            gpu_params_map_cpu_copy,
            cpu_copys_map_gpu_param,
            param_to_fp32_param,
        )

    def _sync_sub_optimizers_state_to_hdo(self):
        """
        Update HDO state attribute to sub-optimizers.
        """

        # optimizer.state:
        # {
        #    torch.nn.Parameter: {
        #        str: Any,
        #    },
        #    ...
        # }
        new_state = defaultdict(dict)
        for optimizer in self.sub_optimizers:
            for param in optimizer.state:
                orig_param = self.inner_param_to_orig_param[param]
                new_state[orig_param] = optimizer.state[param]
                if self.param_update_in_fp32:
                    new_state[orig_param]["master_param"] = param
        self.state = new_state

    def _sync_hdo_state_to_sub_optimizers(self):
        for optimizer in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    orig_param = self.inner_param_to_orig_param[param]
                    new_state[param] = self.state[orig_param]
            optimizer.state = new_state
        self._update_fp32_params_by_new_state()
        self._move_new_state_to_right_device()

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
        param_in_param_group_index = {}
        for i, group in enumerate(self.param_groups):
            for p_id, param in enumerate(group["params"]):
                inner_param = self.param_to_inner_param[param]
                param_in_param_group_index[inner_param] = (i, p_id)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                # After sync-up the sub-optimizer last update, we need to sync-up the
                # HDO new param_groups attributes to the sub-optimizer.
                assert len(group["params"]) > 0, "param_groups should not be empty"
                group_id, _ = param_in_param_group_index[group["params"][0]]
                update_group_attrs = self.param_groups[group_id].copy()
                del update_group_attrs["params"]
                new_group.update(update_group_attrs)

                new_param_groups.append(new_group)
            optimizer.param_groups = new_param_groups

    def _move_new_state_to_right_device(self):
        for optimizer in self.sub_optimizers:
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    orig_param = self.inner_param_to_orig_param.get(param, param)
                    if isinstance(optimizer, self.defaults["cpu_optimizer_cls"]):
                        self.state[orig_param][k] = state[k] = v.to("cpu")
                    else:
                        self.state[orig_param][k] = state[k] = v.to("cuda")

    def _update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param[param]
            fp32_param.data.copy_(v["master_param"])

    def update_fp32_param_by_new_param(self):
        """
        Update the fp32 parameters by the new parameters.
        """
        for param, fp32_param in self.param_to_fp32_param.items():
            fp32_param.data.copy_(param)

    def _register_load_state_dict_hooks(self):
        def pre_load_state_dict_hook(self, state_dict):
            """
            Pre-load state dictionary hook to prevent loss of precision in
            mixed-precision training.

            When loading a state dictionary with `torch.load_state_dict`,
            optimizer states are reset and cast from `float32` to `bfloat16`/`float16`,
            potentially losing precision. This hook replaces parameters with
            their `float32` copies to mitigate this issue.

            Args:
                state_dict (dict): The state dictionary to be loaded.

            Returns:
                dict: The modified state dictionary with `float32` parameters.
            """
            if not self.param_update_in_fp32:
                return state_dict

            new_state = {}
            for param, v in self.state.items():
                param = self.param_to_fp32_param.get(param, param)
                new_state[param] = v
            self.state = new_state

            for group in self.param_groups:
                for i, param in enumerate(group["params"]):
                    group["params"][i] = self.param_to_fp32_param.get(param, param)

            return state_dict

        self.register_load_state_dict_pre_hook(pre_load_state_dict_hook)

        def post_load_state_dict_hook(self):
            # 1. Replace the temporarily replaced fp32 parameters back. Please
            # refer to the documentation in `pre_load_state_dict_hook`.
            if self.param_update_in_fp32:
                new_state = {}
                for param, v in self.state.items():
                    orig_param = self.fp32_param_to_orig_param.get(param, param)
                    new_state[orig_param] = v
                self.state = new_state

                for group in self.param_groups:
                    for i, param in enumerate(group["params"]):
                        group["params"][i] = self.fp32_param_to_orig_param.get(param, param)

            # 2. After loading state_dict, the parameters may change, and we need to
            # reinitialize the sub-optimizers to regenerate the new parameters and
            # cpu copy pairs.
            self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero or zero to none the gradients of all the parameters in the model.
        """
        super(HybridDeviceOptimizer, self).zero_grad(set_to_none)
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, "decoupled_grad"):
                    if set_to_none:
                        param.decoupled_grad = None
                    else:
                        param.decoupled_grad.zero_()

    def dummy_step(self):
        """
        The dummy step can be used to initialize the potential optimizer.state,
        which can solve the problem of checkpoint loading for an inplace operation
        such as loading a torch distributed checkpoint, for example.
        """
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param)
        self.step()
        self.zero_grad()

    @property
    def sub_optimizers(self):
        """
        Return the list of sub-optimizers.
        """
        if self.gpu_optimizer is not None:
            return self.cpu_optimizers + [self.gpu_optimizer]
        return self.cpu_optimizers
