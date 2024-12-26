import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, TypeAlias, Union

import torch

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class HybridDeviceOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        offload_fraction=0.5,
        cpu_optimizer_cls=None,
        gpu_optimizer_cls=None,
        pin_cpu_grads: bool = True,
        **kwargs
    ):
        super(HybridDeviceOptimizer, self).__init__(
            params,
            defaults={
                "cpu_optimizer_cls": cpu_optimizer_cls,
                "gpu_optimizer_cls": gpu_optimizer_cls,
                "offload_fraction": offload_fraction,
                "pin_cpu_grads": pin_cpu_grads,
                **kwargs,
            },
        )
        self.pin_cpu_grads = pin_cpu_grads
        self.sub_optimizer_kwargs = kwargs

        self._init_sub_optimizers(params)

        self._sync_sub_optimizers_param_groups_to_hdo()
        self._register_state_dict_hooks()
        self._register_optimizer_step_hooks()

    def register_grad_cpu_copy_hook(self):
        def grad_cpu_copy_hook_closure():
            def grad_cpu_copy_hook(optimizer, args, kwargs):
                if self.cpu_optimizer is None:
                    return

                self._data_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._data_stream):
                    for group in self.cpu_optimizer.param_groups:
                        for param in group["params"]:
                            gpu_param = self.cpu_copys_map_gpu_param[param]
                            if param not in self.cpu_copy_map_grad:
                                self.cpu_copy_map_grad[param] = torch.empty(
                                    gpu_param.grad.shape,
                                    dtype=gpu_param.grad.dtype,
                                    pin_memory=self.pin_cpu_grads,
                                )

                            if hasattr(gpu_param, "grad"):
                                self.cpu_copy_map_grad[param].data.copy_(
                                    gpu_param.grad, non_blocking=True
                                )
                                param.grad = self.cpu_copy_map_grad[param]
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                    self._data_event = self._data_stream.record_event()

            return grad_cpu_copy_hook

        self.register_step_pre_hook(grad_cpu_copy_hook_closure())

    def register_param_copy_back_gpu_hook(self):
        def param_copy_back_gpu_hook_closure():
            def param_copy_back_gpu_hook(optimizer, args, kwargs):
                # self._data_stream.wait_stream(torch.cuda.current_stream())
                # with torch.cuda.stream(self._data_stream):
                for cpu_copy, gpu_param in self.cpu_copys_map_gpu_param.items():
                    gpu_param.data.copy_(cpu_copy.data)
                # self._data_stream.record_event().wait(torch.cuda.current_stream())

            return param_copy_back_gpu_hook

        if self.cpu_optimizer:
            self.cpu_optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())

    def step(self, closure=None):
        self._step_stream.wait_stream(torch.cuda.current_stream())
        if self.gpu_optimizer:
            with torch.cuda.stream(self._step_stream):
                self.gpu_optimizer.step(closure)
        self._step_stream.record_event().wait(torch.cuda.current_stream())
        if self._data_event is not None:
            self._data_event.synchronize()
            self._data_event = None
        if self.cpu_optimizer:
            self.cpu_optimizer.step(closure)
        torch.cuda.synchronize()

    def _init_sub_optimizers(self, params):
        # params = self.params
        # offload_fraction = self.offload_fraction
        # cpu_optimizer_cls = self.cpu_optimizer_cls
        # gpu_optimizer_cls = self.gpu_optimizer_cls
        # kwargs = self.sub_optimizer_kwargs
        offload_fraction = self.defaults["offload_fraction"]
        cpu_optimizer_cls = self.defaults["cpu_optimizer_cls"]
        gpu_optimizer_cls = self.defaults["gpu_optimizer_cls"]
        kwargs = self.sub_optimizer_kwargs

        (
            self.cpu_params,
            self.gpu_params,
            self.gpu_params_map_cpu_copy,
            self.cpu_copys_map_gpu_param,
        ) = self._split_parameters_updated_on_the_cpu_and_gpu(params, offload_fraction)

        self.sub_optimizers = []
        if len(self.cpu_params) > 0:
            self.cpu_optimizer = cpu_optimizer_cls(self.cpu_params, **kwargs)
            self.sub_optimizers.append(self.cpu_optimizer)
        else:
            self.cpu_optimizer = None
        if len(self.gpu_params) > 0:
            self.gpu_optimizer = gpu_optimizer_cls(self.gpu_params, **kwargs)
            self.sub_optimizers.append(self.gpu_optimizer)
        else:
            self.gpu_optimizer = None

        self.cpu_copy_map_grad: Dict[torch.Tensor, torch.Tensor] = defaultdict(torch.Tensor)
        self._data_stream = torch.cuda.Stream()
        self._step_stream = torch.cuda.Stream()
        self._data_event: torch.cuda.Event = None

        self.register_grad_cpu_copy_hook()
        self.register_param_copy_back_gpu_hook()

    def _split_parameters_updated_on_the_cpu_and_gpu(
        self, params: ParamsT, offload_fraction: float
    ):
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
                param_cpu_copy = param.detach().cpu().pin_memory()
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
                group_defaults.pop("_param_sub_optimizer_attrs", None)
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

            return (
                cpu_param_groups,
                gpu_param_groups,
                gpu_params_map_cpu_copy,
                cpu_copys_map_gpu_param,
            )

        return cpu_params, gpu_params, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param

    def _sync_sub_optimizers_param_groups_to_hdo(self):
        """This function updates the latest sub-optimizer changes to
        HybridDeviceOptimizer param_groups.
        """
        param_in_sub_optimizers_index = {}
        for i, optimizer in enumerate(self.sub_optimizers):
            for group_id, group in enumerate(optimizer.param_groups):
                for param in group["params"]:
                    gpu_param = self.cpu_copys_map_gpu_param.get(param, param)
                    param_in_sub_optimizers_index[gpu_param] = (i, group_id)

        # optimizer.param_groups:
        # [
        #    {
        #        'params': [torch.nn.Parameter, ...],
        #        str: Any,
        #    },
        #    ...
        # ]
        new_param_groups = []
        for group in self.param_groups:
            new_group = group.copy()
            sub_optimizer_update_attrs = {}
            for param_id, param in enumerate(new_group["params"]):
                sub_opt_id, group_id = param_in_sub_optimizers_index[param]
                update_group_attrs = self.sub_optimizers[sub_opt_id].param_groups[group_id].copy()
                del update_group_attrs["params"]

                sub_optimizer_update_attrs[param_id] = update_group_attrs
            new_group["_param_sub_optimizer_attrs"] = sub_optimizer_update_attrs
            new_param_groups.append(new_group)
        self.param_groups = new_param_groups

    def _sync_sub_optimizers_state_and_param_groups_to_hdo(self):
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
                gpu_param = self.cpu_copys_map_gpu_param.get(param, param)
                new_state[gpu_param] = optimizer.state[param]
        self.state = new_state
        self._sync_sub_optimizers_param_groups_to_hdo()

    def _sync_hdo_state_to_sub_optimizers(self):
        for optimizer in self.sub_optimizers:
            new_state = defaultdict(dict)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    gpu_param = self.cpu_copys_map_gpu_param.get(param, param)
                    new_state[param] = self.state[gpu_param]
            optimizer.state = new_state

    def _sync_hdo_param_groups_to_sub_optimizers(self):
        """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
        param_in_param_group_index = {}
        for i, group in enumerate(self.param_groups):
            for p_id, param in enumerate(group["params"]):
                param = self.gpu_params_map_cpu_copy.get(param, param)
                param_in_param_group_index[param] = (i, p_id)

        for optimizer in self.sub_optimizers:
            new_param_groups = []
            for group in optimizer.param_groups:
                new_group = group.copy()
                for param in new_group["params"]:
                    group_id, param_id = param_in_param_group_index[param]
                    hdo_group = self.param_groups[group_id]
                    new_group.update(hdo_group["_param_sub_optimizer_attrs"][param_id])

                # After sync-up the sub-optimizer last update, we need to sync-up the
                # HDO new param_groups attributes to the sub-optimizer.
                assert len(group["params"]) > 0, "param_groups should not be empty"
                group_id, _ = param_in_param_group_index[group["params"][0]]
                update_group_attrs = self.param_groups[group_id].copy()
                del update_group_attrs["params"]
                update_group_attrs.pop("_param_sub_optimizer_attrs", None)
                new_group.update(update_group_attrs)

                new_param_groups.append(new_group)
            optimizer.param_groups = new_param_groups

    def _register_state_dict_hooks(self):
        def post_load_state_dict_hook(self):
            # After loading state_dict, the parameters may change, and we need to
            # reinitialize the sub-optimizers to regenerate the new parameters and
            # cpu copy pairs.
            self._init_sub_optimizers(self.param_groups)
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    def _register_optimizer_step_hooks(self):
        def pre_step_hook(self, args, kwargs):
            # Sync param_groups to sub-optimizers before each step to make sure
            # the lr, wd, etc. are up-to-date.
            self._sync_hdo_param_groups_to_sub_optimizers()

        self.register_step_pre_hook(pre_step_hook)

        def post_step_hook(self, args, kwargs):
            # Sync state and param_groups to HDO after each step.
            # NOTE: It is possible for the optimizer to change the properties
            #   in param_groups.
            self._sync_sub_optimizers_state_and_param_groups_to_hdo()

        self.register_step_post_hook(post_step_hook)

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.sub_optimizers:
            optimizer.zero_grad(set_to_none)

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
