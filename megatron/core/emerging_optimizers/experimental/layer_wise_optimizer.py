import math
from typing import Tuple, Callable, Any

from absl import logging

import torch

from megatron.core import optimizer as mcore_optimizer
from megatron.core.optimizer import clip_grads, Adam as AdamW
from megatron.core.models.gpt import gpt_model as mcore_gpt_model

from megatron.core.emerging_optimizers.soap.soap import SOAP
from megatron.core.emerging_optimizers.orthogonalized_optimizers.muon import Muon


def default_params_group_func(mcore_te_model: mcore_gpt_model.GPTModel) -> Tuple[list[torch.nn.Parameter], ...]:
    """Default parameter group function for LayerWiseDistributedOptimizer.

    It is designed to be used with Megatron-core models with transformer engine backend. Filtering is strongly
    based on layer names.
    """
    # Strip out module inside DDP and other wrappers
    module = mcore_te_model.module if hasattr(mcore_te_model, "module") else mcore_te_model
    while hasattr(module, "module"):
        module = module.module
    assert hasattr(module, "decoder"), "MCore GPTModel should have a decoder"

    linear_params_list = []
    norm_params_list = []
    linear_names = []  # Also collect layer names for debugging purposes
    for name, param in module.decoder.named_parameters():
        if "norm" not in name and name.endswith("weight"):
            linear_params_list.append(param)
            linear_names.append(name.replace(".weight", ""))
        elif "norm" in name:
            norm_params_list.append(param)
        else:
            raise ValueError(f"Unsupported parameter: {name}")
    logging.info(f"{len(linear_params_list)} linear layers collected in total.")
    logging.debug(f"Linear layer names: {linear_names}")

    # TODO(skyw): Revisit whether we should put positional embedding on sperate GPU or together with word embedding
    embed_params_list = [
        [module.embedding.word_embeddings.weight],
        [module.output_layer.weight],
    ]
    if hasattr(module.embedding, "position_embeddings"):
        embed_params_list[-1].append(module.embedding.position_embeddings.weight)
    if hasattr(module.embedding, "token_type_embeddings"):
        embed_params_list[0].append(module.embedding.token_type_embeddings.weight)

    return linear_params_list, norm_params_list, embed_params_list


class LayerWiseDistributedOptimizer(mcore_optimizer.ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Warning:
        This is a experimental optimizer that distributes optimizers of linear and embedding layers to different ranks
        with the option to use different optimizer for different layer types.

    Args:
        module: A megatron core model chunk.
        linear_optimizer: Optimizer class for linear layers.
        linear_optimizer_kwargs: Keyword arguments for linear layer optimizer.
        embed_optimizer: Optimizer class for embedding layers.
        embed_optimizer_kwargs: Keyword arguments for embedding layer optimizer.
        default_optimizer: Optimizer class for other layers (e.g., norm layers).
        default_optimizer_kwargs: Keyword arguments for default optimizer.
        mcore_optimizer_config: Megatron-core optimizer configuration.
        grad_comm_pg: Gradient communication process group. It is not named data parallel group because gradient
            communication also across context parallel and maybe other groups.
        params_group_func: A function that returns a list of parameters iterables.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        linear_optimizer_cls: type[torch.optim.Optimizer],
        linear_optimizer_kwargs: dict,
        embed_optimizer_cls: type[torch.optim.Optimizer],
        embed_optimizer_kwargs: dict,
        default_optimizer_cls: type[torch.optim.Optimizer],
        default_optimizer_kwargs: dict,
        mcore_optimizer_config: mcore_optimizer.OptimizerConfig,
        grad_comm_pg: torch.distributed.ProcessGroup,
        params_group_func: Callable[[torch.nn.Module], Tuple[list[torch.nn.Parameter], ...]] | None = None,
    ) -> None:
        if isinstance(module, list):
            assert len(module) == 1, "LayerWiseDistributedOptimizer only supports one model chunk"
            module = module[0]
        # TODO(skyw): Add consistency check for optimizer kwargs and optimizer config.
        self.grad_comm_pg = grad_comm_pg
        world_size = grad_comm_pg.size()

        if params_group_func is None:
            params_group_func = default_params_group_func
        linear_params_list, norm_params_list, embed_params_list = params_group_func(module)
        logging.info(
            f"Collected {len(linear_params_list)} linear parameters and {len(norm_params_list)} norm parameters "
            f"and {len(embed_params_list)} embedding parameters."
        )

        # List of linear or embedding layers updated on one GPU that needs to be broadcasted to all others
        self.broadcast_params_list = []

        if mcore_optimizer_config.bf16:
            optimizer_wrapper = mcore_optimizer.Float16OptimizerWithFloat16Params
            wrapper_args = [mcore_optimizer_config, None, None]
        elif not mcore_optimizer_config.fp16:
            optimizer_wrapper = mcore_optimizer.FP32Optimizer
            wrapper_args = [mcore_optimizer_config, None]
        else:
            raise ValueError("Unsupported optimizer precision")

        layer_wise_optimizer = None
        num_embed_layers = len(embed_params_list)
        assert (
            world_size > num_embed_layers
        ), f"{self.__class__.__name__} requires at least 3 ranks to distribute embedding, output projection and linear layers"
        for i, params in enumerate(embed_params_list):
            self.broadcast_params_list.append(params)
            if grad_comm_pg.rank() == i:
                layer_wise_optimizer = optimizer_wrapper(
                    embed_optimizer_cls(params, **embed_optimizer_kwargs), *wrapper_args
                )

        # treat linear layers
        num_linear_params_per_rank = math.ceil(len(linear_params_list) / (world_size - num_embed_layers))
        for i, offset in enumerate(range(0, len(linear_params_list), num_linear_params_per_rank)):
            local_linear_params_list = linear_params_list[offset : offset + num_linear_params_per_rank]
            self.broadcast_params_list.append(local_linear_params_list)
            if grad_comm_pg.rank() == i + num_embed_layers:
                layer_wise_optimizer = optimizer_wrapper(
                    linear_optimizer_cls(local_linear_params_list, **linear_optimizer_kwargs),
                    *wrapper_args,
                )

                logging.info(f"Rank {grad_comm_pg.rank()} has {num_linear_params_per_rank} layers")

        # treat norm layers
        norm_optimizer = optimizer_wrapper(
            default_optimizer_cls(norm_params_list, **default_optimizer_kwargs), *wrapper_args
        )

        assert (
            l := len(self.broadcast_params_list)
        ) <= world_size, f"Broadcast params list ({l}) should be smaller than world size ({world_size})"
        assert norm_optimizer, "Norm optimizer should not be empty."

        all_optimizers = [norm_optimizer]
        if layer_wise_optimizer is not None:
            all_optimizers.insert(0, layer_wise_optimizer)
        super().__init__(all_optimizers)

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer

        Note:
            A lot of code from ChainedOptimizer.step() is copied here because there are bug fixes after core 0.12
        """
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, "is_stub_optimizer") and optimizer.is_stub_optimizer:
                continue

            if optimizer.config.clip_grad > 0.0:
                clip_grads.clip_grad_by_total_norm_fp32(
                    optimizer.get_parameters(),
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=optimizer.config.use_precision_aware_optimizer,
                )

        # # Count the zeros in the grads.
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None

        update_successful = self.step_with_ready_grads()

        # Broadcast linear layer weights to all other ranks.
        # This may not be slower than PyTorch allgatherv which calls broadcast internally.
        # TODO(skyw): Profile and implement more efficient version.
        for i, params_list in enumerate(self.broadcast_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.grad_comm_pg, i)
            for param in params_list:
                torch.distributed.broadcast(param, src_global_rank, self.grad_comm_pg)

        return update_successful, grad_norm, num_zeros_in_grad

    def get_grad_norm(self) -> float:
        """Get the gradient norm of the optimizer"""
        if len(self.chained_optimizers) > 1:
            layer_wise_grad_norm = self.chained_optimizers[0].get_grad_norm()
        else:
            layer_wise_grad_norm = 0.0
        out_list = [0 for _ in range(self.grad_comm_pg.size())]

        # TODO(skyw): Determine whether it is too big of an overhead to use all_gather_object.
        torch.distributed.all_gather_object(out_list, layer_wise_grad_norm**2, group=self.grad_comm_pg)

        # Add the gradient norm of the norm layers which is duplicated among all ranks.
        duplicated_grad_norm = self.chained_optimizers[-1].get_grad_norm()
        final_norm = math.sqrt(sum(out_list) + duplicated_grad_norm**2)

        return final_norm

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer.

        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer."""
        super().load_state_dict(torch.load(filename))


def get_shower_optimizer_for_mcore(
    model: torch.nn.Module, config: mcore_optimizer.OptimizerConfig, linear_optimizer: str = "soap", **kwargs: Any
) -> LayerWiseDistributedOptimizer:
    """Get a layer-wise optimizer for a MCore GPTModel.

    Warning:
        Megatron monkey patches a lot of attributes to variety of objects.
        This function tries to encapsulate fragile functionalities that depends on those fragile patches and
        isolate them from the main LayerWiseDistributedOptimizer class.

    Args:
        model: A MCore GPTModel.
        config: A Megatron-core optimizer configuration.
        linear_optimizer: The linear optimizer to use. Must be one of ['soap', 'muon'].
        **kwargs: Additional arguments for LayerWiseOptimizer.

    Returns:
        A LayerWiseOptimizer.
    """
    linear_optimizer_cls: type[torch.optim.Optimizer]
    if linear_optimizer == "soap":
        linear_optimizer_cls = SOAP
        linear_opt_kwargs = dict(
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
        soap_config_keys = [
            "shampoo_beta",
            "eps",
            "use_decoupled_weight_decay",
            "use_nesterov",
            "precondition_frequency",
            "precondition_warmup_steps",
            "adam_warmup_steps",
            "precondition_1d",
            "trace_normalization",
            "normalize_preconditioned_grads",
            "correct_bias",
            "fp32_matmul_prec",
            "use_eigh",
            "qr_fp32_matmul_prec",
            "use_adaptive_criteria",
            "adaptive_update_tolerance",
            "power_iter_steps",
            "max_update_rms",
        ]
        for attr_name in soap_config_keys:
            if hasattr(config, attr_name):
                linear_opt_kwargs[attr_name] = getattr(config, attr_name)
            else:
                logging.warning(f"Config attribute {attr_name} not found in config.")
    elif linear_optimizer == "muon":
        linear_optimizer_cls = Muon
        linear_opt_kwargs = dict(lr=config.lr, weight_decay=config.weight_decay)
        muon_config_keys = [
            "momentum_beta",
            "use_nesterov",
            "num_ns_steps",
            "ns_backend",
            "use_decoupled_weight_decay",
            "scale_mode",
        ]
        for attr_name in muon_config_keys:
            if hasattr(config, attr_name):
                linear_opt_kwargs[attr_name] = getattr(config, attr_name)
            else:
                logging.warning(f"Config attribute {attr_name} not found in config.")
    else:
        raise ValueError(f"Unsupported linear optimizer: {linear_optimizer}, must be one of ['soap', 'muon']")

    # strip out arguments
    embed_adam_kwargs = {
        "lr": config.decoupled_lr if config.decoupled_lr is not None else config.lr,
        "weight_decay": config.weight_decay,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
    }

    norm_adam_kwargs = {
        "lr": config.lr,
        "weight_decay": 0.0,  # No weight decay for norm layers
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
    }

    logging.debug(f"embed_adam_kwargs: {embed_adam_kwargs}")
    logging.debug(f"norm_adam_kwargs: {norm_adam_kwargs}")
    logging.debug(f"linear_opt_kwargs: {linear_opt_kwargs}")

    optimizer = LayerWiseDistributedOptimizer(
        model,
        linear_optimizer_cls,
        linear_opt_kwargs,
        AdamW,
        embed_adam_kwargs,
        AdamW,
        norm_adam_kwargs,
        config,
        **kwargs,
    )

    # Patch necessary keys that MCore depends on.
    for param_group in optimizer.param_groups:
        param_group["min_lr"] = config.min_lr
        param_group["max_lr"] = config.lr
        param_group["is_decoupled_lr"] = False
        if config.decoupled_lr is not None and getattr(
            param_group["params"][0], "is_embedding_or_output_parameter", False
        ):
            param_group["min_lr"] = config.decoupled_min_lr
            param_group["max_lr"] = config.decoupled_lr
            param_group["is_decoupled_lr"] = True

    return optimizer
