# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import typing
import types
from typing import Any, Callable, Optional
from argparse import ArgumentParser, _ArgumentGroup, Namespace
import inspect
import itertools
import builtins
import ast
import enum
from dataclasses import Field, fields
import torch.nn.functional as F
import torch

from megatron.training.config import (
    DistributedInitConfig, 
    PretrainConfigContainer, 
    SchedulerConfig, 
    TrainingConfig, 
    ValidationConfig, 
    RNGConfig, 
    LoggerConfig,
    StragglerDetectionConfig,
    RerunStateMachineConfig, CheckpointConfig, ProfilingConfig
)
# TODO: support arg renames

class TypeInferenceError(Exception):
    """Custom exception type to be conditionally handled by ArgumentGroupFactory."""
    pass

class ArgumentGroupFactory:
    """Utility that adds an argument group to an ArgumentParser based on the attributes of a dataclass.

    This utility uses dataclass metadata including type annotations and docstrings to automatically
        infer the type, default, and other argparse keyword arguments.

    You can override or supplement the automatically inferred argparse kwargs for any 
        dataclass field by providing an "argparse_meta" key in the field's metadata dict.
        The value should be a dict of kwargs that will be passed to ArgumentParser.add_argument().
        These metadata kwargs take precedence over the automatically inferred values.

        Example:
            @dataclass
            class YourConfig:
                your_attribute: int | str | None = field(
                    default=None,
                    metadata={
                        "argparse_meta": {
                            "arg_names": ["--your-arg-name1", "--your-arg-name2"],
                            "type": str,
                            "nargs": "+",
                            "default": "foo",
                        }
                    },
                )

        In this example, inferring the type automatically would fail, as Unions are
        not supported. However the metadata is present, so that takes precedence.
        Any keyword arguments to `ArgumentParser.add_argument()` can be included in
        the "argparse_meta" dict, as well as "arg_names" for the argument flag name.

    This class can also be used as a base class and extended as needed to support dataclasses
        that require some customized or additional handling.

    Args:
        src_cfg_class: The source dataclass type (not instance) whose fields will be 
            converted into command-line arguments. Each field's type annotation determines 
            the argument type, default values become argument defaults, and field-level 
            docstrings are extracted to populate argument help text.
        exclude: Optional list of attribute names from `src_cfg_class` to exclude from 
            argument generation. Useful for omitting internal fields, computed properties,
            or attributes that should be configured through other means. If None, all 
            dataclass fields will be converted to command-line arguments. Default: None.
    """

    def __init__(self, src_cfg_class: type, exclude: Optional[list[str]] = None) -> None:
        self.src_cfg_class = src_cfg_class
        self.field_docstrings = self._get_field_docstrings(src_cfg_class)
        self.exclude = set(exclude) if exclude is not None else set()

    def _format_arg_name(self, config_attr_name: str, prefix: Optional[str] = None) -> str:
        """Convert dataclass name into appropriate argparse flag name.

        Args:
            config_attr_name: dataclass attribute name
            prefix: prefix string to add to the dataclass attribute name. e.g. 'no' for bool 
                settings that are default True. A hyphen is added after the prefix. Default: None
        """
        arg_name = config_attr_name
        if prefix:
            arg_name = prefix + '_' + arg_name
        arg_name = "--" + arg_name.replace("_", "-")
        return arg_name

    def _get_enum_kwargs(self, config_type: enum.EnumMeta) -> dict[str, Any]:
        """Build kwargs for Enums.

        With these settings, the user must provide a valid enum value, e.g.
            'flash', for `AttnBackend.flash`.
        """
        def enum_type_handler(cli_arg):
            return config_type[cli_arg]

        return {"type": enum_type_handler, "choices": list(config_type)}

    def _extract_type(self, config_type: type) -> dict[str, Any]:
        """Determine the type, nargs, and choices settings for this argument.

        Args:
            config_type: attribute type from dataclass
        """
        origin = typing.get_origin(config_type)
        type_tuple = typing.get_args(config_type)

        if isinstance(config_type, type) and issubclass(config_type, enum.Enum):
            return self._get_enum_kwargs(config_type)

        # Primitive type
        if origin is None:
            return {"type": config_type}

        if origin in [types.UnionType, typing.Union]:
            # Handle Optional and Union
            if type_tuple[1] == type(None): # Optional type. First element is value inside Optional[]
                return self._extract_type(type_tuple[0])
            else:
                raise TypeInferenceError(f"Unions not supported by argparse: {config_type}")

        elif origin is list:
            if len(type_tuple) == 1:
                kwargs = self._extract_type(type_tuple[0])
                kwargs["nargs"] = "+"
                return kwargs
            else:
                raise TypeInferenceError(f"Multi-type lists not supported by argparse: {config_type}")

        elif origin is typing.Literal:
            choices_types = [type(choice) for choice in type_tuple]
            assert all([t == choices_types[0] for t in choices_types]), "Type of each choice in a Literal type should all be the same."
            kwargs = {"type": choices_types[0], "choices": type_tuple}
            return kwargs
        else:
            raise TypeInferenceError(f"Unsupported type: {config_type}")


    def _build_argparse_kwargs_from_field(self, attribute: Field) -> dict[str, Any]:
        """Assemble kwargs for add_argument().

        Args:
            attribute: dataclass attribute
        """
        argparse_kwargs = {}
        argparse_kwargs["arg_names"] = [self._format_arg_name(attribute.name)]
        argparse_kwargs["dest"] = attribute.name
        argparse_kwargs["help"] = self.field_docstrings[attribute.name] if attribute.name in self.field_docstrings else ""

        # dataclasses specifies that both should not be set
        if isinstance(attribute.default, type(dataclasses.MISSING)):
            # dataclasses specified default_factory must be a zero-argument callable
            argparse_kwargs["default"] = attribute.default_factory()
        else:
            argparse_kwargs["default"] = attribute.default

        attr_argparse_meta = None
        if attribute.metadata != {} and "argparse_meta" in attribute.metadata:
            # save metadata here, but update at the end so the metadata has highest precedence
            attr_argparse_meta = attribute.metadata["argparse_meta"]


        # if we cannot infer the argparse type, all of this logic may fail. we try to defer
        # to the developer-specified metadata if present
        try:
            argparse_kwargs.update(self._extract_type(attribute.type))

            # use store_true or store_false action for enable/disable flags, which doesn't accept a 'type'
            if argparse_kwargs["type"] == bool:
                argparse_kwargs["action"] = "store_true" if attribute.default == False else "store_false"
                argparse_kwargs.pop("type")

                # add '--no-*' and '--disable-*' prefix if this is a store_false argument
                if argparse_kwargs["action"] == "store_false":
                    argparse_kwargs["arg_names"] = [self._format_arg_name(attribute.name, prefix="no"), self._format_arg_name(attribute.name, prefix="disable")] 
        except TypeInferenceError as e:
            if attr_argparse_meta is not None:
                print(
                    f"WARNING: Inferring the appropriate argparse argument type from {self.src_cfg_class} "
                    f"failed for {attribute.name}: {attribute.type}.\n"
                    "Deferring to attribute metadata. If the metadata is incomplete, 'parser.add_argument()' may fail.\n"
                    f"Original failure: {e}"
                )
            else:
                raise e

        # metadata provided by field takes precedence 
        if attr_argparse_meta is not None:
            argparse_kwargs.update(attr_argparse_meta)

        return argparse_kwargs

    def build_group(self, parser: ArgumentParser, title: Optional[str] = None) -> _ArgumentGroup:
        """Entrypoint method that adds the argument group to the parser.

        Args:
            parser: The parser to add arguments to
            title: Title for the argument group
        """
        arg_group = parser.add_argument_group(title=title, description=self.src_cfg_class.__doc__)
        for attr in fields(self.src_cfg_class):
            if attr.name in self.exclude or attr.init is False:
                continue

            add_arg_kwargs = self._build_argparse_kwargs_from_field(attr)

            arg_names = add_arg_kwargs.pop("arg_names")
            arg_group.add_argument(*arg_names, **add_arg_kwargs)

        return arg_group

    def _get_field_docstrings(self, src_cfg_class: type) -> dict[str, str]:
        """Extract field-level docstrings from a dataclass by inspecting its AST.

        Recurses on parent classes of `src_cfg_class`.

        Args:
            src_cfg_class: Dataclass to get docstrings from.
        """
        source = inspect.getsource(src_cfg_class)
        tree = ast.parse(source)
        root_node = tree.body[0]

        assert isinstance(root_node, ast.ClassDef), "Provided object must be a class."

        field_docstrings = {}

        # Iterate over body of the dataclass using 2-width sliding window.
        # When 'a' is an assignment expression and 'b' is a constant, the window is
        # lined up with an attribute-docstring pair. The pair can be saved to our dict.
        for a, b in itertools.pairwise(root_node.body):
            a_cond = isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name)
            b_cond = isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)

            if a_cond and b_cond:
                # These should be guaranteed by typechecks above, but assert just in case
                assert isinstance(a.target.id, str), "Dataclass attribute not in the expected format. Name is not a string."
                assert isinstance(b.value.value, str), "Dataclass attribute docstring is not a string."

                # Formatting
                docstring = inspect.cleandoc(b.value.value)
                docstring = ' '.join(docstring.split())

                field_docstrings[a.target.id] = docstring

        # recurse on parent class
        base_classes = src_cfg_class.__bases__
        if len(base_classes) > 0:
            parent_class = base_classes[0]
            if parent_class.__name__ not in builtins.__dict__:
                field_docstrings.update(self._get_field_docstrings(base_classes[0]))

        return field_docstrings


def core_transformer_config_from_args(args, config_class=None):
    from megatron.core.activations import squared_relu
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
    from megatron.core.transformer import MLATransformerConfig, TransformerConfig
    from megatron.core.transformer.heterogeneous.heterogeneous_config import (
        HeterogeneousTransformerConfig,
    )
    from megatron.core.quantization.utils import (
        kitchen_quantization_recipe_config,
        load_quantization_recipe,
    )

    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    if args.heterogeneous_layers_config_path is not None:
        assert not args.multi_latent_attention, "Multi latent attention with heterogeneous layers is not supported."
        config_class = HeterogeneousTransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['num_layers_in_first_pipeline_stage']= args.decoder_first_pipeline_num_layers
    kw_args['num_layers_in_last_pipeline_stage']= args.decoder_last_pipeline_num_layers
    kw_args['fp8_param'] = args.fp8_param_gather
    kw_args['fp4_param'] = args.fp4_param_gather
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args['activation_func'] = squared_relu
    elif args.quick_geglu:
        assert not args.swiglu
        kw_args['gated_linear_unit'] = True
        kw_args['activation_func'] = quick_gelu
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir
    if args.rope_type is None:
        # Pop 'rope_type' to let the config class use the default value.
        kw_args.pop('rope_type', None)
    else:
        assert (args.multi_latent_attention or args.rope_type == 'rope'), (
            f'Common attention only support rope_type="rope", but got {args.rope_type}.'
        )

    if len(args.cp_comm_type) == 1:
        kw_args['cp_comm_type'] = args.cp_comm_type[0]
    if args.hybrid_layer_pattern is not None:
        kw_args['is_hybrid_model'] = True
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
        if Symbols.DS_ATTENTION in args.hybrid_layer_pattern:
            kw_args['experimental_attention_variant'] = 'dsa'

    kw_args['inference_sampling_seed'] = args.seed

    # handle quantization config
    # NOTE: Kitchen arguments are only added to the namespace when
    # Kitchen library is available.
    if hasattr(args, "kitchen_config_file") and args.kitchen_config_file is not None:
        kw_args['use_kitchen'] = True
        kw_args['quant_recipe'] = load_quantization_recipe(args.kitchen_config_file)
    elif hasattr(args, 'kitchen_recipe_number') and args.kitchen_recipe_number is not None:
        kw_args['use_kitchen'] = True
        kw_args['quant_recipe'] = kitchen_quantization_recipe_config(args.kitchen_recipe_number)

    kw_args['moe_latent_size'] = args.moe_latent_size

    if args.te_precision_config_file:
        assert not 'quant_recipe' in kw_args, "Quantization recipe already configured."
        # TODO(kwyss): Prohibit fp8_params or fp4_params with this flexibility
        kw_args['quant_recipe'] = load_quantization_recipe(args.te_precision_config_file)

    if hasattr(args, "use_kitchen_attention"):
        kw_args['use_kitchen_attention'] = args.use_kitchen_attention
    if hasattr(args, "kitchen_attention_backend"):
        kw_args['kitchen_attention_backend'] = args.kitchen_attention_backend

    # Return config.
    return config_class(**kw_args)


def _default_config_from_args(cls: type, args: Namespace, return_instance: bool = True) -> Any:
    """Create a config dataclass from the appropriate values in the `args` Namespace.

    This is generic, i.e. it will work if dataclass attribute names map 1-to-1 with
    names in `args`. Some classes might require additional logic.
    """
    kwargs = {}
    for f in fields(cls):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)

    if return_instance:
        return cls(**kwargs)
    else:
        return kwargs

def pretrain_cfg_container_from_args(args: Namespace) -> PretrainConfigContainer:
    """Build a PretrainConfigContainer from the argparse arguments."""
    from megatron.training.training import get_megatron_ddp_config, get_megatron_optimizer_config

    ckpt_kwargs = _default_config_from_args(CheckpointConfig, args, return_instance=False)
    ckpt_kwargs["save_optim"] = not args.no_save_optim
    ckpt_kwargs["save_rng"] = not args.no_save_rng
    ckpt_kwargs["load_optim"] = not args.no_load_optim
    ckpt_kwargs["load_rng"] = not args.no_load_rng
    ckpt_kwargs["fully_parallel_save"] = args.ckpt_fully_parallel_save
    ckpt_kwargs["fully_parallel_load"] = args.ckpt_fully_parallel_load

    prof_kwargs = _default_config_from_args(ProfilingConfig, args, return_instance=False)
    prof_kwargs["use_nsys_profiler"] = args.profile

    rerunsm_kwargs = _default_config_from_args(RerunStateMachineConfig, args, return_instance=False)
    rerunsm_kwargs["check_for_nan_in_loss"] = args.check_for_nan_in_loss_and_grad

    optim_cfg, _ = get_megatron_optimizer_config(args)
    ddp_config = get_megatron_ddp_config(args)

    cfg = PretrainConfigContainer(
        train=_default_config_from_args(TrainingConfig, args),
        validation=_default_config_from_args(ValidationConfig, args),
        optimizer=optim_cfg,
        scheduler=_default_config_from_args(SchedulerConfig, args),
        ddp=ddp_config,
        dist=_default_config_from_args(DistributedInitConfig, args),
        rng=_default_config_from_args(RNGConfig, args),
        logger=_default_config_from_args(LoggerConfig, args),
        checkpoint=CheckpointConfig(**ckpt_kwargs),
        profiling=ProfilingConfig(**prof_kwargs),

        rerun_state_machine=RerunStateMachineConfig(**rerunsm_kwargs),
        straggler=_default_config_from_args(StragglerDetectionConfig, args),
    )

    return cfg
