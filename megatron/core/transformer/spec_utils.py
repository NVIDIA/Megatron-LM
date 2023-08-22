import types
from dataclasses import dataclass, field
from typing import Tuple, Union


@dataclass
class ModuleSpec:
    module_path_or_module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})


def import_module(module_path: Tuple[str]):
    """Import a named object from a module in the context of this function.

    TODO: make this importer module more robust, at least make sure there
    are no side effects of using this as is
    """
    base_path, name = module_path
    try:
        module = __import__(base_path, globals(), locals(), [name])
    except ImportError as e:
        print(f"couldn't import module due to {e}")
        return None
    return vars(module)[name]


def get_module(spec_or_module: Union[ModuleSpec, type], **additional_kwargs):
    # If a module clas is already provided return it as is
    if isinstance(spec_or_module, (type, types.FunctionType)):
        return spec_or_module

    # If the module is provided instead of module path, then return it as is
    if isinstance(spec_or_module.module_path_or_module, (type, types.FunctionType)):
        return spec_or_module.module_path_or_module

    # Otherwise, return the dynamically imported module from the module path
    return import_module(spec_or_module.module_path_or_module)


def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    print(spec_or_module)
    # If the module provided is a `Function` or if the module path provided is
    # a `Function`, written is as it is
    if (
        isinstance(spec_or_module, types.FunctionType)
        or hasattr(spec_or_module, "module_path_or_module")
        and isinstance(spec_or_module.module_path_or_module, types.FunctionType)
    ):
        return spec_or_module

    # Check if a module class is provided as a spec or if the module path
    # itself is a class
    if isinstance(spec_or_module, type):
        module = spec_or_module
    elif hasattr(spec_or_module, "module_path_or_module") and isinstance(
        spec_or_module.module_path_or_module, type
    ):
        module = spec_or_module.module_path_or_module
    else:
        # Otherwise, dynamically import the module from the module path
        module = import_module(spec_or_module.module_path_or_module)

    # Finally return the initialized module with params from the spec as well
    # as those passed as **kwargs from the code
    return module(
        *args, **spec_or_module.params if hasattr(spec_or_module, "params") else {}, **kwargs
    )
