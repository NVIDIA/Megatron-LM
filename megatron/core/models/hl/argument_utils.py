import dataclasses
import inspect
from argparse import ArgumentParser
from typing import Any, Literal, Type, get_args, get_origin


def make_args_container(**dataclses) -> Type:
    """
    Example:
    When calling
    ```python
    make_args_container(
        hl_model_config=HLModelConfig,
        common_layer_config=CommonLayerConfig,
        extra_args=ExtraArgs,
    )
    ```
    , the returned class is equivalent to the following:
    ```python
    from dataclasses import dataclass, field

    @dataclass
    class ArgsContainer:
        hl_model_config: HLModelConfig = field(default_factory=HLModelConfig)
        common_layer_config: CommonLayerConfig = field(default_factory=CommonLayerConfig)
        extra_args: ExtraArgs = field(default_factory=ExtraArgs)
    ```
    """
    fields = []
    for name, datacls in dataclses.items():
        if not dataclasses.is_dataclass(datacls):
            raise TypeError(
                f"can only create argument container from dataclasses, but `{datacls}` is not a "
                f"dataclass"
            )

        # Use class instead of instance.
        if not inspect.isclass(datacls):
            datacls = type(datacls)

        field = (name, datacls, dataclasses.field(default_factory=datacls))
        fields.append(field)

    return dataclasses.make_dataclass('ArgsContainer', fields)


# If we use `tyro`, we don't need the functions below.


def add_arguments(parser: ArgumentParser, datacls: Any, group_title: str | None = None) -> None:
    try:
        fields = dataclasses.fields(datacls)
    except TypeError:
        raise TypeError("need to supply a dataclass for argument provisioning")

    if group_title is None:
        group_title = datacls.__name__ if inspect.isclass(datacls) else type(datacls).__name__
    parser.add_argument_group(title=group_title)

    for field in fields:
        _add_argument(parser, field)


def _add_argument(parser: ArgumentParser, field: dataclasses.Field) -> None:
    arg_kwargs = {}
    arg_name = f"--{field.name}"

    if field.type is bool:
        if field.default is True:
            arg_name = f"--no-{field.name}"
        elif field.default is False:
            arg_name = f"--is-{field.name}"
    elif field.type is str:
        pass
    # Literal type
    elif get_origin(field.type) is Literal:
        assert hasattr(field.type, "__args__")
        arg_kwargs[choices]
        choices = get_args(field.type)
    else:
        raise TypeError(f"cannot convert field '{field.name}' with type `{field.type}` to argument")

    parser.add_argument(arg_name, **arg_kwargs)
