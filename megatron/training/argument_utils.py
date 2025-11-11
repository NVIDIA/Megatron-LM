# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import typing
from typing import Any, Optional
from argparse import ArgumentParser
import inspect
import itertools
import builtins
import ast
from dataclasses import Field, fields

# TODO: support arg renames, bool name invert
# TODO: if metadata handles types, ignore exceptions from _extract_type()

class ArgumentGroupFactory:
    """Utility that adds an argument group to an ArgumentParser based on the attributes of a dataclass.

    This class can be overriden as needed to support dataclasses
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

    def _format_arg_name(self, config_attr_name: str) -> str:
        """Convert dataclass name into appropriate argparse flag name.

        Args:
            config_attr_name: dataclass attribute name
        """
        arg_name = "--" + config_attr_name.replace("_", "-")
        return arg_name

    def _extract_type(self, config_type: type) -> dict[str, Any]:
        """Determine the type, nargs, and choices settings for this argument.

        Args:
            config_type: attribute type from dataclass
        """
        origin = typing.get_origin(config_type)
        type_tuple = typing.get_args(config_type)

        # Primitive type
        if origin is None:
            return {"type": config_type}

        if origin is typing.Union:
            # Handle Optional and Union
            if type_tuple[1] == type(None): # Optional type. First element is value inside Optional[]
                return self._extract_type(type_tuple[0])
            else:
                raise TypeError(f"Unions not supported by argparse: {config_type}")

        elif origin is list:
            if len(type_tuple) == 1:
                kwargs = self._extract_type(type_tuple[0])
                kwargs["nargs"] = "+"
                return kwargs
            else:
                raise TypeError(f"Multi-type lists not supported by argparse: {config_type}")

        elif origin is typing.Literal:
            choices_types = [type(choice) for choice in type_tuple]
            assert all([t == choices_types[0] for t in choices_types]), "Type of each choice in a Literal type should all be the same."
            kwargs = {"type": choices_types[0], "choices": type_tuple}
            return kwargs
        else:
            raise TypeError(f"Unsupported type: {config_type}")


    def _build_argparse_kwargs_from_field(self, attribute: Field) -> dict[str, Any]:
        """Assemble kwargs for add_argument().

        Args:
            attribute: dataclass attribute
        """
        argparse_kwargs = {}
        argparse_kwargs["arg_name"] = self._format_arg_name(attribute.name)
        argparse_kwargs["dest"] = attribute.name
        argparse_kwargs["default"] = attribute.default
        argparse_kwargs["help"] = self.field_docstrings[attribute.name]

        argparse_kwargs.update(self._extract_type(attribute.type))

        # use store_true or store_false action for enable/disable flags, which doesn't accept a 'type'
        if argparse_kwargs["type"] == bool:
            argparse_kwargs["action"] = "store_true" if attribute.default == False else "store_false"
            argparse_kwargs.pop("type")

        # metadata provided by field takes precedence 
        if attribute.metadata != {} and "argparse_meta" in attribute.metadata:
            argparse_kwargs.update(attribute.metadata["argparse_meta"])

        return argparse_kwargs

    def build_group(self, parser: ArgumentParser, title: Optional[str] = None) -> ArgumentParser:
        """Entrypoint method that adds the argument group to the parser.

        Args:
            parser: The parser to add arguments to
            title: Title for the argument group
        """
        arg_group = parser.add_argument_group(title=title, description=self.src_cfg_class.__doc__)
        for attr in fields(self.src_cfg_class):
            if attr.name in self.exclude:
                continue

            add_arg_kwargs = self._build_argparse_kwargs_from_field(attr)

            arg_name = add_arg_kwargs.pop("arg_name")
            arg_group.add_argument(arg_name, **add_arg_kwargs)

        return parser

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
                field_docstrings.update(get_field_docstrings(base_classes[0]))

        return field_docstrings
