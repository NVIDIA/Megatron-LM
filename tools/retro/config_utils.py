# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Config utils."""

import argparse
from collections import namedtuple, OrderedDict
import dataclasses
import enum
import inspect
import os
import re
import types
import typing as T


PARAM_KEYWORDS = {
    "param",
    "parameter",
    "arg",
    "argument",
    "attribute",
    "key",
    "keyword",
}
RAISES_KEYWORDS = {"raises", "raise", "except", "exception"}
DEPRECATION_KEYWORDS = {"deprecation", "deprecated"}
RETURNS_KEYWORDS = {"return", "returns"}
YIELDS_KEYWORDS = {"yield", "yields"}
EXAMPLES_KEYWORDS = {"example", "examples"}


class ParseError(RuntimeError):
    """Base class for all parsing related errors."""


class DocstringStyle(enum.Enum):
    """Docstring style."""

    REST = 1
    GOOGLE = 2
    NUMPYDOC = 3
    EPYDOC = 4
    AUTO = 255


class RenderingStyle(enum.Enum):
    """Rendering style when unparsing parsed docstrings."""

    COMPACT = 1
    CLEAN = 2
    EXPANDED = 3


class DocstringMeta:
    """Docstring meta information.

    Symbolizes lines in form of

        :param arg: description
        :raises ValueError: if something happens
    """

    def __init__(
        self, args: T.List[str], description: T.Optional[str]
    ) -> None:
        """Initialize self.

        :param args: list of arguments. The exact content of this variable is
            dependent on the kind of docstring; it's used to distinguish
            between custom docstring meta information items.
        :param description: associated docstring description.
        """
        self.args = args
        self.description = description


class DocstringParam(DocstringMeta):
    """DocstringMeta symbolizing :param metadata."""

    def __init__(
        self,
        args: T.List[str],
        description: T.Optional[str],
        arg_name: str,
        type_name: T.Optional[str],
        is_optional: T.Optional[bool],
        default: T.Optional[str],
    ) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.arg_name = arg_name
        self.type_name = type_name
        self.is_optional = is_optional
        self.default = default


class DocstringReturns(DocstringMeta):
    """DocstringMeta symbolizing :returns or :yields metadata."""

    def __init__(
        self,
        args: T.List[str],
        description: T.Optional[str],
        type_name: T.Optional[str],
        is_generator: bool,
        return_name: T.Optional[str] = None,
    ) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.type_name = type_name
        self.is_generator = is_generator
        self.return_name = return_name


class DocstringRaises(DocstringMeta):
    """DocstringMeta symbolizing :raises metadata."""

    def __init__(
        self,
        args: T.List[str],
        description: T.Optional[str],
        type_name: T.Optional[str],
    ) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.type_name = type_name
        self.description = description


class DocstringDeprecated(DocstringMeta):
    """DocstringMeta symbolizing deprecation metadata."""

    def __init__(
        self,
        args: T.List[str],
        description: T.Optional[str],
        version: T.Optional[str],
    ) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.version = version
        self.description = description


class DocstringExample(DocstringMeta):
    """DocstringMeta symbolizing example metadata."""

    def __init__(
        self,
        args: T.List[str],
        snippet: T.Optional[str],
        description: T.Optional[str],
    ) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.snippet = snippet
        self.description = description


class Docstring:
    """Docstring object representation."""

    def __init__(
        self,
        style=None,  # type: T.Optional[DocstringStyle]
    ) -> None:
        """Initialize self."""
        self.short_description = None  # type: T.Optional[str]
        self.long_description = None  # type: T.Optional[str]
        self.blank_after_short_description = False
        self.blank_after_long_description = False
        self.meta = []  # type: T.List[DocstringMeta]
        self.style = style  # type: T.Optional[DocstringStyle]

    @property
    def params(self) -> T.List[DocstringParam]:
        """Return a list of information on function params."""
        return {m.arg_name:m for m in self.meta if isinstance(m, DocstringParam)}

    @property
    def raises(self) -> T.List[DocstringRaises]:
        """Return a list of information on the exceptions that the function
        may raise.
        """
        return [
            item for item in self.meta if isinstance(item, DocstringRaises)
        ]

    @property
    def returns(self) -> T.Optional[DocstringReturns]:
        """Return a single information on function return.

        Takes the first return information.
        """
        for item in self.meta:
            if isinstance(item, DocstringReturns):
                return item
        return None

    @property
    def many_returns(self) -> T.List[DocstringReturns]:
        """Return a list of information on function return."""
        return [
            item for item in self.meta if isinstance(item, DocstringReturns)
        ]

    @property
    def deprecation(self) -> T.Optional[DocstringDeprecated]:
        """Return a single information on function deprecation notes."""
        for item in self.meta:
            if isinstance(item, DocstringDeprecated):
                return item
        return None

    @property
    def examples(self) -> T.List[DocstringExample]:
        """Return a list of information on function examples."""
        return [
            item for item in self.meta if isinstance(item, DocstringExample)
        ]


class SectionType(enum.IntEnum):
    """Types of sections."""

    SINGULAR = 0
    """For sections like examples."""

    MULTIPLE = 1
    """For sections like params."""

    SINGULAR_OR_MULTIPLE = 2
    """For sections like returns or yields."""


class Section(namedtuple("SectionBase", "title key type")):
    """A docstring section."""


GOOGLE_TYPED_ARG_REGEX = re.compile(r"\s*(.+?)\s*\(\s*(.*[^\s]+)\s*\)")
GOOGLE_ARG_DESC_REGEX = re.compile(r".*\. Defaults to (.+)\.")
MULTIPLE_PATTERN = re.compile(r"(\s*[^:\s]+:)|([^:]*\]:.*)")

DEFAULT_SECTIONS = [
    Section("Arguments", "param", SectionType.MULTIPLE),
    Section("Args", "param", SectionType.MULTIPLE),
    Section("Parameters", "param", SectionType.MULTIPLE),
    Section("Params", "param", SectionType.MULTIPLE),
    Section("Raises", "raises", SectionType.MULTIPLE),
    Section("Exceptions", "raises", SectionType.MULTIPLE),
    Section("Except", "raises", SectionType.MULTIPLE),
    Section("Attributes", "attribute", SectionType.MULTIPLE),
    Section("Example", "examples", SectionType.SINGULAR),
    Section("Examples", "examples", SectionType.SINGULAR),
    Section("Returns", "returns", SectionType.SINGULAR_OR_MULTIPLE),
    Section("Yields", "yields", SectionType.SINGULAR_OR_MULTIPLE),
]


class GoogleDocstringParser:
    """Parser for Google-style docstrings."""

    def __init__(
        self, sections: T.Optional[T.List[Section]] = None, title_colon=True
    ):
        """Setup sections.

        :param sections: Recognized sections or None to defaults.
        :param title_colon: require colon after section title.
        """
        if not sections:
            sections = DEFAULT_SECTIONS
        self.sections = {s.title: s for s in sections}
        self.title_colon = title_colon
        self._setup()

    def _setup(self):
        if self.title_colon:
            colon = ":"
        else:
            colon = ""
        self.titles_re = re.compile(
            "^("
            + "|".join(f"({t})" for t in self.sections)
            + ")"
            + colon
            + "[ \t\r\f\v]*$",
            flags=re.M,
        )

    def _build_meta(self, text: str, title: str) -> DocstringMeta:
        """Build docstring element.

        :param text: docstring element text
        :param title: title of section containing element
        :return:
        """

        section = self.sections[title]

        if (
            section.type == SectionType.SINGULAR_OR_MULTIPLE
            and not MULTIPLE_PATTERN.match(text)
        ) or section.type == SectionType.SINGULAR:
            return self._build_single_meta(section, text)

        if ":" not in text:
            # raise ParseError(f"Expected a colon in {text!r}.")
            return None

        # Split spec and description
        before, desc = text.split(":", 1)
        if desc:
            desc = desc[1:] if desc[0] == " " else desc
            if "\n" in desc:
                first_line, rest = desc.split("\n", 1)
                desc = first_line + "\n" + inspect.cleandoc(rest)
            desc = desc.strip("\n")

        return self._build_multi_meta(section, before, desc)

    @staticmethod
    def _build_single_meta(section: Section, desc: str) -> DocstringMeta:
        if section.key in RETURNS_KEYWORDS | YIELDS_KEYWORDS:
            return DocstringReturns(
                args=[section.key],
                description=desc,
                type_name=None,
                is_generator=section.key in YIELDS_KEYWORDS,
            )
        if section.key in RAISES_KEYWORDS:
            return DocstringRaises(
                args=[section.key], description=desc, type_name=None
            )
        if section.key in EXAMPLES_KEYWORDS:
            return DocstringExample(
                args=[section.key], snippet=None, description=desc
            )
        if section.key in PARAM_KEYWORDS:
            raise ParseError("Expected paramenter name.")
        return DocstringMeta(args=[section.key], description=desc)

    @staticmethod
    def _build_multi_meta(
        section: Section, before: str, desc: str
    ) -> DocstringMeta:
        if section.key in PARAM_KEYWORDS:
            match = GOOGLE_TYPED_ARG_REGEX.match(before)
            if match:
                arg_name, type_name = match.group(1, 2)
                if type_name.endswith(", optional"):
                    is_optional = True
                    type_name = type_name[:-10]
                elif type_name.endswith("?"):
                    is_optional = True
                    type_name = type_name[:-1]
                else:
                    is_optional = False
            else:
                arg_name, type_name = before, None
                is_optional = None

            match = GOOGLE_ARG_DESC_REGEX.match(desc)
            default = match.group(1) if match else None

            return DocstringParam(
                args=[section.key, before],
                description=desc,
                arg_name=arg_name,
                type_name=type_name,
                is_optional=is_optional,
                default=default,
            )
        if section.key in RETURNS_KEYWORDS | YIELDS_KEYWORDS:
            return DocstringReturns(
                args=[section.key, before],
                description=desc,
                type_name=before,
                is_generator=section.key in YIELDS_KEYWORDS,
            )
        if section.key in RAISES_KEYWORDS:
            return DocstringRaises(
                args=[section.key, before], description=desc, type_name=before
            )
        return DocstringMeta(args=[section.key, before], description=desc)

    def add_section(self, section: Section):
        """Add or replace a section.

        :param section: The new section.
        """

        self.sections[section.title] = section
        self._setup()

    def parse(self, text: str) -> Docstring:
        """Parse the Google-style docstring into its components.

        :returns: parsed docstring
        """
        ret = Docstring(style=DocstringStyle.GOOGLE)
        if not text:
            return ret

        # Clean according to PEP-0257
        text = inspect.cleandoc(text)

        # Find first title and split on its position
        match = self.titles_re.search(text)
        if match:
            desc_chunk = text[: match.start()]
            meta_chunk = text[match.start() :]
        else:
            desc_chunk = text
            meta_chunk = ""

        # Break description into short and long parts
        parts = desc_chunk.split("\n", 1)
        ret.short_description = parts[0] or None
        if len(parts) > 1:
            long_desc_chunk = parts[1] or ""
            ret.blank_after_short_description = long_desc_chunk.startswith(
                "\n"
            )
            ret.blank_after_long_description = long_desc_chunk.endswith("\n\n")
            ret.long_description = long_desc_chunk.strip() or None

        # Split by sections determined by titles
        matches = list(self.titles_re.finditer(meta_chunk))
        if not matches:
            return ret
        splits = []
        for j in range(len(matches) - 1):
            splits.append((matches[j].end(), matches[j + 1].start()))
        splits.append((matches[-1].end(), len(meta_chunk)))

        chunks = OrderedDict()  # type: T.Mapping[str,str]
        for j, (start, end) in enumerate(splits):
            title = matches[j].group(1)
            if title not in self.sections:
                continue

            # Clear Any Unknown Meta
            # Ref: https://github.com/rr-/docstring_parser/issues/29
            meta_details = meta_chunk[start:end]
            unknown_meta = re.search(r"\n\S", meta_details)
            if unknown_meta is not None:
                meta_details = meta_details[: unknown_meta.start()]

            chunks[title] = meta_details.strip("\n")
        if not chunks:
            return ret

        # Add elements from each chunk
        for title, chunk in chunks.items():
            # Determine indent
            indent_match = re.search(r"^\s*", chunk)
            if not indent_match:
                raise ParseError(f'Can\'t infer indent from "{chunk}"')
            indent = indent_match.group()

            # Check for singular elements
            if self.sections[title].type in [
                SectionType.SINGULAR,
                SectionType.SINGULAR_OR_MULTIPLE,
            ]:
                part = inspect.cleandoc(chunk)
                ret.meta.append(self._build_meta(part, title))
                continue

            # Split based on lines which have exactly that indent
            _re = "^" + indent + r"(?=\S)"
            c_matches = list(re.finditer(_re, chunk, flags=re.M))
            if not c_matches:
                raise ParseError(f'No specification for "{title}": "{chunk}"')
            c_splits = []
            for j in range(len(c_matches) - 1):
                c_splits.append((c_matches[j].end(), c_matches[j + 1].start()))
            c_splits.append((c_matches[-1].end(), len(chunk)))
            for j, (start, end) in enumerate(c_splits):
                part = chunk[start:end].strip("\n")
                ret.meta.append(self._build_meta(part, title))

        return ret


def verify_and_get_config_attr_descs(config_cls, strict_docstring_match=True):

    assert dataclasses.is_dataclass(config_cls), f"uh oh <{config_cls.__name__}>."

    # Parse docstring.
    try:
        docstring = GoogleDocstringParser().parse(config_cls.__doc__)
    except Exception as e:
        raise Exception(f"error parsing {config_cls.__name__} docstring.")
    
    # Get attributes and types.
    config_attrs = docstring.params
    config_types = config_cls.__annotations__

    # Verify attribute names.
    config_attr_keys = set(config_attrs.keys())
    config_type_keys = set(config_types.keys())
    missing_attr_keys = config_type_keys - config_attr_keys
    extra_attr_keys = config_attr_keys - config_type_keys
    if strict_docstring_match:
        assert not missing_attr_keys and not extra_attr_keys, f"{config_cls.__name__} docstring is either missing attributes ({', '.join(missing_attr_keys) if missing_attr_keys else '--'}) or contains extra attributes ({', '.join(extra_attr_keys) if extra_attr_keys else '--'})."

    # @todo
    # Verify attribute type names.
    # for key in config_attr_keys:
    #     ... todo ...

    # Verify base class attributes.
    attrs = {k:v for base_cls in config_cls.__bases__ if dataclasses.is_dataclass(base_cls) for k,v in verify_and_get_config_attr_descs(base_cls, strict_docstring_match=strict_docstring_match).items()}
    for key in config_attr_keys:
        if key in config_types:
            attrs[key] = {
                "desc" : config_attrs[key].description,
                "type" : config_types[key],
            }

    return attrs


def add_config_args(parser, config_cls):
    attrs = verify_and_get_config_attr_descs(config_cls, strict_docstring_match=False)
    for key, attr in attrs.items():
        _type = attr["type"]
        if dataclasses.is_dataclass(_type):
            group = parser.add_argument_group(title=attr["desc"])
            add_config_args(group, _type)
        else:

            default_value = getattr(config_cls, key)
            args = {
                "help" : attr["desc"],
                "default" : default_value,
            }

            if _type == bool:
                assert isinstance(args["default"], (bool, type(None))), \
                    f"boolean attribute '{key}' of {config_cls.__name__} " \
                    "has non-boolean default value."

                # When default=True, add 'no-{key}' arg.
                if default_value:
                    args["action"] = "store_false"
                    args["dest"] = key
                    key = "no-" + key
                else:
                    args["action"] = "store_true"

            elif _type in (int, float):
                args["type"] = _type

            elif _type == list:
                args["nargs"] = "*"

            # else: ....... treat as string arg
            #     raise Exception(f"specialize action for '{key}', type <{_type}>.")

            try:
                parser.add_argument(f"--{key.replace('_', '-')}", **args)
            except argparse.ArgumentError as e:
                pass


def get_config_leaf_field_names(config_cls):
    names = set()
    for field in dataclasses.fields(config_cls):
        if dataclasses.is_dataclass(field.type):
            names.update(get_config_leaf_field_names(field.type))
        else:
            names.add(field.name)
    return names


def config_from_args(args, config_cls, add_custom_args=False):

    # Collect config data in a dict.
    data = {}
    for field in dataclasses.fields(config_cls):
        if dataclasses.is_dataclass(field.type):
            data[field.name] = config_from_args(args, field.type)
        else:
            data[field.name] = getattr(args, field.name)

    # Add custom args. (e.g., for tools, tasks)
    if add_custom_args:

        config_keys = get_config_leaf_field_names(config_cls)
        arg_keys = set(vars(args).keys())
        custom_keys = arg_keys - config_keys

        custom_data = {k:v for k, v in vars(args).items() if k in custom_keys}
        custom_config_cls = dataclasses.make_dataclass(
            "CustomConfig",
            [(k, type(v)) for k, v in custom_data.items()])
        custom_config = custom_config_cls(**custom_data)
        data["custom"] = custom_config

    # Create config. [ todo: programmatically create dataclass that inherits
    # TransformerConfig. ]
    config = config_cls(**data)

    return config


def flatten_config(config, base_config_cls=None):

    # Lift sub-config data.
    flat_config = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            flat_config = { **flat_config, **flatten_config(value) }
        else:
            flat_config[field.name] = value

    # Convert to dataclass.
    if base_config_cls:
        base_keys = set(field.name for field in dataclasses.fields(base_config_cls))
        flat_config_cls = dataclasses.make_dataclass(
            cls_name="FlatMegatronConfig",
            fields=[(k, T.Any, dataclasses.field(default=None))
                    for k, v in flat_config.items()
                    if k not in base_keys],
            bases=(base_config_cls,))
        flat_config = flat_config_cls(**flat_config)

    return flat_config
