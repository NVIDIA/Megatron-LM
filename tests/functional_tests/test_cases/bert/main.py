import os
import sys
from typing import Any, Dict, List

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode

# --- Pass 1: Placeholders and Custom Loader ---


class Reference:
    """A placeholder object for a !reference tag."""

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __repr__(self):
        return f"Reference(keys={self.keys})"


class ConfigLoader(yaml.SafeLoader):
    """
    Custom YAML Loader that handles !include, creates
    placeholders for !reference, and defers merge (<<) resolution.
    """

    def __init__(self, stream):
        try:
            self._root = os.path.dirname(stream.name)
        except AttributeError:
            self._root = os.path.abspath('.')
        super().__init__(stream)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> Dict[Any, Any]:
        """
        Override default mapping constructor to *not* call flatten_mapping.
        This prevents PyYAML from trying to resolve '<<' in Pass 1.
        """
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None, "expected a mapping node, but found %s" % node.id, node.start_mark
            )

        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key (%s)" % exc,
                    key_node.start_mark,
                )

            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key (%s)" % key,
                    key_node.start_mark,
                )

            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    def construct_merge(self, node: ScalarNode) -> str:
        """
        Handles the 'tag:yaml.org,2002:merge' tag (the '<<' key)
        by just treating it as a plain string.
        """
        return self.construct_scalar(node)


def include_constructor(loader: ConfigLoader, node: yaml.Node) -> Any:
    """Handles !include by recursively loading with the same loader."""
    filename = os.path.join(loader._root, loader.construct_scalar(node))
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Included file not found: {filename}")

    with open(filename, 'r') as f:
        return yaml.load(f, Loader=type(loader))


def reference_constructor(loader: ConfigLoader, node: yaml.Node) -> Reference:
    """Handles !reference by creating a Reference placeholder."""
    keys = loader.construct_sequence(node)
    return Reference(keys)


# Register all custom constructors with our loader
yaml.add_constructor('!include', include_constructor, Loader=ConfigLoader)
yaml.add_constructor('!reference', reference_constructor, Loader=ConfigLoader)
yaml.add_constructor('tag:yaml.org,2002:merge', ConfigLoader.construct_merge, Loader=ConfigLoader)


# --- Pass 2: Resolver Functions ---


def _lookup(keys: List[str], data: Dict[str, Any]) -> Any:
    """Helper to look up a value from a nested dict via a key path."""
    current = data
    for k in keys:
        current = current[k]
    return current


def resolve_refs(node: Any, root: Dict[str, Any]) -> Any:
    """
    Recursively traverses the data structure, resolving
    Reference placeholders and manually handling '<<' merges.
    """
    # 1. Resolve a Reference placeholder
    if isinstance(node, Reference):
        # Look up the value and *recursively resolve it*
        # in case it's another Reference or contains one.
        found_val = _lookup(node.keys, root)
        return resolve_refs(found_val, root)

    # 2. Recurse into a list
    if isinstance(node, list):
        return [resolve_refs(item, root) for item in node]

    # 3. Recurse into a dict (and handle merges)
    if isinstance(node, dict):
        new_dict = {}

        # Handle the YAML merge key '<<' first.
        if '<<' in node:
            # Resolve the merge source (which could be a Reference)
            merge_source = resolve_refs(node['<<'], root)

            if isinstance(merge_source, dict):
                # Must resolve the *contents* of the merged dict too
                new_dict.update(resolve_refs(merge_source, root))
            elif isinstance(merge_source, list):
                for d in merge_source:
                    if not isinstance(d, dict):
                        raise TypeError(f"YAML merge '<<' list item not a dict: {type(d)}")
                    new_dict.update(resolve_refs(d, root))
            elif merge_source is not None:
                raise TypeError(
                    f"YAML merge key '<<' resolved to invalid type: {type(merge_source)}"
                )

        # Process/override with the rest of the keys
        for key, value in node.items():
            if key == '<<':
                continue
            new_dict[key] = resolve_refs(value, root)

        return new_dict

    # 4. It's a primitive, return as-is
    return node


# --- Main Execution ---


def load_config(main_config_path: str) -> Dict[str, Any]:
    """
    Loads, parses, and fully resolves the two-pass YAML config.
    """
    try:
        # --- Pass 1: Load with custom loader ---
        print("--- Running Pass 1 (Loading) ---")
        pass1_data = None
        with open(main_config_path, 'r') as f:
            pass1_data = yaml.load(f, Loader=ConfigLoader)

        print("Result after Pass 1 (with placeholders):")
        print(pass1_data)

        # --- Pass 2: Resolve references ---
        print("\n--- Running Pass 2 (Resolving) ---")
        # The 'root' for resolution is the entire data structure itself.
        final_data = resolve_refs(pass1_data, pass1_data)

        print("\n--- Final, fully resolved data ---")
        return final_data

    except Exception as e:
        print(f"\nAn error occurred while parsing YAML: {e}", file=sys.stderr)
        # Depending on your app, you might want to re-raise or sys.exit
        raise


if __name__ == "__main__":
    # --- Create dummy files for a self-contained test ---
    # (You would remove this and just call load_config in your real app)

    # This is the main function call

    # Pretty-print the final result
    print(
        yaml.dump(
            load_config('tp1_pp2/model_config.yaml'), default_flow_style=False, sort_keys=False
        )
    )
