# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
Megatron Core API Compatibility Checker

Simple checker using Griffe to find breaking changes between two versions.
Objects decorated with @internal_api or @deprecated are excluded from checks.

Usage:
    python scripts/check_api_backwards_compatibility.py --baseline core_v0.14.0
"""

import argparse
import logging
import os
import re
import sys
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

try:
    import griffe
    try:
        from griffe.dataclasses import Object
    except (ImportError, AttributeError):
        from griffe import Object
except ImportError as e:
    logger.error(f"griffe not installed: {e}")
    logger.error("Install with: pip install griffe")
    sys.exit(2)

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Decorators that exempt objects from compatibility checks
EXEMPT_DECORATORS = ['internal_api', 'deprecated']


def has_exempt_decorator(obj: Object) -> bool:
    """Check if a Griffe object has any exempt decorator.
    
    Args:
        obj: A Griffe Object to check for exempt decorators
        
    Returns:
        bool: True if the object has any decorator matching EXEMPT_DECORATORS list
    """
    if not hasattr(obj, 'decorators'):
        return False
    if not obj.decorators:
        return False
    for decorator in obj.decorators:
        # Get the actual decorator name from the value attribute
        dec_value = str(getattr(decorator, 'value', ''))
        if any(exempt in dec_value for exempt in EXEMPT_DECORATORS):
            return True
    return False


def get_filtered_paths(package: Object, package_name: str) -> set:
    """Recursively collect all object paths with exempt decorators from a package.
    
    This function traverses the entire package tree and identifies objects that are
    decorated with any of the EXEMPT_DECORATORS, building a set of their full paths.
    
    Args:
        package: The Griffe package object to traverse
        package_name: The full package name (e.g., "megatron.core") for path construction
        
    Returns:
        set: A set of full object paths (e.g., "megatron.core.ModelParallelConfig") 
             that should be filtered from compatibility checks
    """
    filtered = set()
    visited = set()
    
    def visit(obj, path, depth=0, is_root=False):
        # Prevent infinite recursion
        if depth > 20 or id(obj) in visited:
            return
        visited.add(id(obj))
        
        # For root object, use the provided path; for children, append obj.name
        if is_root:
            current_path = path
        else:
            current_path = f"{path}.{obj.name}" if path else obj.name
        
        # Skip aliases (imported objects)
        if hasattr(obj, 'is_alias') and obj.is_alias:
            return
        
        # Skip private members
        if obj.name.startswith('_') and not obj.name.startswith('__'):
            return
            
        # Check for exempt decorator
        if has_exempt_decorator(obj):
            filtered.add(current_path)
            logger.info(f"  ⏭️  Exempt: {current_path}")
        
        # Visit children
        if hasattr(obj, 'members'):
            for member in obj.members.values():
                visit(member, current_path, depth + 1, is_root=False)
    
    # Start with the full package name (e.g., "megatron.core")
    visit(package, package_name, is_root=True)
    return filtered


def strip_ansi_codes(text):
    """Remove ANSI escape codes (terminal formatting) from text.
    
    Griffe includes ANSI codes for terminal formatting in some strings,
    which breaks string matching. This strips them out.
    
    Args:
        text: String potentially containing ANSI escape codes
        
    Returns:
        str: Clean text with ANSI codes removed
    """
    if not text:
        return text
    # Pattern to match ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def get_object_path(change) -> str:
    """Extract the full object path from a Griffe breaking change.
    
    Tries multiple sources to get the object path:
    1. Direct path attributes (new_path, old_path, path)
    2. Path from new_value or old_value objects
    3. Parse from the explanation string as last resort
    
    Args:
        change: A Griffe breaking change object
        
    Returns:
        str: The full object path (e.g., "megatron.core.ModelParallelConfig.__init__")
             or None if unable to extract
    """
    # Try different attributes
    path = (getattr(change, 'new_path', None) or 
            getattr(change, 'old_path', None) or
            getattr(change, 'path', None))
    
    if path:
        return strip_ansi_codes(path)
    
    # Try from values
    if hasattr(change, 'new_value') and change.new_value:
        path = getattr(change.new_value, 'path', None)
        if path:
            return strip_ansi_codes(path)
    
    if hasattr(change, 'old_value') and change.old_value:
        path = getattr(change.old_value, 'path', None)
        if path:
            return strip_ansi_codes(path)
    
    # Last resort: parse from explanation
    # Format: "filepath:line: object_path: description"
    # Example: "megatron/core/model_parallel_config.py:338: ModelParallelConfig.cpu_offloading_weights: Attribute value was changed"
    try:
        explanation = change.explain()
        # Split by ": " and get the second part (object path)
        parts = explanation.split(': ')
        if len(parts) >= 2:
            # Get the part after "filepath:line" but before the description
            # It's usually the second part
            object_path = parts[1]
            
            # Extract the module path from file path (first part)
            file_part = parts[0].split(':')[0]  # Get just the file path, remove line number
            
            # Convert file path to module path
            # e.g., "megatron/core/model_parallel_config.py" -> "megatron.core.model_parallel_config"
            module_path = file_part.replace('/', '.').replace('\\', '.').replace('.py', '')
            
            # If object_path doesn't start with module, prepend it
            if not object_path.startswith(module_path):
                full_path = f"{module_path}.{object_path}"
            else:
                full_path = object_path
            
            return strip_ansi_codes(full_path)
    except Exception:
        pass
    
    return None


def should_skip_change(change, filtered_paths: set) -> bool:
    """Determine if a breaking change should be skipped based on exempt decorators.
    
    A change is skipped if:
    - The changed object itself is in filtered_paths (exact match)
    - The changed object is a child of an exempt object (prefix match)
    
    Args:
        change: A Griffe breaking change object
        filtered_paths: Set of paths with exempt decorators
        
    Returns:
        bool: True if the change should be skipped (filtered out)
    """
    path = get_object_path(change)
    if not path:
        return False
    
    # Strip parameter names from path for matching
    # e.g., "Class.__init__(param)" -> "Class.__init__"
    clean_path = path.split('(')[0] if '(' in path else path
    
    # Check exact match
    if clean_path in filtered_paths or path in filtered_paths:
        return True
    
    # Check if it's a child of a filtered object
    # e.g., MyClass.__init__ is child of MyClass, MyClass.attr is child of MyClass
    for filtered_path in filtered_paths:
        if clean_path.startswith(filtered_path + '.'):
            return True
        # Also check the original path in case parameter names matter
        if path.startswith(filtered_path + '.'):
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Check API backwards compatibility')
    parser.add_argument('--baseline', required=True, help='Baseline git ref (tag/branch/commit)')
    parser.add_argument('--current', default=None, help='Current git ref (default: working directory)')
    parser.add_argument('--package', default='megatron.core', help='Package to check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    try:
        package_name = args.package
        
        logger.info(f"\n{'='*80}\nAPI COMPATIBILITY CHECK: {package_name}\n{'='*80}\n")
        
        # Load baseline
        logger.info(f"📦 Loading baseline @ {args.baseline}...")
        baseline = griffe.load_git(
            package_name, ref=args.baseline, resolve_aliases=False, 
            resolve_external=False, allow_inspection=False)
        logger.info(f"   ✓ Loaded")
        
        # Load current
        logger.info(f"\n📦 Loading current @ {args.current or 'working directory'}...")
        if args.current:
            current = griffe.load_git(
                package_name, ref=args.current, resolve_aliases=False,
                resolve_external=False, allow_inspection=False)
        else:
            current = griffe.load(
                package_name, search_paths=[os.getcwd()], resolve_aliases=False,
                resolve_external=False, allow_inspection=False)
        logger.info(f"   ✓ Loaded")
        
        # Get filtered paths from CURRENT version only
        logger.info(f"\n🔍 Finding exempt objects in current version...")
        filtered_paths = get_filtered_paths(current, package_name)
        logger.info(f"   Found {len(filtered_paths)} exempt objects")
        
        # Find breaking changes
        logger.info(f"\n🔍 Comparing versions...")
        all_changes = list(griffe.find_breaking_changes(baseline, current))
        logger.info(f"   Found {len(all_changes)} potential breaking changes")
        
        # Filter out exempt changes  
        breaking_changes = []
        skipped_count = 0
        
        # DEBUG: Print first 5 breaking changes for debugging
        print("\n===TEST DEBUG (first 5 changes)===")
        print(f"Filtered paths: {filtered_paths}")
        for i, change in enumerate(all_changes[:5]):
            path = get_object_path(change)
            clean_path = path.split('(')[0] if path and '(' in path else path
            print(f"\nChange {i+1}: {path}")
            print(f"  Clean: {clean_path}")
            print(f"  Clean repr: {repr(clean_path)}")
            
            # Test matching
            matched = False
            for fpath in filtered_paths:
                if clean_path and (clean_path == fpath or clean_path.startswith(fpath + '.')):
                    print(f"  ✓ MATCH with: {fpath}")
                    matched = True
                    break
            if not matched:
                print(f"  ✗ NO MATCH")
        print("\n===END TEST DEBUG===\n")
        
        for change in all_changes:
            if should_skip_change(change, filtered_paths):
                skipped_count += 1
            else:
                breaking_changes.append(change)
        
        logger.info(f"\n   Skipped {skipped_count} exempt | Reporting {len(breaking_changes)} breaking changes")
        
        # Print results
        if not breaking_changes:
            logger.info(f"\n✅ No breaking changes detected!")
            return 0
        
        # Count by type
        change_types = Counter(change.kind.value for change in breaking_changes)
        logger.info(f"\n📊 Breaking changes by type:")
        for change_type, count in sorted(change_types.items(), key=lambda x: -x[1]):
            logger.info(f"   • {change_type}: {count}")
        
        # Print detailed changes
        print(f"\n❌ Found {len(breaking_changes)} breaking change(s):\n{'='*80}")
        
        for i, change in enumerate(breaking_changes, 1):
            path = get_object_path(change)
            path_info = f"\n   Object: {path}" if path else ""
            print(f"\n{i}. {change.kind.value}\n   Package: {package_name}{path_info}\n   → {change.explain()}\n{'-'*80}")
        
        print(f"\n{'='*80}\nSUMMARY\n{'='*80}\nTotal breaking changes: {len(breaking_changes)}\n{'='*80}\n")
        
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
