#!/usr/bin/env python3
"""
Megatron Core API Compatibility Checker

This script uses griffe to check for breaking changes in the Megatron Core API
between two git references (branches, tags, or commits). It supports filtering
to exclude functions marked with specific decorators or patterns.

Usage:
    # Compare current code against latest release (default: megatron.core)
    python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0
    
    # Compare two specific branches
    python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --current main
    
    # Check multiple packages
    python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --package megatron.core megatron.training
    
    # Verbose output
    python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 -v

Exit codes:
    0 - No breaking changes detected
    1 - Breaking changes found
    2 - Script error

For more information, see: docs/api-backwards-compatibility-checking.md
"""

import argparse
import os
import sys
from typing import Set, List

try:
    import griffe
    try:
        from griffe.dataclasses import Object
    except (ImportError, AttributeError):
        # Newer versions of griffe
        from griffe import Object
except ImportError as e:
    print(f"ERROR: griffe is not installed or import failed: {e}", file=sys.stderr)
    print("Install it with: pip install griffe", file=sys.stderr)
    sys.exit(2)


# ============================================================================
# Configuration - Customize these as needed
# ============================================================================

# Packages to check (can specify multiple)
PACKAGE_NAMES = [
    "megatron.core",
    # Add more packages here if needed:
    # "megatron.training",
    # "megatron.post_training",
]

# Decorators that exempt functions from compatibility checks
EXEMPT_DECORATORS = {
    "exempt_from_compat_check",
    "experimental",
    "internal_api",
    "private",
}

# Exclude functions/classes starting with these prefixes
EXCLUDE_PREFIXES = {
    "test_",   # Test functions
}

# Exclude specific paths/modules
EXCLUDE_PATHS = {
    "tests",
    "test",
    "experimental",
    "internal",
    ".legacy",  # Legacy code
}


# ============================================================================
# Filtering logic
# ============================================================================

def should_skip_object(obj: Object) -> bool:
    """
    Determine if an object should be skipped from compatibility checks.
    
    Args:
        obj: A griffe Object (Function, Class, etc.)
    
    Returns:
        True if object should be skipped, False otherwise
    """
    # Skip aliases to external packages (e.g., import torch)
    if obj.kind.value == "alias":
        try:
            # Try to check if it's an external import
            if hasattr(obj, 'target_path'):
                target = str(obj.target_path)
                # Skip if it's pointing to external packages
                if not target.startswith('megatron'):
                    print(f"  ⏭️  Skipping {obj.path} (external alias)", file=sys.stderr)
                    return True
        except Exception:
            # If we can't resolve it, skip it
            print(f"  ⏭️  Skipping {obj.path} (unresolvable alias)", file=sys.stderr)
            return True
    
    # Check decorators
    try:
        if hasattr(obj, 'decorators') and obj.decorators:
            decorator_names = {
                dec.value.name if hasattr(dec.value, 'name') else str(dec.value)
                for dec in obj.decorators
            }
            if decorator_names & EXEMPT_DECORATORS:
                print(f"  ⏭️  Skipping {obj.path} (has exempt decorator)", file=sys.stderr)
                return True
    except Exception as e:
        # If we can't check decorators (e.g., alias resolution error), skip it
        print(f"  ⏭️  Skipping {obj.path} (decorator check failed: {type(e).__name__})", file=sys.stderr)
        return True
    
    # Check name prefixes
    for prefix in EXCLUDE_PREFIXES:
        if obj.name.startswith(prefix):
            print(f"  ⏭️  Skipping {obj.path} (test)", file=sys.stderr)
            return True
    
    # Check path exclusions
    for exclude_path in EXCLUDE_PATHS:
        if exclude_path in obj.path:
            print(f"  ⏭️  Skipping {obj.path} (excluded path)", file=sys.stderr)
            return True
    
    return False


def filter_objects(obj: Object, filtered: Set[str], visited: Set[str] = None):
    """
    Recursively filter objects and mark which should be skipped.
    
    Args:
        obj: A griffe Object to examine
        filtered: Set to populate with paths of objects to skip
        visited: Set of already visited paths to prevent infinite recursion
    """
    if visited is None:
        visited = set()
    
    # Hard limit on path length - catch malformed circular paths early
    if len(obj.path) > 150:
        filtered.add(obj.path)
        return
    
    # Prevent infinite recursion by tracking visited paths
    if obj.path in visited:
        return
    visited.add(obj.path)
    
    # Detect circular references by checking for repeated path patterns
    path_parts = obj.path.split('.')
    for length in range(2, min(len(path_parts) // 2 + 1, 10)):
        for start in range(len(path_parts) - 2 * length + 1):
            pattern = '.'.join(path_parts[start:start + length])
            rest_of_path = '.'.join(path_parts[start + length:])
            if rest_of_path.startswith(pattern):
                filtered.add(obj.path)
                if len(visited) < 5:  # Print first few for visibility
                    print(f"  ⏭️  Skipping {obj.path[:80]}... (circular: {pattern[:40]})", file=sys.stderr)
                return
    
    # Skip aliases (imports) - they're not real API definitions
    if obj.kind.value == "alias":
        try:
            if hasattr(obj, 'target_path'):
                target = str(obj.target_path)
                if not target.startswith('megatron'):
                    filtered.add(obj.path)
        except Exception:
            filtered.add(obj.path)
        return
    
    # Check if object should be skipped (decorators, path patterns, etc.)
    if should_skip_object(obj):
        filtered.add(obj.path)
        return
    
    # Recurse into members (for classes, modules, etc.)
    if hasattr(obj, 'members'):
        for member in obj.members.values():
            filter_objects(member, filtered, visited)


# ============================================================================
# Main comparison logic
# ============================================================================

def extract_path_from_explanation(change) -> str:
    """
    Extract object path from a breakage's explanation text.
    
    Format: "filepath:line: object_path: description"
    Example: "megatron/core/model_parallel_config.py:338: ModelParallelConfig.cpu_offloading_weights: Attribute value was changed"
    
    Returns the object path or None if not found.
    """
    try:
        explanation = change.explain()
        # Split by ": " and get the second part (object path)
        parts = explanation.split(': ')
        if len(parts) >= 2:
            # First part is "filepath:line", second is object path
            return parts[1]
    except:
        pass
    return None


def load_and_filter(package_name: str, ref: str = None, verbose: bool = False) -> tuple:
    """
    Load package and apply filters.
    
    Args:
        package_name: Name of package to load
        ref: Git reference (branch/tag/commit), None for current
        verbose: Enable verbose output
    
    Returns:
        Tuple of (griffe Object, set of filtered object paths)
    """
    ref_label = f" @ {ref}" if ref else " (current)"
    print(f"📦 Loading {package_name}{ref_label}", file=sys.stderr)
    
    try:
        # Load the package (without resolving aliases to prevent circular refs)
        if ref:
            package = griffe.load_git(package_name, ref=ref, resolve_aliases=False, resolve_external=False)
        else:
            # For current version, load from working directory using static analysis
            package = griffe.load(
                package_name,
                search_paths=[os.getcwd()],  # Search in current directory
                resolve_aliases=False,
                resolve_external=False
            )
    except Exception as e:
        print(f"❌ Error loading {package_name}{ref_label}: {e}", file=sys.stderr)
        raise
    
    # Find objects to filter
    filtered = set()
    if verbose:
        print(f"   Applying filters...", file=sys.stderr)
    filter_objects(package, filtered)
    
    if filtered:
        print(f"   Filtered {len(filtered)} objects", file=sys.stderr)
    
    # Note: We don't physically remove filtered objects from the tree because
    # the circular structure makes tree traversal extremely slow (300K+ objects).
    # Instead, we'll filter breaking changes after comparison.
    
    return package, filtered


def print_breaking_changes(changes: List, verbose: bool = False) -> bool:
    """
    Print breaking changes in a formatted way.
    
    Args:
        changes: List of breaking changes from griffe
        verbose: Enable verbose output
    
    Returns:
        True if no breaking changes, False otherwise
    """
    if not changes:
        print("\n✅ No breaking changes detected!")
        return True
    
    print(f"\n❌ Found {len(changes)} breaking change(s):\n")
    print("=" * 80)
    
    for i, change in enumerate(changes, 1):
        print(f"\n{i}. {change.kind.value}")
        
        # Different breakage types have different path attributes
        old_path = getattr(change, 'old_path', None) or getattr(change, 'path', None)
        new_path = getattr(change, 'new_path', None) or getattr(change, 'path', None)
        obj_path = old_path or new_path or "Unknown"
        print(f"   Object: {obj_path}")
        
        # Print location if available
        if hasattr(change, 'old_value') and hasattr(change.old_value, 'filepath'):
            print(f"   File: {change.old_value.filepath}")
        
        # Print explanation
        explanation = change.explain()
        for line in explanation.split('\n'):
            if line.strip():
                print(f"   → {line}")
        
        print("-" * 80)
    
    return False


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check Megatron Core API compatibility using griffe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare current code against release tag (default: megatron.core)
  python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0
  
  # Compare two specific references
  python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --current main
  
  # Check multiple packages
  python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 --package megatron.core megatron.training
  
  # Verbose output
  python scripts/check_api_backwards_compatibility.py --baseline core_r0.8.0 -v

Configuration:
  Edit the script to customize:
  - PACKAGE_NAMES: List of packages to check
  - EXEMPT_DECORATORS: Decorators that mark functions as exempt
  - EXCLUDE_PREFIXES: Name prefixes to exclude (e.g., "test_")
  - EXCLUDE_PATHS: Paths to exclude (e.g., "tests", "experimental")

Decorators:
  Use @exempt_from_compat_check to mark functions that should not be checked:
  
    from megatron.core.backwards_compatibility_decorators import exempt_from_compat_check
    
    @exempt_from_compat_check
    def experimental_feature():
        pass

Exit codes:
  0 = No breaking changes (CI passes)
  1 = Breaking changes detected (CI fails)
  2 = Script error
        """
    )
    
    parser.add_argument(
        '--baseline', '-b',
        required=True,
        help='Baseline git reference (e.g., core_r0.8.0, v1.0.0, main)'
    )
    
    parser.add_argument(
        '--current', '-c',
        default=None,
        help='Current git reference (default: working directory)'
    )
    
    parser.add_argument(
        '--package', '-p',
        nargs='+',
        default=PACKAGE_NAMES,
        help=f'Package name(s) to check (default: {", ".join(PACKAGE_NAMES)}). Can specify multiple packages separated by spaces.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Ensure packages is a list
    packages = args.package if isinstance(args.package, list) else [args.package]
    
    try:
        all_breaking_changes = []
        all_success = True
        
        # Check each package
        for package_name in packages:
            print(f"\n{'=' * 80}", file=sys.stderr)
            print(f"📦 Checking package: {package_name}", file=sys.stderr)
            print(f"{'=' * 80}", file=sys.stderr)
            
            # Load baseline version
            baseline, baseline_filtered = load_and_filter(package_name, ref=args.baseline, verbose=args.verbose)
            
            # Load current version
            current, current_filtered = load_and_filter(package_name, ref=args.current, verbose=args.verbose)
            
            # Combine filtered sets
            all_filtered = baseline_filtered | current_filtered
            
            # Find breaking changes
            print(f"\n🔍 Comparing {package_name}:", file=sys.stderr)
            print(f"   Baseline: {args.baseline}", file=sys.stderr)
            print(f"   Current:  {args.current or 'working directory'}", file=sys.stderr)
            print(f"   Running comparison...", file=sys.stderr)
            
            all_breaking_changes_raw = list(griffe.find_breaking_changes(baseline, current))
            print(f"   ✓ Comparison complete", file=sys.stderr)
            
            # Filter out breaking changes involving filtered objects
            breaking_changes = []
            for change in all_breaking_changes_raw:
                # Different breakage types have different path attributes
                old_path = (getattr(change, 'old_path', None) or 
                           getattr(change, 'path', None) or
                           (getattr(change, 'old_value', None) and getattr(change.old_value, 'path', None)) or
                           extract_path_from_explanation(change))
                new_path = (getattr(change, 'new_path', None) or 
                           getattr(change, 'path', None) or
                           (getattr(change, 'new_value', None) and getattr(change.new_value, 'path', None)) or
                           extract_path_from_explanation(change))
                
                # Skip if either path is in our filtered set
                if old_path and old_path in all_filtered:
                    continue
                if new_path and new_path in all_filtered:
                    continue
                
                # Skip paths with circular/repeated segments (false positives from import cycles)
                # e.g., "megatron.core.pipeline_parallel.p2p_communication.core.megatron.training"
                should_skip = False
                for path in [old_path, new_path]:
                    if path:
                        parts = path.split('.')
                        # Check for repeated 2-3 segment patterns
                        for length in range(2, min(4, len(parts) // 2 + 1)):
                            for start in range(len(parts) - 2 * length + 1):
                                pattern = '.'.join(parts[start:start + length])
                                rest = '.'.join(parts[start + length:])
                                if rest.startswith(pattern):
                                    should_skip = True
                                    break
                            if should_skip:
                                break
                    if should_skip:
                        break
                
                if should_skip:
                    continue
                
                breaking_changes.append(change)
            
            if len(all_breaking_changes_raw) > len(breaking_changes):
                filtered_count = len(all_breaking_changes_raw) - len(breaking_changes)
                print(f"   Filtered out {filtered_count} changes (excluded code + circular paths)", file=sys.stderr)
            
            if breaking_changes:
                all_breaking_changes.extend([(package_name, change) for change in breaking_changes])
                all_success = False
            
            # Print results for this package
            if breaking_changes:
                print(f"\n❌ Found {len(breaking_changes)} breaking change(s) in {package_name}:\n")
                print("=" * 80)
                for i, change in enumerate(breaking_changes, 1):
                    print(f"\n{i}. {change.kind.value}")
                    print(f"   Package: {package_name}")
                    
                    # Different breakage types have different path attributes
                    old_path = (getattr(change, 'old_path', None) or 
                               getattr(change, 'path', None) or
                               (getattr(change, 'old_value', None) and getattr(change.old_value, 'path', None)) or
                               extract_path_from_explanation(change))
                    new_path = (getattr(change, 'new_path', None) or 
                               getattr(change, 'path', None) or
                               (getattr(change, 'new_value', None) and getattr(change.new_value, 'path', None)) or
                               extract_path_from_explanation(change))
                    obj_path = old_path or new_path or "Unknown"
                    print(f"   Object: {obj_path}")
                    
                    # Print location if available
                    if hasattr(change, 'old_value') and hasattr(change.old_value, 'filepath'):
                        print(f"   File: {change.old_value.filepath}")
                    
                    # Print explanation
                    explanation = change.explain()
                    for line in explanation.split('\n'):
                        if line.strip():
                            print(f"   → {line}")
                    
                    print("-" * 80)
            else:
                print(f"\n✅ No breaking changes in {package_name}")
        
        # Print summary
        print(f"\n{'=' * 80}", file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)
        print(f"Packages checked: {len(packages)}", file=sys.stderr)
        print(f"Total breaking changes: {len(all_breaking_changes)}", file=sys.stderr)
        
        if all_success:
            print("\n✅ No breaking changes detected across all packages!")
        else:
            print(f"\n❌ Breaking changes detected in {sum(1 for pkg in packages if any(pkg == p for p, _ in all_breaking_changes))} package(s)")
        
        # Exit with appropriate code
        sys.exit(0 if all_success else 1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()

