#!/usr/bin/env python3
"""
Megatron Core API Compatibility Checker

Simple checker using Griffe to find breaking changes between two versions.
Objects decorated with @exempt_from_compat_check are excluded from checks.

Usage:
    python scripts/check_api_backwards_compatibility.py --baseline core_v0.14.0
"""

import argparse
import os
import sys
from collections import Counter

try:
    import griffe
    try:
        from griffe.dataclasses import Object
    except (ImportError, AttributeError):
        from griffe import Object
except ImportError as e:
    print(f"ERROR: griffe not installed: {e}", file=sys.stderr)
    print("Install with: pip install griffe", file=sys.stderr)
    sys.exit(2)

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Decorators that exempt objects from compatibility checks
EXEMPT_DECORATORS = ['exempt_from_compat_check', 'deprecated', 'internal_api']


def has_exempt_decorator(obj: Object) -> bool:
    """Check if object has any exempt decorator."""
    if not hasattr(obj, 'decorators'):
        return False
    for decorator in obj.decorators:
        dec_str = str(decorator)
        if any(exempt in dec_str for exempt in EXEMPT_DECORATORS):
            return True
    return False


def get_filtered_paths(package: Object) -> set:
    """Get all paths that should be filtered (have exempt decorators)."""
    filtered = set()
    
    def visit(obj, path):
        current_path = f"{path}.{obj.name}" if path else obj.name
        
        # Skip private members
        if obj.name.startswith('_') and not obj.name.startswith('__'):
            return
            
        # Check for exempt decorator
        if has_exempt_decorator(obj):
            filtered.add(current_path)
            print(f"  ⏭️  Exempt: {current_path}", file=sys.stderr)
        
        # Visit children
        if hasattr(obj, 'members'):
            for member in obj.members.values():
                visit(member, current_path)
    
    visit(package, "")
    return filtered


def get_object_path(change) -> str:
    """Extract the object path from a breaking change."""
    # Try different attributes
    path = (getattr(change, 'new_path', None) or 
            getattr(change, 'old_path', None) or
            getattr(change, 'path', None))
    
    if path:
        return path
    
    # Try from values
    if hasattr(change, 'new_value') and change.new_value:
        return getattr(change.new_value, 'path', None)
    if hasattr(change, 'old_value') and change.old_value:
        return getattr(change.old_value, 'path', None)
    
    return None


def should_skip_change(change, filtered_paths: set) -> bool:
    """Check if a breaking change should be skipped based on filters."""
    path = get_object_path(change)
    if not path:
        return False
    
    # Check exact match
    if path in filtered_paths:
        return True
    
    # Check if it's a child of a filtered object
    # e.g., MyClass.__init__ is child of MyClass
    for filtered_path in filtered_paths:
        if path.startswith(filtered_path + '.') or path.startswith(filtered_path + '('):
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Check API backwards compatibility')
    parser.add_argument('--baseline', required=True, help='Baseline git ref (tag/branch/commit)')
    parser.add_argument('--current', default=None, help='Current git ref (default: working directory)')
    parser.add_argument('--package', default='megatron.core', help='Package to check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Clean up stale worktrees before starting
    cleanup_git_worktrees()
    
    try:
        package_name = args.package
        
        print(f"\n{'='*80}\nAPI COMPATIBILITY CHECK: {package_name}\n{'='*80}\n", file=sys.stderr)
        
        # Load baseline
        print(f"📦 Loading baseline @ {args.baseline}...", file=sys.stderr)
        baseline = griffe.load_git(
            package_name, ref=args.baseline, resolve_aliases=False, 
            resolve_external=False, allow_inspection=False)
        print(f"   ✓ Loaded", file=sys.stderr)
        
        # Load current
        print(f"\n📦 Loading current @ {args.current or 'working directory'}...", file=sys.stderr)
        if args.current:
            current = griffe.load_git(
                package_name, ref=args.current, resolve_aliases=False,
                resolve_external=False, allow_inspection=False)
        else:
            current = griffe.load(
                package_name, search_paths=[os.getcwd()], resolve_aliases=False,
                resolve_external=False, allow_inspection=False)
        print(f"   ✓ Loaded", file=sys.stderr)
        
        # Get filtered paths from CURRENT version only
        print(f"\n🔍 Finding exempt objects in current version...", file=sys.stderr)
        filtered_paths = get_filtered_paths(current)
        print(f"   Found {len(filtered_paths)} exempt objects", file=sys.stderr)
        
        # Find breaking changes
        print(f"\n🔍 Comparing versions...", file=sys.stderr)
        all_changes = list(griffe.find_breaking_changes(baseline, current))
        print(f"   Found {len(all_changes)} potential breaking changes", file=sys.stderr)
        
        # Filter out exempt changes
        breaking_changes = []
        skipped_count = 0
        for change in all_changes:
            if should_skip_change(change, filtered_paths):
                skipped_count += 1
            else:
                breaking_changes.append(change)
        
        print(f"   Skipped {skipped_count} exempt | Reporting {len(breaking_changes)} breaking changes", file=sys.stderr)
        
        # Print results
        if not breaking_changes:
            print(f"\n✅ No breaking changes detected!", file=sys.stderr)
            return 0
        
        # Count by type
        change_types = Counter(change.kind.value for change in breaking_changes)
        print(f"\n📊 Breaking changes by type:", file=sys.stderr)
        for change_type, count in sorted(change_types.items(), key=lambda x: -x[1]):
            print(f"   • {change_type}: {count}", file=sys.stderr)
        
        # Print detailed changes
        print(f"\n❌ Found {len(breaking_changes)} breaking change(s):\n{'='*80}")
        
        for i, change in enumerate(breaking_changes, 1):
            path = get_object_path(change)
            path_info = f"\n   Object: {path}" if path else ""
            print(f"\n{i}. {change.kind.value}\n   Package: {package_name}{path_info}\n   → {change.explain()}\n{'-'*80}")
        
        print(f"\n{'='*80}\nSUMMARY\n{'='*80}\nTotal breaking changes: {len(breaking_changes)}\n{'='*80}\n")
        
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
