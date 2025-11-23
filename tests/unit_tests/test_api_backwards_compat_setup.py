# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3
"""
Sanity test script to verify API backward compatibility checker setup

This script tests that:
1. Griffe is installed
2. The checker script can be imported
3. The decorator module exists
4. Basic functionality works

Usage:
    python tests/unit_tests/test_api_backwards_compat_setup.py
"""

import io
import sys
from pathlib import Path

# Configure UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def test_griffe_installed():
    """Test that griffe is installed"""
    print("1. Testing griffe installation...", end=" ")
    try:
        import griffe

        print("✅ griffe is installed")
        print(f"   Version: {griffe.__version__ if hasattr(griffe, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("❌ griffe is NOT installed")
        print("   Install with: pip install griffe")
        return False


def test_decorator_module():
    """Test that the compat decorator module exists"""
    print("\n2. Testing decorator module...", end=" ")
    try:
        from megatron.core.utils import deprecated, internal_api

        print("✅ Decorator module found")
        print("   Available: @internal_api, @deprecated")
        return True
    except ImportError as e:
        print("❌ Decorator module NOT found")
        print(f"   Error: {e}")
        return False


def test_checker_script():
    """Test that the checker script exists"""
    print("\n3. Testing checker script...", end=" ")
    script_path = Path("scripts/check_api_backwards_compatibility.py")
    if script_path.exists():
        print("✅ Checker script found")
        print(f"   Location: {script_path}")
        return True
    else:
        print("❌ Checker script NOT found")
        print(f"   Expected: {script_path}")
        return False


def test_workflow():
    """Test that the GitHub Actions workflow exists"""
    print("\n4. Testing GitHub Actions workflow...", end=" ")
    workflow_path = Path(".github/workflows/check_api_backwards_compatibility_workflow.yml")
    if workflow_path.exists():
        print("✅ Workflow found")
        print(f"   Location: {workflow_path}")
        return True
    else:
        print("❌ Workflow NOT found")
        print(f"   Expected: {workflow_path}")
        return False


def test_decorators_work():
    """Test that decorators can be applied"""
    print("\n5. Testing decorator functionality...", end=" ")
    try:
        from megatron.core.utils import deprecated, internal_api

        # Test internal_api decorator
        @internal_api
        def test_func1():
            pass

        assert hasattr(test_func1, '_internal_api')

        # Test deprecated decorator
        @deprecated(version="1.0", removal_version="2.0", alternative="new_func")
        def test_func2():
            pass

        assert hasattr(test_func2, '_deprecated')

        print("✅ Decorators work correctly")
        return True
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        return False


def test_basic_comparison():
    """Test basic griffe comparison"""
    print("\n6. Testing griffe comparison...", end=" ")
    try:
        import griffe

        # Create two simple code snippets
        old_code = """
def example_func(x, y):
    pass
"""

        new_code = """
def example_func(x, y, z=None):
    pass
"""

        # This would normally use griffe.load_git, but we'll skip the actual test
        # since it requires a git repo. Just verify griffe has the function.
        assert hasattr(griffe, 'find_breaking_changes')

        print("✅ Griffe comparison available")
        return True
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


def main():
    print("=" * 70)
    print("API Backward Compatibility Checker Setup Sanity Test")
    print("=" * 70)

    results = []
    results.append(test_griffe_installed())
    results.append(test_decorator_module())
    results.append(test_checker_script())
    results.append(test_workflow())
    results.append(test_decorators_work())
    results.append(test_basic_comparison())

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"✅ All tests passed ({passed}/{total})")
        print("\nYou're ready to use the API compatibility checker!")
        print("\nNext steps:")
        print("  1. Run: python scripts/check_api_backwards_compatibility.py --baseline <ref>")
        print("  2. See: docs/api-backwards-compatibility-check.md for full documentation")
        return 0
    else:
        print(f"❌ Some tests failed ({passed}/{total} passed)")
        print("\nPlease fix the issues above before using the checker.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
