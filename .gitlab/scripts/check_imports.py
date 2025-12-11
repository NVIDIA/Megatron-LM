# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
"""
Import checker script for megatron.hub package.

This script recursively discovers all Python modules in the specified package
and attempts to import them, reporting any import errors.
"""

import importlib
import os
import sys
import traceback
from typing import Dict, List, Tuple

import click


class ImportChecker:
    """Check imports for all modules in a package."""

    def __init__(self, package_name: str = "megatron.core", verbose: bool = False):
        self.package_name = package_name
        self.success_count = 0
        self.failure_count = 0
        self.graceful_count = 0
        self.skipped_count = 0
        self.failures: Dict[str, str] = {}
        self.successes: List[str] = []
        self.graceful_failures: Dict[str, str] = {}
        self.skipped: List[str] = []

        # Modules to skip (known problematic ones)
        self.skip_patterns = {
            "__pycache__",
            ".pytest_cache",
            ".git",
            "test_",
            "_test",
        }

        # Add current directory to Python path if not already there
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    def should_skip_module(self, module_name: str) -> bool:
        """Check if a module should be skipped."""
        for pattern in self.skip_patterns:
            if pattern in module_name:
                return True
        return False

    def discover_modules(self, package_path: str) -> List[str]:
        """Discover all Python modules in the given package path."""
        modules = []

        package = importlib.import_module(package_path)
        package_path = package.__path__[0]

        # Walk through all Python files
        for root, dirs, files in os.walk(package.__path__[0]):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    # Convert file path to module name
                    rel_path = os.path.relpath(os.path.join(root, file), package_path)
                    module_parts = rel_path.replace(os.sep, ".").replace(".py", "")

                    # Handle __init__.py files
                    if module_parts.endswith(".__init__"):
                        module_parts = module_parts[:-9]  # Remove .__init__

                    full_module_name = (
                        f"{self.package_name}.{module_parts}"
                        if module_parts
                        else self.package_name
                    )

                    if not self.should_skip_module(full_module_name):
                        modules.append(full_module_name)

            # Remove duplicates and sort
            modules = sorted(list(set(modules)))

        return modules

    def import_module(self, module_name: str) -> Tuple[str, str]:
        """
        Try to import a module and return success status and error message.

        Returns:
            Tuple of (status: str, error_message: str)
            status can be: "success", "graceful", or "failed"
        """
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]

            importlib.import_module(module_name)
            return "success", ""

        except Exception:
            tb = traceback.format_exc()
            if "UnavailableError" in tb:
                return "graceful", "UnavailableError detected during import"
            return "failed", f"{str(tb)}"

    def check_all_imports(self):
        """Check imports for all discovered modules."""
        print(f"Discovering modules in package '{self.package_name}'...")
        modules = self.discover_modules(self.package_name)

        if not modules:
            print("No modules found!")
            return

        print(f"Found {len(modules)} modules to check")
        print("=" * 60)

        for i, module_name in enumerate(modules, 1):
            status, error_msg = self.import_module(module_name)

            if status == "success":
                self.success_count += 1
                self.successes.append(module_name)
            elif status == "graceful":
                self.graceful_count += 1
                self.graceful_failures[module_name] = error_msg
            else:  # failed
                self.failure_count += 1
                self.failures[module_name] = error_msg

        """Print a summary of the import check results."""
        total = (
            self.success_count
            + self.failure_count
            + self.graceful_count
            + self.skipped_count
        )

        print("\n" + "=" * 60)
        print("IMPORT CHECK SUMMARY")
        print("=" * 60)
        print(f"Total modules checked: {total}")
        print(
            f"Successful imports:    {self.success_count} ({self.success_count / total * 100:.1f}%)"
        )
        print(
            f"Gracefully handled:    {self.graceful_count} ({self.graceful_count / total * 100:.1f}%)"
        )
        print(
            f"Failed imports:        {self.failure_count} ({self.failure_count / total * 100:.1f}%)"
        )
        if self.skipped_count > 0:
            print(
                f"Skipped modules:       {self.skipped_count} ({self.skipped_count / total * 100:.1f}%)"
            )

        if self.graceful_failures:
            print(f"\n🟡 GRACEFULLY HANDLED ({len(self.graceful_failures)}):")
            print("-" * 40)

        if self.failures:
            print(f"\n❌ FAILED IMPORTS ({len(self.failures)}):")
            print("-" * 40)
            for module_name, error_msg in self.failures.items():
                print(f"\n• {module_name}")
                # Show only the first few lines of error to keep output manageable
                error_lines = error_msg.split("\n")
                for line in error_lines:
                    # if self.package_name.replace(".", os.sep) not in line:
                    #     continue
                    if line.strip():
                        print(f"  {line}")

        return self.failure_count == 0


@click.command()
@click.option(
    "--package-name",
    required=True,
    help="Package name to check imports for",
)
def main(package_name: str):
    """Main entry point."""
    checker = ImportChecker(package_name=package_name)
    successful = checker.check_all_imports()
    exit(0 if successful else 1)


if __name__ == "__main__":
    main()
