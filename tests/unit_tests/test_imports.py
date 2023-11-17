# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import importlib
import inspect
import os
import traceback

import torch
import wrapt

from megatron.core.transformer.module import MegatronModule


def import_class_by_path(path: str):
    paths = path.split('.')
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod


def _build_import_path(subdomains: list, imp):
    import_path = ["megatron", "core"]
    import_path.extend(subdomains)
    import_path.append(imp)
    path = ".".join(import_path)
    return path


def _get_class_from_path(subdomains, imp):
    path = _build_import_path(subdomains, imp)
    print(path)
    class_ = None
    result = None
    try:
        class_ = import_class_by_path(path)
        if inspect.isclass(class_):
            if isinstance(class_, wrapt.FunctionWrapper):
                class_ = class_.__wrapped__
            if issubclass(class_, (MegatronModule, torch.nn.Module)):
                result = class_
        else:
            class_ = None
        error = None
    except Exception:
        error = traceback.format_exc()
    return class_, result, error


def _test_domain_module_imports(module, subdomains: list):
    module_list = []
    failed_list = []
    error_list = []

    error = None
    if len(subdomains) > 0:
        basepath = module.__path__[0]
        megatron_index = basepath.rfind("megatron")
        basepath = basepath[megatron_index:].replace(os.path.sep, ".")
        new_path = '.'.join([basepath, *subdomains])

        try:
            module = importlib.import_module(new_path)
        except Exception:
            print(f"Could not import `{new_path}` ; Traceback below :")
            error = traceback.format_exc()
            error_list.append(error)

    if error is None:
        for imp in dir(module):
            class_, result, error = _get_class_from_path(
                subdomains, imp)

            if result is not None:
                module_list.append(class_)

            elif class_ is not None:
                failed_list.append(class_)

            if error is not None:
                error_list.append(error)

    for module in module_list:
        print("Module successfully imported :", module)

    print()
    for module in failed_list:
        print(
            "Module did not match a valid signature of Megatron core Model (hence ignored):", module)

    print()
    if len(error_list) > 0:
        print("Imports crashed with following traceback !")

        for error in error_list:
            print("*" * 100)
            print()
            print(error)
            print()
            print("*" * 100)
            print()

    if len(error_list) > 0:
        return False
    else:
        return True


###############################


def test_domain_mcore():
    import megatron.core as mcore

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['models'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['pipeline_parallel'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['tensor_parallel'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['transformer'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['fusions'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['distributed'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['datasets'])

    all_passed = _test_domain_module_imports(
        mcore,  subdomains=['dist_checkpointing'])

    if not all_passed:
        exit(1)


if __name__ == '__main__':
    test_domain_mcore()
