# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import abc
from abc import ABC


class TensorBoardAnalysis(ABC):

    def __init__(self):
        self._name = None
        self._label_name = None
        self._csv_name = None

    @abc.abstractmethod
    def set_names(self, path_name):
        ...

    @abc.abstractmethod
    def get_label_name(self):
        ...

    @abc.abstractmethod
    def get_csv_filename(self):
        ...

    @abc.abstractmethod
    def path_regex(self):
        ...
