# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from abstract_analysis import TensorBoardAnalysis


class UniversalCheckpointingAnalysis(TensorBoardAnalysis):

    def __init__(self):
        self._name = "universal_checkpointing"

    def set_names(self, path_name):
        match = re.match(self.path_regex(), path_name)
        if not match:
            raise ValueError(f"Path ({path_name}) did not match regex ({self.path_regex()})")
        tp, pp, dp, sp = match.groups()

        self._label_name = f"Training Run: TP: {tp}, PP: {pp}, DP: {dp}"
        self._csv_name = f"uc_out_tp{tp}_pp{pp}_dp{dp}_sp{sp}"

    def get_label_name(self):
        return self._label_name

    def get_csv_filename(self):
        return self._csv_name

    def path_regex(self):
        return '.*tp(\d+).*pp(\d+).*dp(\d+).*sp(\d+)'
