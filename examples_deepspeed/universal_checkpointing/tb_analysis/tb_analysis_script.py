# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import get_analyzer, find_files
from arguments import parser

args = parser.parse_args()

if args.use_sns:
    import seaborn as sns
    sns.set()

def main():
    target_affix = 'events.out.tfevents'
    tb_log_paths = find_files(args.tb_dir, target_affix)

    analyzer = get_analyzer(args.analyzer)

    for tb_path in tb_log_paths:
        print(f"Processing: {tb_path}")
        analyzer.set_names(tb_path)

        event_accumulator = EventAccumulator(tb_path)
        event_accumulator.Reload()

        events = event_accumulator.Scalars(args.tb_event_key)

        x = [x.step for x in events]
        y = [x.value for x in events]

        plt.plot(x, y, label=f'{analyzer.get_label_name()}')

        if not args.skip_csv:
            df = pd.DataFrame({"step": x, "value": y})
            df.to_csv(f"{args.csv_name}{analyzer.get_csv_filename()}.csv")

    if not args.skip_plot:
        plt.legend()
        plt.title(args.plot_title)
        plt.xlabel(args.plot_x_label)
        plt.ylabel(args.plot_y_label)
        plt.savefig(args.plot_name)

if __name__ == "__main__":
    main()
