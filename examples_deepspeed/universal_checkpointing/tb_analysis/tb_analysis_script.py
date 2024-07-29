# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import get_analyzer, find_files_prefix, find_files_suffix
from arguments import parser

args = parser.parse_args()

if args.use_sns:
    import seaborn as sns
    sns.set()

def main():
    target_prefix = 'events.out.tfevents'
    tb_log_paths = find_files_prefix(args.tb_dir, target_prefix)

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

    plt.grid(True)

    if not args.skip_plot:
        plt.legend()
        plt.title(args.plot_title)
        plt.xlabel(args.plot_x_label)
        plt.ylabel(args.plot_y_label)
        plt.savefig(args.plot_name)

def plot_csv():
    target_suffix = 'csv'
    csv_log_files = find_files_suffix(args.csv_dir, target_suffix)

    analyzer = get_analyzer(args.analyzer)

    for csv_file in csv_log_files:
        analyzer.set_names(csv_file)

        x, y = [], []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == 'step':
                    continue
                x.append(int(row[1]))  # Assuming the first column contains x values
                y.append(float(row[2]))  # Assuming the second column contains y values

        plt.plot(x, y, label=f'{analyzer.get_label_name()}')

    plt.grid(True)
    plt.legend()
    plt.title(args.plot_title)
    plt.xlabel(args.plot_x_label)
    plt.ylabel(args.plot_y_label)
    plt.savefig(args.plot_name)

if __name__ == "__main__":
    if args.plot_only:
        plot_csv()
    else:
        main()
