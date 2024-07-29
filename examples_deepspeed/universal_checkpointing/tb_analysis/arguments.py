# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--tb_dir", required=True, type=str, help="Directory for tensorboard output")
parser.add_argument("--analyzer", default="universal_checkpointing", type=str, choices=["universal_checkpointing"], help="Specify the analyzer to use")
parser.add_argument("--tb_event_key", required=False, default="lm-loss-training/lm loss", type=str, help="Optional override of the TensorBoard event key")
parser.add_argument("--plot_title", required=False, default="Megatron-GPT Universal Checkpointing", type=str, help="Optional override of the plot title")
parser.add_argument("--plot_x_label", required=False, default="Training Step", type=str, help="Optional override of the plot x-label")
parser.add_argument("--plot_y_label", required=False, default="LM Loss", type=str, help="Optional override of the plot y-label")
parser.add_argument("--plot_name", required=False, default="uni_ckpt_char.png", type=str, help="Optional override of the plot file name")
parser.add_argument("--skip_plot", action='store_true', help="Skip generation of plot file")
parser.add_argument("--skip_csv", action='store_true', help="Skip generation of csv files")
parser.add_argument("--use_sns", action='store_true', help="Use the SNS library to format plot")
parser.add_argument("--csv_name", required=False, default="", type=str, help="Unique name for CSV files")
parser.add_argument("--plot_only", action='store_true', help="Plot only using csv files")
parser.add_argument("--csv_dir", required=False, type=str, help="Directory for csv files")
