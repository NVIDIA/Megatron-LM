#!/usr/bin/env python

# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

# this script converts results.json:
#
#   "results": {
#     "arc_challenge": {
#       "acc": 0.24232081911262798,
#       "acc_stderr": 0.01252159329580012,
#       "acc_norm": 0.2764505119453925,
#       "acc_norm_stderr": 0.013069662474252425
#     },
#
# into a format expected by a spreadsheet, which is:
#
#   task          metric   value    err
#   arc_challenge acc      xxx      yyy
#   arc_challenge acc_norm xxx      yyy
#   arc_challenge f1       xxx      yyy
#
# usage:
# report-to-csv.py results.json


import sys
import json
import io
import csv

results_file = sys.argv[1]

csv_file = results_file.replace("json", "csv")

print(f"Converting {results_file} to {csv_file}")

with io.open(results_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

with io.open(csv_file, 'w', encoding='utf-8') as f:

    writer = csv.writer(f)
    writer.writerow(["task", "metric", "value", "err", "version"])

    versions = results["versions"]

    for k,v in sorted(results["results"].items()):
        if k not in versions:
            versions[k] = -1

        if "acc" in v:
            writer.writerow([k, "acc", v["acc"], v["acc_stderr"], versions[k]])
        if "acc_norm" in v:
            writer.writerow([k, "acc_norm", v["acc_norm"], v["acc_norm_stderr"], versions[k]])
        if "f1" in v:
            writer.writerow([k, "f1", v["f1"], v["f1_stderr"] if "f1_stderr" in v else "", versions[k]])
        # if "ppl" in v:
        #     writer.writerow([k, "ppl", v["ppl"], v["ppl_stderr"], versions[k]])
        # if "em" in v:
        #     writer.writerow([k, "em", v["em"], v["em_stderr"] if "em_stderr" in v else "", versions[k]])
