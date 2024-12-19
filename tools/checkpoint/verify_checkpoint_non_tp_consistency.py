# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from collections import namedtuple
from dataclasses import dataclass
import os
import re
import glob
import tqdm
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))


ParallelConfig = namedtuple('ParallelConfig', 'dp_degree tp_degree pp_degree')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", default=None, type=str, help="Checkpoint folder"
    )
    parser.add_argument(
        "--model_type",
        default="LLAMA",
        type=str,
        help="Type of the model",
        choices=["LLAMA"],
    )
    args = parser.parse_args()
    print(f"args = {args}")
    return args


class MLMCheckpoint:
    def __init__(self, folder, args) -> None:
        if hasattr(args, "use_dist_ckpt"):
            self.use_dist_ckpt = args.use_dist_ckpt
        if hasattr(args, "dist_ckpt_format"):
            self.dist_ckpt_format = args.dist_ckpt_format
        if hasattr(args, "use_distributed_optimizer"):
            self.use_distributed_optimizer = args.use_distributed_optimizer
        if not os.path.basename(folder).startswith("iter_"):
            filename = os.path.join(folder, "latest_checkpointed_iteration.txt")
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    latest_checkpointed_iteration = int(f.readline().rstrip())
                folder = os.path.join(folder, f"iter_{latest_checkpointed_iteration:07d}")
        self.ckpt_folder = folder
        if hasattr(args, "data_parallel_size") and hasattr(args, "tensor_model_parallel_size") and hasattr(args, "pipeline_model_parallel_size"):
            dp = args.data_parallel_size
            tp = args.tensor_model_parallel_size
            pp = args.pipeline_model_parallel_size
        else:
            files = glob.glob(os.path.join(folder, 'mp_rank_*'))
            if hasattr(args, "use_dist_ckpt"):
                self.use_dist_ckpt = len(files) == 0
            if not self.use_dist_ckpt:
                files = [os.path.basename(file).split("mp_rank_")[1].split("_") for file in files]
                degree_map = {}
                for file in files:
                    for i in range(len(file)):
                        if file[i] not in degree_map.keys():
                            degree_map[file[i]] = 1
                        else:
                            degree_map[file[i]] += 1
                inv_degree_map = {}
                for key, val in degree_map.items():
                    if val not in inv_degree_map.keys():
                        inv_degree_map[val] = []
                    inv_degree_map[val].append(key)
                tp = None
                pp = None
                for key, val in inv_degree_map.items():
                    if len(inv_degree_map[key]):
                        if len(val[0]) == 2:
                            tp = len(inv_degree_map[key])
                        if len(val[0]) == 3:
                            pp = len(inv_degree_map[key])
                dp = 'x'
        self.distrib_optim_filename = "distrib_optim.pt"
        self.model_optim_rng_filename = "model_optim_rng.pt"
        self.parallel_config = ParallelConfig(dp_degree=dp, tp_degree=tp, pp_degree=pp)

    def validate_files(self):
        if not self.use_dist_ckpt:
            for tensor_rank in range(self.parallel_config.tp_degree):
                for pipeline_rank in range(self.parallel_config.pp_degree):
                    if self.parallel_config.pp_degree != 1:
                        folder_path = os.path.join(self.ckpt_folder,
                            f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')
                    else:
                        folder_path = os.path.join(self.ckpt_folder,
                            f'mp_rank_{tensor_rank:02d}')
                    assert os.path.exists(folder_path), f"{folder_path=} does not exist, {self.parallel_config.pp_degree=}, {self.parallel_config.tp_degree=}"
                    files = os.listdir(folder_path)
                    # Filtering only the files.
                    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
                    num_files = 1
                    if self.use_distributed_optimizer:
                        num_files += 1
                    assert len(files) >= num_files
                    if self.use_distributed_optimizer:
                        assert self.distrib_optim_filename in files
                    assert self.model_optim_rng_filename in files


def show_3d(mlm_checkpoint):
    parallel_config = mlm_checkpoint.parallel_config
    dp, tp, pp = parallel_config.dp_degree, parallel_config.tp_degree, parallel_config.pp_degree
    print(f"3D configuration: DP={dp} TP={tp} PP={pp}")


def get_model_optim_rng_patterns_for_non_sharded(model_type):
    if model_type == "LLAMA":
        return [
            r"embedding.word_embeddings.bias",
            r"embedding.position_embeddings.weight",
            r"decoder.layers.+\d+.input_layernorm.weight",
            r"decoder.layers.+\d+.input_layernorm.bias",
            r"decoder.layers.+\d+.self_attention.linear_qkv.bias",
            r"decoder.layers.+\d+.self_attention.linear_proj.bias",
            r"decoder.layers.+\d+.pre_mlp_layernorm.weight",
            r"decoder.layers.+\d+.pre_mlp_layernorm.bias",
            r"decoder.layers.+\d+.mlp.linear_fc1.bias",
            r"decoder.layers.+\d+.mlp.linear_fc2.bias",
            r"decoder.final_layernorm.weight",
            r"decoder.final_layernorm.bias",
        ]


@dataclass
class ParamInfo:
    pp: int
    tp: int
    dp: int
    data: torch.Tensor
    numel: int


def verify_equal_params(params, tp):
    failed = 0
    report = {}
    for name, info in params.items():
        n = len(info)
        if n != tp:
            ok = False
            print(f"{name}: FAILED expected n={n} == tp={tp}")
        elif n == 1:
            ok = True
        else:
            ok = all([(x.numel == info[0].numel) for x in info[1:]])
            if not ok:
                print(f"{name}: FAILED numel comparison [n={n}]")
            else:
                ok = all([x.data.eq(info[0].data).all().item() for x in info[1:]])
                if not ok:
                    print(f"{name}: FAILED data comparison [n={n}]")
        failed += ok == False
        report[name] = (ok, n)
        if ok:
            print(f"{name}: OK [n={n}]")
    return failed, report


def update_model_optim_rng_non_sharded_params(params, model_type, filename, pp_index, tp_index):
    sd = torch.load(filename, map_location=torch.device("cpu"))['model']
    model_optim_rng_patterns = get_model_optim_rng_patterns_for_non_sharded(model_type)
    for key in sd.keys():
        if not any(re.match(model_optim_rng_pattern, key) for model_optim_rng_pattern in model_optim_rng_patterns):
            continue
        if key not in params:
            params[key] = []
        info = ParamInfo(
            pp=pp_index, tp=tp_index, dp=-1, data=sd[key], numel=sd[key].numel()
        )
        params[key].append(info)
    return params


def verify_model_optim_rng_files(mlm_checkpoint, model_type):
    parallel_config = mlm_checkpoint.parallel_config
    tp, pp = parallel_config.tp_degree, parallel_config.pp_degree

    total_failed = 0
    if not mlm_checkpoint.use_dist_ckpt:
        for pp_index in range(pp):
            print(f"\nChecking pp_stage={pp_index}")
            params = {}
            for tp_index in range(tp):
                if pp != 1:
                    folder = os.path.join(mlm_checkpoint.ckpt_folder, f'mp_rank_{tp_index:02d}_{pp_index:03d}')
                else:
                    folder = os.path.join(mlm_checkpoint.ckpt_folder, f'mp_rank_{tp_index:02d}')
                filename = os.path.join(folder, mlm_checkpoint.model_optim_rng_filename)
                update_model_optim_rng_non_sharded_params(
                    params, model_type, filename, pp_index, tp_index
                )
            failed, report = verify_equal_params(params, tp)
            total_failed += failed
    return total_failed


def verify_checkpoint(folder, model_type, args=None):
    mlm_checkpoint = MLMCheckpoint(folder, args=args)
    mlm_checkpoint.validate_files()
    show_3d(mlm_checkpoint)

    if not mlm_checkpoint.use_dist_ckpt:
        print("\nVerify ** model_optim_rng ** files")
    total_failed_model_optim_rng = verify_model_optim_rng_files(mlm_checkpoint, model_type)
    if total_failed_model_optim_rng == 0:
        if not mlm_checkpoint.use_dist_ckpt:
            print("\nCheckpoint model_optim_rng files OK")
    else:
        if not mlm_checkpoint.use_dist_ckpt:
            print(f"\nCheckpoint model_optim_rng files BAD with total_failed={total_failed_model_optim_rng}")

    # TODO: if possible find/explore a way to verify distributed optimizer ckpt also

    return total_failed_model_optim_rng == 0


def main():
    print(f"Verify Checkpoint consistency for non-TP-sharded parameters")
    args = parse_arguments()
    assert (
        verify_checkpoint(args.folder, args.model_type, args) is True
    ), "Checkpoint verification failed"


if __name__ == "__main__":
    main()
