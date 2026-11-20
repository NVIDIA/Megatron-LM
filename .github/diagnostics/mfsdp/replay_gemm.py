# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Replay one captured grouped GEMM without a model or distributed processes."""

import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    directory = Path(sys.argv[1])
    manifest = json.loads((directory / "manifest.json").read_text())
    for key, value in manifest["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ["CUBLASLT_LOG_LEVEL"] = "5"
    os.environ["CUBLASLT_LOG_FILE"] = str(directory / "replay-cublaslt.log")

    import torch
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

    torch.use_deterministic_algorithms(manifest["deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = manifest["allow_tf32"]
    loaded = {}

    def load(entry):
        if entry["file"] not in loaded:
            raw = (directory / entry["file"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            loaded[entry["file"]] = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        tensor = loaded[entry["file"]].reshape(entry["shape"]).cuda()
        if list(tensor.stride()) != entry["stride"]:
            tensor = torch.empty_strided(
                entry["shape"], entry["stride"], dtype=torch.bfloat16, device="cuda"
            ).copy_(tensor)
        return tensor

    operands_a = [load(entry) for entry in manifest["A"]]
    operands_b = [load(entry) for entry in manifest["B"]]
    expected = [load(entry) for entry in manifest["outputs"]]
    rows = []
    for mode, indices in [("grouped", list(range(len(expected)))), ("single", [2])]:
        for repeat in range(3):
            outputs = [torch.empty_like(expected[index]) for index in indices]
            metadata = dict(manifest["metadata"])
            metadata["m_splits"] = [metadata["m_splits"][index] for index in indices]
            result = general_grouped_gemm(
                [operands_a[index] for index in indices],
                [operands_b[index] for index in indices],
                outputs,
                None,
                torch.bfloat16,
                **metadata,
            )[0]
            torch.cuda.synchronize()
            for index, actual in zip(indices, result):
                raw = actual.cpu().view(torch.uint8).numpy().tobytes()
                filename = f"replay-{mode}-{repeat}-expert{index}.bin"
                (directory / filename).write_bytes(raw)
                rows.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "expert_index": index,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "matches_training": raw
                        == (directory / manifest["outputs"][index]["file"]).read_bytes(),
                        "different_elements": int(
                            (actual.view(torch.int16) != expected[index].view(torch.int16))
                            .sum()
                            .item()
                        ),
                    }
                )
    report = {"gpu": torch.cuda.get_device_name(), "results": rows}
    (directory / "replay-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
