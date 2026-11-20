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

"""Run one controlled training arm; never edit production source."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path('/workspace')
request = json.loads(Path(sys.argv[1]).read_text())
label = request['label']
result = root / 'results' / label
steps = request.get('steps', 4)
os.environ.update(
    HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',
    HF_HOME='/workspace/cache/hf',
    UV_PROJECT_ENVIRONMENT='/opt/venv',
    UV_CACHE_DIR='/workspace/cache/uv',
    PYTHONPATH='/opt/megatron-lm',
    PYTHONDONTWRITEBYTECODE='1',
    PYTHONUNBUFFERED='1',
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    RECORD_CHECKPOINTS='false',
)
for key in ['CUDA_DEVICE_MAX_CONNECTIONS', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS']:
    os.environ.pop(key, None)
for key, value in request.get('env', {}).items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)
with (result / 'runtime.json').open('w') as output:
    subprocess.run([sys.executable, str(root / 'runtime_probe.py')], stdout=output, check=True)
config = yaml.safe_load((root / 'model_config.yaml').read_text())
config['MODEL_ARGS']['--exit-interval'] = steps
for key, value in request.get('model_args', {}).items():
    if value is None:
        config['MODEL_ARGS'].pop(key, None)
    else:
        config['MODEL_ARGS'][key] = value
if request.get('trace'):
    os.environ['COLLECTIVE_TRACE_DIR'] = str(result / 'collectives')
    script = '/workspace/trace_pretrain.py'
else:
    script = 'pretrain_gpt.py'
config_path = result / 'model_config.yaml'
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
(root / 'empty-checkpoints').mkdir(exist_ok=True)
args = {
    'DATA_PATH': '/mnt/artifacts',
    'DATA_CACHE_PATH': '/workspace/cache/data',
    'OUTPUT_PATH': str(result),
    'TENSORBOARD_PATH': str(result / 'tensorboard'),
    'CHECKPOINT_SAVE_PATH': str(result / 'checkpoints'),
    'CHECKPOINT_LOAD_PATH': '/workspace/empty-checkpoints',
    'TRAINING_SCRIPT_PATH': script,
    'TRAINING_PARAMS_PATH': str(config_path),
    'REPEAT': '1',
    'RUN_NUMBER': '1',
    'IS_NEMO_TEST': 'false',
    'RECORD_CHECKPOINTS': 'false',
    'GPUS_PER_NODE': '4',
    'NUM_NODES': '1',
}
started = time.monotonic()
with (result / 'training.log').open('w') as log:
    process = subprocess.run(
        [
            'bash',
            'tests/functional_tests/shell_test_utils/_run_training.sh',
            *[f'{key}={value}' for key, value in args.items()],
        ],
        cwd='/opt/megatron-lm',
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
report = dict(
    request,
    training_exit_code=process.returncode,
    elapsed_seconds=time.monotonic() - started,
    metrics={},
)
if (result / 'tensorboard').exists():
    events = EventAccumulator(str(result / 'tensorboard'), size_guidance={'scalars': 0})
    events.Reload()
    for name in ['lm loss', 'num-zeros', 'mem-allocated-bytes']:
        if name in events.Tags()['scalars']:
            report['metrics'][name] = {
                str(event.step): event.value for event in events.Scalars(name)
            }
loss = report['metrics'].get('lm loss', {})
report['complete'] = process.returncode == 0 and all(
    str(step) in loss and math.isfinite(loss[str(step)]) for step in range(1, steps + 1)
)
if report['complete']:
    for name in ['github']:
        expected = json.loads((root / f'reference-{name}.json').read_text())['lm loss']['values']
        report[f'mismatches_vs_{name}'] = [
            step for step in range(1, steps + 1) if loss[str(step)] != expected[str(step)]
        ]
(result / 'result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
sys.exit(0 if report['complete'] else 1)
