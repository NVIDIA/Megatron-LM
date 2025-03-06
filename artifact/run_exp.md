# Artifact Evaluation
This documents provides the guides to reproduce the experiment in the paper
**Balancing Pipeline Parallelism with Vocabulary Parallelism**.

The code repository can be found in [VocabularyParallelism](https://github.com/sail-sg/VocabularyParallelism).

This evaluation consists of 2 parts:
- **Quick Experiment** to quickly verify the result on a specific case.
- **Full Experiment** to run all cases on a 8-GPU server.

### Environment Setup
Run a container:
```shell
docker run --name vocab_torch24 \
    --network=host -d \
    --runtime=nvidia --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --privileged=true \
    nvcr.io/nvidia/pytorch:24.03-py3 sleep infinity
```

Get inside the container, clone the codes:
```shell
# Enter the container
docker exec -it vocab_torch24 bash
# Clone the codes
git clone https://github.com/sail-sg/VocabularyParallelism.git
cd VocabularyParallelism
```

Note that all the following commands should be run inside the `VocabularyParallelism` directory.

### Quick Experiment
The quick experiment runs all the methods (*baseline*, *redis*, *interlaced*, *vocab-1*, *vocab-2*) on a specific setting in the paper:
- Sequence Length: 4096
- Vocabulary Size: 256k

The experiment will show 2 key results:
- **Peak Memory**
- **MFU**

Run all the methods one by one:
```shell
bash artifact/quick_exp.sh run baseline
bash artifact/quick_exp.sh run redis
bash artifact/quick_exp.sh run interlaced
bash artifact/quick_exp.sh run vocab-1
bash artifact/quick_exp.sh run vocab-2
```

This will automatically download the dataset from huggingface and run the training experiments.
The log containing the result will locate in `quick-logs/<method>/stdout.log`.

Each method should take a few minutes to complete.

Then run this to collect the results:
```shell
bash artifact/quick_exp.sh show-result
```

The data should show similar result as:
```
Method: baseline
Peak Memory: 33.7246 GB
MFU: 26.1384 %

Method: redis
Peak Memory: 33.7227 GB
MFU: 44.6888 %

Method: interlaced
Peak Memory: 31.207 GB
MFU: 42.5455 %

Method: vocab-1
Peak Memory: 27.3848 GB
MFU: 54.0276 %

Method: vocab-2
Peak Memory: 26.1094 GB
MFU: 53.5131 %
```

### Full Experiment
This will run all experiments on single server with 8 A100 GPUs.

The whole experiment will take around 3 hours to complete.
```shell
bash artifact/full_exp.sh artifact/exp_one_host.csv
```
