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

Each method should take around 6 minutes to complete.

Then run this to collect the results:
```shell
bash artifact/quick_exp.sh show-result
```

The data should show similar result as:
<details>
<summary>Click to expand result</summary>


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

</details>

### Full Experiment
This will run all experiments on single server with 8 A100 GPUs.

The whole experiment will take around 3 hours to complete.
```shell
bash artifact/full_exp.sh artifact/exp_one_host.csv
```

Print results:
```shell
python artifact/show_result_full_exp.py
```

The result should roughly match the following:
<details>
<summary>Click to expand result</summary>

```
Seq Length Vocab Size     Method    MFU  Peak Memory (GB)
        2k        32k   baseline 47.52%             16.01
        2k        32k      redis 47.52%             16.01
        2k        32k    vocab-2 52.56%             15.87
        2k        32k    vocab-1 53.10%             16.64
        2k        32k interlaced 54.01%             17.87
---------------------------------------------------------
        2k        64k   baseline 40.82%             17.74
        2k        64k      redis 49.15%             17.74
        2k        64k    vocab-2 52.36%             16.64
        2k        64k    vocab-1 52.74%             17.48
        2k        64k interlaced 53.81%             18.63
---------------------------------------------------------
        2k       128k   baseline 33.70%             21.00
        2k       128k      redis 44.12%             21.00
        2k       128k    vocab-2 51.95%             17.56
        2k       128k    vocab-1 52.22%             18.40
        2k       128k interlaced 53.55%             19.70
---------------------------------------------------------
        2k       256k   baseline 17.29%             35.56
        2k       256k      redis 38.21%             27.63
        2k       256k    vocab-2 51.20%             18.85
        2k       256k    vocab-1 51.73%             19.44
        2k       256k interlaced 53.14%             21.15
---------------------------------------------------------
        4k        32k   baseline 52.82%             22.23
        4k        32k      redis 52.81%             22.23
        4k        32k    vocab-2 58.23%             23.39
        4k        32k    vocab-1 58.52%             24.97
        4k        32k interlaced 59.39%             27.38
---------------------------------------------------------
        4k        64k   baseline 45.78%             23.98
        4k        64k      redis 53.81%             23.98
        4k        64k    vocab-2 57.94%             23.14
        4k        64k    vocab-1 58.22%             24.58
        4k        64k interlaced 59.01%             27.43
---------------------------------------------------------
        4k       128k   baseline 37.13%             27.22
        4k       128k      redis 49.56%             27.22
        4k       128k    vocab-2 57.40%             24.56
        4k       128k    vocab-1 57.87%             26.10
        4k       128k interlaced 58.61%             29.04
---------------------------------------------------------
        4k       256k   baseline 27.63%             33.86
        4k       256k      redis 42.45%             33.72
        4k       256k    vocab-2 56.61%             26.25
        4k       256k    vocab-1 57.31%             27.38
        4k       256k interlaced 58.04%             30.72
```

</details>

The result should also roughly match the 2 rows in `Table 5. Comparison of Methods on 1F1B` in the paper:
- 8GPU, SEQ LENGTH 2048
- 8GPU, SEQ LENGTH 4096
