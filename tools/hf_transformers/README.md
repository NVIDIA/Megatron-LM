# Conversion of Megatron checkpoints to HF transformers
These scripts support MQA.
Only supports 1-way tensor/pipeline parallelism for now (use `checkpoint_util` to merge checkpoints if needed).

Convert a megatron checkpoint to a HF-transformer checkpoint that can be directly pushed to the hub:
```
python -m tools.hf_transformers.convert_checkpoint --path_to_checkpoint /checkpoint_dir/iter_{num_iter}/mp_rank_00/model_optim_rng.pt --output-dir /checkpoint_dir/hf_checkpoints/iter_{num_iter}
```

Convert all checkpoints and push them to the hub:
```
python -m tools.hf_transformers.push_checkpoints --exp_dir /path/to/experiment --repo_name org/repo --branch_name main --iter_interval 20000"
```


