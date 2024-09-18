#!/bin/bash
#SBATCH -J download_pile_dataset

export HF_DATASETS_CACHE="/N/scratch/jindjia/.cache/huggingface/datasets"

cd /N/slate/jindjia/bash_scripts/data_prepare/thepile

python download.py