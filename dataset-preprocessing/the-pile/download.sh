#!/bin/bash
#SBATCH -J download_pile_dataset

export HF_DATASETS_CACHE="path" #TODO

cd PATH #TODO

python download.py