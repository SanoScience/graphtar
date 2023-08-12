#!/bin/bash

#SBATCH -J miraw_autoencoder
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=05:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="miraw_autoencoder.txt"
#SBATCH --array=418,627,960,426,16,523,708,541,747,897,714,515,127,657,662,284,595,852,734,136,394,321,200,502,786,817,411,264,929,407

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, data_split_seed, lr, batch_size, epochs_num, model_dir
# deepmirtar BS=128
# miraw BS=64
# mirtarraw BS=128
python3 experiments/miraw/autoencoder.py data_modules/configs/miraw_config.json $SLURM_ARRAY_TASK_ID 0.001 16 1000 experiments/miraw/models
