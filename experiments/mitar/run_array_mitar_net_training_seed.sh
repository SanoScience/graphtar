#!/bin/bash

#SBATCH -J mitar_net
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH -A plgsano4-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="mitar_net.txt"
#SBATCH --array=418,627,960,426,16,523,708,541,747,897,714,515,127,657,662,284,595,852,734,136,394,321,200,502,786,817,411,264,929,407

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, data_split_seed, lr, batch_size, epochs_num
# deepmirtar BS=16
# miraw BS=64
# mirtarraw BS=64
python3 experiments/mitar/mitar_net.py data_modules/configs/mirtarraw_config.json $SLURM_ARRAY_TASK_ID 0.001 128 1000 experiments/mitar/models