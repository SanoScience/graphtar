#!/bin/bash

#SBATCH -J mitar_net
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=04:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="mitar_net.txt"

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, data_split_seed, lr, batch_size, epochs_num
python3 experiments/mitar/mitar_net.py data_modules/configs/mitar_config.json 1234 0.001 128 1000 experiments/mitar/models
