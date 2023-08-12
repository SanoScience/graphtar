#!/bin/bash

#SBATCH -J miraw_autoencoder
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=05:00:00
#SBATCH -A plgsano4-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="miraw_autoencoder.txt"
#SBATCH --array=16,32,64,128,256,512

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/miraw/autoencoder.py data_modules/configs/miraw_config.json 1234 0.001 $SLURM_ARRAY_TASK_ID 1000 experiments/miraw/models
