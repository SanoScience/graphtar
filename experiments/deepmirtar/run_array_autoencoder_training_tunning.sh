#!/bin/bash

#SBATCH -J deepmirtar_autoencoder
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=02:30:00
#SBATCH -A plgsano4-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="deepmirtar_autoencoder.txt"
#SBATCH --array=16,32,64,128,256,512
#17072,613,2169,2697,7488,5258,2687,2775,8604,4435,4886,9997,3162,7189,8246,4506,8778,8566,4481,2389,9906,7142,9830,6295,4277,7053,8532,705,641,809
module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, data_split_seed, lr, batch_size, epochs_num, model_path
python3 experiments/deepmirtar/sd_autoencoder.py data_modules/configs/deepmirtar_config.json 1234 0.001 $SLURM_ARRAY_TASK_ID 1000 experiments/deepmirtar/models
