#!/bin/bash

#SBATCH -J deepmirtar_ann
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=02:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="deepmirtar_ann.txt"
#SBATCH --array=418,627,960,426,16,523,708,541,747,897,714,515,127,657,662,284,595,852,734,136,394,321,200,502,786,817,411,264,929,407

module load plgrid/apps/cuda/11.3

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# config_path, model, data_split_seed, lr, batch_size, epochs_num, model_path
# deepmirtar BS=512
# miraw BS=512
# mirtarraw BS=256
python3 experiments/deepmirtar/ann.py data_modules/configs/deepmirtar_config.json $SLURM_ARRAY_TASK_ID 0.001 512 1000 experiments/deepmirtar/models
