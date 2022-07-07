#!/bin/bash

#SBATCH -J deepmirtar
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=01:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="deepmirtar.txt"

module load plgrid/apps/cuda/11.3

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# config_path, data_split_seed, lr, batch_size, epochs_num
python3 experiments/training/deepmirtar/sd_autoencoder.py "../../data_modules/configs/deepmirtar_config.json" 1234 0.001 128 10
