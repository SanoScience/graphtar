#!/bin/bash

#SBATCH -J venv_setup
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=1:00:00
#SBATCH -A plgsano4-cpu
#SBATCH -p plgrid-now
#SBATCH --output="output.txt"

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

rm -r venv
VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pytorch-lightning==1.5.10
pip3 install torchmetrics==0.8.2
pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip3 install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip3 install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip3 install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip3 install torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
