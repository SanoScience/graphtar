#!/bin/bash

#SBATCH -J graphtar_net_sage_add
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=12:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="graphtar_net.txt"

module load plgrid/apps/cuda/10.2

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/graphtar/gnn.py data_modules/configs/graphtar_config_deepmirtar.json GRAPHSAGE ADD 64 2 64 2 0.4 1234 0.001 128 1000 experiments/graph/models
