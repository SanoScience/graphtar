#!/bin/bash

#SBATCH -J graphtar_net
#SBATCH -N 1
#SBATCH --tasks-per-node=3
#SBATCH --time=12:00:00
#SBATCH -A plgsano3
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="graphtar_net.txt"
#SBATCH --array=418,627,960,426,16,523,708,541,747,897,714,515,127,657,662,284,595,852,734,136,394,321,200,502,786,817,411,264,929,407

module load plgrid/apps/cuda/11.3

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

VENV_DIR="./venv"
[ ! -d $VENV_DIR ] && python3 -m venv venv

source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# config_path, data_split_seed, lr, batch_size, epochs_num
# deepmirtar BS=x
# miraw BS=x
# mirtarraw BS=x
# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/graphtar/gnn_w2v.py data_modules/configs/graphtar_config_deepmirtar_w2v.json GCN MAX 64 2 64 2 0.4 $SLURM_ARRAY_TASK_ID 0.001 128 1000 experiments/graph/models
