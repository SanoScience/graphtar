#!/bin/bash

#SBATCH -J graphtar_net_graphsage_add
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH -A plgsano4-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="graphtar_net.txt"
#SBATCH --array=16,32,64,128,256,512

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/graphtar/gnn.py data_modules/configs/graphtar_config_deepmirtar.json GRAPHSAGE ADD $SLURM_ARRAY_TASK_ID 7 64 4 0.4 1234 0.001 128 1000 experiments/graphtar/models
