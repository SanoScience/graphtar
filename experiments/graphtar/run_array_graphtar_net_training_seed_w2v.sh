#!/bin/bash

#SBATCH -J graphtar_net_gat
#SBATCH -N 1
#SBATCH --tasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH -A plgsano4-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output="graphtar_net.txt"
#SBATCH --array=418,627,960,426,16,523,708,541,747,897,714,515,127,657,662,284,595,852,734,136,394,321,200,502,786,817,411,264,929,407

module add .plgrid
module add plgrid/apps/cuda/11.1.1-gcc-10.2.0
module add plgrid/tools/python/3.8.6-gcccore-10.2.0

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

source ./venv/bin/activate

# GCN MAX, GAT AND SAGE ADD
# N_GNN_LAYERS GAT: 5, SAGE:5, GCN: 3
# N_FC_LAYERS:   GAT: 2 , SAGE: 3, GCN: 2
# GNN_HIDDEN_SIZE: GAT: 256, SAGE: 256, GCN: 128
# FC_HIDDEN_SIZE: GAT: 128, SAGE: 256, GCN: 512
# BATCH_SIZE: GAT: 128, SAGE: 128, GCN: 128 
# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
# BATCH_SIZE: Deepmirtar: GAT: 128, SAGE: 128, GCN: 128 
#             miraw: GAT: 512, SAGE: 32, GCN: 64 
#               mitarraw: GAT: 512, SAGE: 256, GCN: 64

python3 experiments/graphtar/gnn_w2v.py data_modules/configs/graphtar_config_mirtarraw_w2v.json GAT ADD 256 5 128 2 0.4 $SLURM_ARRAY_TASK_ID 0.001 512 1000 experiments/graphtar/models
