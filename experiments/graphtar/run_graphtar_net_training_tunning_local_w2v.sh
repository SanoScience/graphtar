#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

# config_path, gnn_layer_type (GCN, GRAPHSAGE, GAT), global_pooling (MAX,MEAN,ADD), gnn_hidden_size, n_gnn_layers, fc_hidden_size, n_fc_layers, dropout_rate, data_split_seed, lr, batch_size, epochs_num, model_dir
python3 experiments/graphtar/gnn_w2v.py data_modules/configs/graphtar_config_deepmirtar_w2v.json GAT ADD 64 2 64 2 0.4 1234 0.001 128 1000 experiments/graph/models
