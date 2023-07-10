#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

# config_path, data_split_seed, lr, batch_size, epochs_num
python3 experiments/mitar/mitar_net.py data_modules/configs/mitar_config.json 1234 0.001 128 1000 experiments/mitar/models