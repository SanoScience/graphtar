#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

# config_path, data_split_seed, lr, batch_size, epochs_num
python3 experiments/miraw/ann.py data_modules/configs/miraw_config.json 1234 0.001 128 1000 experiments/miraw/models