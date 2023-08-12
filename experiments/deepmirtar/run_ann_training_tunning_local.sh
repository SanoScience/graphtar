#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

# config_path, model, data_split_seed, lr, batch_size, epochs_num, model_path
python3 experiments/deepmirtar/ann.py data_modules/configs/deepmirtar_config.json 1234 0.001 128 1000 experiments/deepmirtar/models