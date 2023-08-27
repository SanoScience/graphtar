#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:./"

python3 experiments/graphtar/gnn_w2v_test_ablation.py
