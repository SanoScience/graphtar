#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

python3 experiments/miraw/ann_test.py
