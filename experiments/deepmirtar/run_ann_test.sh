#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

python3 experiments/deepmirtar/ann_test.py
