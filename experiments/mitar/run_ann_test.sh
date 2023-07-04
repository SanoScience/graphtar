#!/bin/bash

cd ../../

export PYTHONPATH="${PYTHONPATH}:../"

python3 experiments/mitar/mitar_net_test.py
