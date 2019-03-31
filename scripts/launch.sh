#!/usr/bin/env bash
nohup python3 multitrain.py --config config_farm_tests_softplus.csv --gpu_idx 1 > softplus.out 2>&1 &
