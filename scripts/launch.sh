#!/usr/bin/env bash
nohup python3 multitrain.py --config config_ultra_chaos_test01.csv --gpu_idx 3 > test01.out 2>&1 &
nohup python3 multitrain.py --config config_ultra_chaos_test02.csv --gpu_idx 2 > test02.out 2>&1 &
