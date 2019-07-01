#!/usr/bin/env bash
CSV=$1
GPUs=$2
nohup python3 multitrain.py --config $CSV --gpu_idx $GPUs --disable_cache > $CSV.out 2>&1&