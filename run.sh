#!/usr/bin/env bash
CSV=$1
nohup python3 multitrain.py --config $CSV --disable_cache > $CSV.out 2>&1&