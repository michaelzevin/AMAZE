#!/bin/bash

gw_path="/PATH/TO/gwevents"
flow_path="/PATH/TO/flow_models/"

python calculate_KL.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
    --gw-path ${gw_path} \
    --no-samps 10000