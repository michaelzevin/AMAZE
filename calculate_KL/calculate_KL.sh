#!/bin/bash

gw_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/GWTC-3/events"
flow_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/flow_models/mixed_models/"

/data/wiay/2297403c/conda_envs/amaze/bin/python calculate_KL.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
    --gw-path ${gw_path} \
    --no-samps 10000