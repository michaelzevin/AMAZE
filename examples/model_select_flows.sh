#! /bin/bash
#
# Runs model inference with normalising flows, training flow models and using these for continuous inference

seed=12

model_path="/PATH/TO/models_reduced.hdf5"
gw_path="/PATH/TO/gwevents"
flow_path="/PATH/TO/flow_models/"

python /PATH/TO/model_select\
        --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	    --verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT'\
        --spline-bins 5 4 4 5 4 \
        --epochs 15000 10000 10000 10000 10000 \
        --use-flows \
        --device 'cuda:0' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --prior 'p_theta_jcb' \
        --regularisation_N '990903' \
        --Nsamps 11058 \
        --name seed${seed} \
        --random-seed ${seed} \