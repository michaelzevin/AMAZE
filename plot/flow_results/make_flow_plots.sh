#!/bin/bash
flow_path="/PATH_TO/flow_models/"

python make_paper_plots.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' \
    --hyperparam-idxs 0 3 \
    --conditional 0.0 2.0 \
    --plot-flow-corner \
    --plot-dataspace-result \
    --justplot \
    --plot-KDE \
    --KDE-result-path '/PATH/TO/KDE_results/' \
    --discrete-result-path '/PATH/TO/discrete_flow_results/' \
    --cont-result-path '/PATH/TO/continuous_flow_results/' \
    --outdir '/pdfs' \