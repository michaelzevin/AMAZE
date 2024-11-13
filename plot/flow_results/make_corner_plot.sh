flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_project_rescources/rns/Flows_071024_trainNSC/flow_models/"

python make_corner_plot.py \
    --flow-path ${flow_path} \
    --channel-label 'NSC' \
    --hyperparam-idxs 1 \
    --conditional 0.1 \