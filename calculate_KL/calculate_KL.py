import numpy as np
from scipy.stats import entropy
import argparse
from tqdm import tqdm

import sys
sys.path.append('../')
from populations.bbh_models import read_hdf5
import populations.bbh_models as read_models
from populations import gw_obs

#arguments
argp = argparse.ArgumentParser()
argp.add_argument("--channel-label", type=str, nargs="+", default="CE", help="")
argp.add_argument("--Neventsamps", type=int, default="11058", help="")
argp.add_argument("--no-samps", type=int, default="1000", help="")
argp.add_argument("--flow-path", type=str, default=None, help="Directory from where to load flow models. Default=None.")
argp.add_argument("--gw-path", type=str, default=None, help="Directory from where to load GW observations. Default=None.")

args = argp.parse_args()

params = ['mchirp','q', 'chieff', 'z']
no_params = len(params)
channels = args.channel_label
chi_b = [0.0,0.1,0.2,0.5]
alpha_CE = [0.2,0.5,1.0,2.,5.]
no_samples = args.no_samps

gw_path = args.gw_path
#_basepath='data/wiay/2297403c'
models_path =f'{_basepath}/models_reduced.hdf5'
np.random.seed(12)
observations, obsdata, p_theta, events = gw_obs.generate_observations(params, gw_path, \
                                        args.Neventsamps, 'posteriors', prior='p_theta_jcb')

for channel_label in channels:
    #read in popsynth models
    popsynth_outputs = read_hdf5(models_path, channel_label) # read all data from hdf5 file
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())
    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][params]
        weights_dict[key]= popsynth_outputs[key]['weight']

    #initialise flow and KDE models
    model_names, flow = read_models.get_models(models_path, [channel_label], params, use_flows=True, device='cuda:0')
    _, KDE = read_models.get_models(models_path, [channel_label], params, use_flows=False, device='cpu')

    flow_path=args.flow_path
    flow[channel_label].load_model(flow_path, device='cuda:0')

    #get model names for grabbing KDE models
    hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
    Nhyper = np.max([len(x.split('/')) for x in hyperparams])
    # construct dict that relates submodels to their index number
    submodels_dict = {} #dummy index dict keys:0,1,2,3, items: particular models
    ctr=0 #associates with either chi_b or alpha (0 or 1)
    while ctr < Nhyper:
        submodels_dict[ctr] = {}
        hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
        for idx, model in enumerate(hyper_set): #idx associates with 0,1,2,3,(4) keys
            submodels_dict[ctr][idx] = model
        ctr += 1

    if channel_label == 'CE':
        flow_KDE_KL = np.zeros((4,5))
        KDE_flow_KL = np.zeros((4,5))

        model_samples = np.zeros((4,5,no_samples,4))
        model_weights = np.zeros((4,5,no_samples))

        for chi_b_id, xb in enumerate(chi_b):
            for alpha_id, a in enumerate(alpha_CE):
                model_samples_idx = np.random.choice(np.shape(models_dict[(chi_b_id,alpha_id)])[0], no_samples)#, \
                    #p=weights_dict[(chi_b_id,alpha_id)]/np.sum(weights_dict[(chi_b_id,alpha_id)]))
                model_samples[chi_b_id,alpha_id,:,:] = np.array(models_dict[(chi_b_id,alpha_id)])[model_samples_idx]
                model_weights[chi_b_id,alpha_id,:] = np.array(weights_dict[(chi_b_id,alpha_id)])[model_samples_idx]

        print(f'cacluating flow||models and KDE||models given the model observations')
        flow_KL = np.zeros((4,5))
        KDE_KL = np.zeros((4,5))
        for chi_b_id, xb in enumerate(chi_b):
            for alpha_id, a in enumerate(alpha_CE):
                p_flow = flow[channel_label](np.reshape(model_samples[chi_b_id,alpha_id],(no_samples,1,4)), np.array([xb,np.log(a)]), 990903)
                flow_KL[chi_b_id,alpha_id] = -np.mean(p_flow)
                p_kde = KDE[channel_label][submodels_dict[0][chi_b_id]][submodels_dict[1][alpha_id]](np.reshape(model_samples[chi_b_id,alpha_id],(no_samples,1,4)), 990903)
                KDE_KL[chi_b_id,alpha_id] = -np.mean(np.log(p_kde))

        print(f'saving flow_KL-kde_KL for {channel_label}')
        np.save(f'/data/{channel_label}_flow_KL.npy', flow_KL)
        np.save(f'/data/{channel_label}_KDE_KL.npy', KDE_KL)
    else:
        flow_KDE_KL = np.zeros((4))
        KDE_flow_KL = np.zeros((4))
        model_samples = np.zeros((4,no_samples,4))
        model_weights = np.zeros((4,no_samples))

        for chi_b_id, xb in enumerate(chi_b):
            model_samples_idx = np.random.choice(np.shape(models_dict[(chi_b_id)])[0], no_samples)#, \
                #p=weights_dict[(chi_b_id,alpha_id)]/np.sum(weights_dict[(chi_b_id,alpha_id)]))
            model_samples[chi_b_id,:,:] = np.array(models_dict[(chi_b_id)])[model_samples_idx]
            model_weights[chi_b_id,:] = np.array(weights_dict[(chi_b_id)])[model_samples_idx]

        flow_KL = np.zeros((4))
        KDE_KL = np.zeros((4))
        for chi_b_id, xb in enumerate(chi_b):
            p_flow = flow[channel_label](np.reshape(model_samples[chi_b_id],(no_samples,1,4)), np.array([xb]), 990903)
            flow_KL[chi_b_id] = -np.mean(p_flow)
            p_kde = KDE[channel_label][submodels_dict[0][chi_b_id]](np.reshape(model_samples[chi_b_id],(no_samples,1,4)), 990903)
            KDE_KL[chi_b_id] = -np.mean(np.log(p_kde))

        print(f'saving flow_KL-kde_KL for {channel_label}')
        np.save(f'/data/{channel_label}_flow_KL.npy', flow_KL)
        np.save(f'/data/{channel_label}_KDE_KL.npy', KDE_KL)
