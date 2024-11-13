import matplotlib.pyplot as plt
import corner as corner
import pandas as pd
import numpy as np
import h5py
import torch
import time
from scipy.special import logsumexp
import matplotlib.cm as cm
from pylab import rcParams

import sys
sys.path.append('../../../')
import populations.bbh_models as read_models
from populations import gw_obs
from populations.bbh_models import read_hdf5
from sample import sample

_chi_b = [0.0,0.1,0.2,0.5]
_alpha = [0.2,0.5,1.,2.,5.]

def load_models(channel, params, no_bins, flow_path, use_unityweights):
    #get pop synth samples and GW observations
    file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
    gw_path = '/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events'
    observations, obsdata, p_theta, events = gw_obs.generate_observations(params, gw_path, \
                                                1, 'posteriors', None)
    popsynth_outputs = read_hdf5(file_path, channel)

    model_names, flow = read_models.get_models(file_path, [channel], params, use_flows=True, use_unityweights=use_unityweights, device='cpu', no_bins=[no_bins])
    model_names, KDE = read_models.get_models(file_path, [channel], params, use_flows=False, use_unityweights=use_unityweights, device='cpu')

    #inputs: x, data, pop_models, submodels_dict, channels, use_flows
    hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
    Nhyper = np.max([len(x.split('/')) for x in hyperparams])
    channels = sorted(list(set([x.split('/')[0] for x in model_names])))

    # construct dict that relates submodels to their index number
    submodels_dict = {} #dummy index dict keys:0,1,2,3, items: particular models
    ctr=0 #associates with either chi_b or alpha (0 or 1)
    while ctr < Nhyper:
        submodels_dict[ctr] = {}
        hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
        for idx, model in enumerate(hyper_set): #idx associates with 0,1,2,3,(4) keys
            submodels_dict[ctr][idx] = model
        ctr += 1

    #load trained flow
    flow[channel].load_model(f'{flow_path}/flow_models/', channel)

    return popsynth_outputs, obsdata, p_theta, flow, KDE, submodels_dict

def calc_llh_ratio_nonCE(channel, obsdata, p_theta, flow, KDE, submodels_dict, use_reg):
    

    llh_ratio_kde_flow = np.zeros((4,obsdata.shape[0]))
    lnlike_flow_theta = np.zeros((4,obsdata.shape[0]))

    for chibid in range(4):
        for i,obs in enumerate(obsdata):
            obs = np.reshape(obs, (1,1,4))
            lnlike_flow_theta[chibid,i] = sample.lnlike([chibid], obs, flow, submodels_dict, [channel], use_flows=True, prior_pdf=np.array([p_theta[i]]))
            lnlike_kde_theta = sample.lnlike([chibid],obs, KDE, submodels_dict, [channel], use_flows=False, prior_pdf=[p_theta[i]])
            llh_ratio_kde_flow[chibid,i] = lnlike_flow_theta[chibid,i]-lnlike_kde_theta
    return llh_ratio_kde_flow

def plot_llh_ratio_nonCE(file_path, flow_directory, channel, flow_bins, use_unityweights, use_reg, params = ['mchirp','q', 'chieff', 'z']):
    
    popsynth_outputs, obsdata, p_theta, flow, KDE, submodels_dict = load_models(channel, params, flow_bins, flow_directory, use_unityweights)
    llh_ratio_kde_flow = calc_llh_ratio_nonCE(channel, obsdata, p_theta, flow, KDE, submodels_dict, use_reg)
    

    plt.rcParams["figure.figsize"] = (10,18)
    fig, ax = plt.subplots(4,2)

    for chibid in range(4):

        p_theta = corner.hist2d(np.array(popsynth_outputs[chibid]['mchirp']), np.array(popsynth_outputs[chibid]['q']), zorder=-999, \
            weights=popsynth_outputs[chibid]['weight'], color='grey', plot_density=True, ax=ax[chibid,0])
        ax[chibid,0].scatter(obsdata[:,:,0],obsdata[:,:,1], c=llh_ratio_kde_flow[chibid], cmap=cm.magma, label=fr'$\chi_b$={_chi_b[chibid]}', zorder=200)
        ax[chibid,0].set_xlabel(r'$\mathcal{M}_{chirp} /M_{\odot}$')
        ax[chibid,0].set_ylabel(r'$q$')
        ax[chibid,0].set_ylim(0.1,1)
        ax[chibid,0].set_xlim(0,65)
        ax[chibid,0].legend()
        corner.hist2d(np.array(popsynth_outputs[chibid]['chieff']), np.array(popsynth_outputs[chibid]['z']), bins=50, zorder=-999,\
            weights=popsynth_outputs[chibid]['weight'], color='grey', plot_density=True, ax=ax[chibid,1])
        sp=ax[chibid,1].scatter(obsdata[:,:,2],obsdata[:,:,3], c=llh_ratio_kde_flow[chibid], cmap=cm.magma, label=fr'$\chi_b$={_chi_b[chibid]}', zorder=200)
        ax[chibid,1].set_ylim(0.,1)
        ax[chibid,1].set_xlim(-0.8,1.0)
        ax[chibid,1].set_xlabel(r'$\chi_{eff}$')
        ax[chibid,1].set_ylabel(r'$z$')
        ax[chibid,1].legend()
        #fig.colorbar(ax=ax[chibid,1], label='log(ptheta_flow)-log(ptheta_kde)')
        cbar = fig.colorbar(sp,ax=ax[chibid,1])
        cbar.set_label(fr'log(p(x|{channel},flow)-log(p(x|{channel},KDE))')

        fig.tight_layout(pad=1.3)
        fig.savefig(f'{file_path}/plots/llh_ratio/{channel}_KDE_flow_ratio.pdf')

def calc_llh_ratio_CE(channel, obsdata, p_theta, flow, KDE, submodels_dict, use_reg):

    llh_ratio_kde_flow = np.zeros((4,5,obsdata.shape[0]))
    for chibid in range(4):
        for alphaid in range(5):
            for i,obs in enumerate(obsdata):
                obs = np.reshape(obs, (1,1,4))
                lnlike_flow_theta=sample.lnlike([chibid,alphaid],obs, flow, submodels_dict, [channel], use_flows=True, use_reg=use_reg, prior_pdf=np.array([p_theta[i]]))
                lnlike_kde_theta=sample.lnlike([chibid,alphaid],obs, KDE, submodels_dict, [channel], use_flows=False, use_reg=use_reg, prior_pdf=[p_theta[i]])
                llh_ratio_kde_flow[chibid,alphaid,i] = lnlike_flow_theta-lnlike_kde_theta
    return llh_ratio_kde_flow

def plot_llh_ratio_CE(file_path, flow_directory, channel, flow_bins, use_unityweights, use_reg, params = ['mchirp','q', 'chieff', 'z']):

    popsynth_outputs, obsdata, p_theta, flow, KDE, submodels_dict = load_models(channel, params, flow_bins, flow_directory, use_unityweights)
    llh_ratio_kde_flow = calc_llh_ratio_CE(channel, obsdata, p_theta, flow, KDE, submodels_dict, use_reg)
    
    plt.rcParams["figure.figsize"] = (10,18)
    for alphaid in range(5):
        fig, ax = plt.subplots(4,2)
        for chibid in range(4):

            p_theta = corner.hist2d(np.array(popsynth_outputs[chibid,alphaid]['mchirp']), np.array(popsynth_outputs[chibid,alphaid]['q']), bins=50,\
            weights=popsynth_outputs[chibid, alphaid]['weight'], color='grey', plot_density=True, ax=ax[chibid,0])
            ax[chibid,0].scatter(obsdata[:,:,0],obsdata[:,:,1], c=llh_ratio_kde_flow[chibid,alphaid], cmap=cm.cividis, label=fr'$\chi_b$={_chi_b[chibid]},$\alpha$={_alpha[alphaid]}')
            ax[chibid,0].set_xlabel(r'$\mathcal{M}_{chirp} /M_{\odot}$')
            ax[chibid,0].set_ylabel(r'$q$')
            ax[chibid,0].set_ylim(0.1,1)
            ax[chibid,0].set_xlim(0,65)
            ax[chibid,0].legend()
            corner.hist2d(np.array(popsynth_outputs[chibid,alphaid]['chieff']), np.array(popsynth_outputs[chibid,alphaid]['z']), bins=50,\
            weights=popsynth_outputs[chibid, alphaid]['weight'], color='grey', plot_density=True, ax=ax[chibid,1])
            sp=ax[chibid,1].scatter(obsdata[:,:,2],obsdata[:,:,3], c=llh_ratio_kde_flow[chibid,alphaid], cmap=cm.cividis, label=fr'$\chi_b$={_chi_b[chibid]},$\alpha$={_alpha[alphaid]}')
            ax[chibid,1].set_ylim(0.,1)
            ax[chibid,1].set_xlim(-0.5,0.9)
            ax[chibid,1].set_xlabel(r'$\chi_{eff}$')
            ax[chibid,1].set_ylabel(r'$z$')
            ax[chibid,1].legend()
            #fig.colorbar(ax=ax[chibid,1], label='log(ptheta_flow)-log(ptheta_kde)')
            cbar = fig.colorbar(sp,ax=ax[chibid,1])
            cbar.set_label(fr'log(p(x|{channel},flow)-log(p(x|{channel},KDE))')

        fig.tight_layout(pad=1.3)
        fig.savefig(f'{file_path}/plots/llh_ratio/{channel}_llhratio_alpha{alphaid}.pdf')

def plot1Dsamps_nonCE(file_path, flow_directory, channel, flow_bins, use_unityweights, params = ['mchirp','q', 'chieff', 'z']):

    popsynth_outputs, obsdata, p_theta, flow, KDE, submodels_dict = load_models(channel, params, flow_bins, flow_directory, use_unityweights)
    
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())


    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][params]
        if use_unityweights:
            weights_dict[key] = np.ones(popsynth_outputs[key]['weight'].shape)
        else:
            weights_dict[key]= popsynth_outputs[key]['weight']

    plt.rcParams["figure.figsize"] = (15,5)
    no_samples = 100000
    no_bins = 60

    fig_mchirp, ax_m = plt.subplots(1,4)
    fig_q, ax_q = plt.subplots(1,4)
    fig_c, ax_c = plt.subplots(1,4)
    fig_z, ax_z = plt.subplots(1,4)

    for chi_b_id, xb in enumerate(_chi_b):
        flow_samples_stack = flow[channel].flow.sample(np.array([xb]),no_samples)
        flow_mchirp = flow[channel].expistic(flow_samples_stack[:,0], flow[channel].mappings[0], flow[channel].mappings[1])
        flow_q = flow[channel].expistic(flow_samples_stack[:,1], flow[channel].mappings[2])
        flow_chieff = np.tanh(flow_samples_stack[:,2])
        flow_z = flow[channel].expistic(flow_samples_stack[:,3], flow[channel].mappings[4], flow[channel].mappings[5])

        mapped_flow_samples = [flow_mchirp,flow_q,flow_chieff,flow_z]

        kde_samples = KDE[channel][submodels_dict[0][chi_b_id]].sample(no_samples)

        for i, ax in enumerate([ax_m,ax_q,ax_c,ax_z]):
            flow_distr, bin_edges = np.histogram(mapped_flow_samples[i], bins=no_bins, density=True)
            kde_distr, bin_edges_KDE = np.histogram(kde_samples[:,i], bins=no_bins, density=True)
            known_distr, bin_edges_known =np.histogram(models_dict[chi_b_id][:][params[i]], bins=no_bins, density=True, weights=weights_dict[chi_b_id])
            ax[chi_b_id].step(np.linspace(bin_edges[0],bin_edges[-1],no_bins),flow_distr, label='flow')
            ax[chi_b_id].step(np.linspace(bin_edges_KDE[0],bin_edges_KDE[-1],no_bins),kde_distr,label='KDE', color='purple')
            ax[chi_b_id].step(np.linspace(bin_edges_known[0],bin_edges_known[-1],no_bins),known_distr,label='underlying')
            ax[chi_b_id].set_title(fr'$\chi_b$={xb}')
            ax[chi_b_id].set_xlabel(f'{params[i]}')
            ax[chi_b_id].set_ylabel(f'p({params[i]})')
            #ax[chi_b_id].set_yscale('log')
            ax[chi_b_id].legend()
            plt.tight_layout(pad=1.3)
        fig_mchirp.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[0]}.pdf')
        fig_q.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[1]}.pdf')
        fig_c.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[2]}.pdf')
        fig_z.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[3]}.pdf')

def plot1Dsamps_CE(file_path, flow_directory, flow_bins, use_unityweights, params = ['mchirp','q', 'chieff', 'z']):
    channel = 'CE'
    popsynth_outputs, obsdata, p_theta, flow, KDE, submodels_dict = load_models(channel, params, flow_bins, flow_directory, use_unityweights)
    
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())

    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][params]
        if use_unityweights:
            weights_dict[key] = np.ones(popsynth_outputs[key]['weight'].shape)
        else:
            weights_dict[key]= popsynth_outputs[key]['weight']

    plt.rcParams["figure.figsize"] = (15,15)
    plt.rcParams.update({'font.size': 15})
    no_samples = 100000
    no_bins = 60

    fig_mchirp, ax_m = plt.subplots(4,5)
    fig_q, ax_q = plt.subplots(4,5)
    fig_c, ax_c = plt.subplots(4,5)
    fig_z, ax_z = plt.subplots(4,5)

    param_label = ['$\mathcal{M}$ /$M_{\odot}$','q', '$\chi_{eff}$', 'z']

    for chi_b_id, xb in enumerate(_chi_b):
        for alpha_id, a in enumerate(_alpha):
            flow_samples_stack = flow[channel].flow.sample(np.array([xb,a]), no_samples)
            flow_mchirp = flow[channel].expistic(flow_samples_stack[:,0], flow[channel].mappings[0], flow[channel].mappings[1])
            flow_q = flow[channel].expistic(flow_samples_stack[:,1], flow[channel].mappings[2])
            flow_chieff = np.tanh(flow_samples_stack[:,2])
            flow_z = flow[channel].expistic(flow_samples_stack[:,3], flow[channel].mappings[4], flow[channel].mappings[5])

            mapped_flow_samples = [flow_mchirp,flow_q,flow_chieff,flow_z]

            kde_samples = KDE['CE'][submodels_dict[0][chi_b_id]][submodels_dict[1][alpha_id]].sample(no_samples)

            for i, ax in enumerate([ax_m,ax_q,ax_c,ax_z]):
                flow_distr, bin_edges = np.histogram(mapped_flow_samples[i], bins=no_bins, density=True)
                kde_distr, bin_edges_KDE = np.histogram(kde_samples[:,i], bins=no_bins, density=True)
                known_distr, bin_edges_known =np.histogram(models_dict[(chi_b_id,alpha_id)][:][params[i]], bins=no_bins, density=True, weights=weights_dict[(chi_b_id,alpha_id)][:])
                ax[chi_b_id,alpha_id].step(np.linspace(bin_edges[0],bin_edges[-1],no_bins),flow_distr, label='flow')
                ax[chi_b_id,alpha_id].step(np.linspace(bin_edges_KDE[0],bin_edges_KDE[-1],no_bins),kde_distr,label='KDE', color='purple')
                ax[chi_b_id,alpha_id].step(np.linspace(bin_edges_known[0],bin_edges_known[-1],no_bins),known_distr,label='underlying')
                ax[chi_b_id,alpha_id].set_title(fr'$\chi_b$={xb} and $\alpha$={a}')
                ax[chi_b_id,alpha_id].set_xlabel(fr'{param_label[i]}')
                ax[chi_b_id,alpha_id].set_ylabel(fr'P({param_label[i]}|$\chi_b$={xb}, $\alpha$={a})')
                #ax[chi_b_id,alpha_id].set_yscale('log')
                #ax[chi_b_id,alpha_id].legend()
                fig_mchirp.tight_layout(pad=1.3)
                fig_c.tight_layout(pad=1.3)
                fig_q.tight_layout(pad=1.3)
                fig_z.tight_layout(pad=1.3)

            fig_mchirp.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[0]}.pdf')
            fig_q.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[1]}.pdf')
            fig_c.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[2]}.pdf')
            fig_z.savefig(f'{file_path}/plots/flow_KDE_samples/{channel}_{params[3]}.pdf')