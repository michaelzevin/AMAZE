import numpy as np
import pandas as pd
import os
import operator
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import corner
from tqdm import tqdm
import matplotlib.lines as mlines
from scipy.stats import dirichlet
from scipy.stats import loguniform

from matplotlib import gridspec

import sys
sys.path.append('../../')
from populations.bbh_models import *
from populations.Pop_Flows import FlowModel
from sample.sample import lnlike_disc

colors = sns.color_palette("colorblind", n_colors=10)
cp = [colors[0], colors[2], colors[4], colors[1], colors[3], colors[6], colors[9], colors[5], colors[8]]
_basepath, _ = os.path.split(os.path.realpath(__file__))
plt.style.use(_basepath+"/mpl.sty")

_params = ['mchirp','q', 'chieff', 'z']
_channels = ['CE','CHE','GC','NSC','SMT']
_chi_b=[0.0,0.1,0.2,0.5]
_alpha_CE=[0.2,0.5,1.0,2.0,5.0]

_param_label = ['Chirp Mass /$M_{\odot}$','Mass Ratio', 'Effective Spin', 'Redshift']
_param_bounds = {"mchirp": (0,70), "q": (0.25,1), "chieff": (-0.5,1), "z": (0,1.25)}
_param_ticks = {"mchirp": [0,10,20,30,40,50,60,70], "q": [0.25,0.5,0.75,1], "chieff": [-0.5,0,0.5,1], "z": [0,0.25,0.5,0.75,1.0,1.25]}
_pdf_bounds = {"mchirp": (0,0.075), "q": (0,32), "chieff": (0,17), "z": (0,4)}
_pdf_ticks = {"mchirp": [0.0,0.025,0.050,0.075], "q": [0,10,20,30], "chieff": [0,4,8,12,16], "z": (0,1,2,3,4)}
_labels_dict = {"mchirp": r"$\mathcal{M}_{\rm c}$ [$M_{\odot}$]", "q": r"$q$", \
"chieff": r"$\chi_{\rm eff}$", "z": r"$z$", "chi00": r"$\chi_\mathrm{b}=0.0$", \
"chi01": r"$\chi_\mathrm{b}=0.1$", "chi02": r"$\chi_\mathrm{b}=0.2$", \
"chi05": r"$\chi_\mathrm{b}=0.5$", "alpha02": r"$\alpha_\mathrm{CE}=0.2$", \
"alpha05": r"$\alpha_\mathrm{CE}=0.5$", "alpha10": r"$\alpha_\mathrm{CE}=1.0$", \
"alpha20": r"$\alpha_\mathrm{CE}=2.0$", "alpha50": r"$\alpha_\mathrm{CE}=5.0$", \
"CE": r"$\texttt{CE}$", "CHE": r"$\texttt{CHE}$", "GC": r"$\texttt{GC}$", \
"NSC": r"$\texttt{NSC}$", "SMT": r"$\texttt{SMT}$", \
"chi_b": r"$\chi_\mathrm{b}$", "alpha_CE": r"$\alpha_\mathrm{CE}$"}
_Nsamps = 100000
_channel_label =[r'$\beta_{\mathrm{CE}}$',r'$\beta_{\mathrm{CHE}}$',r'$\beta_{\mathrm{GC}}$',r'$\beta_{\mathrm{NSC}}$',r'$\beta_{\mathrm{SMT}}$']


_models_path ='/data/wiay/2297403c/models_reduced.hdf5'

pt = 1./72.27
jour_sizes = {"AAS": {"onecol": 242.26653*pt, "twocol": 242.26653*2*pt},
              # Add more journals below. Can add more properties to each journal
             }

figure_width = jour_sizes["AAS"]["onecol"]


_base_corner_kwargs = dict(
    bins=64,
    smooth=0.9,
    #quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=True,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=False,
    hist_kwargs=dict(density=True),
    labels=_param_label,
)

def load_result_samps(filenames, discrete_result=False, Nhyper=2, Nchannels=5, detectable=False):
    """
    Loads hyperposterior samples from like of hdf5 files
    filenames : list, array
    """
    samples_allchains = np.array([])
    for i, filename in enumerate(filenames):
        try:
            result = h5py.File(filename, 'r')
        except:
            continue
        if detectable:
            result_key = 'detectable_samples'
        else:
            result_key = 'samples'
        if discrete_result:
            samples_file =np.hstack([result['model_selection'][result_key]['block1_values'], result['model_selection'][result_key]['block0_values']])
        else:
            samples_file = result['model_selection'][result_key]['block0_values']
        samples_allchains = np.append(samples_allchains, samples_file)
        samples_allchains = np.reshape(samples_allchains, (-1, Nhyper+Nchannels))

    return samples_allchains

def sample_pop_corner(flow_dir, channel_label, conditional, KDE_hyperparam_idxs=None):
    
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file

    weighted_flow = FlowModel(channel_label, popsynth_outputs, _params, flow_path=flow_dir,  device='cpu', sensitivity='midhighlatelow')
    model_names, KDE_models = get_models(_models_path, [channel_label], _params, use_flows=False, normalize=False, detectable=False)

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

    weighted_flow.load_model(flow_dir, device='cpu')

    #sample flow

    print('sampling flow...')
    if channel_label=='CE':
        flow_samples_stack = weighted_flow.sample(np.array([conditional[0], conditional[1]]),_Nsamps)
    else:
        flow_samples_stack = weighted_flow.sample(np.array([conditional[0]]),_Nsamps)

    print('sampling KDE...')
    if type(KDE_hyperparam_idxs) == type(None):
        pass
    else:
        if channel_label=='CE':
            kde_samples = KDE_models[channel_label][submodels_dict[0][KDE_hyperparam_idxs[0]]][submodels_dict[1][KDE_hyperparam_idxs[1]]].sample(_Nsamps)
        else:
            kde_samples = KDE_models[channel_label][submodels_dict[0][KDE_hyperparam_idxs[0]]].sample(_Nsamps)
        np.save(f"{_basepath}/data/{channel_label}_KDEs_cornersample.npy", kde_samples)
        
    print('saving samples...')
    np.save(f"{_basepath}/data/{channel_label}_flows_cornersample.npy", flow_samples_stack)
    print('samples saved')


def make_pop_corner(channel_label, hyperparam_idxs, justplot=True, flow_dir=None, conditional=None, plot_KDE=True):

    #get samples from models, either sampling first or loading from file
    if justplot==False:
        sample_pop_corner( flow_dir, channel_label, conditional, KDE_hyperparam_idxs=hyperparam_idxs)
    if type(hyperparam_idxs) == type(None):
        pass
    else:
        kde_samples = np.load(f"{_basepath}/data/{channel_label}_KDEs_cornersample.npy")
    flow_samples= np.load(f"{_basepath}/data/{channel_label}_flows_cornersample.npy")

    #get training population samples
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())

    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][_params]
        weights_dict[key]= popsynth_outputs[key]['weight']

    #set colours for plot
    colors=['C1', 'purple', 'royalblue']
    labels=['Underlying Model', 'KDE', 'Normalising Flow']

    model_kwargs = deepcopy(_base_corner_kwargs)
    model_kwargs["color"] = colors[0]
    model_kwargs["hist_kwargs"]["color"] = colors[0]
    corner_kwargs_kde = deepcopy(_base_corner_kwargs)
    corner_kwargs_kde["color"] = colors[1]
    corner_kwargs_kde["hist_kwargs"]["color"] = colors[1]
    corner_kwargs_flow = deepcopy(_base_corner_kwargs)
    corner_kwargs_flow["color"] = colors[2]
    corner_kwargs_flow["hist_kwargs"]["color"] = colors[2]

    plt.rcParams['figure.figsize'] = [figure_width, figure_width]

    #plot flow samples without training models or KDEs if off of training model grid
    if type(hyperparam_idxs) == type(None):
        fig=corner.corner(flow_samples, **corner_kwargs_flow)
    #else plot training models, KDE if plotting, and flow last
    else:
        if channel_label == 'CE':
            fig =corner.corner(models_dict[tuple(hyperparam_idxs)],  weights=weights_dict[tuple(hyperparam_idxs)], **model_kwargs)
        else:
            fig =corner.corner(models_dict[hyperparam_idxs[0]],  weights=weights_dict[hyperparam_idxs[0]], **model_kwargs)
        if plot_KDE==True:
            corner.corner(kde_samples, fig=fig, **corner_kwargs_kde)
        corner.corner(flow_samples, fig=fig, **corner_kwargs_flow)
    #add legend
    plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=labels[i])
                for i in range(3)
            ],
            frameon=False,
            bbox_to_anchor=(1, 4), loc="upper right"
        )
    #save figure
    plt.savefig(f"{_basepath}/pdfs/{channel_label}_flowKDEmodel_corner_chib{hyperparam_idxs[0]}.pdf")

def make_1D_result_discrete(filenames, second_files=None, labels = [None,None], figure_name='Discrete'):
    plt.rcParams['figure.figsize'] = [figure_width, figure_width/4]
    channels = _channels
    submodels_dict= {0: {0: 'chi00', 1: 'chi01', 2: 'chi02', 3: 'chi05'}, \
    1: {0: 'alpha02', 1: 'alpha05', 2: 'alpha10', 3: 'alpha20', 4: 'alpha50'}}

    Nhyper =2
    _concentration = np.ones(len(channels))
    beta_p0 =  dirichlet.rvs(_concentration, size=100000)

    fig, ax_margs = plt.subplots(2,5)
    fig.tight_layout(h_pad=3)

    #add together samples from multiple files
    samples_allchains = load_result_samps(filenames, discrete_result=True)
    if second_files:
        samples_allchains_comp = load_result_samps(second_files, discrete_result=True)

    #loop over astrophysical parameters
    for hyper_idx in [0, 1]:
        #loop over population models and plot histograms
        for midx, model in submodels_dict[hyper_idx].items():
            smdl_locs = np.argwhere(samples_allchains[:,hyper_idx]==midx).flatten()
            if second_files:
                comp_smdl_locs = np.argwhere(samples_allchains_comp[:,hyper_idx]==midx).flatten()

            for cidx, channel in enumerate(channels):
                factor = 50/(len(samples_allchains[:, cidx+Nhyper]))
                h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains[smdl_locs, cidx+Nhyper], \
                    histtype='step', color=cp[midx], bins=50, ls='-', lw=1.5, \
                    label=_labels_dict[model]+labels[0],\
                    weights=factor*np.ones_like(samples_allchains[smdl_locs, cidx+Nhyper]))
                if second_files:
                    factor_comp = 50/(len(samples_allchains_comp[:, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='stepfilled', color=cp[midx], bins=50, \
                        alpha=0.3, label=_labels_dict[model]+labels[1],\
                        weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='step', color=cp[midx], bins=50, \
                        alpha=0.7, weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))

        # format plot
        for cidx, (channel, ax_marg) in enumerate(zip(channels, ax_margs.T)):
            #median branching fractions
            lower_5 = np.percentile(samples_allchains[:, cidx+Nhyper], 5)
            upper_95 = np.percentile(samples_allchains[:, cidx+Nhyper], 95)
            median = np.percentile(samples_allchains[:, cidx+Nhyper], 50)

            ax_marg[hyper_idx].vlines([lower_5, median, upper_95], 0,20, color='black', alpha=0.5, lw=0.5)

            #plot prior
            h, bins, _ = ax_marg[hyper_idx].hist(beta_p0[:,cidx], \
                    histtype='step', color='grey', bins=20, alpha=0.7, density=True)
            #plot total BF
            h, bins, _ = ax_marg[hyper_idx].hist(samples_allchains[:, cidx+Nhyper], \
                    histtype='step', color='black', bins=50, ls='--', lw=1.0, \
                    alpha=0.7, density=True)

            ax_marg[1].set_xlabel(_channel_label[cidx], fontsize=15)
            ax_marg[hyper_idx].set_yscale('log')

            ax_marg[hyper_idx].set_xlim(0,1)
            ax_marg[hyper_idx].set_ylim(int(1e-3),20)
            if cidx == 0:
                ax_marg[hyper_idx].set_ylabel(r"p($\beta$)", fontsize=15)
            else:
                ax_marg[hyper_idx].tick_params(labelleft=False)
        # legend
        if hyper_idx == 0:
            ax_margs[0,0].legend(loc='lower left', bbox_to_anchor=(0.5, 1.02), ncol=4, prop={'size':10})
        if hyper_idx ==1:
            ax_margs[1,0].legend(loc='lower left', bbox_to_anchor=(-1.0, 1.02), ncol=5, prop={'size':10})
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{_basepath}/pdfs/{figure_name}_flowKDE_infresults.pdf")
        

def make_1D_result_continuous(filenames, second_files=None, figure_name='Continuous', detectable=False):
    channels = _channels
    plt.rcParams['figure.figsize'] = [figure_width, figure_width/2]
    colors = ['royalblue','lightskyblue','darkblue']
    _concentration = np.ones(len(channels))
    beta_p0 =  dirichlet.rvs(_concentration, size=100000)
    #alphaCE_p0 =  loguniform.rvs(_concentration, size=100000)
    Nhyper =2

    fig = plt.figure(layout='constrained')
    subfigs = fig.subfigures(2, 1, height_ratios=[1.,1.])
    ax_chibalpha = subfigs[0].subplots(1, 2)
    ax_margs = subfigs[1].subplots(1, 5)

    #add together samples from multiple files
    samples_allchains = load_result_samps(filenames, detectable=detectable)
    if second_files:
        samples_allchains_comp = load_result_samps(second_files, detectable=detectable)
    else:
        samples_allchains_comp = np.array([])

    for i, samples in enumerate(np.array([samps for samps in [samples_allchains, samples_allchains_comp] if len(samps)>0])):
        h, bins, _ = ax_chibalpha[0].hist(samples[:, 0], density=True,\
            histtype='step', color=colors[i], bins=50, ls='-', lw=1.5)
        h, bins, _ = ax_chibalpha[1].hist(samples[:, 1], density=True,\
            histtype='step', color=colors[i], bins=50, ls='-', lw=1.5)

        for cidx, channel in enumerate(channels):
            h, bins, _ = ax_margs[cidx].hist(samples[:,cidx+Nhyper], density=True,\
                histtype='step', color=colors[i], bins=50, ls='-', lw=1.5)

    # format plot
    chi_b_lim = ax_chibalpha[0].get_ylim()[1] + 50
    alpha_CE_lim = ax_chibalpha[1].get_ylim()[1] +2
    #plot training lines
    ax_chibalpha[0].vlines(_chi_b, ax_chibalpha[0].get_ylim()[0], chi_b_lim, color='black', alpha=0.5)
    ax_chibalpha[1].vlines(_alpha_CE, ax_chibalpha[1].get_ylim()[0], alpha_CE_lim, color='black', alpha=0.5)
    #plot alpha_CE prior
    """ax_chibalpha[1].hist(beta_p0[:,cidx], \
                histtype='step', color='grey', bins=20, alpha=0.7, density=True)"""

    ax_chibalpha[0].autoscale(tight=True, axis='y')
    ax_chibalpha[1].autoscale(tight=True, axis='y')
    ax_chibalpha[0].set_xlabel(_labels_dict['chi_b'])
    ax_chibalpha[1].set_xlabel(_labels_dict['alpha_CE'])
    ax_chibalpha[0].set_ylabel('p('+_labels_dict['chi_b']+')')
    ax_chibalpha[1].set_ylabel('p('+_labels_dict['alpha_CE']+')')

    for i, ax_marg in enumerate(np.append(ax_chibalpha,ax_margs).flatten()):

        #median branching fractions
        q_mid = np.percentile(samples_allchains[:, i], 50)
        q_m = q_mid - np.percentile(samples_allchains[:, i], 5)
        q_p = np.percentile(samples_allchains[:, i], 95) - q_mid

        title_fmt=".2f"
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))
        ax_marg.set_yscale('log')
        ax_marg.set_title(fr'${title}$')

    for cidx, channel in enumerate(channels):

        #plot prior
        h, bins, _ = ax_margs[cidx].hist(beta_p0[:,cidx], \
                histtype='step', color='grey', bins=20, alpha=0.7, density=True)

        ax_margs[cidx].set_xlabel(_channel_label[cidx])

        ax_margs[cidx].set_xlim(0,1)
        if cidx == 0:
            ax_margs[cidx].set_ylabel(r"p($\beta$)")
        else:
            ax_margs[cidx].tick_params(labelleft=False)
    if detectable:
        subfigs[0].delaxes(ax_chibalpha[0])
        subfigs[0].delaxes(ax_chibalpha[1])
        fig.sca(subfigs[1])
    plt.savefig(f"{_basepath}/pdfs/{figure_name}_flowKDE_infresults.pdf")

def save_detectable_betas(filenames, analysis_name):
    #in case detectable betas weren't saved during model_select run, save them now
    #for continuous results only

    params = ['mchirp','q', 'chieff', 'z']
    channels =['CE', 'CHE', 'GC', 'NSC', 'SMT']

    #initialise flows
    model_names, flow = get_models(_models_path, channels, params, use_flows=True, device='cpu', sensitivity='midhighlatelow')

    #read all samples
    samples_allchains = load_result_samps(filenames)
    
    converted_betas = np.zeros((samples_allchains[:,2:].shape[0], samples_allchains[:,2:].shape[1]))
        
    alphas = np.zeros((samples_allchains.shape[0], len(channels)))
    #get alpha for 5 channels given chi_b, alpha_CE in each sample
    for i, samp in enumerate(tqdm(samples_allchains)):
        for cidx, chnl in enumerate(channels):
            smdl = flow[chnl]
            if chnl == 'CE':
                alphas[i, cidx] = smdl.get_alpha(samp[:2])
            else:
                alphas[i, cidx] = smdl.get_alpha([samp[:1][0], 1.])

    converted_betas = (samples_allchains[:,2:] * alphas)
    #divide by sum across channels
    converted_betas /= converted_betas.sum(axis=1, keepdims=True)

    columns = ['p0','p1']
    for channel in channels:
        columns.append('beta_'+channel)
        
    df = pd.DataFrame(np.hstack([samples_allchains[:,:2],converted_betas]), columns=columns)
    df.to_hdf(f'{_basepath}/data/{analysis_name}_detectable_betas.hdf5', key='model_selection/detectable_samples')

def calc_llh_ratio_CE(flow_dir):

    channel_label = 'CE'
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file

    flow = FlowModel(channel_label, popsynth_outputs, _params, flow_path=flow_dir, sensitivity='midhighlatelow', device='cpu')
    model_names, KDE_models = get_models(_models_path, [channel_label], _params, use_flows=False, normalize=False,\
         detectable=False, senseitivity='midhighlatelow')

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

    flow.load_model(flow_dir, device='cpu')

    mchirps = np.linspace(4.,49.9,20)
    qs = np.linspace(0.01,0.99,20)

    p_mchirpq_unreg = np.zeros((20,20))
    p_mchirpq_kde_unreg = np.zeros((20,20))
    p_mchirpq_reg = np.zeros((20,20))
    p_mchirpq_kde_reg = np.zeros((20,20))
    chi_b_id = 0
    alpha_id = 4

    for  i, m in enumerate(tqdm(mchirps)):
        for j, q in enumerate(qs):
            sample = np.reshape([m, q,0.05,0.2], (1,1,4))
            p_mchirpq_reg[i, j] = lnlike_disc([chi_b_id,alpha_id], sample, flow, submodels_dict, ['CE'], use_flows=True,\
                 prior_pdf=None, smallest_N=990903)
            p_mchirpq_kde_reg[i, j] = lnlike_disc([chi_b_id,alpha_id], sample, KDE_models, submodels_dict, ['CE'], use_flows=False,\
                 prior_pdf=None, smallest_N=990903)
            
            p_mchirpq_unreg[i, j] = lnlike_disc([chi_b_id,alpha_id], sample, flow, submodels_dict, ['CE'], use_flows=True,\
                 prior_pdf=None, smallest_N=None)
            p_mchirpq_kde_unreg[i, j] = lnlike_disc([chi_b_id,alpha_id], sample, KDE_models, submodels_dict, ['CE'], use_flows=False,\
                 prior_pdf=None, smallest_N=None)

    llh_ratio_kde_flow_reg = np.log10(np.exp(p_mchirpq_reg-p_mchirpq_kde_reg))
    llh_ratio_kde_flow_unreg = np.log10(np.exp(p_mchirpq_unreg-p_mchirpq_kde_unreg))

    #save ratios
    np.save(f"{_basepath}/data/llh_ratio_kde_flow_reg.npy", llh_ratio_kde_flow_reg)
    np.save(f"{_basepath}/data/llh_ratio_kde_flow_unreg.npy", llh_ratio_kde_flow_unreg)

    return mchirps, qs, llh_ratio_kde_flow_reg, llh_ratio_kde_flow_unreg

def plot_llh_ratio_CE(flow_dir):
    plt.rcParams['figure.figsize'] = [figure_width/2, figure_width/2]
    
    mchirps, qs, llh_ratio_kde_flow_reg, llh_ratio_kde_flow_unreg = calc_llh_ratio_CE(flow_dir)

    popsynth_outputs = read_hdf5(_models_path, channel_label)
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())

    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][_params]
        weights_dict[key]= popsynth_outputs[key]['weight']

    chi_b_id = 0
    alpha_id = 4
    
    fig, ax = plt.subplots(2,2)

    for row, ratio in enumerate[llh_ratio_kde_flow_reg, llh_ratio_kde_flow_unreg]:
        c = ax[row,0].imshow(np.swapaxes(ratio, 0,1), extent=(mchirps[0], mchirps[-1], qs[0], qs[-1]), origin='lower',\
            vmin=-90, vmax=90, aspect='auto', cmap='RdBu')
        cbar = fig.colorbar(c, ax=ax[0], cmap='RdBu')
        cbar.set_label(r'ln $p_\mathrm{flow} \minus$ln $p_\mathrm{KDE}$')
        #c = ax[chibid,1].imshow(np.swapaxes(oldratio, 0,1), extent=(mchirps[0], mchirps[-1], qs[0], qs[-1]), origin='lower', vmin=-6, vmax=6, aspect='auto')

        ax[row,0].set_xlabel(r'$\mathcal{M}_{\mathrm{c}}$ /$M_{\odot}$')
        ax[row,0].set_ylabel(r'$q$')

        bin_width = 0.05
        chieffs = popsynth_outputs[(chi_b_id,alpha_id)][:]['chieff']
        zs = popsynth_outputs[(chi_b_id,alpha_id)][:]['z']
        bin_chieff = np.logical_and(chieffs>0.0 - 2*bin_width,  chieffs< 0.0 + 2*bin_width)
        bin_z = np.logical_and(zs>0.2 - 10*bin_width, zs < 0.2 + 10*bin_width)
        bin_conditions = np.logical_and(bin_chieff, bin_z)

        c = ax[row, 1].imshow(np.swapaxes(ratio,0,1), extent=(mchirps[0], mchirps[-1], qs[0], qs[-1]), origin='lower',\
            vmin=-90, vmax=90, aspect='auto', zorder=-200, cmap='RdBu')

        corner.hist2d(np.array(popsynth_outputs[(chi_b_id,alpha_id)][bin_conditions]['mchirp']),\
            np.array(popsynth_outputs[(chi_b_id,alpha_id)][bin_conditions]['q']), \
            weights=np.array(weights_dict[(chi_b_id,alpha_id)][bin_conditions]), alpha=0.5, density=True, ax=ax[1], no_fill_contours=True)
        ax[row,1].set_xlim(mchirps[0], mchirps[-1])
        ax[row,1].set_ylim(qs[0], qs[-1])
    ax[1,1].set_xlabel(r'$\mathcal{M}_{\mathrm{c}}$ /$M_{\odot}$')
    fig.tight_layout(pad=1.3)
    fig.savefig(f'{_basepath}/pdfs/CE2Dmchirpq_llhratio.pdf')

