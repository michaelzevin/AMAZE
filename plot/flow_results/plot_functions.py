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
sys.path.append('/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/')
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

_param_bounds = {"mchirp": (-1.,70), "q": (0.08,1.01), "chieff": (-0.6,1.), "z": (-0.1,3.5)}
_param_ticks = {"mchirp": [0,10,20,30,40,50,60,70], "q": [0.25,0.5,0.75,1], "chieff": [-0.5,0,0.5,1], "z": [0,0.25,0.5,0.75,1.0,1.25]}
_pdf_bounds = {"mchirp": (0,0.075), "q": (0,32), "chieff": (0,17), "z": (0,4)}
_pdf_ticks = {"mchirp": [0.0,0.025,0.050,0.075], "q": [0,10,20,30], "chieff": [0,4,8,12,16], "z": (0,1,2,3,4)}
_labels_dict = {"mchirp": r"$\mathcal{M}$ /$M_{\odot}$", "q": r"$q$", \
"chieff": r"$\chi_{\rm eff}$", "z": r"$z$", "chi00": r"$\chi_\mathrm{b}=0.0$", \
"chi01": r"$\chi_\mathrm{b}=0.1$", "chi02": r"$\chi_\mathrm{b}=0.2$", \
"chi05": r"$\chi_\mathrm{b}=0.5$", "alpha02": r"$\alpha_\mathrm{CE}=0.2$", \
"alpha05": r"$\alpha_\mathrm{CE}=0.5$", "alpha10": r"$\alpha_\mathrm{CE}=1.0$", \
"alpha20": r"$\alpha_\mathrm{CE}=2.0$", "alpha50": r"$\alpha_\mathrm{CE}=5.0$", \
"CE": r"$\texttt{CE}$", "CHE": r"$\texttt{CHE}$", "GC": r"$\texttt{GC}$", \
"NSC": r"$\texttt{NSC}$", "SMT": r"$\texttt{SMT}$", \
"chi_b": r"$\chi_\mathrm{b}$", "alpha_CE": r"$\alpha_\mathrm{CE}$"}
_param_label = [_labels_dict["mchirp"],_labels_dict["q"], _labels_dict["chieff"], _labels_dict["z"]]
_Nsamps = 100000
_channel_label =[r'$\beta_{\mathrm{CE}}$',r'$\beta_{\mathrm{CHE}}$',r'$\beta_{\mathrm{GC}}$',r'$\beta_{\mathrm{NSC}}$',r'$\beta_{\mathrm{SMT}}$']
_channel_label_det =[r'$\beta_{\mathrm{CE}}^{\mathrm{det}}$',r'$\beta_{\mathrm{CHE}}^{\mathrm{det}}$',r'$\beta_{\mathrm{GC}}^{\mathrm{det}}$',\
    r'$\beta_{\mathrm{NSC}}^{\mathrm{det}}$',r'$\beta_{\mathrm{SMT}}^{\mathrm{det}}$']
_beta_det_label = r'$p(\beta^{\mathrm{det}})$'

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

def load_result_samps(filenames, Nhyper=2, Nchannels=5, detectable=False):
    """
    Loads hyperposterior samples from like of hdf5 files
    filenames : list, array
    """
    samples_allchains = np.array([])
    for i, filename in enumerate(filenames):
        try:
            result = h5py.File(filename, 'r')
        except:
            print('file not found')
            continue
        if detectable:
            result_key = 'detectable_samples'
        else:
            result_key = 'samples'
        samples_file = result['model_selection'][result_key]['block0_values']
        samples_allchains = np.append(samples_allchains, samples_file)
        samples_allchains = np.reshape(samples_allchains, (-1, Nhyper+Nchannels))

    return samples_allchains

def sample_pop_corner(flow_dir, channel_label, conditional, KDE_hyperparam_idxs=None, outdir=_basepath):
    
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file

    weighted_flow = FlowModel(channel_label, popsynth_outputs, _params, flow_path=flow_dir,  device='cpu', sensitivity='midhighlatelow_network')
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
        np.save(f"{outdir}/data/{channel_label}_KDEs_cornersample.npy", kde_samples)
        
    print('saving samples...')
    np.save(f"{outdir}/data/{channel_label}_flows_cornersample.npy", flow_samples_stack)
    print('samples saved')


def make_pop_corner(channel_label, hyperparam_idxs, justplot=True, flow_dir=None, conditional=None, plot_KDE=True, outdir=_basepath):

    #get samples from models, either sampling first or loading from file
    if justplot==False:
        sample_pop_corner( flow_dir, channel_label, conditional, KDE_hyperparam_idxs=hyperparam_idxs, outdir=outdir)
    if type(hyperparam_idxs) == type(None):
        pass
    else:
        kde_samples = np.load(f"{outdir}/data/{channel_label}_KDEs_cornersample.npy")
    flow_samples= np.load(f"{outdir}/data/{channel_label}_flows_cornersample.npy")

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
    print('plotting samples')
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
    fig.set_size_inches(figure_width*2, figure_width*2)
    plt.savefig(f"{outdir}/pdfs/{channel_label}_flowKDEmodel_corner_chib{hyperparam_idxs[0]}.pdf")

def make_1D_result_discrete(filenames, second_files=None, labels = [None,None], figure_name='Discrete', outdir=_basepath):
    plt.rcParams['figure.figsize'] = [figure_width*2, figure_width]
    channels = _channels
    submodels_dict= {0: {0: 'chi00', 1: 'chi01', 2: 'chi02', 3: 'chi05'}, \
    1: {0: 'alpha02', 1: 'alpha05', 2: 'alpha10', 3: 'alpha20', 4: 'alpha50'}}

    Nhyper =2
    _concentration = np.ones(len(channels))
    beta_p0 =  dirichlet.rvs(_concentration, size=100000)
    beta_bins = np.linspace(0,1,45)

    fig, ax_margs = plt.subplots(2,5, gridspec_kw={'wspace': 0.22, 'hspace': 0.5})
    fig.tight_layout(h_pad=3, w_pad=0.05)

    #add together samples from multiple files
    samples_allchains = load_result_samps(filenames)
    if second_files:
        samples_allchains_comp = load_result_samps(second_files)

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
                    histtype='step', color=cp[midx], bins=beta_bins, ls='-', lw=1.2, \
                    label=_labels_dict[model],\
                    weights=factor*np.ones_like(samples_allchains[smdl_locs, cidx+Nhyper]))
                if second_files:
                    factor_comp = 50/(len(samples_allchains_comp[:, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='stepfilled', color=cp[midx], bins=beta_bins, \
                        alpha=0.4, \
                        #label=_labels_dict[model]+labels[1],\
                        weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='step', color=cp[midx], bins=beta_bins, \
                        alpha=0.0, weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))

        # format plot
        for cidx, (channel, ax_marg) in enumerate(zip(channels, ax_margs.T)):
            #median branching fractions
            lower_5 = np.percentile(samples_allchains[:, cidx+Nhyper], 5)
            upper_95 = np.percentile(samples_allchains[:, cidx+Nhyper], 95)
            median = np.percentile(samples_allchains[:, cidx+Nhyper], 50)

            #ax_marg[hyper_idx].vlines([lower_5, median, upper_95], 0,20, color='black', alpha=0.5, lw=0.5)

            #plot prior
            h, bins, _ = ax_marg[hyper_idx].hist(beta_p0[:,cidx], \
                    histtype='step', color='grey', bins=20, alpha=0.7, density=True)
            #plot total BF
            """h, bins, _ = ax_marg[hyper_idx].hist(samples_allchains[:, cidx+Nhyper], \
                    histtype='step', color='black', bins=beta_bins, ls='--', lw=1.0, \
                    alpha=0.7, density=True)"""

            ax_marg[1].set_xlabel(_channel_label[cidx])
            ax_marg[hyper_idx].set_yscale('log')

            ax_marg[hyper_idx].set_xlim(0,1)
            ax_marg[hyper_idx].set_ylim(1e-4,80)
            if cidx == 0:
                ax_marg[hyper_idx].set_ylabel(r"p($\beta$)")
            else:
                ax_marg[hyper_idx].tick_params(labelleft=False)
        # legend
        if hyper_idx == 0:
            ax_margs[0,0].legend(loc='lower left', bbox_to_anchor=(.7, 1.02), ncol=4)
        if hyper_idx ==1:
            ax_margs[1,0].legend(loc='lower left', bbox_to_anchor=(-0.05, 1.02), ncol=5)

    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{outdir}/pdfs/{figure_name}_flowKDE_infresults.pdf")
        

def make_1D_result_continuous(filenames, filenames_det=None, figure_name='Continuous', detectable=False, outdir=_basepath):
    channels = _channels
    colors = ['royalblue','lightskyblue','darkblue']
    _concentration = np.ones(len(channels))
    beta_p0 =  dirichlet.rvs(_concentration, size=100000)
    alpha_CE_p0 =  loguniform.rvs(_alpha_CE[0], _alpha_CE[-1], size=100000)
    chib_p0 =  np.random.uniform(0, 0.5, size=100000)
    Nhyper =2

    fig = plt.figure(layout='constrained')
    if detectable==False:
        plt.rcParams['figure.figsize'] = [figure_width*2, figure_width]
        subfigs = fig.subfigures(2, 1, height_ratios=[1.,1.])
        ax_chibalpha = subfigs[0].subplots(1, 2)
        ax_margs = subfigs[1].subplots(1, 5)
        ax_margs_set = [ax_margs]
        channel_labels = [_channel_label]
    else:
        plt.rcParams['figure.figsize'] = [figure_width*2, figure_width*1.5]
        subfigs = fig.subfigures(3, 1, height_ratios=[1.,1., 1.])
        ax_margs_det = subfigs[2].subplots(1, 5)
        ax_chibalpha = subfigs[0].subplots(1, 2)
        ax_margs = subfigs[1].subplots(1, 5)
        ax_margs_set = [ax_margs, ax_margs_det]
        channel_labels = [_channel_label,_channel_label_det]


    #add together samples from multiple files
    samples_allchains = load_result_samps(filenames)
    sample_sets = np.array([samples_allchains])
    if detectable:
        samples_allchains_detectable = load_result_samps([filenames_det], detectable=True)
        sample_sets = [samples_allchains, samples_allchains_detectable]

    #sample_sets = np.array([samps for samps in [samples_allchains, samples_allchains_detectable] if len(samps)>0])

    #plot posteriors on chi_b and alpha_CE
    h, bins, _ = ax_chibalpha[0].hist(samples_allchains[:, 0], density=True,\
        histtype='step', color=colors[0], bins=np.linspace(0,0.5,80), ls='-', lw=1.5)
    h, bins, _ = ax_chibalpha[1].hist(samples_allchains[:, 1], density=True,\
        histtype='step', color=colors[0], bins=np.linspace(0,5.,50), ls='-', lw=1.5)

    for i, (samples,axes) in enumerate(zip(sample_sets, ax_margs_set)):
        for cidx, channel in enumerate(channels):
            h, bins, _ = axes[cidx].hist(samples[:,cidx+Nhyper], density=True,\
                histtype='step', color=colors[0], bins=np.linspace(0,1.,45), ls='-', lw=1.5)

    # format plot
    chi_b_lim = ax_chibalpha[0].get_ylim()[1] + 50
    alpha_CE_lim = ax_chibalpha[1].get_ylim()[1] +2
    #plot training lines
    ax_chibalpha[0].vlines(_chi_b, ax_chibalpha[0].get_ylim()[0], chi_b_lim, color='black', alpha=0.5)
    #plot chib prior
    ax_chibalpha[0].hist(chib_p0, \
                histtype='step', color='grey', bins=1, alpha=0.7, density=True, zorder=-1000)
    ax_chibalpha[1].vlines(_alpha_CE, ax_chibalpha[1].get_ylim()[0], alpha_CE_lim, color='black', alpha=0.5)
    #plot alpha_CE prior
    ax_chibalpha[1].hist(alpha_CE_p0, \
                histtype='step', color='grey', bins=20, alpha=0.7, density=True, zorder=-1000)

    ax_chibalpha[0].autoscale(tight=True, axis='y')
    ax_chibalpha[1].autoscale(tight=True, axis='y')
    ax_chibalpha[0].set_xlabel(_labels_dict['chi_b'])
    ax_chibalpha[1].set_xlabel(_labels_dict['alpha_CE'])
    #ax_chibalpha[1].set_xscale('log')
    ax_chibalpha[0].set_ylabel('p('+_labels_dict['chi_b']+')')
    ax_chibalpha[1].set_ylabel('p('+_labels_dict['alpha_CE']+')')

    for cidx, channel in enumerate(channels):
            #plot prior
            h, bins, _ = ax_margs[cidx].hist(beta_p0[:,cidx], \
                    histtype='step', color='grey', bins=20, alpha=0.7, density=True, zorder=-1000)

    for axes, label, samples in zip(ax_margs_set, channel_labels, sample_sets):
        for i, ax_marg in enumerate(np.append(ax_chibalpha, axes).flatten()):

            #median branching fractions
            q_mid = np.percentile(samples[:, i], 50)
            q_m = q_mid - np.percentile(samples[:, i], 5)
            q_p = np.percentile(samples[:, i], 95) - q_mid

            title_fmt=".2f"
            fmt = "{{0:{0}}}".format(title_fmt).format
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))
            ax_marg.set_yscale('log')
            #ax_marg.set_title(fr'${title}$')

            for cidx, channel in enumerate(channels):
                axes[cidx].set_xlabel(label[cidx])
                axes[cidx].set_xlim(0,1)
                ax_margs_set[0][cidx].set_ylim(1e-4,80)
                ax_margs_set[1][cidx].set_ylim(1e-4,15)
                axes[cidx].get_yaxis().set_tick_params(which='minor', size=0)
                #axes[cidx].set_yticks([1e-1,1e1])
                if cidx == 0:
                    ax_margs_set[0][cidx].set_ylabel(r"p($\beta$)")
                    ax_margs_set[1][cidx].set_ylabel(r'$p(\beta^{\mathrm{det}})$')
                else:
                    axes[cidx].tick_params(labelleft=False)
    
    plt.savefig(f"{outdir}/pdfs/{figure_name}_flowKDE_infresults.pdf")

def save_detectable_betas(filenames, analysis_name, outdir=_basepath):
    #in case detectable betas weren't saved during model_select run, save them now
    #for continuous results only

    params = ['mchirp','q', 'chieff', 'z']
    channels =['CE', 'CHE', 'GC', 'NSC', 'SMT']

    #initialise flows
    model_names, flow = get_models(_models_path, channels, params, use_flows=True, device='cpu', sensitivity='midhighlatelow_network')

    #read all samples
    samples_allchains = load_result_samps(filenames)
    converted_betas = np.zeros((samples_allchains[:,2:].shape[0], samples_allchains[:,2:].shape[1]))
        
    alphas = np.zeros((samples_allchains.shape[0], len(channels)))
    #get alpha for 5 channels given chi_b, alpha_CE in each sample
    for i, samp in enumerate(tqdm(samples_allchains)):
        for cidx, chnl in enumerate(channels):
            smdl = flow[chnl]
            if chnl == 'CE':
                #needs to be log alphaCE!
                alphas[i, cidx] = smdl.get_alpha([[samp[0], np.log(samp[1])]])
            else:
                alphas[i, cidx] = smdl.get_alpha([samp[:1][0], 1.])

    converted_betas = (samples_allchains[:,2:] * alphas)
    #divide by sum across channels
    converted_betas /= converted_betas.sum(axis=1, keepdims=True)

    columns = ['p0','p1']
    for channel in channels:
        columns.append('beta_'+channel)
    
    print('saving detectable betas...')
        
    df = pd.DataFrame(np.hstack([samples_allchains[:,:2],converted_betas]), columns=columns)
    df.to_hdf(f'{outdir}/data/{analysis_name}_detectable_betas.hdf5', key='model_selection/detectable_samples')

def calc_llh_ratio_CE(flow_dir, outdir=_basepath):

    channel_label = 'CE'
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file

    model_names, flow = get_models(_models_path, [channel_label], _params, use_flows=True, \
         detectable=False, senseitivity='midhighlatelow_network', flow_path=flow_dir, device='cpu')
    model_names, KDE_models = get_models(_models_path, [channel_label], _params, use_flows=False, normalize=False,\
         detectable=False, senseitivity='midhighlatelow_network')

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

    flow[channel_label].load_model(flow_dir, device='cpu')

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
    np.save(f"{outdir}/data/llh_ratio_kde_flow_reg.npy", llh_ratio_kde_flow_reg)
    np.save(f"{outdir}/data/llh_ratio_kde_flow_unreg.npy", llh_ratio_kde_flow_unreg)

    return mchirps, qs, llh_ratio_kde_flow_reg, llh_ratio_kde_flow_unreg

def plot_llh_ratio_CE(flow_dir, outdir, justplot=False):
    plt.rcParams['figure.figsize'] = [figure_width, figure_width*1.2]
    channel_label = 'CE'
    
    if justplot:
        mchirps = np.linspace(4.,49.9,20)
        qs = np.linspace(0.01,0.99,20)
        llh_ratio_kde_flow_reg = np.load(f"{outdir}/data/llh_ratio_kde_flow_reg.npy")
        llh_ratio_kde_flow_unreg = np.load(f"{outdir}/data/llh_ratio_kde_flow_unreg.npy")
    else:
        mchirps, qs, llh_ratio_kde_flow_reg, llh_ratio_kde_flow_unreg = calc_llh_ratio_CE(flow_dir, outdir)

    popsynth_outputs = read_hdf5(_models_path, channel_label)
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())

    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][_params]
        weights_dict[key]= popsynth_outputs[key]['weight']

    chi_b_id = 0
    alpha_id = 4
    
    fig, ax = plt.subplots(2,1)
    #cbar_scales = [50,2]

    for row, ratio in enumerate([llh_ratio_kde_flow_unreg, llh_ratio_kde_flow_reg]):
        #cbar_scale= cbar_scales[row]
        cbar_scale = np.max(np.abs(ratio))
        c = ax[row].imshow(np.swapaxes(ratio, 0,1), extent=(mchirps[0], mchirps[-1], qs[0], qs[-1]), origin='lower',\
            vmin=-cbar_scale, vmax=cbar_scale, aspect='auto', cmap='RdBu')
        cbar = fig.colorbar(c, ax=ax[row], cmap='RdBu')
        cbar.set_label(r'log$_{10}$ (p$_\mathrm{flow}/$p$_\mathrm{KDE}$)')

        ax[row].set_ylabel(_labels_dict['q'])

        bin_width = 0.05
        chieffs = popsynth_outputs[(chi_b_id,alpha_id)][:]['chieff']
        zs = popsynth_outputs[(chi_b_id,alpha_id)][:]['z']
        bin_chieff = np.logical_and(chieffs>0.0 - 2*bin_width,  chieffs< 0.0 + 2*bin_width)
        bin_z = np.logical_and(zs>0.2 - 10*bin_width, zs < 0.2 + 10*bin_width)
        bin_conditions = np.logical_and(bin_chieff, bin_z)

        c = ax[row].imshow(np.swapaxes(ratio,0,1), extent=(mchirps[0], mchirps[-1], qs[0], qs[-1]), origin='lower',\
            vmin=-cbar_scale, vmax=cbar_scale, aspect='auto', zorder=-200, cmap='RdBu')

        corner.hist2d(np.array(popsynth_outputs[(chi_b_id,alpha_id)][bin_conditions]['mchirp']),\
            np.array(popsynth_outputs[(chi_b_id,alpha_id)][bin_conditions]['q']), bins =16, \
            levels=(.50, .90, .99), \
            weights=np.array(weights_dict[(chi_b_id,alpha_id)][bin_conditions]), contour_kwargs=dict(lw=1.0), \
            pcolor_kwargs=dict(alpha=0.0), density=True, ax=ax[row], no_fill_contours=True,\
            plot_datapoints=False)
        ax[row].set_xlim(mchirps[0], mchirps[-1])
        ax[row].set_ylim(qs[0], qs[-1])
        ax[1].set_xlabel(_labels_dict['mchirp'])


    fig.tight_layout(pad=1.3)
    fig.savefig(f'{outdir}/pdfs/CE2Dmchirpq_llhratio.pdf')

def save_dataspace_samps(filenames, flow_dir, outdir=_basepath):
    channels = _channels
    params=_params

    print('loading results samples')
    hyper_posts = load_result_samps(filenames)

    print('loading flows')
    model_names, flow = get_models(_models_path, channels, _params, use_flows=True, device='cpu',\
     sensitivity='midhighlatelow_network', flow_path=flow_dir)
    for channel in channels:
        flow[channel].load_model(flow_dir, device='cpu')   

    print('sampling over marginalised hyperparameters')
    #Marginalising over hyperposterior samples
    no_total_samps = 1000000
    samps_per_hyperposts = 10
    no_hypersamps = int(no_total_samps/samps_per_hyperposts)

    samps_filled = np.zeros(6, dtype=int)
    channel_samps = [[],[],[],[],[]]

    for i, hyperpost_idx in enumerate(tqdm(np.random.choice(np.arange(np.shape(hyper_posts)[0]), no_hypersamps, replace=False))):
        hyperpost_samp = hyper_posts[hyperpost_idx,:]

        #evauluate channel as weighted choice according to branching fractions
        channel_idx = np.random.choice(np.arange(len(channels)), p=hyperpost_samp[2:])
        channel = channels[channel_idx]

        #sample flow
        if channel == 'CE':
            hyperpost_samp[1] = np.log(hyperpost_samp[1])
            samps = flow[channel].flow.sample(np.array([hyperpost_samp[:2]]), samps_per_hyperposts)
        else:
            samps = flow[channel].flow.sample(np.array([hyperpost_samp[:1]]), samps_per_hyperposts)

        samps[:,0] = flow[channel].expistic(samps[:,0], flow[channel].mappings[0], flow[channel].mappings[1])
        samps[:,1] = flow[channel].expistic(samps[:,1], flow[channel].mappings[2])
        samps[:,2] = np.tanh(samps[:,2])
        samps[:,3] = flow[channel].expistic(samps[:,3], flow[channel].mappings[4], flow[channel].mappings[5])
        
        channel_samps[channel_idx].append(samps)

    #evaluate the cumulative sum of samples with each channel
    no_channel_samps = [np.shape(channel_samps[chnl])[0] for chnl in np.arange(len(channels))]
    samps_filled[:5] = np.cumsum(no_channel_samps, axis=0)*samps_per_hyperposts

    #ordering total samples by channel
    total_samps_ordered = np.zeros((no_total_samps, len(params)))
    for cidx in range(len(channels)):
        channel_samps_red = np.reshape(np.array(channel_samps[cidx]),(-1,len(params)))
        total_samps_ordered[samps_filled[cidx-1]:samps_filled[cidx],:] = channel_samps_red
    
    np.save(f'{outdir}/data/flow_samps_dataspace.npy', total_samps_ordered)
    np.save(f'{outdir}/data/no_channelsamps_dataspace.npy', samps_filled)

def plot_samps_dataspace(filenames=None, flow_dir=None, outdir=_basepath, justplot=True):
    channels=_channels
    params=_params
    
    if justplot==False:
        save_dataspace_samps(filenames, flow_dir, outdir=outdir)
    total_samps_ordered = np.load(f"{outdir}/data/flow_samps_dataspace.npy")
    samps_filled = np.load(f'{outdir}/data/no_channelsamps_dataspace.npy')

    plt.rcParams["figure.figsize"] = (figure_width*2,figure_width*2/3)
    fig,ax=plt.subplots(1,4)
    cmap = plt.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, 6))

    print('loading parametric samples')
    comb_intrins_samps = pd.read_hdf('/data/wiay/2297403c/GW_ChirpSim/binary_param_generation/60_intrins_samps_zmax1_35.hdf5', key='all_intrins_samps')
    mchirp_samps = comb_intrins_samps['m1']*(comb_intrins_samps['q']**3/(1+comb_intrins_samps['q']))
    chieff_samps = ((comb_intrins_samps['chi_1']*comb_intrins_samps['costilt_1'])+(comb_intrins_samps['q']*comb_intrins_samps['chi_2']*comb_intrins_samps['costilt_2']))\
    /(1+comb_intrins_samps['q'])

    no_total_samps = np.shape(total_samps_ordered)[0]
    PLPP_samps = [mchirp_samps, comb_intrins_samps['q'], chieff_samps, comb_intrins_samps['z']]
    no_bins=50
    bins =np.array([np.linspace(_param_bounds[pidx][0],_param_bounds[pidx][1],no_bins) for pidx in params])
    ax_ylims = [0.6*no_total_samps,0.2*no_total_samps,0.6*no_total_samps,0.12*no_total_samps]
    mask_min = [np.min(total_samps_ordered[:,0]), 0,-1,0]
    mask_max = [100, 1,1,1.35]

    for cidx, channel in enumerate(channels):
        for pidx,param in enumerate(params):
            ax[pidx].hist(total_samps_ordered[samps_filled[cidx-1]:samps_filled[cidx],pidx], color=colors[cidx], histtype='step', bins=bins[pidx], lw=1,\
                label=channel)

    for pidx,param in enumerate(params):
        ax[pidx].hist(total_samps_ordered[:,pidx], color='black', histtype='step', bins=bins[pidx], lw=1, ls='-', label='Total', alpha=.7)
        mask = np.logical_and(mask_min[pidx]<total_samps_ordered[:,pidx],total_samps_ordered[:,pidx]<mask_max[pidx])
        ax[pidx].hist(PLPP_samps[pidx], color='deeppink', histtype='step',\
            weights=np.ones_like(PLPP_samps[pidx])*len(total_samps_ordered[mask,pidx])/len(PLPP_samps[pidx]),\
            bins=bins[pidx], lw=1, ls='-', label='Parametric', alpha=0.7, zorder=-100)
        ax[pidx].set_yscale('log')
        ax[pidx].set_xlabel(_labels_dict[param])
        ax[pidx].set_ylim(5,ax_ylims[0])
        ax[pidx].set_xlim(_param_bounds[param][0],_param_bounds[param][1])
        if pidx>0:
            ax[pidx].tick_params(labelleft=False)
    ax[0].set_ylabel('No. samples')
    plt.legend(loc='lower center', bbox_to_anchor=(-1.4, 1.02), ncol=7, columnspacing=1.)
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout(pad=1.3)
    fig.savefig(f'{outdir}/pdfs/1D_dataspace_samps_marg.pdf')




