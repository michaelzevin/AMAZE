"""
Plotting functions so we don't bog down the executable
"""

import numpy as np
import pandas as pd
import os
import pdb
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set_style("whitegrid")

from populations import *
from utils import helper_functions

cp = sns.color_palette("colorblind", 6)
_basepath, _ = os.path.split(os.path.realpath(__file__))
plt.style.use(_basepath+"/.MATPLOTLIB_RCPARAMS.sty")

_Nsamps = 1000
_Neval = 100
_marg_kde_bandwidth = 0.01


def plot_1D_kdemodels(model_names, true_model, models, model0, \
                        event_parameters, population_parameters, \
                        observations, samples, \
                        dirname=None, fixed_parameters=None, \
                        plot_obs=False, plot_obs_samples=False, \
                        verbose=False):
    """
    Plots all the KDEs for each channel in each model, as well as the *true* model described by the input branching fraction.
    """
    # warn about fixed parameters if model0 is specified...
    if model0 and fixed_parameters:
        warnings.warn("Since you are assuming a true mock model, fixed parameters will be set to the true values of the mock model.", stacklevel=2)
        fixed_parameters = true_model

    # plot for each fixed hyperparameter
    channels = list(models.keys())
    params = list(event_parameters.keys())

    for fixed_name, fixed_val in fixed_parameters.items():
        # downselect models that will be plotted
        models_to_plot = [x for x in model_names if fixed_val in x]

        # get the names of the models that are varied in the plot
        model_variations = [x.split('/', 1)[1] for x in models_to_plot]
        model_variations = [x.split('/') for x in model_variations]
        model_variations = [item for sublist in model_variations for item in sublist]
        model_variations = list(set(model_variations))
        model_variations.remove(fixed_val)
        model_variations.sort()

        # get varied population parameter
        varied_name = []
        for key in population_parameters.keys():
            if model_variations[0] in list(population_parameters[key]['values'].keys()):
                varied_name.append(key)
        varied_name = list(set(varied_name))[0]

        Nchannels = int(len(models.keys()))
        Nparams = int(len(event_parameters.keys()))
        Nsbmdls = int(len(models_to_plot)/Nchannels)

        fig, axs = plt.subplots(Nsbmdls, Nparams, figsize=(7*Nparams, 6*Nsbmdls))

        # loop over all models...
        if verbose:
            print('  plotting population models for fixed {:s}={:0.1f}...'.format(fixed_name, population_parameters[fixed_name]['values'][fixed_val]))
        for cidx, channel in tqdm(enumerate(channels), total=len(channels)):
            channel_smdls = [x for x in models_to_plot if channel+'/' in x]
            for idx, (model, varied_val) in enumerate(zip(channel_smdls, model_variations)):
                kde = helper_functions.GetFromDict(models, model.split('/'))

                # if this kde is in model0, allocate array for samples
                if model0 and (kde.label == model0[channel].label):
                    channel_model0_samples = np.zeros(shape=(int(kde.rel_frac*_Nsamps),Nparams))

                # loop over all parameters...
                for pidx, param in enumerate(params):
                    if axs.ndim == 1:
                        ax = axs[idx]
                    else:
                        ax = axs[idx,pidx]

                    # get event parameter dictionary for marginalization
                    marg_parameters = {param: event_parameters[param]}

                    # marginalize the kde (this redoes the KDE in 1D)
                    # make sure to set alpha=1 so each channel is evenly weighted in plot
                    marg_kde = kde.marginalize(\
                        params=[param], \
                        bandwidth=_marg_kde_bandwidth, \
                        detectable=True)

                    # evaluate the marginalized kde over the param range
                    eval_pts = np.linspace(*marg_parameters[param]['limits'], _Neval)
                    eval_pts = eval_pts.reshape(_Neval,1,1)
                    pdf = marg_kde(eval_pts, smallest_N=None)

                    # if this model is in model0, sample the marginalized KDE
                    if model0 and (kde.label == model0[channel].label):
                        channel_model0_samples[:,pidx] = marg_kde.sample(int(kde.rel_frac*_Nsamps)).flatten()

                    # labels and legend
                    if model0:
                        if idx==int(np.argwhere(np.array(model_variations) == true_model[varied_name])) \
                                    and pidx==Nparams-1:
                            label = channel+r" ($\beta$={0:0.2f})".format(model0[channel].rel_frac)
                        else:
                            label = None
                    elif idx==0 and pidx==Nparams-1 and cidx==Nchannels-1:
                        label = channel 
                    else:
                        label = None

                    # plot the kde
                    ax.plot(eval_pts.flatten(), pdf, color=cp[cidx], label=label)

                    # Format plot
                    if label is not None:
                        ax.legend(prop={'size':35}, loc='center', bbox_to_anchor=(1.0,0.5))
                    if cidx==Nchannels-1:
                        ax.set_xlim(*marg_parameters[param]['limits'])
                        ax.set_ylim(bottom=0)
                        if idx==Nsbmdls-1:
                            ax.set_xlabel(marg_parameters[param]['fullname'], fontsize=40)
                        if pidx==0:
                            ylbl = population_parameters[varied_name]['fullname'] + \
                                ' = {:0.1f}'.format(population_parameters[varied_name]['values'][varied_val]) + \
                                '\n(' + population_parameters[fixed_name]['fullname'] + \
                                ' = {:0.1f}'.format(population_parameters[fixed_name]['values'][fixed_val]) + ')'
                            ax.set_ylabel(ylbl, fontsize=50)

            # append the (detectable) draws from model0 from all channels
            if model0:
                if cidx==0:
                    model0_samples = channel_model0_samples
                else:
                    model0_samples = np.concatenate((model0_samples, channel_model0_samples))


        # Plot model0, obsdata, and formatting
        if verbose:
            print('    plotting model0 and observations...')
        for idx in np.arange(Nsbmdls):
            for pidx, param in enumerate(params):
                if axs.ndim == 1:
                    ax = axs[idx]
                else:
                    ax = axs[idx,pidx]

                # get event parameter dictionary for marginalization
                marg_parameters = {param: event_parameters[param]}

                # construct combined KDE model and plot
                if model0:
                    # NOTE: samples are already drawn from detectable KDEs
                    combined_samples = pd.DataFrame(model0_samples[:,pidx].flatten(), columns=[param])
                    combined_kde = KDEModel.from_samples(\
                        label='combined_kde', \
                        samples=combined_samples, \
                        param_dict=marg_parameters, \
                        sensitivity=None, \
                        max_samps=_Nsamps, \
                        kde_bandwidth=_marg_kde_bandwidth, \
                        store_optimal_snrs=False)
                    eval_pts = np.linspace(*marg_parameters[param]['limits'], 100)
                    eval_pts = eval_pts.reshape(100,1,1)
                    pdf = combined_kde(eval_pts, smallest_N=None)

                    ax.plot(eval_pts.flatten(), pdf, color='k', linestyle='--')

                # plot the observations, if specified
                if plot_obs:
                    for obs in observations:
                        # delta function observations
                        ax.axvline(obs[pidx], ymax=0.1, color='b', \
                                                    alpha=0.4, zorder=-10)

                if plot_obs_samples:
                    for obs in samples:
                        ax.axvline(np.median(obs[:,pidx]), ymax=0.1, \
                                            color='b', alpha=0.4, zorder=-20)
                        # construct KDE from observations
                        obs_samps = pd.DataFrame(obs[:,pidx], columns=[param])
                        obs_kde = KDEModel.from_samples(\
                            label='obs_kde', \
                            samples=obs_samps, \
                            param_dict=marg_parameters, \
                            sensitivity=None)
                        eval_pts = np.linspace(obs_samps.min(), \
                                                obs_samps.max(), 100)
                        eval_pts = eval_pts.reshape(100,1,1)
                        pdf = obs_kde(eval_pts)

                        # scale down the pdf
                        pdf = 0.2 * pdf/(pdf.max()/pdf_max)

                        ax.fill_between(eval_pts.flatten(), \
                        y1=np.zeros_like(pdf), y2=pdf, color='b', alpha=0.05)

        # Titles and saving
        if model0:
            model0_name = ""
            for key,val in fixed_parameters.items():
                model0_name += population_parameters[key]['fullname'] + \
                    ' = {:0.1f}, '.format(population_parameters[key]['values'][val])
        else:
            model0_name='GW observations'
        plt.suptitle("Sampled model: {0:s}".format(model0_name), fontsize=50, y=0.99)

        if dirname:
            fname = os.path.join(dirname, 'marginalized_kdes_fixed_{:s}.png'.format(fixed_name))
        else:
            fname = './marginalized_kdes_fixed_{:s}.png'.format(fixed_name)

        plt.tight_layout()
        plt.savefig(fname)
        plt.close()




def plot_samples(samples, submodels_dict, model_names, channels, model0, name=None, hyper_idx=0, detectable_beta=False):
    """
    Plots the models that the chains are exploring, and histograms of the 
    branching fraction recovered for each model.

    :hyper_marg_idx: defines the index of the hyperparaeter in submodels_dict
    you wish to plot, marginalizing over the other parameters
    """

    Nhyper = len(submodels_dict)

    # setup the plots
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(len(channels), 3, wspace=0.2, hspace=0.2)
    ax_chains, ax_margs = [], []
    for cidx, channel in enumerate(channels):
        ax_chains.append(fig.add_subplot(gs[cidx, :2]))
        ax_margs.append(fig.add_subplot(gs[cidx, -1]))

    # plot the chains moving in beta space, colored by their model
    for chain in samples:
        for midx, model in submodels_dict[hyper_idx].items():
            smdl_locs = np.argwhere(chain[:,hyper_idx]==midx)[:,0]
            steps = np.arange(chain.shape[0])
            for cidx, channel in enumerate(channels):
                ax_chains[cidx].scatter(steps[smdl_locs], \
                    chain[smdl_locs,cidx+Nhyper], color=cp[midx], s=0.5, alpha=0.2)

    # plot the histograms on beta for each model
    # compactify all the chains in samples
    samples_allchains = np.reshape(samples, (samples.shape[0]*samples.shape[1], samples.shape[2]))
    basemdl_samps = len(np.argwhere(samples_allchains[:,hyper_idx]==0).flatten())
    h_max = 0
    for midx, model in submodels_dict[hyper_idx].items():
        smdl_locs = np.argwhere(samples_allchains[:,hyper_idx]==midx).flatten()
        mdl_samps = len(smdl_locs)
        if basemdl_samps > 0:
            BF = float(mdl_samps)/basemdl_samps
        else:
            BF = float(mdl_samps)
        for cidx, channel in enumerate(channels):
            h, bins, _ = ax_margs[cidx].hist(samples_allchains[smdl_locs, cidx+Nhyper], \
                orientation='horizontal', histtype='step', color=cp[midx], bins=50, \
                alpha=0.7, label=model+', BF={0:0.1e}'.format(BF))
            h_max = h.max() if h.max() > h_max else h_max


    # format plot
    for cidx, (channel, ax_chain, ax_marg) in enumerate(zip(channels, \
                                                ax_chains, ax_margs)):

        # plot the injected value
        if model0:
            if detectable_beta==True:
                ax_chain.axhline(model0[channel].rel_frac_detectable, color='k', \
                        linestyle='--', alpha=0.7)
                ax_marg.axhline(model0[channel].rel_frac_detectable, color='k', \
                        linestyle='--', alpha=0.7)
            else:
                ax_chain.axhline(model0[channel].rel_frac, color='k', \
                        linestyle='--', alpha=0.7)
                ax_marg.axhline(model0[channel].rel_frac, color='k', \
                        linestyle='--', alpha=0.7)

        # tick labels
        if cidx != len(channels)-1:
            ax_chain.set_xticklabels([])
            ax_marg.set_xticklabels([])
        ax_chain.set_yticks([0,0.5,1.0])
        ax_marg.set_yticklabels([])
        ax_chain.tick_params(axis='both', labelsize=20)
        ax_marg.tick_params(axis='both', labelsize=20)

        # legend
        if cidx == 0:
            ax_marg.legend(loc='center', bbox_to_anchor=[1.0,1.0], prop={'size':10})

        if cidx == len(channels)-1:
            ax_chain.set_xlabel('Step', fontsize=30)
            ax_marg.set_xlabel(r"p($\beta$)", fontsize=30)

        ax_chain.set_ylabel(r"$\beta_{%s}$" % format(channel), fontsize=30)
        ax_chain.set_xlim(0,samples.shape[1])
        ax_chain.set_ylim(0,1)
        ax_marg.set_xlim(0,h_max+10)
        ax_marg.set_ylim(0,1)


    # title
    if model0:
        # find the deepest model
        channel_depth = 0
        for channel in channels:
            if len(model0[channel].label.split('/')) > channel_depth:
                channel_depth = len(model0[channel].label.split('/'))
                deepest_channel = channel
        model0_name = model0[deepest_channel].label.split('/', 1)[1]
    else:
        model0_name='GW observations'
    plt.suptitle("True model: {0:s}".format(model0_name), fontsize=40)
    fname = 'samples'
    if detectable_beta==True:
        fname = 'samples_detectable'
    elif detectable_beta==False:
        fname = 'samples_underlying'
    if name:
        fname = fname+'_hyperidx'+str(hyper_idx)+'_'+name+'.png'
    else:
        fname = fname+'_hyperidx'+str(hyper_idx)+'.png'
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(fname)
    plt.close()




