import sys
import numpy as np
import scipy as sp
from scipy.stats import dirichlet
from scipy.stats import loguniform
import pandas as pd
from functools import reduce
import operator
import pdb
from tqdm import tqdm
import time
from scipy.special import logsumexp

import emcee
from emcee import EnsembleSampler


_valid_samplers = {'emcee': EnsembleSampler}

_sampler = 'emcee'
_prior = 'emcee_lnp'
_likelihood = 'emcee_lnlike'
_posterior = 'emcee_lnpost'

_nwalkers = 250
_nsteps = 1000
_fburnin = 0.4

_hyperparam_bounds = [[0.,0.5],[0.2,5.]]

"""
Class for initializing and running the sampler.
"""

class Sampler(object):
    """
    Sampler class.
    """
    def __init__(self, model_names, **kwargs):
        """
        model_names : list of str
            channel, chib, alpha of each submodel of form
            'CE/chi00/alpha02' or 'SMT/chi00'
        """

        # Store the number of population hyperparameters and formation channels
        hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
        Nhyper = np.max([len(x.split('/')) for x in hyperparams])
        channels = sorted(list(set([x.split('/')[0] for x in model_names])))

        # construct dict that relates submodels to their index number
        submodels_dict = {} #index dict. keys:0,1,2,3, items: particular models
        ctr=0 #associates with either chi_b or alpha (0 or 1)
        while ctr < Nhyper:
            submodels_dict[ctr] = {}
            hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
            for idx, model in enumerate(hyper_set): #idx associates with 0,1,2,3,(4) keys
                submodels_dict[ctr][idx] = model
            ctr += 1

        # note that ndim is (Nchannels-1) + Nhyper for the model indices -- branching fractions minus 1 plus number of hyperparams
        ndim = (len(channels)-1) + Nhyper

        # store as attributes
        self.Nhyper = Nhyper
        self.model_names = model_names
        self.channels = channels
        self.ndim = ndim
        self.submodels_dict = submodels_dict


        # kwargs
        self.sampler_name = kwargs['sampler'] if 'sampler' in kwargs \
                                                            else _sampler
        if self.sampler_name not in _valid_samplers.keys():
            raise NameError("Sampler {0:s} is unknown, check valid \
samplers!".format(self.sampler_name))
        self.sampler = _valid_samplers[self.sampler_name]

        self.prior_name = kwargs['prior'] if 'prior' in kwargs else _prior
        if self.prior_name not in _valid_priors.keys():
            raise NameError("Prior function {0:s} is unknown, check valid \
priors!".format(self.prior_name))
        self.prior = _valid_priors[self.prior_name]

        self.likelihood_name = kwargs['likelihood'] if 'likelihood' in kwargs \
                                                            else _likelihood
        if self.likelihood_name not in _valid_likelihoods.keys():
            raise NameError("Likelihood function {0:s} is unknown, check \
valid likelihoods!".format(self.likelihood_name))
        self.likelihood = _valid_likelihoods[self.likelihood_name]

        self.posterior_name = kwargs['posterior'] if 'posterior' in kwargs \
                                                            else _posterior
        if self.posterior_name not in _valid_posteriors.keys():
            raise NameError("Posterior function {0:s} is unknown, check valid \
posteriors!".format(self.posterior_name))
        self.posterior = _valid_posteriors[self.posterior_name]

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs \
                                                            else _nwalkers
        self.nsteps = kwargs['nsteps'] if 'nsteps' in kwargs else _nsteps
        self.fburnin = kwargs['fburnin'] if 'fburnin' in kwargs else _fburnin


        self.hyperparam_bounds = kwargs['hyperparam_bounds'] if 'hyperparam_bounds' in kwargs else _hyperparam_bounds

    def sample(self, pop_models, obsdata, use_flows, smallest_N, continuous_sample, prior_pdf, verbose=True):
        """
        Initialize and run the sampler

        pop_models : dict
            contains the normalising flow or KDE model instances for each channel/population model
        obsdata : array
            posterior samples of observations or mock observations for which to calculate the likelihoods,
            shape[Nobs x Nsample x Nparams]
        use_flows : bool
            True if using normalising flows for inference. If False, uses KDEs
        smallest_N : int
            Value by which to regularise the population distributions.
            See eq. 2 in Colloms et al.
        continuous_sample : bool
            True if performing continuous inference with normalising flows flows. If False, performs discrete inference (flows or KDEs)
        prior_pdf : array
            Prior value on each GW posterior sample in obsdata
        """

        # --- Set up initial point for the walkers
            #ndim encompasses the population hyperparameters and the branching fractions between channels
        p0 = np.empty(shape=(self.nwalkers, self.ndim))

        # first, for the population hyperparameters
        #selects points in uniform prior for chi_b
        p0[:,0] = np.random.uniform(self.hyperparam_bounds[0][0], self.hyperparam_bounds[0][1], size=self.nwalkers)
        #select points in log uniform in alphaCE (uniform over alphaCE indices, or loguniform in alphaCE values)
        if continuous_sample==False:
            p0[:,1] = np.random.uniform(self.hyperparam_bounds[1][0], self.hyperparam_bounds[1][1], size=self.nwalkers)
        else:
            p0[:,1] = loguniform.rvs(self.hyperparam_bounds[1][0], self.hyperparam_bounds[1][1], size=self.nwalkers)
        # second, for the branching fractions (we have Nchannel-1 betasin the inference because of the implicit constraint that Sum(betas) = 1
        _concentration = np.ones(len(self.channels))
        beta_p0 =  dirichlet.rvs(_concentration, p0.shape[0])
        p0[:,self.Nhyper:] = beta_p0[:,:-1]

        # --- Do the sampling
        #set arguments to pass to self.posterior
        posterior_args = [obsdata, pop_models, self.submodels_dict, self.channels, _concentration, use_flows, smallest_N, prior_pdf, self.hyperparam_bounds, continuous_sample]
        if verbose:
            print("Sampling...")
        #initialise emcee sampler with self.posterior as probability function
        sampler = self.sampler(self.nwalkers, self.ndim, self.posterior, args=posterior_args)
        
        #run sampling
        for idx, result in enumerate(sampler.sample(p0, iterations=self.nsteps)):
            if verbose:
                if (idx+1) % (self.nsteps/200) == 0:#progress bar
                    sys.stderr.write("\r  {0}% (N={1})".\
                                format(float(idx+1)*100. / self.nsteps, idx+1))
        if verbose:
            print("\nSampling complete!\n")

        # remove the burnin -- this removes some hyperpost samples at the start of the run before sampler equilibrates
        burnin_steps = int(self.nsteps * self.fburnin)
        self.Nsteps_final = self.nsteps - burnin_steps
        #chain output is of shape [number of chain, point in chain, value at that point]
        samples = sampler.chain[:,burnin_steps:,:]
        lnprb = sampler.lnprobability[:,burnin_steps:]

        # synthesize last betas, since they sum to unity
        last_betas = (1.0-np.sum(samples[...,self.Nhyper:], axis=2))
        last_betas = np.expand_dims(last_betas, axis=2)
        samples = np.concatenate((samples, last_betas), axis=2)

        self.samples = samples
        self.lnprb = lnprb



# --- Define the likelihood and prior

def lnp(x, submodels_dict, _concentration, hyperparam_bounds, continuous_sample):
    """
    Log of the prior. 
    Returns logL of -inf for points outside hyperparam_bounds.
    Prior is uniform within bounds for chi_b, and uniform over alpha_CE indices, log uniform over alpha_CE values.
    Dirichlet prior on betas given by _concentraion, conditional on the sum of the betas being one.
    """
    # first get prior on the hyperparameters, flat between the hyperparameter boundaries
    for hyper_idx in list(submodels_dict.keys()):
        hyperparam = x[hyper_idx]
        if ((hyperparam < hyperparam_bounds[hyper_idx][0]) | (hyperparam > hyperparam_bounds[hyper_idx][1])):
            return -np.inf

    # second, get the prior on the betas as a Dirichlet prior
    betas_tmp = np.asarray(x[len(submodels_dict):])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp)) #synthesize last beta
    if np.any(betas_tmp < 0.0):
        return -np.inf
    if np.sum(betas_tmp) != 1.0:
        return -np.inf

    if continuous_sample==False:
        alpha_CE_prior = 0
    else:
        alpha_CE_prior = loguniform.logpdf(x[1],a=hyperparam_bounds[1][0], b=hyperparam_bounds[1][1])

    # Dirchlet distribution prior for betas, plus uniform prior on log(alphaCE) values
    return dirichlet.logpdf(betas_tmp, _concentration) + alpha_CE_prior


def lnlike_cont(x, data, pop_models, submodels_dict, channels, prior_pdf, smallest_N, **kwargs): #data here is obsdata previously, and x is the point in log hyperparam space
    """
    Log of the likelihood for continuous model selection, using normalising flows. 
    Selects on model, then tests beta.

    x: array
        current position of walker in parameter space of hyperparameter *indices*
        shape [Nhyperparameters=2] even for channels with 1 hyperparameter
    data: array
        GW posterior samples or mock observations
        [Nobs x Nsample x Nparams]
    pop_models: dict
        contains the normalising flow or KDE model instances for each channel/population model
    submodels_dict: dict
        stores submodels to related to their index number by keys [0 or 1][0,1,2,3,4]
        where first is either chi_b or alpha, and the other is hyperparameter value
    channels: array/list of str
        list of formation channels used in inference in form 'CE'
    prior_pdf : array
            Prior value on each GW posterior sample in obsdata
    smallest_N : int
        Value by which to regularise the population distributions.
        See eq. 2 in Colloms et al.

    Returns
        log likelihood summed over events, accounting for detection efficiency
    """
    # get betas
    betas = np.asarray(x[len(submodels_dict):])
    betas = np.append(betas, 1-np.sum(betas))

    # Likelihood
    lnprob = np.zeros(data.shape[0])-np.inf

    # Detection effiency for this hypermodel
    alpha = 0

    #find log of alpha_CE for finding likelihood
    model_hyperparams = [x[0],np.log(x[1])]

    # Iterate over channels in this submodel, return likelihood of population model
    for channel, beta in zip(channels, betas):
        #get corresponding flow to channel
        smdl = pop_models[channel]
        #sum likelihood over channels, keep track of detection efficiency
        lnprob = logsumexp([lnprob, np.log(beta) + smdl(data, model_hyperparams[:smdl.conditionals], smallest_N, prior_pdf=prior_pdf)], axis=0)
        alpha += beta * smdl.get_alpha([model_hyperparams[:smdl.conditionals]])

    #returns lnprob summed over events (probability multiplied over events, divided by detection efficiency)
    return (lnprob-np.log(alpha)).sum()

def lnlike_disc(x, data, pop_models, submodels_dict, channels, prior_pdf, use_flows, smallest_N, **kwargs): #data here is obsdata previously, and x is the point in log hyperparam space
    """
    Log of the likelihood for discrete model selection, using either normalising flows or KDEs.
    Selects on model, then tests beta.

    x: array
        current position of walker in parameter space of hyperparameter *indices*
        shape [Nhyperparameters=2] even for channels with 1 hyperparameter
    data: array
        GW posterior samples or mock observations
        [Nobs x Nsample x Nparams]
    submodels_dict: dictionary
        stores submodels to related to their index number by keys [0 or 1][0,1,2,3,4]
        where first is either chi_b or alpha, and the other is hyperparameter value
    channels: array/list of str
        list of formation channels used in inference in form 'CE'
    prior_pdf : array
            Prior value on each GW posterior sample in obsdata
    use_flows : bool
        True if using normalising flows for inference. If False, uses KDEs
    smallest_N : int
        Value by which to regularise the population distributions.
        See eq. 2 in Colloms et al.

    
    Returns
        log likelihood summed over events, accounting for detection efficiency
    """
    model_list = []
    hyperparam_idxs = []
    for hyper_idx in list(submodels_dict.keys()):
        hyperparam_idxs.append(int(np.floor(x[hyper_idx])))
        model_list.append(submodels_dict[hyper_idx][int(np.floor(x[hyper_idx]))]) #finds where walker is in hyperparam space

    # get betas
    betas = np.asarray(x[len(submodels_dict):])
    betas = np.append(betas, 1-np.sum(betas))

    # Likelihood
    lnprob = np.zeros(data.shape[0])-np.inf

    # Detection effiency for this hypermodel
    alpha = 0

    # Iterate over channels in this submodel, return likelihood of population model
    #can't vectorise over this unless its a numpy array of flows, which doesn't seem like the best coding practice
    for channel, beta in zip(channels, betas):

        model_list_tmp = model_list.copy()
        model_list_tmp.insert(0,channel) #list with channel, 2 hypermodels (chi_b, alpha)
        
        #calls popModels to return likelihood of data given model and add contribution from this channel
        if use_flows==True:
            smdl = pop_models[channel]
            #LSE over channels
            #keep lnprob as shape [Nobs]
            conditional_hps = [smdl.hps[i][hyperparam_idxs[i]] for i in range(smdl.conditionals)]
            if len(conditional_hps) >1:
                conditional_hps = [conditional_hps[0],np.log(conditional_hps[1])]
            lnprob = logsumexp([lnprob, np.log(beta) + smdl(data, conditional_hps, smallest_N, prior_pdf=prior_pdf)], axis=0)
            #for CE alpha dictionary key is tuple, but for non-CE, keys are ints
            if channel == 'CE':
                alpha += beta * smdl.alpha[tuple(hyperparam_idxs)]
            else:
                alpha += beta * smdl.alpha[hyperparam_idxs[0]]
        else:
            smdl = reduce(operator.getitem, model_list_tmp, pop_models) #grabs correct submodel
            lnprob = logsumexp([lnprob, np.log(beta) + np.log(smdl(data, smallest_N))], axis=0)
            alpha += beta * smdl.alpha

    #returns lnprob summed over events (probability multiplied over events - see one channel eq D13 for full likelihood calc)
    return (lnprob-np.log(alpha)).sum()

def lnpost(x, data, pop_models, submodels_dict, channels, _concentration, use_flows, smallest_N, prior_pdf, hyperparam_bounds, continuous_sample):
    """
    Combines the prior and likelihood to give a log posterior probability 
    at a given point

    x : np array
        walker points in hyperparameters space to sample probability
    data : array
        GW observations of shape [Nobs, Nsamps, Nparams]
    pop_models : Dict
        population models represented by either KDEs or Flows
    submodels_dict: dictionary
        stores submodels to related to their index number by keys [0 or 1][0,1,2,3,4]
        where first is either chi_b or alpha, and the other is hyperparameter value
    channels: array of str
        list of formation channels used in inference in form 'CE'
    _concentration: list
        concentration to use for prior on Betas
    use_flows : bool
        True if using normalising flows for inference. If False, uses KDEs
    smallest_N : int
        Value by which to regularise the population distributions.
        See eq. 2 in Colloms et al.
    prior_pdf : array
            Prior value on each GW posterior sample in obsdata
    hyperparam_bounds : array
        lower and upper limits on the priors for chi_b and alpha_CE or their model indices
    continuous_sample : bool
        True if performing continuous inference with normalising flows flows. If False, performs discrete inference (flows or KDEs)
    
    Returns
        log likelihood plus log prior
    """
    # Prior
    log_prior = lnp(x, submodels_dict, _concentration, hyperparam_bounds, continuous_sample)
    if not np.isfinite(log_prior):
        return log_prior

    # Likelihood
    if continuous_sample==False:
        log_like = lnlike_disc(x, data, pop_models, submodels_dict, channels, prior_pdf, use_flows, smallest_N)
    else:
        log_like = lnlike_cont(x, data, pop_models, submodels_dict, channels, prior_pdf, smallest_N)
    
    return log_like + log_prior #evidence is divided out




_valid_priors = {'emcee_lnp': lnp}
_valid_likelihoods = {'emcee_lnlike': lnlike_cont, 'emcee_lnlike_disc': lnlike_disc}
_valid_posteriors = {'emcee_lnpost': lnpost}
