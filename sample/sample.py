import sys
import numpy as np
import scipy as sp
from scipy.stats import dirichlet
from scipy.stats import loguniform
import pandas as pd

from functools import reduce
import operator
import pdb
from scipy.special import logsumexp

from emcee import EnsembleSampler


# default sampler settings 
_sampler = 'emcee'
_prior = 'emcee_lnp'
_likelihood = 'emcee_lnlike'
_posterior = 'emcee_lnpost'
_nwalkers = 250
_nsteps = 1000
_fburnin = 0.2


"""
Class for initializing and running the sampler.
"""

class Sampler(object):
    """
    Sampler class.
    """
    def __init__(self, model_names, pop_param_dict, use_flows, \
                 continuous_sampling, **kwargs):
        """
        model_names : list of str
            channel, chib, alpha of each submodel of form
            'CE/chi00/alpha02' or 'SMT/chi00'
        pop_param_dict : dictionary
            contains infor about population hyperparameters
        use_flows : bool
            indicates whether normalizing flows are being used instead of KDEs
        continuous_sampling : bool
            indicates if continuous sampling is used instead of discrete
            only valid if using normalizing flows
        """

        # Check that use_flows and continuous_sampling are compatible
        if use_flows==False and continuous_sampling==True:
            raise NameError("Cannot perform continuous sampling unless using flows!")

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
        self.use_flows = use_flows
        self.continuous_sampling = continuous_sampling

        # determine whether to sample each parameter in log
        log_sampling = [pop_param_dict[p]['logsampling'] for p in pop_param_dict.keys()]
        if continuous_sampling==False and any(log_sampling)==True:
            raise ValueError("Cannot perform log sampling when doing discrete model sampling!")
        self.log_sampling = log_sampling

        # kwargs
        self.sampler_name = kwargs['sampler'] if 'sampler' in kwargs else _sampler
        if self.sampler_name not in _valid_samplers.keys():
            raise NameError("Sampler {0:s} is unknown, check valid samplers!".format(self.sampler_name))
        self.sampler = _valid_samplers[self.sampler_name]

        self.prior_name = kwargs['prior'] if 'prior' in kwargs else _prior
        if self.prior_name not in _valid_priors.keys():
            raise NameError("Prior function {0:s} is unknown, check valid priors!".format(self.prior_name))
        self.prior = _valid_priors[self.prior_name]

        self.likelihood_name = kwargs['likelihood'] if 'likelihood' in kwargs else _likelihood
        if self.likelihood_name not in _valid_likelihoods.keys():
            raise NameError("Likelihood function {0:s} is unknown, check valid likelihoods!".format(self.likelihood_name))
        self.likelihood = _valid_likelihoods[self.likelihood_name]

        self.posterior_name = kwargs['posterior'] if 'posterior' in kwargs else _posterior
        if self.posterior_name not in _valid_posteriors.keys():
            raise NameError("Posterior function {0:s} is unknown, check valid posteriors!".format(self.posterior_name))
        self.posterior = _valid_posteriors[self.posterior_name]

        self.nwalkers = kwargs['nwalkers'] if 'nwalkers' in kwargs else _nwalkers
        self.nsteps = kwargs['nsteps'] if 'nsteps' in kwargs else _nsteps
        self.fburnin = kwargs['fburnin'] if 'fburnin' in kwargs else _fburnin

        # set bounds for hyperparameters
        if continuous_sampling:
            hyperparam_bounds = []
            for p in pop_param_dict.keys():
                pmin = min(list(pop_param_dict[p]['values'].values()))
                pmax = max(list(pop_param_dict[p]['values'].values()))
                hyperparam_bounds.append([pmin,pmax])
        else:
            hyperparam_bounds = [[0, len(pop_param_dict[p]['values'].keys())] \
                                      for p in pop_param_dict.keys()]
        self.hyperparam_bounds = hyperparam_bounds

    def sample(self, models, obsdata, prior_pdf, smallest_N, verbose=True):
        """
        Initialize and run the sampler

        models : dict
            contains the normalising flow or KDE model instances for each channel/population model
        obsdata : array
            posterior samples of observations or mock observations for which to calculate the likelihoods,
            shape[Nobs x Nsample x Nparams]
        prior_pdf : array
            Prior value on each GW posterior sample in obsdata
        smallest_N : int
            Value by which to regularise the population distributions.
            See eq. 2 in Colloms et al.
        """

        # --- Set up initial point for the walkers
            #ndim encompasses the population hyperparameters and the branching fractions between channels
        p0 = np.empty(shape=(self.nwalkers, self.ndim))

        # first, for the population hyperparameters
        #selects points in uniform prior over hyperparameter indices (discrete case) or transformed hyperparameter values (continuous case)
        for hpidx in range(self.Nhyper):
            p0[:,hpidx] = np.random.uniform(self.hyperparam_bounds[hpidx][0], self.hyperparam_bounds[hpidx][1], size=self.nwalkers)
        # second, for the branching fractions (we have Nchannel-1 betasin the inference because of the implicit constraint that Sum(betas) = 1
        _concentration = np.ones(len(self.channels))
        beta_p0 =  dirichlet.rvs(_concentration, p0.shape[0])
        p0[:,self.Nhyper:] = beta_p0[:,:-1]

        # --- Do the sampling
        #set arguments to pass to self.posterior
        posterior_args = [obsdata, models, self.submodels_dict, self.channels, \
                prior_pdf, self.hyperparam_bounds, self.use_flows, self.continuous_sampling, \
                self.log_sampling, smallest_N, _concentration]
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

def lnp(x, submodels_dict, _concentration, hyperparam_bounds, \
        continuous_sampling, log_sampling):
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

    # Dirchlet distribution prior for betas, plus uniform prior on log(alphaCE) values
    return dirichlet.logpdf(betas_tmp, _concentration)


def lnlike(x, data, models, submodels_dict, channels, prior_pdf, \
           use_flows, continuous_sampling, smallest_N, **kwargs):
    """
    Log of the likelihood for model selection, using either KDEs or normalising flows. 
    Selects on model, then tests beta.

    x: array
        current position of walker in parameter space of hyperparameter *indices*
        shape [Nhyperparameters=2] even for channels with 1 hyperparameter
    data: array
        GW posterior samples or mock observations
        [Nobs x Nsample x Nparams]
    models: dict
        contains the normalising flow or KDE model instances for each channel/population model
    submodels_dict: dict
        stores submodels to related to their index number by keys [0 or 1][0,1,2,3,4]
        where first is either chi_b or alpha, and the other is hyperparameter value
    channels: array/list of str
        list of formation channels used in inference in form 'CE'
    prior_pdf : array
        Prior value on each GW posterior sample in obsdata
    use_flows : bool
        True if using normalising flows for inference. If False, uses KDEs
    continuous_sampling : bool
        True if doing continuous inference on hyperparameters, only valid if using flows
    smallest_N : int
        Value by which to regularise the population distributions.
        See eq. 2 in Colloms et al.

    Returns
        log likelihood summed over events, accounting for detection efficiency
    """

    # get betas
    betas = np.asarray(x[len(submodels_dict):])
    betas = np.append(betas, 1-np.sum(betas))

    # allocate likelihood 
    lnprob = np.zeros(data.shape[0])-np.inf

    # initialize detection effiency for this hypermodel
    alpha = 0

    # FIXME: Do we need to pass on log_sampling here?
    if continuous_sampling:
        model_hyperparams = x[:len(submodels_dict)]
    else:
        model_list = []
        #find hyperparameter index of walker
        hyperparam_idxs = []
        for hyper_idx in list(submodels_dict.keys()):
            hyperparam_idxs.append(int(np.floor(x[hyper_idx])))
            model_list.append(submodels_dict[hyper_idx][int(np.floor(x[hyper_idx]))])

    # Iterate over channels in this submodel, return likelihood of population model
    for channel, beta in zip(channels, betas):

        if continuous_sampling:
            # continuous hyperparameter sampling with normalizing flows
            smdl = models[channel]  # get corresponding flow to channel
            #sum likelihood over channels, keep track of detection efficiency
            lnprob = logsumexp([lnprob, np.log(beta) + smdl(data, model_hyperparams[:smdl.conditionals], smallest_N, data_prior=prior_pdf)], axis=0)
            alpha += beta * smdl.get_alpha(model_hyperparams[:smdl.conditionals])

        elif use_flows==True:
            # discrete hyperparameter sampling with normalizing flows
            smdl = models[channel]
            #identify submodel conditional values
            conditional_hps = [smdl.hp_vals[i][hyperparam_idxs[i]] for i in range(smdl.conditionals)]
            #LSE over channels
            lnprob = logsumexp([lnprob, np.log(beta) + smdl(data, conditional_hps, smallest_N, data_prior=prior_pdf)], axis=0)
            #for multiple hyperparameters, dictionary key is tuple, but for single hyperparameters, keys are ints
            if smdl.conditionals > 1:
                hyperparam_idxs = tuple(hyperparam_idxs)
            else:
                hyperparam_idxs = hyperparam_idxs[0]
            alpha += beta * smdl.alpha[hyperparam_idxs]

        else:
            model_list_tmp = model_list.copy()
            model_list_tmp.insert(0,channel) #list with channel and hypermodels
            smdl = reduce(operator.getitem, model_list_tmp, models) #grabs correct submodel
            lnprob = logsumexp([lnprob, np.log(beta) + np.log(smdl(data, smallest_N, data_prior=prior_pdf))], axis=0)
            alpha += beta * smdl.alpha

    #returns lnprob summed over events (probability multiplied over events - see one channel eq D13 for full likelihood calc)
    return (lnprob-np.log(alpha)).sum()


def lnpost(x, data, models, submodels_dict, channels, prior_pdf, hyperparam_bounds, \
           use_flows, continuous_sampling, log_sampling, smallest_N, _concentration):
    """
    Combines the prior and likelihood to give a log posterior probability 
    at a given point

    x : np array
        walker points in hyperparameters space to sample probability
    data : array
        GW observations of shape [Nobs, Nsamps, Nparams]
    models : Dict
        population models represented by either KDEs or Flows
    submodels_dict: dictionary
        stores submodels to related to their index number by keys [0 or 1][0,1,2,3,4]
        where first is either chi_b or alpha, and the other is hyperparameter value
    channels: array of str
        list of formation channels used in inference in form 'CE'
    prior_pdf : array
        Prior value on each GW posterior sample in obsdata
    hyperparam_bounds : array
        lower and upper limits on the priors for chi_b and alpha_CE or their model indices
    use_flows : bool
        True if using normalising flows for inference. If False, uses KDEs
    continuous_sampling : bool
        True if performing continuous inference with normalising flows flows. If False, performs discrete inference (flows or KDEs)
    log_sampling : list of bool
        Same length as number of hyperparameters, with indices indicating whether sampling in logspace
    smallest_N : int
        Value by which to regularise the population distributions.
        See eq. 2 in Colloms et al.
    _concentration: list
        concentration to use for Dirichlet prior on Betas

    Returns
        log likelihood plus log prior
    """
    # Prior
    log_prior = lnp(x, submodels_dict, _concentration, \
                    hyperparam_bounds, continuous_sampling, log_sampling)
    if not np.isfinite(log_prior):
        return log_prior

    # Likelihood
    log_like = lnlike(x, data, models, submodels_dict, channels, \
                      prior_pdf, use_flows, continuous_sampling, smallest_N)
    
    return log_like + log_prior #evidence is divided out




_valid_samplers = {'emcee': EnsembleSampler}
_valid_priors = {'emcee_lnp': lnp}
_valid_likelihoods = {'emcee_lnlike': lnlike}
_valid_posteriors = {'emcee_lnpost': lnpost}
