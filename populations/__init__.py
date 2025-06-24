import sys
import os
import pickle
import itertools
import copy
from tqdm import tqdm
import multiprocessing
from functools import partial
import warnings
import pdb
import json

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.special import logit
from scipy.special import logsumexp
from scipy.special import expit
from sklearn.model_selection import train_test_split
from .population_utils.flow import NFlow
from .population_utils.bounded_Nd_kde import Bounded_Nd_kde
from .population_utils.transform import mtotq_to_mchirp, mtoteta_to_mchirpq, eta_to_q, mchirpq_to_m1m2
from .population_utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# mock uncertainty defaults
_posterior_sigmas = {"mchirp": 1.512, "q": 0.166, "chieff": 0.1043, "z": 0.0463}   # TOUPDATE
_snrscale_sigmas = {"mchirp": 0.04, "eta": 0.03, "chieff": 0.14}      # TOUPDATE
# projection factor interpolant from Dominik et al. 2015
proj_factor = projection_factor_Dominik2015_interp()

# default values if not provided
_normalization_bounds_defaults = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}
_kde_bandwidth_default = 0.01
_max_samps_default = int(1e5)
_store_optimal_snrs_default = False

"""
Set of classes used to construct statistical models of populations.
"""

class Model(object):
    """
    Base model class. Mostly used to root the inheritance tree.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        return None

    def get_alpha(self, samples, sensitivity):
        # check that the provdided sensitivity series is in the dataframe
        if sensitivity is not None:
            if 'pdet_'+sensitivity not in samples.columns:
                raise ValueError(f"{sensitivity} was specified for your detection weights, but cannot find the column 'pdet_{sensitivity}' in the samples datafarme!")
            # get *\alpha* for each model, defined as 
            #   \int p(\theta|\lambda) Pdet(\theta) d\theta
            # if cosmological weights are provided, do mock draws from the pop
            if 'weight' in samples.keys():
                mock_samp = samples.sample(int(1e6), \
                    weights=(samples['weight']/len(samples)), replace=True)
            else:
                mock_samp = samples.sample(int(1e6), replace=True)
            alpha = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
        else:
            alpha = 1.0
        return alpha

class KDEModel(Model):
    @staticmethod
    def from_samples(label, samples, param_dict, sensitivity=None, **kwargs):
        """
        Generate a KDE model instance from `samples`, where `params` are \
        series in the `samples` dataframe. Additional *kwargs* can be passed \
        specifying KDE bandwidth. If `weight` is a column in your population \
        model, will assume this is the cosmological weight of each sample, and \
        will include this in the construction of all your KDEs. If `sensitivity` \
        is provided, samples used to generate the detection-weighted KDE will be \
        weighted according to the key in the argument `pdet_${sensitivity}`.

        Inputs:
        label : str
            submodel label of form CE/chi00/alpha02
        samples : pandas Dataframe
            binary samples from population synthesis.
        param_dict : dict
            dictionary of event-level parameters, including parameter bounds
        sensitivity : str
            dataframe column name for detection probability: 'pdet_${sensitivity}'
        max_samps : int
            maximum number of samples to use for each KDE
        kde_bandwidth : float
            bandwidth of KDEs
        store_optimal_snrs : bool
            whether to store optimal SNRs for each sample (only used if mock uncertainty is SNR-dependent)
        """

        max_samps = kwargs['max_samps'] if 'max_samps' in kwargs else _max_samps_default
        kde_bandwidth = kwargs['kde_bandwidth'] if 'kde_bandwidth' in kwargs else _kde_bandwidth_default
        store_optimal_snrs = kwargs['store_optimal_snrs'] if 'store_optimal_snrs' in kwargs else _store_optimal_snrs_default
        
        #get alpha from parent class
        alpha = Model().get_alpha(samples, sensitivity)

        # specify the series that we plan to keep along, adding weights and detection info to this
        series_to_keep = list(param_dict.keys())
        series_to_keep.extend(['weight'])
        if 'weight' not in samples.keys():
            samples['weight'] = np.ones(len(samples))

        if not sensitivity:
            samples['pdet_'] = np.ones(len(samples))
            series_to_keep.extend(['pdet_'])
        else:
            series_to_keep.extend(['pdet_'+sensitivity])

        # get optimal SNRs for this sensitivity, if using SNR-dependent mock measurement uncertainty
        if store_optimal_snrs:
            if 'snropt_'+sensitivity not in samples.columns:
                raise ValueError(f"To use SNR-dependent mock measurement uncertainty, you also need to supply optimal SNRs with the key 'snropt_{sensitivity}'")
            series_to_keep.extend(['snropt_'+sensitivity])
        else:
            samples['snropt_'] = np.nan*np.ones(len(samples))
            series_to_keep.extend(['snropt_'])

        # downsample population
        N_samps = max_samps if max_samps else _max_samps_default
        if len(samples) > N_samps:
            samples = samples.sample(N_samps)

        # get normalization, revert to defaults if not specified
        params = list(param_dict.keys())
        normalization_bounds = {}
        for p in params:
            normalization_bounds[p] = param_dict[p]['limits'] \
                if param_dict[p]['limits'] \
                else _normalization_bounds_defaults[p]

        # get KDE bandwidth, revert to defaults if not specified
        bandwidth = kde_bandwidth if kde_bandwidth else _kde_bandwidth_default

        # get samples for the parameters in question, as well as weights/pdets/snrs
        samples = pd.DataFrame(samples[series_to_keep])

        return KDEModel(label, samples, params, bandwidth, sensitivity, \
                                alpha, normalization_bounds)


    def __init__(self, label, samples, params, bandwidth, \
                    sensitivity, alpha, normalization_bounds, \
                    detectable=False):
        super()
        self.label = label
        self.samples = samples
        self.params = params
        self.bandwidth = bandwidth
        self.sensitivity = sensitivity
        self.alpha = alpha
        self.normalization_bounds = normalization_bounds
        self.detectable = detectable

        # Save range of each parameter
        self.sample_range = {}
        for param in params:
            self.sample_range[param] = (samples[param].min(), samples[param].max())

        # Normalize data s.t. they all are on the unit cube
        bounds = list(normalization_bounds.values())
        self.bounds = bounds
        kde_samples = normalize_samples(np.asarray(samples[params]), bounds)
        # also need to scale pdf by parameter range, so save this
        pdf_scale = scale_to_unity(bounds)
        self.pdf_scale = pdf_scale

        # add a little bit of scatter to samples that have the exact same values, as this will freak out the KDE generator
        for idx, param in enumerate(params):
            if len(np.unique(kde_samples[:,idx]))==1:
                kde_samples[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=kde_samples.shape[0])

        # Get the KDE objects, specify function for pdf
        # This custom KDE handles multiple dimensions, bounds, and weights, and takes in samples (Ndim x Nsamps)
        if detectable==True:
            w = samples['weight'] * samples['pdet_'+sensitivity]
        else:
            w = samples['weight']
        kde = Bounded_Nd_kde(kde_samples.T, weights=w, bw_method=bandwidth, bounds=[(0,1)]*len(params))
        self.pdf = lambda x: kde(normalize_samples(x, bounds).T) / pdf_scale
        self.kde = kde
        self.kde_samples = kde_samples

        self.cached_values = None

    def sample(self, N=1):
        """
        Samples KDE and denormalizes sampled data
        """
        kde = self.kde
        samps = denormalize_samples(kde.bounded_resample(N).T, self.bounds)
        return samps

    def rel_frac(self, beta):
        """
        Stores the relative fraction of samples that are drawn from this KDE model
        """
        self.rel_frac = beta

    def rel_frac_detectable(self, beta_det):
        """
        Stores the relative detectable fraction of samples that are drawn from this KDE model
        """
        self.rel_frac_detectable = beta_det

    def Nobs_from_beta(self, Nobs):
        """
        Stores the branching fraction of the underlying population
        """
        self.Nobs_from_beta = Nobs

    def freeze(self, data, smallest_N, data_prior=None, multiproc=False):
        """
        Caches the values of the model likelihood at the data points provided. This
        is useful to construct the hierarchal model likelihood since it
        is evaluated many times, but only needs to be once
        because it's a fixed value, dependent only on the observations
        """
        self.cached_values = None
        data_prior = data_prior if data_prior is not None else np.ones((data.shape[0],data.shape[1]))
        likelihood_vals = []

        if multiproc == False:
            for (d,d_prior) in tqdm(zip(data,data_prior), total=len(data)):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                d_prior = d_prior.reshape((1, d_prior.shape[0]))
                likelihood_vals.append(self(d, smallest_N, d_prior))
        else:
            # FIXME: this is not working
            processes = []
            for (d,d_prior) in zip(data,data_prior):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                d_prior = d_prior.reshape((1, d_prior.shape[0]))
                p = multiprocessing.Process(target=self, args=(d, smallest_N, d_prior))
                processes.append(p)

            for p in tqdm(processes):
                p.start()
            
            #func = partial(self, smallest_N=smallest_N)
            # get data in correct format for initializing multiple processes
            #with multiprocessing.Pool(processes=4) as pool:
                # run the likelihood function in parallel
            #    likelihood_vals = list(tqdm(pool.starmap(func, zip(data, data_prior)), total=len(data)))
            # get data in correct format for initializing multiple processes
            """multiproc_data = []
            for (d,d_pdf) in zip(data,data_prior):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                d_pdf = d_pdf.reshape((1, d_pdf.shape[0])) 
                multiproc_data.append((d, smallest_N, d_pdf))"""

            # run multiprocessing
            #with multiprocessing.Pool(processes=Nproc) as pool:
            #    likelihood_vals = list(tqdm(pool.starmap(test, multiproc_data), \
            #                                total=len(multiproc_data)))
            #pool = multiprocessing.Pool(processes=Nproc)
            #manager = multiprocessing.Manager()
            #return_dict = manager.dict()
            """processes = []
            for idx, (d,d_pdf) in tqdm(enumerate(zip(data,data_prior)), total=len(data)):
                d = d.reshape((1, d.shape[0], d.shape[1]))
                d_pdf = d_pdf.reshape((1, d_pdf.shape[0]))
                p = multiprocessing.Process(target=self, args=(d,smallest_N,d_pdf,))
                processes.append(p)
                p.start()
            for process in processes:
                process.join()"""

            """for i in sorted(list(return_dict.keys())):
                likelihood_vals.append(return_dict[i])"""

        likelihood_vals = np.asarray(likelihood_vals).flatten()
        self.cached_values = likelihood_vals

    def __call__(self, data, smallest_N=None, data_prior=None):#, proc_idx=None, return_dict=None):
        """
        Calculate the likelihood of the observations give a particular hypermodel. \
        The expectation is that "data" is a [Nobs x Nsample x Nparams] array. \
        If data_prior is None, each observation is expected to have equal \
        posterior probability. Otherwise, the prior weights should be \
        provided as the dimemsions [samples(Nobs), samples(Nsamps)].
        """
        if self.cached_values is not None:
            return self.cached_values

        likelihood = np.ones(data.shape[0]) * 1e-50
        data_prior = data_prior if data_prior is not None else np.ones((data.shape[0],data.shape[1]))
        data_prior[data_prior==0] = 1e-50

        #SC: can this be vectorised?
        for idx, (obs, d_pdf) in enumerate(zip(np.atleast_3d(data),data_prior)):
            # Evaluate the KDE at the samples
            likelihood_per_samp = self.pdf(obs) / d_pdf
            likelihood[idx] += (1.0/len(obs)) * np.sum(likelihood_per_samp)
        # store value for multiprocessing TODELETE
        #if return_dict is not None:
        #    return_dict[proc_idx] = likelihood

        if smallest_N is not None:
            # population probability plus uniform regularisation
            pi_reg = 1/(smallest_N+1)
            q_weight = smallest_N/(smallest_N+1)
            likelihood = (q_weight * likelihood) + pi_reg
        return likelihood

    def marginalize(self, params, bandwidth=None, detectable=False):
        """
        Generate a new, lower dimensional, KDEModel from the parameters in [params]
        """
        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        norm_bounds = {}
        for p in params:
            norm_bounds[p] = self.normalization_bounds[p]

        return KDEModel(label, self.samples, params, \
                        bandwidth, sensitivity=self.sensitivity, \
                        alpha=1.0, normalization_bounds=norm_bounds, \
                        detectable=detectable)


    def generate_observations(self, Nobs, verbose=False):
        """
        Generates samples from density estimate model. This will generated Nobs samples, 
          storing the attribute 'self.observations' with dimensions [Nobs x Nparam] 
        """
        if verbose:
            print("  drawing {} observations from channel {}...".format(Nobs, self.label))

        # get SNR threshold
        self.snr_thresh = _PSD_defaults['snr_network'] if 'network' in self.sensitivity \
            else _PSD_defaults['snr_single']

        # choose detected systems based on cosmological weight and detection probability
        obs = self.samples.sample(n=Nobs, weights=(self.samples['pdet_'+self.sensitivity]*self.samples['weight']))

        # reset dataframe indices for observed samples
        obs = obs.reset_index(drop=True)

        self.observations = obs
        return obs


    def measurement_uncertainty(self, Nsamps, method='delta', observation_noise=False, verbose=False):
        """
        Mocks up measurement uncertainty from observations using specified method
        """
        if verbose:
            print("    mocking up observation uncertainties for the {} channel using the '{}' method...".format(self.label, method))

        params = self.params

        if method=='delta':
            # assume a delta function measurement
            obsdata = np.expand_dims(self.observations, 2)
            return obsdata

        # set up obsdata as [obs, params, samples]
        obsdata = np.zeros((self.observations.shape[0], Nsamps, len(params)))
        
        # for 'gwevents', assume snr-independent measurement uncertainty based on the typical values for events in the catalog
        if method == "events":
            for idx, obs in self.observations.iterrows():
                for pidx, param in enumerate(self.params):
                    mu = obs[param]
                    sigma = _posterior_sigmas[param]
                    low_lim = self.normalization_bounds[param][0]
                    high_lim = self.normalization_bounds[param][1]

                    # construnct gaussian and drawn samples
                    dist = norm(loc=mu, scale=sigma)

                    # if observation_noise is specified, wiggle around the observed value
                    if observation_noise==True:
                        mu_obs = dist.rvs()
                        dist = norm(loc=mu_obs, scale=sigma)

                    samps = dist.rvs(Nsamps)

                    # reflect samples if drawn past the parameters bounds
                    above_idxs = np.argwhere(samps>high_lim)
                    samps[above_idxs] = high_lim - (samps[above_idxs]-high_lim)
                    below_idxs = np.argwhere(samps<low_lim)
                    samps[below_idxs] = low_lim + (low_lim - samps[below_idxs])

                    obsdata[idx, :, pidx] = samps


        # for 'snr', use SNR-dependent measurement uncertainty following procedures from Fishbach, Holz, & Farr 2018 (2018ApJ...863L..41F)
        # NOTE: this method is a bit of a hack, look into this more
        if method == "snr":
            for idx, obs in self.observations.iterrows():

                # to use SNR-dependent uncertainty, we need to make sure that chirp mass/mass ratio parameters are supplied
                if set(['mchirp','q']).issubset(set(params)):
                    mc_true = obs['mchirp']
                    q_true = obs['q']
                elif set(['mtot','q']).issubset(set(params)):
                    mc_true = mtotq_to_mchirp(obs['mtot'], obs['q'])
                    q_true = obs['q']
                elif set(['mtot','eta']).issubset(set(params)):
                    mc_true, q_true = mtoteta_to_mchirpq(obs['mtot'], obs['q'])
                else:
                    raise ValueError("You need to have a mass and mass ratio parameter to use SNR-weighted uncertainty!")

                z_true = obs['z']
                mcdet_true = mc_true*(1+z_true)
                eta_true = q_true * (1+q_true)**(-2)
                dL_true = cosmo.luminosity_distance(z_true).to(u.Gpc).value
                # randomly choose projection factor according to the distribution from Dominik et al. 2015
                Theta_true = proj_factor(np.random.random())
                snr_true = obs['snropt_'+self.sensitivity] * Theta_true

                # apply Gaussian noise to SNR
                snr_obs = snr_true + np.random.normal(loc=0, scale=1)

                # get the snr-weighted sigma for the detector-frame chirp mass, and draw samples
                mc_sigma = _snrscale_sigmas['mchirp']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    mcdet_obs = float(10**(np.log10(mcdet_true) + norm.rvs(loc=0, scale=mc_sigma, size=1)))
                else:
                    mcdet_obs = mcdet_true
                mcdet_samps = 10**(np.log10(mcdet_obs) + norm.rvs(loc=0, scale=mc_sigma, size=Nsamps))

                # get the snr-weighted sigma for eta, and draw samples
                eta_sigma = _snrscale_sigmas['eta']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    eta_obs = float(truncnorm.rvs(a=(0-eta_true)/eta_sigma, b=(0.25-eta_true)/eta_sigma, loc=eta_true, scale=eta_sigma, size=1))
                else:
                    eta_obs = eta_true
                eta_samps = truncnorm.rvs(a=(0-eta_obs)/eta_sigma, b=(0.25-eta_obs)/eta_sigma, loc=eta_obs, scale=eta_sigma, size=Nsamps)

                # get samples for projection factor (use the true value as the observed value)
                # Note that our Theta is the projection factor (between 0 and 1), rather than the Theta from Finn & Chernoff 1993
                snr_opt = obs['snropt_'+self.sensitivity]
                Theta_sigma = 0.3 / (1.0 + snr_opt/self.snr_thresh)
                Theta_samps = truncnorm.rvs(a=(0-Theta_true)/Theta_sigma, b=(1-Theta_true)/Theta_sigma, loc=Theta_true, scale=Theta_sigma, size=Nsamps)

                # get luminosity distance and redshift observed samples
                dL_samps = dL_true * (Theta_samps/Theta_true)
                z_samps = np.asarray([z_at_value(cosmo.luminosity_distance, d) for d in dL_samps*u.Gpc])

                # get source-frame chirp mass and other mass parameters
                mc_samps = mcdet_samps / (1+z_samps)
                q_samps = eta_to_q(eta_samps)
                m1_samps, m2_samps = mchirpq_to_m1m2(mc_samps,q_samps)
                mtot_samps = (m1_samps + m2_samps)

                for pidx, param in enumerate(params):
                    if param=='mchirp':
                        obsdata[idx, :, pidx] = mc_samps
                    elif param=='mtot':
                        obsdata[idx, :, pidx] = mtot_samps
                    elif param=='q':
                        obsdata[idx, :, pidx] = q_samps
                    elif param=='eta':
                        obsdata[idx, :, pidx] = eta_samps
                    elif param=='chieff':
                        chieff_true = obs['chieff']
                        chieff_sigma = _snrscale_sigmas['chieff']*self.snr_thresh / snr_obs
                        if observation_noise==True:
                            chieff_obs = float(truncnorm.rvs(a=(-1-chieff_true)/chieff_sigma, b=(1-chieff_true)/chieff_sigma, loc=chieff_true, scale=chieff_sigma, size=1))
                        else:
                            chieff_obs = chieff_true
                        chieff_samps = truncnorm.rvs(a=(-1-chieff_obs)/chieff_sigma, b=(1-chieff_obs)/chieff_sigma, loc=chieff_obs, scale=chieff_sigma, size=Nsamps)
                        obsdata[idx, :, pidx] = chieff_samps
                    elif param=='z':
                        obsdata[idx, :, pidx] = z_samps

        return obsdata


def normalize_samples(samples, bounds):
    """
    Normalizes samples to range [0,1] for the purposes of KDE construction
    """
    norm_samples = np.transpose([((x-b[0])/(b[1]-b[0])) for x, b in \
                                        zip(samples.T, bounds)])
    return norm_samples


def denormalize_samples(norm_samples, bounds):
    """
    Denormalizes samples that are drawn from the normalzed KDE
    """
    samples = np.transpose([(x*(b[1]-b[0]) + b[0]) for x, b in \
                                        zip(norm_samples.T, bounds)])
    return samples


def scale_to_unity(bounds):
    """
    Provides scale factor to renormalize pdf evaluation on the original 
    bounds of the data
    """
    ranges = [b[1]-b[0] for b in bounds]
    scale_factor = np.prod(ranges)
    return scale_factor

class FlowModel(Model):
    @staticmethod
    def from_samples(channel, samples, param_dict, channel_hyperparams, smdl_indxs_combos, random_seed, sensitivity=None):
        """
        Generate a normalising flow model instance from `samples`, where the keys in `params_dict` are a series in the `samples` dataframe. 
        
        If `weight` is a column in the population model, will assume this is the cosmological weight of each sample,
        and will include this in the training of the flow. 
        If `sensitivity` is provided, the detection efficiency will be calculated according to `pdet_*sensitivity*`.
        Parameters
        ----------
        channel : str
            channel label to identify channel in flow configuration file
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters, cosmo_weights, pdet weights, optimal snrs
        param_dict : dict
            dictionary to identify the binary parameters in samples 
            structured {'parameter key' : {'limits':(X,Y), 'fullname':''}}
        channel_hyperparams : dict
            dictionary with the corresponding hyperparameters to this formation channel
            structured {'hyperparameter name' : {'values':{'submodel key':submodel value}, 'fullname':''}
        smdl_indxs_combos : list
            The combinations of hyperparameter indices used as keys in the samples, combined_weights, and alpha dictionaries
        sensitivity : str
            Desired detector sensitivity consistent with the string following the `pdet` and `snropt` columns in the population dataframes.
            Used to construct a detection-weighted population model, as well as for drawing samples from the underlying population
            to calculate the detection efficiency
        
        Returns
        ----------
        FlowModel : obj
        """

        #set model keys for specified training data
        model_keys = smdl_indxs_combos

        #initialise dictionaries of alpha, cosmo_weights, pdet, optimal_snrs, and combined_weights for each submodel
        alpha = dict.fromkeys(samples.keys())
        cosmo_weights= dict.fromkeys(samples.keys())
        combined_weights= dict.fromkeys(samples.keys())

        for model_idxs in model_keys:
            #grab samples for each submodel according to key in samples dict
            if model_idxs.ndim > 0:
                dict_key = tuple(model_idxs)
            else:
                dict_key = model_idxs
            sbml_samps = samples[dict_key]

            #check that defined sensitivity exists in submodel dataframe
            if sensitivity is not None:
                if 'pdet_'+sensitivity not in sbml_samps.columns:
                    raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")

            # get *\alpha* for each model, defined as \int p(\theta|\lambda) Pdet(\theta) d\theta
            alpha[dict_key]=Model().get_alpha(sbml_samps, sensitivity)

            ### GET WEIGHTS ###
            # if cosmological weights are provided...
            if 'weight' in sbml_samps.keys():
                cosmo_weights[dict_key] = np.asarray(sbml_samps['weight']) 
            else:
                cosmo_weights[dict_key] = np.ones(len(sbml_samps))

            # Normalise the cosmological weights. If wanted detection weighted samples, cosmo weigths could be combined with pdets
            if (cosmo_weights[dict_key] is not None):
                combined_weights[dict_key] = (cosmo_weights[dict_key] / np.sum(cosmo_weights[dict_key]))
            else:
                combined_weights[dict_key] = np.ones(len(sbml_samps))
        return FlowModel(channel, samples, param_dict, channel_hyperparams, combined_weights, alpha, model_keys)


    def __init__(self, channel, samples, param_dict, channel_hyperparams, combined_weights, alpha, model_keys):
        """
        Initialisation for FlowModel object.
        Parameters
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters, cosmo_weights, pdet weights, optimal snrs
        param_dict : dict
            dictionary to identify the binary parameters in samples 
            structured {'parameter key' : {'limits':(X,Y), 'fullname':''}}
        channel_hyperparams : dict
            dictionary with the corresponding hyperparameters to this formation channel
            structured {'hyperparameter name' : {'values':{'submodel key':submodel value}, 'fullname':''}
        combined_weights : dict
            cosmo weights corresponding to the model samples
        alpha : dict
            detection efficiency values for each submodel
        model_keys : list
            The combinations of hyperparameter indices used as keys in the samples, combined_weights, and alpha dictionaries
        """

        super()
        self.channel_label = channel
        self.samples = samples
        self.param_dict = copy.deepcopy(param_dict)

        #initialises list of population hyperparameters model names and values from hyperparameter dictionary
        #set hyperparameters to log of the original values if transform is specified as log in dictionary
        #Note that if loading a flow model with a hyperparameter specified with a log transform, the flow needs
        #to have previously been trained with that hyperparameter specified with a log transform.
        self.hyperparam_models = []
        self.hp_vals = []
        for hp in channel_hyperparams:
            self.hyperparam_models.append(list(channel_hyperparams[hp]['values'].keys()))
            if channel_hyperparams[hp]['transform'] == 'log':
                self.hp_vals.append(list(np.log(list(channel_hyperparams[hp]['values'].values()))))
            else:
                self.hp_vals.append(list(channel_hyperparams[hp]['values'].values()))

        #number of binary parameters
        self.no_params = len(param_dict.keys())
        #dimensionailty of non-branching ratio hyperparameters
        self.conditionals = len(self.hyperparam_models)

        #set weights and detection efficienies as class properties
        self.combined_weights = combined_weights
        self.alpha = alpha

        #initialise the keys for the samples dicts and how many training submodels exist for this channel
        self.model_keys = model_keys
        self.total_smdls = len(model_keys)


    def map_samples(self):
        """
        Maps samples with logistic or arctanh mapping and 
        divide samples, conditionals, and weights into training and validation sets.
        
        Returns
        -------
        training_data : array
            data samples to be used for training the normalising flow.
            [samples:hyperparameters:weights]
        val_data : array
            data samples to be used for validating the normalising flow.
            [samples, hyperparameters, weights]
        """

        print('Mapping population synthesis samples for training...')

        #measure no_samples in models and identify samples with weights below fmin
        model_size = np.zeros(self.total_smdls)
        cumulsize = np.zeros(self.total_smdls)
        weights_idxs = []

        for i, model_idxs in enumerate(self.model_keys):
            #find corresponding submodel key
            if model_idxs.ndim > 0:
                dict_key = tuple(model_idxs)
            else:
                dict_key = model_idxs
            weights_temp=np.asarray(self.combined_weights[dict_key])
            weights_idxs.append(np.argwhere((weights_temp) > np.finfo(np.float32).tiny))
            model_size[i] = np.shape(weights_idxs[i])[0]
            cumulsize[i] = np.sum(model_size)

        #initialise array to move samples and weights to
        self.no_binaries = int(cumulsize[-1])
        models = np.zeros((self.no_binaries, self.no_params))
        weights = np.zeros((self.no_binaries, 1))
        cumulsize = np.append(cumulsize, 0)

        #move samples and weights from dictionaries into array
        for i, model_idxs in enumerate(self.model_keys):
            if model_idxs.ndim > 0:
                dict_key = tuple(model_idxs)
            else:
                dict_key = model_idxs
            models[int(cumulsize[i-1]):int(cumulsize[i])]=np.reshape(np.asarray(self.samples[dict_key][self.param_dict.keys()])[weights_idxs[i]],(-1,len(self.param_dict.keys())))
            weights[int(cumulsize[i-1]):int(cumulsize[i])]=np.asarray(self.combined_weights[dict_key])[weights_idxs[i]]

        models_stack = np.copy(models)
        #map samples with logistic mapping before dividing into training and validation data
        for pidx, param in enumerate(self.param_dict):
            try:
                if self.param_dict[param]['transf'] == 'logit':
                    models_stack[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'] = self.logistic(models_stack[:,pidx],\
                        wholedataset=True, rescale_max=self.param_dict[param]['limits'][1])
                elif self.param_dict[param]['transf'] == 'tanh':
                    models_stack[:,pidx] = np.arctanh(models_stack[:,pidx])
            except:
                raise KeyError(f'Must specify how to transform {param} with `transf` entry in `event-parameter-dict`.')

        #repeat subpopulation hyperparameter values Nsamps times for each combination of hyperparameter values
        hp_combos = np.squeeze(list(product(*self.hp_vals)))
        hps_stack = np.repeat(hp_combos, (model_size).astype(int), axis=0)

        #reshape conditionals and weights
        hps_stack = np.reshape(hps_stack,(-1,self.conditionals))
        weights = np.reshape(weights,(-1,1))

        #shuffle and split the training data, conditional data, and sample weights into training and validation sets
        train_models_stack, validation_models_stack, train_weights, validation_weights, training_hps_stack, validation_hps_stack = \
                train_test_split(models_stack, weights, hps_stack, shuffle=True, train_size=0.8)

        #concatenate training and validation data in order: samples, hyperparameters, weights
        training_data = np.concatenate((train_models_stack, training_hps_stack, train_weights), axis=1)
        val_data = np.concatenate((validation_models_stack, validation_hps_stack, validation_weights), axis=1)

        return(training_data, val_data)

    def sample(self, conditional, N=1):
        """
        Samples Flow
        Parameters
        ----------
        conditional : array of length self.conditionals
            the values of the model hyperparameters for the sampled channel (e.g. [chi_b,alpha_CE])
        N : int
            number of samples to draw
        Returns
        ----------
        samps : array
            samples in shape [N, Nparams]
        """

        #sample from flow - this returns samples in the logistically mapped space
        logit_samps = self.flow.sample(conditional,N)

        samps = np.zeros(np.shape(logit_samps))

        #map samples back from logit space according to type of transform specified in `self.param_dict[param]['transf']``
        for pidx, param in enumerate(self.param_dict):
            if self.param_dict[param]['transf'] == 'logit':
                samps[:,pidx] = self.expistic(logit_samps[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'])
            elif self.param_dict[param]['transf'] == 'tanh':
                samps[:,pidx] = np.tanh(logit_samps[:,pidx])
            else:
                print(f'No transformation type specified for {param} dimension, attempting expistic transform')
                samps[:,pidx] = self.expistic(logit_samps[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'])

        return samps

    def __call__(self, data, conditional_hps, smallest_N, data_prior=None):
        """
        Calculate the regularised likelihood of the observations give some hyperparameters.
        This is used to calculate the hyperlikelihood.
        Parameters
        ----------
        data : array
            Posterior samples of observations or mock observations for which to calculate the likelihoods,
            shape[Nobs x Nsample x Nparams]
        conditional_hps : array
            Values of hyperparameters for require submodel, of shape [self.conditionals]
        smallest_N : int
            The constant by which to add a regularisation factor, in order to give an approximately constant 
            probability of 1/smallest_N in the distribution tails 
        data_prior : array
            Prior on the data of shape [Nobs x Nsample]
            If data_prior is None, each observation is given equal
            posterior probability.
        Returns
        -------
        likelihood : array
        the log likelihoods obtained from the flow model for each event, shape [Nobs]
        """

        #initialise log likelihood as -infnity
        likelihood = np.ones(data.shape[0]) * -np.inf

        #set equal prior for all samples if prior is not specified
        data_prior = data_prior if data_prior is not None else np.ones((data.shape[0],data.shape[1]))
        #raise error if any samples have prior=0
        if np.any(data_prior == 0.):
            raise Exception('One or more of the prior samples is equal to zero')

        #maps observations into the logistically mapped space
        mapped_obs = self.map_obs(data)

        #conditionals tiled into shape [Nobs x Nsamples x Nconditionals]
        conditional_hps = np.asarray(conditional_hps)
        conditionals = np.repeat([conditional_hps],np.shape(mapped_obs)[1], axis=0)
        conditionals = np.repeat([conditionals],np.shape(mapped_obs)[0], axis=0)

        #calculates likelihoods for all events and all samples
        likelihoods_per_samp = self.flow.get_logprob(data, mapped_obs, self.param_dict, conditionals)

        if smallest_N is not None:
            #sums the population probability plus uniform regularisation
            pi_reg = np.log(1/(smallest_N+1))
            q_weight = np.log(smallest_N/(smallest_N+1))
            likelihoods_per_samp = logsumexp([q_weight + likelihoods_per_samp, pi_reg*np.ones(likelihoods_per_samp.shape)], axis=0)

        #divide by the prior on the data samples
        likelihoods_per_samp = likelihoods_per_samp - np.log(data_prior)

        #checks for nans in likelihood
        if np.any(np.isnan(likelihoods_per_samp)):
            raise Exception('Nans in likelihood.')

        #adds likelihoods from samples together and then sums over events, normalise by number of samples
        #likelihood in shape [Nobs]
        likelihood = logsumexp([likelihood, logsumexp(likelihoods_per_samp, axis=1) - np.log(data.shape[1])], axis=0)

        return likelihood

    def get_latent_samps(self, samps, conditional):
        """
        Maps data into latent space of flow, return samples in latent space
        Parameters
        ----------
        samps : array of shape [Nobs, Nsamps, Nparams]
        conditional : array of length self.conditionals
            the values of the model hyperparameters for the sampled channel
        Returns
        ----------
        samps mapped to latent space
        """

        conditional = np.asarray(conditional)

        #maps observations into the logistically mapped space
        mapped_obs = self.map_obs(samps)

        #conditionals tiled into shape [Nobs x Nsamples x Nconditionals]
        conditional = np.repeat([conditional],np.shape(mapped_obs)[1], axis=0)
        conditional = np.repeat([conditional],np.shape(mapped_obs)[0], axis=0)

        return self.flow.get_latent_samps(mapped_obs, conditional)

    def map_obs(self,data):
        """
        Maps oberservational data into logistically mapped space for flows to handle.
        Each dimension is mapped either by a logistic transform or an arctanh transform according
        to which is specified in `self.param_dict[param]['transf']`
        Parameters
        -------
        data : array
            Observations in array Nobs x Nsamples x Nparams
        Returns
        -------
        mapped_data : array
            observational binary parameters logistically mapped
        Only accounts for full set of parameters in param_dict.
        """
        mapped_data = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2]))

        #compute logistic mappings of data
        for pidx, param in enumerate(self.param_dict):
            if self.param_dict[param]['transf'] == 'logit':
                mapped_data[:,:,pidx],_,_ = self.logistic(data[:,:,pidx], False, self.param_dict[param]['logit_max'], self.param_dict[param]['max'])
            elif self.param_dict[param]['transf'] == 'tanh':
                mapped_data[:,:,pidx] = np.arctanh(data[:,:,pidx])
            else:
                print(f'No transformation type specified for {param} dimension, attempting logistic transform')
                mapped_data[:,:,pidx],_,_ = self.logistic(data[:,:,pidx], False, self.param_dict[param]['logit_max'], self.param_dict[param]['max'])

        return mapped_data


    def logistic(self, data, wholedataset, max =1, rescale_max=1):
        """
        Logistically maps samples to a logistsic space
        If the whole training set is passed to the function, this determines the maximum rescaling values.
        Parameters
        -------
        data : array 
            Samples of observations or mock observations for which to map
        wholedataset : bool
            Whether or not the mapping is of the whole data set, in which case, after the logit transform, divide the samples by the max of logit(data).
            if false, divide data by max
        max : float
            The maxmimum value of the distribution of the logistic data, used to scale the data after the logistic transform
        rescale_max : float
            Initial value by which to normalise the data by such that it lies on a range of 0-1 before the logistic mapping
        Returns
        -------
        logit_data : array
            scaled and logistically mapped samples of data
        max : float
            the value used to scale the data after the logistic mapping
        rescale_max : float
            the value used to normalise the data intially to be on a range from 0 to 1
        """

        #rescales samples so that they lie between 0 to 1, according to the upper bound of the parameter space
        rescale_max = rescale_max
        data_normed = data/rescale_max

        #sample must be within bounds for logistic function to return definite value
        if np.any(data_normed >= 1):
            #artificially increase upper bound for troublesome training samples
            rescale_max = rescale_max*1.001
            data_normed = data/rescale_max
        if np.logical_or(data_normed <= 0, data_normed >= 1).any():
            raise Exception('Data out of bounds for logistic mapping')

        #takes the logistic of sample
        logit_data = logit(data_normed)

        #scales the distribution in logistic space, so that the samples can have spread O(1), easier for flow to learn
        if wholedataset:
            max = np.max(logit_data)
        else:
            max = max
        logit_data /= max

        return([logit_data, max, rescale_max])

    def expistic(self, data, max, rescale_max=None):
        """
        Undoes the logistic transform on logistically mapped data
        Parameters
        -------
        data : array
            scaled and logistically mapped samples of data, shape [Nobs, Nsamps]
        max : float
            the value used to scale the data after the logistic mapping
        rescale_max : float
            the value used to normalise the data intially to be on a range from 0 to 1
        Returns
        -------
        data : array 
            posterior samples of observations or mock observations for which to unmap,
            shape[Nobs x Nsample]
        """
        #times by scaling used to reduce spread of logistically mapped data
        data*=max

        #expit the logistic data
        data = expit(data)

        #times by the initial scaling to rescale the data to its original range
        if rescale_max != None:
            data *=rescale_max
        return(data)

    def train(self, no_trans, no_neurons, no_blocks, no_bins, lr, epochs, batch_no, filepath, device):
        """
        Trains the normalising flow with certain configuration of flow network parameters.
        Saves these network parameters to a json config file, and saves flow post training.
        Parameters
        -------
        no_trans : int
            number of transformations that the flow uses to map the data to the latent space
        no_bins : int
            number of spline bins for each transformation with the spline flow
        no_blocks : int
            number of blocks to divide data into for each transformation
        no_neurons : int
            number of neurons in each block, determining the expressibility of each transform
        lr : float
            the initial learning rate of the flow
        epochs : int
            the number of epochs which to train the flow
        batch_no : int
            the number of samples to use for a batch of training
        filepath : str
            the directory to save the flow models and associated config
        deivce : str
            Device on which to run the flow. Either is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """

        #define flow hyperparams as class properties
        self.no_trans = no_trans
        self.no_neurons = no_neurons
        self.no_blocks = no_blocks
        self.no_bins = no_bins

        #FIXME - make this a settable parameter
        batch_size=10000

        #initislise flow network as NFlow class
        self.flow = NFlow(self.no_trans, self.no_neurons, self.no_blocks, self.no_bins, self.no_params, self.conditionals, batch_size, 
                    self.total_smdls, RNVP=False, device=device)

        #map the training and validation samples
        training_data, val_data = self.map_samples()

        #FIXME: take network parameter and mapping info from ini file rather than config
        #write or append channel config to json file
        channel_config = {'transforms':no_trans, 'neurons':no_neurons,'blocks':no_blocks,'bins':no_bins}

        #set mapping parameters into channel config
        for param in self.param_dict:
            if self.param_dict[param]['transf'] == 'logit':
                channel_config[param] = {'logit_max':self.param_dict[param]['logit_max'], 'max':self.param_dict[param]['max']}

        channel_json = {}
        channel_json[self.channel_label] = channel_config

        #check if config file exists (e.g. containing other channels), and update this channel to current config
        if os.path.isfile(f'{filepath}/flowconfig.json'):
            with open(f'{filepath}/flowconfig.json', 'r') as f:
                old_config = json.load(f)
            #update the old config of this channel
            old_config[self.channel_label] = channel_config
            #load old config of other channels
            channel_json = old_config

        #write this channels config to file
        with open(f'{filepath}/flowconfig.json', 'w') as f:
            json.dump(channel_json, f)

        save_filename = f'{filepath}/{self.channel_label}'
        #train the normalising flow
        self.flow.trainval(lr, epochs, batch_no, save_filename, training_data, val_data)

    def load_model(self, filepath, device='cpu'):
        """
        Load a normalising flow model into self.flow with configuration of flow network parameters from json file if it exists.
        Parameters
        -------
        filepath : str
            directory with saved flow model and config
        deivce : str
            Device on which to run the flow. Either is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """
        #load no. transforms, no. neurons and no. bins from config if config for flows exists
        if os.path.isfile(f'{filepath}/flowconfig.json'):
            with open(f'{filepath}/flowconfig.json', 'r') as f:
                config = json.load(f)
            #load flow hyperparameters, set to defaults if not found
            try:
                self.no_trans = config[self.channel_label]['transforms']
            except:
                self.no_trans = 6
            try:
                self.no_neurons = config[self.channel_label]['neurons']
            except:
                self.no_neurons = 128
            try:
                self.no_blocks = config[self.channel_label]['blocks']
            except:
                self.no_blocks = 2
            try:
                self.no_bins = config[self.channel_label]['bins']
            except:
                self.no_bins = 4
            #load parameter mapings, from config if availble, if not then try mappings.np file
            try:
                for param in self.param_dict:
                    if self.param_dict[param]['transf'] == 'logit':
                        for key in ['logit_max', 'max']:
                            self.param_dict[param][key] = config[self.channel_label][param][key]
            except:
                #deal with old hardcoding mapping saving
                mappings = np.load(f'{filepath}/{self.channel_label}_mappings.npy', allow_pickle=True)
                self.param_dict['mchirp']['transf'] = 'logit'
                self.param_dict['q']['transf'] = 'logit'
                self.param_dict['chieff']['transf'] = 'tanh'
                self.param_dict['z']['transf'] = 'logit'
                i=0
                for param in self.param_dict:
                    if self.param_dict[param]['transf'] == 'logit':
                        for key in ['logit_max', 'max']:
                            self.param_dict[param][key] = mappings[i]
                            i+=1
        else:
            raise Exception("no config available")

        #FIXME: make this a settable parameter
        batch_size=10000

        #initialise NFlow model
        self.flow = NFlow(self.no_trans, self.no_neurons, self.no_blocks, self.no_bins, self.no_params, self.conditionals, batch_size,\
            self.total_smdls, RNVP=False, device=device)

        #load in actual flow model
        self.flow.load_model(f'{filepath}/{self.channel_label}.pt')

    def get_alpha(self, hyperparams):
        """
        Get the detection efficiency at particular hypereparameter values with pchip spline interpolation.
        Parameters
        -------
        hyperparams : array
            hyperparameter values at which to evaluate alpha
        
        Returns
        -------
        alpha : float
            value of detection efficiency for specified hyperparameter values
        """

        #reshape detection efficiency values onto grid the shape of hyperparameter values
        hp_grid_shape = [len(self.hyperparam_models[i]) for i in range(len(self.hyperparam_models))]
        alpha_grid = np.reshape(tuple(self.alpha.values()), (hp_grid_shape))

        #initialise interpolator over hyperparameters to interolate log(detection efficiency)
        alpha_interp = sp.interpolate.RegularGridInterpolator((self.hp_vals), np.log(alpha_grid),\
            bounds_error=False, method='pchip', fill_value=None)
        #find alpha at specified hyperparameter values
        alpha = np.exp(alpha_interp(hyperparams))

        return alpha