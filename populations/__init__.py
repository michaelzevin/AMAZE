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

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from .population_utils.bounded_Nd_kde import Bounded_Nd_kde
from .population_utils.transform import mtotq_to_mchirp, mtoteta_to_mchirpq, eta_to_q, mchirpq_to_m1m2
#SC: is this necessary? the selection effects file has been removed
#from .population_utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults
#proj_factor = projection_factor_Dominik2015_interp()

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# Need to ensure all parameters are normalized over the same range
_posterior_sigmas = {"mchirp": 1.512, "q": 0.166, "chieff": 0.1043, "z": 0.0463}   # TOUPDATE
_snrscale_sigmas = {"mchirp": 0.04, "eta": 0.03, "chieff": 0.14}      # TOUPDATE

_normalization_bounds_defaults = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}
_kde_bandwidth_default = 0.01
_max_samps_default = int(1e5)

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

class KDEModel(Model):
    @staticmethod
    def from_samples(label, samples, param_dict, sensitivity=None, \
                        max_samps=None, kde_bandwidth=None, \
                        store_optimal_snrs=False, **kwargs):
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