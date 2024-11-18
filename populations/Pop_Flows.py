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
import time
import wandb
import json

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.special import logit
from scipy.special import logsumexp
from scipy.special import expit
from sklearn.model_selection import train_test_split
from .utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults
from .utils.flow import NFlow

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# Get the interpolation function for the projection factor in Dominik+2015
# which takes in a random number and spits out a projection factor 'w'
projection_factor_interp = projection_factor_Dominik2015_interp()

_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}

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


class FlowModel(Model):
    @staticmethod
    def from_samples(channel, samples, params, flow_path, sensitivity=None, device='cpu'):
        """
        Generate a Flow model instance from `samples`, where `params` are series in the `samples` dataframe. 
        
        If `weight` is a column in your population model, will assume this is the cosmological weight of each sample,
        and will include this in the construction of all your KDEs. If `sensitivity` 
        is provided, samples used to generate the detection-weighted KDE will be 
        weighted according to the key in the argument `pdet_*sensitivity*`.

        Parameters
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters in 'params' array, cosmo_weights, pdet weights, optimal snrs
        params : list of str
            subset of mchirp, q, chieff, z
        flow_path : str
            directory of the flow models to load network config from if config file exists
        sensitivity : str
            Desired detector sensitivity consistent with the string following the `pdet` and `snropt` columns in the population dataframes.
            Used to construct a detection-weighted population model, as well as for drawing samples from the underlying population
            to calculate the detection efficiency
        deivce : str
            Device on which to run the flow. Either is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        
        Returns
        ----------
        FlowModel : obj
        """
        return FlowModel(channel, samples, params, flow_path, sensitivity, device=device)


    def __init__(self, channel, samples, params, flow_path, sensitivity, device):
        """
        Initialisation for FlowModel object. Sets self.flow as instance of Nflow class, of which FlowModel is wrapper of that object.

        Parameters
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters in 'params' array, cosmo_weights, pdet weights, optimal snrs
        params : list of str
            subset of [mchirp, q, chieff, z]
        flow_path : str
            directory of the flow models to load network config from if config file exists
        sensitivity : str
            Desired detector sensitivity consistent with the string following the `pdet` and `snropt` columns in the population dataframes.
            Used to construct a detection-weighted population model, as well as for drawing samples from the underlying population
            to calculate the detection efficiency
        deivce : str
            Device on which to run the flow. default is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """
        
        super()
        self.channel_label = channel
        self.samples = samples
        self.params = params
        self.sensitivity = sensitivity

        #initialises list of population hyperparameter values
        self.hps = [[0.,0.1,0.2,0.5]]
        self.param_bounds = [_param_bounds[param] for param in self.params]

        #additional alpha dimension for CE channel, else dummy dimension
        if self.channel_label=='CE':
            self.hps.append([0.2,0.5,1.,2.,5.])
        else:
            self.hps.append([1])
        
        #number of binary parameters
        self.no_params = np.shape(params)[0]
        #dimensionailty of non-branching ratio hyperparameters
        self.conditionals = 2 if self.channel_label =='CE' else 1

        #initialise dictionaries of alpha, cosmo_weights, pdet, optimal_snrs, and combined_weights for each submodel
        alpha = dict.fromkeys(samples.keys())
        cosmo_weights= dict.fromkeys(samples.keys())
        pdets= dict.fromkeys(samples.keys())
        optimal_snrs= dict.fromkeys(samples.keys())
        combined_weights= dict.fromkeys(samples.keys())

        #loop over submodels
        for chib_id, chib in enumerate(self.hps[0]):
            for alphaid, alphaCE in enumerate(self.hps[1]):

                #grab samples for each submodel depending on how many hyperparameter dimensions there are (CE vs non-CE)
                if self.channel_label=='CE':
                    sbml_samps = samples[chib_id,alphaid]
                    key = chib_id,alphaid
                else:
                    sbml_samps = samples[chib_id]
                    key= chib_id

                #check that defined sensitivity exists in submodel dataframe
                if sensitivity is not None:
                    if 'pdet_'+sensitivity not in sbml_samps.columns:
                        raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")

                # get *\alpha* for each model, defined as \int p(\theta|\lambda) Pdet(\theta) d\theta
                if sensitivity is not None:
                    # if cosmological weights are provided, do mock draws from the pop
                    if 'weight' in sbml_samps.keys():
                        mock_samp = sbml_samps.sample(int(1e6), weights=(sbml_samps['weight']/len(sbml_samps)), replace=True)
                    else:
                        mock_samp = sbml_samps.sample(int(1e6), replace=True)
                    alpha[key]=np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
                else:
                    alpha[key]=1.0

                ### GET WEIGHTS ###
                # if cosmological weights are provided...
                if 'weight' in sbml_samps.keys():
                    cosmo_weights[key] = np.asarray(sbml_samps['weight'])
                else:
                    cosmo_weights[key] = np.ones(len(sbml_samps))
                # if detection weights are provided...
                if sensitivity is not None:
                    pdets[key] = np.asarray(sbml_samps['pdet_'+sensitivity])
                else:
                    pdets[key] = np.ones(len(sbml_samps))

                # get optimal SNRs for this sensitivity
                if sensitivity is not None:
                    optimal_snrs[key] = np.asarray(sbml_samps['snropt_'+sensitivity])
                else:
                    optimal_snrs[key] = np.nan*np.ones(len(sbml_samps))


                # Normalise the cosmological weights. If wanted detection weighted samples, cosmo weigths could be combined with pdets
                if (cosmo_weights[key] is not None):
                    combined_weights[key] = (cosmo_weights[key] / np.sum(cosmo_weights[key]))
                else:
                    combined_weights[key] = np.ones(len(sbml_samps))

        #sets weights as class properties
        self.combined_weights = combined_weights
        self.pdets = pdets
        self.optimal_snrs = optimal_snrs
        self.alpha = alpha
        self.cosmo_weights = cosmo_weights
        
        #Load flow network parameters from config file if it exists
        if os.path.isfile(f'{flow_path}flowconfig.json'):
            with open(f'{flow_path}flowconfig.json', 'r') as f:
                config = json.load(f)
        else:
            config = {}

        if self.channel_label in list(config.keys()):
            self.no_trans = config[self.channel_label]['transforms']
            self.no_neurons = config[self.channel_label]['neurons']
            self.no_bins = config[self.channel_label]['bins']
        else:
            self.no_trans = 6
            self.no_neurons = 128
            self.no_bins=4
            if self.channel_label=='CE' or self.channel_label=='NSC':
                self.no_bins=5

        batch_size=10000

        #initialise the channel of this flow and how many training submodels exist for this channel
        self.total_hps = np.shape(self.hps[0])[0]*np.shape(self.hps[1])[0]
        channel_ids = {'CE':0, 'CHE':1,'GC':2,'NSC':3, 'SMT':4}
        self.channel_id = channel_ids[self.channel_label] #will be 0, 1, 2, 3, or 4

        #initislises flow network
        flow = NFlow(self.no_trans, self.no_neurons, self.no_params, self.conditionals, batch_size, 
                    self.total_hps, self.channel_label, RNVP=False, device=device, no_bins=self.no_bins)
        self.flow = flow


    def map_samples(self, samples, params, filepath, testCEsmdl=False):
        """
        Maps samples with logistic mapping (mchirp, q, z samples) and tanh (chieff).
        Stacks data by [mchirp,q,chieff,z,weight,chi_b,(alpha)].
        Handles any channel.

        Parameters
        ----------
        samples : dict
            dictionary of data in form 
            ['mchirp', 'q', 'chieff', 'z', 'm1' 'm2' 's1x' 's1y' 's1z' 's2x' 's2y' 's2z'
            'weight' 'pdet_midhighlatelow_network' 'snropt_midhighlatelow_network'
            'pdet_midhighlatelow' 'snropt_midhighlatelow']
        params : list of str
            list of parameters to be used for inference, typically ['mchirp', 'q', 'chieff', 'z']
        filepath : str
            the filepath to the flow models and mappings to be loaded/saved
        testCEsmdl : bool
            Whether or not to remove the CE subpopulation (chi_b=0.1, alphaCE=1.0) as a test population before training.
        
        Returns
        -------
        training_data : array
            data samples to be used for training the normalising flow.
            [mchirp, q, chieff, z, weights, chi_b,(alpha)]
        val_data : array
            data samples to be used for validating the normalising flow.
            for the non-CE channels this is the same as the training data.
            for the CE channel this is set to 2 of the 20 sub-populations
        mappings : array
            constants used to map the mchirp, q, and z distributions with logistic mappings.
        """

        print('Mapping population synthesis samples for training...')
        

        if self.channel_label != 'CE':
            #Channels with 1D hyperparameters: SMT, GC, NSC, CHE


            #measure no_samples in models and identify samples with weights below fmin
            model_size = np.zeros(self.no_params)
            cumulsize = np.zeros(self.no_params)
            weights_idxs = []
            
            for chib_id, xb in enumerate(self.hps[0]):
                weights_temp=np.asarray(self.combined_weights[(chib_id)])
                weights_idxs.append(np.argwhere((weights_temp) > np.finfo(np.float32).tiny))
                model_size[chib_id] = np.shape(weights_idxs[chib_id])[0]
                cumulsize[chib_id] = np.sum(model_size)

            self.no_binaries = int(cumulsize[-1])
            models = np.zeros((self.no_binaries, self.no_params))
            weights = np.zeros((self.no_binaries, 1))
            cumulsize = np.append(cumulsize, 0)

            #moves binary parameter samples and weights from dictionaries into array
            for chib_id, xb in enumerate(self.hps[0]):
                models[int(cumulsize[chib_id-1]):int(cumulsize[chib_id])]=np.reshape(np.asarray(samples[(chib_id)][params])[weights_idxs[chib_id]],(-1,len(params)))
                weights[int(cumulsize[chib_id-1]):int(cumulsize[chib_id])]=np.asarray(self.combined_weights[(chib_id)])[weights_idxs[chib_id]]

            models_stack = np.copy(models)

            #map samples with logistic mapping before dividing into training and validation data

            #mchirp
            models_stack[:,0], max_logit_mchirp, max_mchirp = self.logistic(models_stack[:,0],wholedataset=True, \
                rescale_max=self.param_bounds[0][1])

            #q
            if self.channel_id == 2:
                #add extra tiny amount to GC mass ratios as q=1 samples exist
                models_stack[:,1], max_logit_q, max_q = self.logistic(models_stack[:,1],wholedataset=True, \
                rescale_max=self.param_bounds[1][1]+0.001)
            else:
                models_stack[:,1], max_logit_q, max_q = self.logistic(models_stack[:,1],wholedataset=True, \
                rescale_max=self.param_bounds[1][1])

            #chieff
            models_stack[:,2] = np.arctanh(models_stack[:,2])

            #z
            models_stack[:,3],max_logit_z, max_z = self.logistic(models_stack[:,3],wholedataset=True, \
                rescale_max=self.param_bounds[3][1])
            
            #repeat subpopulation hyperparameter values no samples times for each subpopulation
            #then split the training data, conditional data, and sample weights into training and validation sets
            training_hps_stack = np.repeat(self.hps[0], (model_size).astype(int), axis=0)
            training_hps_stack = np.reshape(training_hps_stack,(-1,self.conditionals))
            weights = np.reshape(weights,(-1,1))
            train_models_stack, validation_models_stack, train_weights, validation_weights, training_hps_stack, validation_hps_stack = \
                    train_test_split(models_stack, weights, training_hps_stack, shuffle=True, train_size=0.8)
            
        else:
            #CE channel with alpha_CE parameter

            #sets test population to remove, test_model_id is the model indices of the removed population
            if testCEsmdl:
                test_model_id = [1,2]
                test_model_id_flat = 7
                self.total_hps = self.total_hps - 1

            #initialise arrays for model size
            model_size = np.zeros((4,5))
            cumulsize = np.zeros(self.total_hps)
            weights_idxs = []

            #tile list of chi_bs and alpha_CEs into liost for each training sub-population
            chi_b_alpha_pairs= np.zeros((self.total_hps, 2))
            chi_b_alpha_pairs[:,0] = np.repeat(self.hps[0],np.shape(self.hps[1])[0])
            chi_b_alpha_pairs[:,1] = np.tile(np.log(self.hps[1]), np.shape(self.hps[0])[0])
            if testCEsmdl:
                chi_b_alpha_pairs = np.delete(chi_b_alpha_pairs, test_model_id_flat, axis=0)

            #meaure no samples in each population, and the cumulative samples in each population used for training
            i=0
            for chib_id in range(4):
                for alpha_id in range(5):
                    if testCEsmdl:
                        if [chib_id, alpha_id] == test_model_id:
                            continue
                    weights_temp=np.asarray(self.combined_weights[(chib_id, alpha_id)])
                    weights_idxs.append(np.argwhere((weights_temp) > np.finfo(np.float32).tiny))
                    model_size[chib_id, alpha_id] = np.shape(weights_idxs[i])[0]
                    cumulsize[i] = np.sum(model_size)
                    i+=1

            self.no_binaries = int(cumulsize[-1])
            models = np.zeros((self.no_binaries, self.no_params))
            weights = np.zeros((self.no_binaries,1))
            cumulsize = np.append(cumulsize, 0)

            #put samples from each model into array of shape [no_samples in channel, no params]
            #and weights into array [no_samples in channel, 1]
            i=0
            for chib_id in range(4):
                for alpha_id in range(5):
                    if testCEsmdl:
                        if [chib_id, alpha_id] == test_model_id:
                            continue
                    models[int(cumulsize[i-1]):int(cumulsize[i])]=np.reshape(np.asarray(samples[(chib_id, alpha_id)][params])[weights_idxs[i]],(-1,len(params)))
                    weights[int(cumulsize[i-1]):int(cumulsize[i])]=np.reshape(np.asarray(self.combined_weights[(chib_id, alpha_id)])[weights_idxs[i]],(-1,1))
                    i+=1

            #reshape the model size array to be 1D
            flat_model_size = np.reshape(model_size, 20)
            if testCEsmdl:
                flat_model_size = np.delete(flat_model_size, test_model_id_flat)

            #repeat the pairs of [chi_b, alphaCE] for the number of samples in the training samples
            all_chi_b_alphas = np.repeat(chi_b_alpha_pairs, (flat_model_size).astype(int), axis=0)


            #scale parameters with logistic mapping, only for full range of parameters
            models_stack = np.copy(models)

            #chirp mass original range 0 to inf
            models_stack[:,0], max_logit_mchirp, max_mchirp = self.logistic(models_stack[:,0], wholedataset=True, \
                rescale_max=self.param_bounds[0][1])

            #mass ratio - original range 0 to 1
            models_stack[:,1], max_logit_q, max_q = self.logistic(models_stack[:,1], wholedataset=True, \
                rescale_max=self.param_bounds[1][1])

            #chieff - original range -1 to +1
            models_stack[:,2] = np.arctanh(models_stack[:,2])

            #redshift - original range 0 to inf
            models_stack[:,3], max_logit_z, max_z = self.logistic(models_stack[:,3],wholedataset=True, \
                rescale_max=self.param_bounds[3][1])

            weights = np.reshape(weights,(-1,1))
            train_models_stack, validation_models_stack, train_weights, validation_weights, training_hps_stack, validation_hps_stack = \
                    train_test_split(models_stack, weights, all_chi_b_alphas, shuffle=True, train_size=0.8)


        #concatenate data and weights and hyperparams
        training_data = np.concatenate((train_models_stack, train_weights, training_hps_stack), axis=1)
        val_data = np.concatenate((validation_models_stack, validation_weights, validation_hps_stack), axis=1)

        #save mapping constants in flow model directory
        mappings = np.asarray([max_logit_mchirp, max_mchirp, max_logit_q, max_q, max_logit_z, max_z])
        np.save(f'{filepath}{self.channel_label}_mappings.npy',mappings)
        
        return(training_data, val_data, mappings)

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
            samples in shape [N, no_params]
        """
        #log alphaCE
        if self.channel_label =='CE':
            conditional[1] = np.log(conditional[1])
        
        #sample from flow - this returns samples in the logistically mapped space
        logit_samps = self.flow.sample(conditional,N)

        #map samples back from logit space
        samps = np.zeros(np.shape(logit_samps))
        samps[:,0] = self.expistic(logit_samps[:,0], self.mappings[0], self.mappings[1])
        samps[:,1] = self.expistic(logit_samps[:,1], self.mappings[2], self.mappings[3])
        samps[:,2] = np.tanh(logit_samps[:,2])
        samps[:,3] = self.expistic(logit_samps[:,3], self.mappings[4], self.mappings[5])
        return samps

    def __call__(self, data, conditional_hps, smallest_N, prior_pdf=None):
        """
        Calculate the likelihood of the observations give a particular hypermodel (given by conditional_hps).
        (this is the hyperlikelihood).

        Parameters
        ----------
        data : array
            posterior samples of observations or mock observations for which to calculate the likelihoods,
            shape[Nobs x Nsample x Nparams]
        conditional_hps : array
            values of hyperparameters for require submodel, of shape [self.conditionals]
        smallest_N : int
            the constant by which to add a regularisation factor, in order to give an approximately constant 
            probability in the distribution tails of 1/smallest_N
        prior_pdf : array
            p(x) prior on the data
            If prior_pdf is None, each observation is expected to have equal
            posterior probability. Otherwise, the prior weights should be
            provided as the dimemsions [samples(Nobs), samples(Nsamps)].

        Returns
        -------
        likelihood : array
        the log likelihoods obtained from the flow model for each event, shape [Nobs]
        """
        
        #initialise log likelihood as -infnity
        likelihood = np.ones(data.shape[0]) * -np.inf

        #set equal prior for all samples if prior is not specified
        prior_pdf = prior_pdf if prior_pdf is not None else np.ones((data.shape[0],data.shape[1]))
        #raise error if any samples have prior=0
        if np.any(prior_pdf == 0.):
            raise Exception('One or more of the prior samples is equal to zero')

        #maps observations into the logistically mapped space
        mapped_obs = self.map_obs(data)

        #conditionals tiled into shape [Nobs x Nsamples x Nconditionals]
        conditional_hps = np.asarray(conditional_hps)
        conditionals = np.repeat([conditional_hps],np.shape(mapped_obs)[1], axis=0)
        conditionals = np.repeat([conditionals],np.shape(mapped_obs)[0], axis=0)

        #calculates likelihoods for all events and all samples
        likelihoods_per_samp = self.flow.get_logprob(data, mapped_obs, self.mappings, conditionals)

        if smallest_N is not None:
            #LSE population probability plus uniform regularisation
            pi_reg = np.log(1/(smallest_N+1))
            q_weight = np.log(smallest_N/(smallest_N+1))
            likelihoods_per_samp = logsumexp([q_weight + likelihoods_per_samp, pi_reg*np.ones(likelihoods_per_samp.shape)], axis=0)

        #divide by the prior on the data samples
        likelihoods_per_samp = likelihoods_per_samp - np.log(prior_pdf)

        #checks for nans in likelihood
        if np.any(np.isnan(likelihoods_per_samp)):
            raise Exception('Obs data is outside of range of samples for channel - cannot logistic map.')

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
            the values of the model hyperparameters for the sampled channel (e.g. [chi_b,alpha_CE])

        Returns
        ----------
        samps mapped to latent space
        """

        #logs alphaCE
        conditional = np.asarray(conditional)
        if self.channel_label =='CE':
            conditional[1] = np.log(conditional[1])
        

        #maps observations into the logistically mapped space
        mapped_obs = self.map_obs(samps)

        #conditionals tiled into shape [Nobs x Nsamples x Nconditionals]
        conditional = np.repeat([conditional],np.shape(mapped_obs)[1], axis=0)
        conditional = np.repeat([conditional],np.shape(mapped_obs)[0], axis=0)

        return self.flow.get_latent_samps(mapped_obs, conditional)

    def map_obs(self,data):
        """
        Maps oberservational data into logistically mapped space for flows to handle.

        Parameters
        -------
        data : array
            observations in array Nobs x Nsamples x Nparams

        Returns
        -------
        mapped_data : array
            observational binary parameters logistically mapped

        Only accounts for full set of parameters [mchirp, q, chieff, z].
        mappings in form [max_logit_mchirp, max_mchirp, max_q, None, max_logit_z, max_z]

        """
        mapped_data = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2]))

        #compute logistic mappings of data
        mapped_data[:,:,0],_,_= self.logistic(data[:,:,0], False, max=self.mappings[0], rescale_max=self.mappings[1])
        mapped_data[:,:,1],_,_= self.logistic(data[:,:,1], False, max=self.mappings[2], rescale_max=self.mappings[3])
        mapped_data[:,:,2]= np.arctanh(data[:,:,2])
        mapped_data[:,:,3],_,_= self.logistic(data[:,:,3], False, max=self.mappings[4], rescale_max=self.mappings[5])

        return mapped_data


    def logistic(self, data, wholedataset, max =1, rescale_max=1):
        """
        Logistically maps sample in non-logistsic space
        input is [Nsamps] shape array
        if the whole training set is passed to the function, this determines the maximum rescaling values

        Parameters
        -------
        data : array 
            posterior samples of observations or mock observations for which to map,
            shape[Nobs x Nsample]
        wholedataset : bool
            whether or not the mapping is of the whole data set, in which case, after the logit transform, divide the samples by the max of logit(data).
            if false, divide data by max
        max : float
        rescale_max : float
            initial value by which to normalise the data by such that it lies on a range of 0-1 before the logistic mapping

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

    def train(self, no_trans, no_bins, no_neurons, lr, epochs, batch_no, filepath, testCEsmdl, use_wandb=False):
        """
        Trains the normalising flow with certain configuration of flow network parameters.
        Saves these network parameters to a json config file, and saves flow post training

        Parameters
        -------
        no_trans : int
            number of transformations that the flow uses to map the data to the latent space
        no_bins : int
            number of spline bins for each transformation with the spline flow
        no_neurons : int
            number of neurons each layer of the neural network gets
        lr : float
            the initial learning rate of the flow
        epochs : int
            the number of epochs which to train the flow
        batch_no : int
            the number of samples to use for a batch of training
        filepath : str
            the directory to save the flow models and associated config
        use_wandb : bool
            whether or not to use Weights and Biases network optimisation to train the flow and track its loss etc
        """

        #write or append channel config to json file
        channel_config = {'transforms':no_trans, 'neurons':no_neurons,'bins':no_bins}
        channel_json = {}
        channel_json[self.channel_label] = channel_config

        #check if config exists e.g. for other channels
        if os.path.isfile(f'{filepath}flowconfig.json'):
            with open(f'{filepath}flowconfig.json', 'r') as f:
                old_config = json.load(f)
            old_config[self.channel_label] = channel_config
            channel_json = old_config

        #write this channels config to file
        with open(f'{filepath}flowconfig.json', 'w') as f:
            json.dump(channel_json, f)

        #map the training samples etc 
        training_data, val_data, self.mappings = self.map_samples(self.samples, self.params, filepath, testCEsmdl)

        save_filename = f'{filepath}{self.channel_label}'
        #train the normalising flow
        self.flow.trainval(lr, epochs, batch_no, save_filename, training_data, val_data, use_wandb)

    def load_model(self, filepath, device='cpu'):
        """
        Loads the normalising flow into self.flow with configuration of flow network parameters from json file if it exists.

        Parameters
        -------
        filepath : str
            directory with saved flow model and config
        """
        #load no. transforms, no. neurons and no. bins from config and reinitialise flow if config for flows exists
        if os.path.isfile(f'{filepath}flowconfig.json'):
            with open(f'{filepath}flowconfig.json', 'r') as f:
                config = json.load(f)
            self.no_neurons = config[self.channel_label]['neurons']
            self.no_bins = config[self.channel_label]['bins']
            batch_size=10000

            self.flow = NFlow(self.no_trans, self.no_neurons, self.no_params, self.conditionals, batch_size,\
                self.total_hps, self.channel_label, RNVP=False, device=device, no_bins=self.no_bins)
        
        #load in actual flow model, and mappings
        self.flow.load_model(f'{filepath}{self.channel_label}.pt')
        self.mappings = np.load(f'{filepath}{self.channel_label}_mappings.npy', allow_pickle=True)

    def get_alpha(self, hyperparams):
        """
        Get the detection efficiency at certain values of chi_b, alpha_CE with pchip spline interpolation.

        Parameters
        -------
        hyperparams : array
            [chi_b] or [chi_b, log(alpha_CE)] depending on non-CE or CE channel
        
        Returns
        -------
        alpha : float
            value of detection efficiency for specified [chi_b, {alpha_CE}]
        """

        #reshape detection efficiency values onto 2D array or shape len(chi_b), len(alpha_CE) if CE channel
        #len(alpha_CE)=1 for non-CE channels
        alpha_grid = np.reshape(tuple(self.alpha.values()), (len(self.hps[0]),len(self.hps[1])))

        #CE case with 2D interpolation
        if self.channel_label == "CE":
            #initialise interpolator over chi_b, log(alpha_CE) to interolate log(detection efficiency)
            alpha_interp = sp.interpolate.RegularGridInterpolator((self.hps[0],np.log(self.hps[1])), np.log(alpha_grid),\
                bounds_error=False, method='pchip', fill_value=None)
            #find alpha at specified chi_b, log(alpha_CE)
            alpha = np.exp(alpha_interp([hyperparams[0][0], hyperparams[0][1]]))

        else:
            #interpolate log alpha over chi_b values
            alpha_interp = sp.interpolate.RegularGridInterpolator([self.hps[0]], np.log(np.reshape(alpha_grid, len(self.hps[0]))),\
            bounds_error=False, method='pchip', fill_value=None)
            #return alpha at specified chi_b
            alpha = np.exp(alpha_interp([hyperparams[0]]))
        return alpha


    def wandb_init(self, epochs):
        """
        Initialises a wandb sweep and then uses a wandb agent to train a flow under that sweeps regimes
        current process optimises the number of epochs over validation loss, 
        the prior for the number of epochs being uniform between argument's number of epochs and 3*args.epochs
        """
        #setting up wandb sweep parameters

        sweep_config = {
            'method': 'bayes'
            }

        metric = {
            'name': 'val_loss',
            'goal': 'minimize'   
            }

        sweep_config['metric'] = metric

        parameters_dict = {
            'lr': {
                'value':0.001
            },
            'epochs': {
                'value':10000
            },
            'batch_no': {
                'value':10
            },
            'no_trans': {
                'distribution': 'int_uniform',
                'min': 4,
                'max': 10
            },
            'no_neurons': {
                'distribution': 'int_uniform',
                'min': 12,
                'max': 128
            },
            'no_bins': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 8
            }                
            }

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(sweep_config, project=f"{self.channel_label}_transneubins_sweep")
        
        wandb.agent(sweep_id, self.wandbtrain, count=20)
    
    def wandbtrain(self, config=None):
        with wandb.init(config=config):
            config = wandb.config

            batch_size=10000
            total_hps = np.shape(self.hps[0])[0]*np.shape(self.hps[1])[0]
            device='cuda:0'

            flow = NFlow(config.no_trans, config.no_neurons, self.no_params, self.conditionals, self.no_binaries, batch_size, 
                    total_hps, self.channel_label, RNVP=False, device=device, no_bins=config.no_bins)
            self.flow = flow
            self.train(config.lr, config.epochs, config.batch_no, f"./wandb_models/{wandb.run.id}_wandb", self.channel_label, True)
