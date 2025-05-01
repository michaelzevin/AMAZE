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
from itertools import product

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.special import logit
from scipy.special import logsumexp
from scipy.special import expit
from sklearn.model_selection import train_test_split
from .population_utils.flow import NFlow


from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# Get the interpolation function for the projection factor in Dominik+2015
# which takes in a random number and spits out a projection factor 'w'
# projection_factor_interp = projection_factor_Dominik2015_interp()
# TODELETE

_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}
# Set based on config file...

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
    def from_samples(channel, samples, param_dict, channel_hyperparams, smdl_indxs_combos, sensitivity=None):
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
            if sensitivity is not None:
                # if cosmological weights are provided, do mock draws from the pop
                if 'weight' in sbml_samps.keys():
                    mock_samp = sbml_samps.sample(int(1e6), weights=(sbml_samps['weight']/len(sbml_samps)), replace=True)
                else:
                    mock_samp = sbml_samps.sample(int(1e6), replace=True)
                alpha[dict_key]=np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
            else:
                alpha[dict_key]=1.0

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
        deivce : str
            Device on which to run the flow. default is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """
        
        super()
        self.channel_label = channel
        self.samples = samples

        #want this to be structured {'mchirp':{'bounds':(0,100.), 'max':(100.)}, 'q':{...}, 'chieff:{...}}
        self.param_dict = param_dict

        #initialises list of population hyperparameters from hyperparameter dictionary
        self.hyperparam_models = []
        self.hp_vals = []
        for hp in channel_hyperparams:
            self.hyperparam_models.append(list(channel_hyperparams[hp]['values'].keys()))
            self.hp_vals.append(list(channel_hyperparams[hp]["values"].items()))
        
        #number of binary parameters
        self.no_params = len(param_dict.keys())
        #dimensionailty of non-branching ratio hyperparameters
        self.conditionals = len(self.hyperparam_models)

        #set weights as class properties
        self.combined_weights = combined_weights
        self.alpha = alpha

        #initialise the channel of this flow and how many training submodels exist for this channel
        self.model_keys = model_keys
        self.total_smdls = len(model_keys)


    def map_samples(self, filepath):
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

        self.no_binaries = int(cumulsize[-1])
        models = np.zeros((self.no_binaries, self.no_params))
        weights = np.zeros((self.no_binaries, 1))
        cumulsize = np.append(cumulsize, 0)

        #moves binary parameter samples and weights from dictionaries into array
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
            if self.param_dict[param]['transf'] == 'logit':
                models_stack[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'] = self.logistic(models_stack[:,pidx],\
                    wholedataset=True, rescale_max=self.param_dict[param]['bounds'][1])
            elif self.param_dict[param]['transf'] == 'tanh':
                models_stack[:,pidx] = np.arctanh(models_stack[:,pidx])
            else:
                print(f'No transformation type specified for {param} dimension, attempting logistic transform')
                models_stack[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'] = self.logistic(models_stack[:,pidx],\
                    wholedataset=True, rescale_max=self.param_dict[param]['bounds'][1])
            
        #repeat subpopulation hyperparameter values Nsamps times for each subpopulation
        hp_combos = np.squeeze(list(product(*self.hp_vals)))
        hps_stack = np.repeat(hp_combos, (model_size).astype(int), axis=0)

        #reshape conditionals and weights
        hps_stack = np.reshape(hps_stack,(-1,self.conditionals))
        weights = np.reshape(weights,(-1,1))

        #split the training data, conditional data, and sample weights into training and validation sets
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
            samples in shape [N, no_params]
        """
        
        #sample from flow - this returns samples in the logistically mapped space
        logit_samps = self.flow.sample(conditional,N)

        #map samples back from logit space
        samps = np.zeros(np.shape(logit_samps))
        for pidx, param in enumerate(self.param_dict):
            if self.param_dict[param]['transf'] == 'logit':
                samps[:,pidx] = self.expistic(logit_samps[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'])
            elif self.param_dict[param]['transf'] == 'tanh':
                samps[:,pidx] = np.tanh(logit_samps[:,pidx])
            else:
                print(f'No transformation type specified for {param} dimension, attempting expistic transform')
                samps[:,pidx] = self.expistic(logit_samps[:,pidx], self.param_dict[param]['logit_max'], self.param_dict[param]['max'])

        return samps

    def __call__(self, data, conditional_hps, smallest_N, prior_pdf=None):
        """
        Calculate the likelihood of the observations give a particular hypermodel (given by conditional_hps).
        This is the hyperlikelihood).

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
        likelihoods_per_samp = self.flow.get_logprob(data, mapped_obs, self.param_dict, conditionals)

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

    def train(self, no_trans, no_neurons, no_blocks, no_bins, lr, epochs, batch_no, filepath, device):
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
        deivce : str
            Device on which to run the flow. Either is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """

        #define flow hyperparams as class properties
        self.no_trans = no_trans
        self.no_neurons = no_neurons
        self.no_blocks = no_neurons
        self.no_bins = no_bins


        #TO CHANGE - make this a settable parameter
        batch_size=10000

        #initislises flow network
        self.flow = NFlow(self.no_trans, self.no_neurons, self.no_blocks, self.no_bins, self.no_params, self.conditionals, batch_size, 
                    self.total_smdls, RNVP=False, device=device)

        #map the training samples etc 
        training_data, val_data = self.map_samples(filepath)

        #write or append channel config to json file
        channel_config = {'transforms':no_trans, 'neurons':no_neurons,'blocks':no_neurons,'bins':no_bins}

        #set mapping parameters into channel config
        for param in self.param_dict:
            if self.param_dict[param]['transf'] == 'logit':
                channel_config[param] = {'logit_max':self.param_dict[param]['logit_max'], 'max':self.param_dict[param]['max']}

        channel_json = {}
        channel_json[self.channel_label] = channel_config

        #check if config exists e.g. for other channels, and update this channel to current config
        if os.path.isfile(f'{filepath}flowconfig.json'):
            with open(f'{filepath}flowconfig.json', 'r') as f:
                old_config = json.load(f)
            #update the old config of this channel
            old_config[self.channel_label] = channel_config
            #load old config of other channels
            channel_json = old_config

        #write this channels config to file
        with open(f'{filepath}flowconfig.json', 'w') as f:
            json.dump(channel_json, f)

        save_filename = f'{filepath}{self.channel_label}'
        #train the normalising flow
        self.flow.trainval(lr, epochs, batch_no, save_filename, training_data, val_data)

    def load_model(self, filepath, device='cpu'):
        """
        Loads the normalising flow into self.flow with configuration of flow network parameters from json file if it exists.

        Parameters
        -------
        filepath : str
            directory with saved flow model and config
        deivce : str
            Device on which to run the flow. Either is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """
        #load no. transforms, no. neurons and no. bins from config and reinitialise flow if config for flows exists
        if os.path.isfile(f'{filepath}flowconfig.json'):
            with open(f'{filepath}flowconfig.json', 'r') as f:
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
                            self.param_dict[param][key] = config[self.channel_label][key]
            except:
                #deal with old hardcoding mapping saving
                mappings = np.load(f'{filepath}{self.channel_label}_mappings.npy', allow_pickle=True)
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

        batch_size=10000
        self.flow = NFlow(self.no_trans, self.no_neurons, self.no_blocks, self.no_bins, self.no_params, self.conditionals, batch_size,\
            self.total_smdls, RNVP=False, device=device)
        
        #load in actual flow model, and mappings
        self.flow.load_model(f'{filepath}{self.channel_label}.pt')

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

        #reshape detection efficiency values onto grid the shape of hyperparameter values
        hp_grid_shape = [len(self.hyperparam_models[i]) for i in range(len(self.hyperparam_models))]
        alpha_grid = np.reshape(tuple(self.alpha.values()), (hp_grid_shape))

        #initialise interpolator over hyperparameters to interolate log(detection efficiency)
        alpha_interp = sp.interpolate.RegularGridInterpolator((self.hp_vals), np.log(alpha_grid),\
            bounds_error=False, method='pchip', fill_value=None)
        #find alpha at specified chi_b, log(alpha_CE)
        alpha = np.exp(alpha_interp(hyperparams))
        return alpha
