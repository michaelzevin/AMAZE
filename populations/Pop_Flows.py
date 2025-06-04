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
            if sensitivity is not None:
                # if cosmological weights are provided, do mock draws from the pop
                if 'weight' in sbml_samps.keys():
                    mock_samp = sbml_samps.sample(int(1e6), weights=(sbml_samps['weight']/len(sbml_samps)), replace=True, random_state=random_seed)
                else:
                    mock_samp = sbml_samps.sample(int(1e6), replace=True, random_state=random_seed)
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
