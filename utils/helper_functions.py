import configparser
import ast
import warnings
import os
import h5py

import numpy as np
np.set_printoptions(legacy='1.25') #so datatypes aren't printed
import pandas as pd
from copy import deepcopy

from functools import reduce
import operator

__all__ = ['GetFromDict','SetInDict','ParseIniFile',\
            'ErrorCheckIni','ErrorCheckModels',\
            'GetDeepestModels','DetectableBranchingFractions',\
            'PrintSummaryStatistics','SaveToDisk']

# --- Useful functions for accessing items in KDE dictionary
def GetFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def SetInDict(dataDict, mapList, value):
    GetFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def ParseIniFile(file_path):
    """
    Parses an INI file and returns a dictionary with sections as keys and
    dictionaries of key-value pairs as values.
    """
    if os.path.isfile(file_path) is False:
        raise Exception("Path to config file does not exist")
     
    config = configparser.ConfigParser()
    config.read(file_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            # Check if the value is a list or a dictionary
            if value.startswith('[') and value.endswith(']'):
                config_dict[section][key] = ast.literal_eval(value)
            elif value.startswith('{') and value.endswith('}'):
                config_dict[section][key] = ast.literal_eval(value)
            # Check if the value is a number
            elif value.isdigit():
                config_dict[section][key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                config_dict[section][key] = float(value)
            # Check if the value is a boolean
            elif value in ['True','true','yes','y']:
                config_dict[section][key] = True
            elif value in ['False','false','no','n']:
                config_dict[section][key] = False
            # Check if the value is None
            elif value in ['None','none','']:
                config_dict[section][key] = None
            else:
                # it's just a string
                config_dict[section][key] = value
    MainSettings = config_dict.get('MainSettings')
    RealObservations = config_dict.get('RealObservations')
    MockObservations = config_dict.get('MockObservations')
    Sensitivity = config_dict.get('Sensitivity')
    Flows = config_dict.get('Flows')
    Sampler = config_dict.get('Sampler')
    ExtraOptions = config_dict.get('ExtraOptions')

    # save all options in arguments dictionary
    settings = {}
    for section in [MainSettings,RealObservations,MockObservations,\
                        Sensitivity,Flows,Sampler,ExtraOptions]:
        for key, value in section.items():
            settings[key] = value

    return settings


def ErrorCheckIni(settings):
    """
    Checks for errors in the INI file sections.
    """
    # Check that output directory does not already exist
    if settings['output-dir'] is not None:
        if os.path.exists(settings['output-dir']):
            warnings.warn("Output directory already exists! Continuing will overwrite files in the existing directory {:s}.".format(settings['output-dir']), stacklevel=2)

    # Check consistency between flows and continuous sample specifications
    if settings['use-flows']==False and settings['continuous-sampling']==True:
        raise ValueError('Cannot use KDEs for continuous inference (you set use-flows==False and continuous-sampling==True).')

    # Check that betas are provided correctly if using mock observations
    if settings['true-model'] is not None:
        # check that keys for channels and branching fractions are consistent
        channels = settings['channels-dict'].keys()
        betas = [float(x) for x in settings['branching-fractions'].values()]
        for c in channels:
            if c not in settings['branching-fractions'].keys():
                raise ValueError(f"Channel {c} is in the channels-dict but not in the branching-fractions dict.")
        # check that numebr of branching fractions provided is consistent with number of channels
        if (len(betas) != len(channels)):
            raise ValueError("Must specify {0:d} branching fractions, you provided {1:d}!".format(len(channels), len(betas)))
        # check that Branching fractions sum to unity
        if ~np.isclose(np.sum(betas), 1):
            raise ValueError("Branching fractions must sum to unity (yours sum to {0:0.2f})!".format(np.sum(betas)))
        # check that Nobs was specified if using mock observations
        if not settings['n-observations']:
            raise ValueError("You need to specify and number of observations to be drawn from the 'true' model if not using observations!")
        # check that the uncertainty method is valid
        valid_uncertainties = ["delta", "events", "snr"]
        if settings['mock-uncertainty'] not in valid_uncertainties:
            raise ValueError("Unspecified measurement uncertainty procedure when using mock observations: '{0:s}' (valid uncertainties: {1:s})".format(settings['mock-uncertainty'], ', '.join(valid_uncertainties)))
        # If 'delta' measurement uncertainty is specified and >1 Nsamps give, spit out warning
        if settings['mock-uncertainty']=='delta' and settings['n-observations']>1:
            warnings.warn("You specified delta-function observations but asked for more than one sample, only one sample will be used for each observations!")
    else:
        # if not using mock observations, make sure event samples are provided
        if settings['event-samples-path'] is None:
            raise ValueError("You need to either specify a true model for mock observations or provide event samples using event-samples-path!")

    # Check that the varied parameters for each population model are in the population parameter dictionary
    for c in settings['channels-dict'].keys():
        for p in settings['channels-dict'][c]['parameters']:
            if p not in settings['population-parameter-dict'].keys():
                raise ValueError(f"Parameter {p} from channel {c} is not in the population parameter dictionary.")

    # Check that true model is provided if using mock observations
    if settings['true-model'] is not None:
        if len(settings['true-model'].keys()) != len(settings['population-parameter-dict'].keys()):
           raise ValueError("The number of parameters in the true model does not match the number of parameters in the population parameter dictionary.")
        # make sure the hyperparameters of the true model are valid
        for key, value in settings['true-model'].items():
            if key not in settings['population-parameter-dict'].keys():
                raise ValueError(f"Parameter {key} from your true model is not in the population parameter dictionary.")
            if value not in settings['population-parameter-dict'][key]['values'].keys():
                raise ValueError(f"Parameter {key} and value {value} from your true model is not in the population parameter dictionary.")

    # Check that optimal SNRs and projection-factor Theta parameters are provided
    #   if using mock observations with 'snr' mock measurement uncertainty
    if settings['true-model'] is not None and settings['mock-uncertainty']=='snr':
        if len(settings['true-model'].keys()) != len(settings['population-parameter-dict'].keys()):
           raise ValueError("The number of parameters in the true model does not match the number of parameters in the population parameter dictionary.")
        # make sure the hyperparameters of the true model are valid
        for key, value in settings['true-model'].items():
            if key not in settings['population-parameter-dict'].keys():
                raise ValueError(f"Parameter {key} from your true model is not in the population parameter dictionary.")
            if value not in settings['population-parameter-dict'][key]['values'].keys():
                raise ValueError(f"Parameter {key} and value {value} from your true model is not in the population parameter dictionary.")


def ErrorCheckModels(model_names, channels, Nhyper, true_model):
    """
    Checks models and hyperparameters
    """
    # check that the true model provided is valid if gwobs not specified
    highest_smdl_ctr=0
    if true_model is not None:
        for channel in channels:
            base_smdls = [s.split('/')[1] for s in model_names if channel+'/' in s]
            highest_smdls = [s.split('/')[-1] for s in model_names if channel+'/' in s]
            # make sure base model is shared across channels
            model0 = [x for x in true_model.values()]
            if model0[0] not in base_smdls:
                raise ValueError("The true model you specified ({0:s}) is not one of the models you loaded in!".format('/'.join(model0)))
            # make sure highest level model is given in at least one channel
            if (model0[-1] in highest_smdls):
                highest_smdl_ctr+=1
        if (highest_smdl_ctr==0):
            raise ValueError("The highest level of the true model you specified ({0:s}) is not used in any of your models!".format('/'.join(model0)))

    # ensure that the number of hyperparameters within each channel
    #   is the same depth
    for channel in channels:
        channel_smdls = [x for x in model_names if channel+'/' in x]
        Nlevels_in_channel = [len(x.split('/')) for x in channel_smdls]
        if not all(x == Nlevels_in_channel[0] for x in Nlevels_in_channel):
            raise ValueError("The formation channel '{0:s}' does not have the same hierarchical levels of hyperparameters across submodels: {1:s}".format(channel, ','.join(channel_smdls)))

    # ensure that models at each level are consistent across formation channels
    # start at 1, which will be the highest-level hyperparameter
    #   since the formation channel is the first parameter
    i=1
    Nhyper_per_model = [len(x.split('/'))-1 for x in model_names]
    while i <= Nhyper:
        models_at_hyperlevel = np.asarray(model_names)[np.asarray(Nhyper_per_model) >= i]
        hyper_set = sorted(set([x.split('/')[i] for x in models_at_hyperlevel]))
        for channel in channels:
            channel_smdls = [x for x in models_at_hyperlevel if channel+'/' in x]
            if len(channel_smdls) > 0:
                channel_set = sorted(set([x.split('/')[i] for x in channel_smdls]))
                if sorted(hyper_set) != sorted(channel_set):
                    raise ValueError("At hyperparameter level {0:d}, the formation channel {1:s} does not have the same hyperparameters as the rest of the models (all models: {2:s}, {1:s}: {3:s}".format(i, channel, ','.join(hyper_set), ','.join(channel_set)))
        i += 1


def GetDeepestModels(model_names, models, hyperparam_dict, use_flows=False):
    # --- Copy kde_models so that they all have the same levels of hyperparameters
    # FIXME: @Storm, is there a way to make it more clear what the flows are doing? 
    #   Maybe a separate while loop if use-flows is true?
    Nhyper = len(hyperparam_dict.keys())
    all_models_at_deepest = all([len(x.split('/')[1:])==Nhyper for x in model_names])

    while all_models_at_deepest==False:
        # loop until all models have the same length
        for model in model_names:
            # See number of hyperparameters in model, subtract one for channel
            Nhyper_in_model = len(model.split('/'))-1
            if use_flows==False:
                kde_hold = GetFromDict(models, model.split('/'))
            # loop until this model has all the hyperparam levels as well
            while Nhyper_in_model < Nhyper:
                if use_flows==False:
                    # remove kde model from old level
                    SetInDict(models, model.split('/'), {})
                model_names.remove(model)
                for new_hyperparam in hyperparam_dict[Nhyper_in_model]:
                    if use_flows==False:
                        # copy the same kde model for the higher hyperparam level
                        new_kde = deepcopy(kde_hold)
                        new_level = model.split('/') + [new_hyperparam]
                        SetInDict(models, new_level, new_kde)
                    # add new model name
                    model_names.append(model+'/'+new_hyperparam)
                Nhyper_in_model += 1
        # see if all models are at deepest level else repeat
        all_models_at_deepest = all([len(x.split('/')[1:])==Nhyper for x in model_names])
    return model_names, models

def DetectableBranchingFractions(samples, model_names, models, submodels_dict, channels_dict, branching_fractions, model0, true_model, use_flows, continuous_sampling):
    """
    Calculates detectable branching fractions after the inference is run
    """
    channels = list(channels_dict.keys())
    detectable_samples = samples.copy()
    smdls = list(set([x.split('/',1)[1] for x in model_names]))
    if use_flows==False:
        # get the conversion factors between the detectable and underlying distributions
        for smdl in sorted(smdls):
            detectable_convfacs = []
            for channel in channels:
                detectable_convfacs.append(GetFromDict(models, [channel]+smdl.split('/')).alpha)
            detectable_convfacs = np.asarray(detectable_convfacs)
            # loop over hyperparams to get samples in this submodel
            hyperparams = smdl.split('/')
            for idx, param in enumerate(hyperparams):
                hyper_idx = list(submodels_dict[idx].keys())[list(submodels_dict[idx].values()).index(param)]
                if idx==0:
                    matching_idxs = np.where(samples[:,idx] == hyper_idx)[0]
                    matching_samps = samples[matching_idxs]
                else:
                    matching_idxs = matching_idxs[np.where(matching_samps[:,idx] == hyper_idx)[0]]
                    matching_samps = samples[matching_idxs]
            # if no samples are in this model, continue
            if len(matching_idxs)==0:
                continue
            # convert hyperparams of these samples accordingly to get the underlying betas
            converted_betas = detectable_samples[matching_idxs,len(hyperparams):] * detectable_convfacs
            converted_betas /= converted_betas.sum(axis=1, keepdims=True)
            detectable_samples[matching_idxs,len(hyperparams):] = converted_betas

            # also save converted relative fractions to model0
            if smdl==true_model:
                converted_rel_fracs = detectable_convfacs * \
                        np.asarray(list(branching_fractions.values()))
                converted_rel_fracs /= np.sum(converted_rel_fracs)
                for cidx, channel in enumerate(channels):
                    model0[channel].rel_frac_detectable(converted_rel_fracs[cidx])
    else:
        alphas = np.zeros((detectable_samples.shape[0], len(channels_dict.keys())))
        #get alpha given hyperparameters in each sample
        for i, samp in enumerate(samples):
            for cidx, chnl in enumerate(channels_dict.keys()):
                smdl = models[chnl]
                #find detection efficiency
                if continuous_sampling:
                    alphas[i, cidx] = smdl.get_alpha([samp[:smdl.conditionals]])
                else:
                    if smdl.conditionals > 1:
                        hyperparam_idxs = tuple(samp[:smdl.conditionals])
                    else:
                        hyperparam_idxs = samp[0]
                    alphas[i, cidx] = smdl.alpha[hyperparam_idxs]
        #multiply by detection efficiency
        detectable_samples = (detectable_samples[:,2:] * alphas)
        #divide by sum across channels
        detectable_samples /= detectable_samples.sum(axis=1, keepdims=True)
        #concatenate detectable branching fractions back with astrophysical parameter samples
        detectable_samples = np.concatenate((samples[:,:2],detectable_samples), axis=1)
    return detectable_samples, model0

def PrintSummaryStatistics(samples, samples_det, model_names, \
                           channels_dict, pop_param_dict, submodels_dict):
    """
    Print summary statistics for inference
    """
    channels = list(channels_dict.keys())
    Nhyper = len(list(pop_param_dict.keys()))

    recovered_vals = {}
    smdls = list(set([x.split('/',1)[1] for x in model_names]))
    for smdl in sorted(smdls):
        recovered_vals[smdl] = {}
        hyperparams = smdl.split('/')
        # loop over hyperparams to get matching samples
        for idx, param in enumerate(hyperparams):
            hyper_idx = list(submodels_dict[idx].keys())[list(submodels_dict[idx].values()).index(param)]
            if idx==0:
                matching_samps = samples[samples[:,idx] == hyper_idx]
                if samples_det is not None:
                    matching_samps_detectable = samples_det[samples_det[:,idx] == hyper_idx]
            else:
                matching_samps = matching_samps[matching_samps[:,idx] == hyper_idx]
                if samples_det is not None:
                    matching_samps_detectable = matching_samps_detectable[matching_samps_detectable[:,idx] == hyper_idx]
        # get counts in this model
        counts = matching_samps.shape[0]
        recovered_vals[smdl]['counts'] = counts
        # get betas for this model from each channel
        recovered_vals[smdl]['betas'] = {}
        recovered_vals[smdl]['betas_detectable'] = {}
        for cidx, channel in enumerate(channels):
            # append beta values for this model
            if counts > 0:
                beta = matching_samps[:,Nhyper+cidx]
                beta = round(np.mean(beta), 3)
                if samples_det is not None:
                    beta_detectable = matching_samps_detectable[:,Nhyper+cidx]
                    beta_detectable = round(np.mean(beta_detectable), 3)
            else:
                beta = np.nan
                if samples_det is not None:
                    beta_detectable = np.nan
            recovered_vals[smdl]['betas'][channel] = beta
            if samples_det is not None:
                recovered_vals[smdl]['betas_detectable'][channel] = beta_detectable

    # print everything
    for smdl in sorted(smdls):
        sample_counts = recovered_vals[smdl]['counts']
        sample_betas = recovered_vals[smdl]['betas']
        sample_betas_detectable = recovered_vals[smdl]['betas_detectable']
        print("  Model {:s}".format(smdl))
        print("    {:d} samples ({:0.2f}%)".format(sample_counts, 100*(sample_counts/len(samples))))
        print("    betas={}".format(list(sample_betas.items())))
        if samples_det is not None:
            print("    detectable betas={}".format(list(sample_betas_detectable.items())))
    print("")

def SaveToDisk(settings, random_seed, model0, submodels_dict, obsdata, \
               samples, probs, events=None, detectable_samples=None):
    """
    Saves all relevant information to disk as hdf5 file
    """
    if settings['output-dir'] is not None:
        fpath = os.path.join(os.getcwd(), settings['output-dir'], f'amaze_output_seed{random_seed}.hdf5')
    else:
        fpath = os.path.join(os.getcwd(), f'amaze_output_seed{random_seed}.hdf5')

    if settings['verbose']:
        print("  writing to disk at {:s}...".format(fpath))

    params = list(settings['event-parameter-dict'].keys())
    channels = list(settings['channels-dict'].keys())

    # set up h5 file
    hfile = h5py.File(fpath, "w")
    bsgrp = hfile.create_group("model_selection")

    # save aspects of the true model
    if settings['true-model'] is not None:
        info = np.append([*[k+': '+v for k,v in settings['true-model'].items()]], \
                [*[key+': '+str(model0[key].rel_frac) for key in model0.keys()]])
    else:
        info = np.append(["Real Observations: "], [*events])
    info = [x.encode('utf-8') for x in info]
    bsgrp.attrs["true-model-params"] = info

    # save strings of the arguments in the config file
    arguments = []
    for key, val in settings.items():
        arguments.append('{}: {}'.format(key,val))
    bsgrp.attrs["config"] = arguments

    # add submodels_dict attribute
    for hyper_idx in submodels_dict.keys():
        conversion = []
        for key, val in submodels_dict[hyper_idx].items():
            conversion.append(str(key)+': '+str(val))
        bsgrp.attrs["p"+str(hyper_idx)+'_conversion_dict'] = conversion
    hfile.close()
    
    # save observations as dataframe
    df = pd.DataFrame()
    saved_obs_idxs = np.repeat(np.arange(obsdata.shape[0]), obsdata.shape[1])
    saved_obs = np.reshape(obsdata, (-1,len(params)))
    df = pd.DataFrame(saved_obs, columns=params, index=saved_obs_idxs)
    df.to_hdf(fpath, key='model_selection/obsdata')

    # save samples as dataframe
    columns = []
    convert_dict = {}
    for hyper_idx in submodels_dict.keys():
        columns.append('p'+str(hyper_idx))
        # save column names to convert model indices to ints
        convert_dict['p'+str(hyper_idx)] = float
    for channel in channels:
        columns.append('beta_'+channel)
    df = pd.DataFrame(samples, columns=columns).astype(convert_dict)
    df.to_hdf(fpath, key='model_selection/samples')

    # save detectable samples as dataframe, if they're calculated
    if detectable_samples is not None:
        columns = []
        convert_dict = {}
        for hyper_idx in submodels_dict.keys():
            columns.append('p'+str(hyper_idx))
            # save column names to convert model indices to ints
            convert_dict['p'+str(hyper_idx)] = float
        for channel in channels:
            columns.append('beta_'+channel)
        df = pd.DataFrame(detectable_samples, columns=columns).astype(convert_dict)
        df.to_hdf(fpath, key='model_selection/detectable_samples')

    # save sample probabilities as dataframe
    df = pd.DataFrame(probs, columns=['lnprb'])
    df.to_hdf(fpath, key='model_selection/lnprb')

    return