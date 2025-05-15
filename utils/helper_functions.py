import configparser
import ast
import warnings
import os

import numpy as np

from functools import reduce
import operator

__all__ = ['GetFromDict','SetInDict','ParseIniFile',\
           'ErrorCheckIni','ErrorCheckModels']

# --- Useful functions for accessing items in KDE dictionary
def GetFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def SetInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def ParseIniFile(file_path):
    """
    Parses an INI file and returns a dictionary with sections as keys and
    dictionaries of key-value pairs as values.
    """
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
    Flows = config_dict.get('Flows')
    ExtraOptions = config_dict.get('ExtraOptions')

    # save all options in arguments dictionary
    settings = {}
    for section in [MainSettings,RealObservations,MockObservations,Flows,ExtraOptions]:
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
            raise ValueError("Output directory already exists! Please choose a different output directory.")

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