import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from itertools import product

from . import *
from .utils.transform import _DEFAULT_TRANSFORMS, _to_chieff, \
_uniform_spinmag, _isotropic_spinmag
from .Pop_Flows import FlowModel



_VALID_SPIN_DISTR = {
    # Uniform - |a| ~ uniform distribution in 0, 1
    "uniform": _uniform_spinmag,
    # Isotropic - |a| ~ a^2 distribution in 0, 1
    "isotropic": _isotropic_spinmag
}

def get_params(df, params):
    inference_params = pd.DataFrame()

    # check if :params: in the dataframe, otherwise perform transformations
    for param in params:
        if param not in df.columns:
            # default transformations
            if param in _DEFAULT_TRANSFORMS.keys():
                df[param] = _DEFAULT_TRANSFORMS[param](df)
            # chieff transformations
            elif param=='chieff':
                df['theta1'] = _DEFAULT_TRANSFORMS['theta1'](df)
                df['theta2'] = _DEFAULT_TRANSFORMS['theta2'](df)
                # check if spin magnitudes have been provided
                if not {'a1','a2'}.issubset(df.columns):
                    if spin_distr in _VALID_SPIN_DISTR:
                        df['a1'],df['a2'] = _VALID_SPIN_DISTR[spin_distr](df)
                    else:
                        raise NameError("Spin magnitudes not provided and valid spin distribution was not specified, so can't generate effective spins!")

                df['chieff'] = _to_chi_eff(df)
            # otherwise, raise an error
            else:
                raise NameError("You specified the parameter {0:s} for inference, but it is not in your population data and you haven't written a transformation to calculate it!".format(param))

    return df

def read_hdf5(path, channel, channel_smdl_names, smdl_indxs_combos):
    """
    For CE channel, returns diction of submodels for all chi_b and alpha_CE values, as keys i,j in dictionary
    For other channels, returns dictionary of submodels varying with chi_b for that channel

    Parameters
    ----------
    path : list of str
        binary parameters used for inference e.g. ['mchirp', 'q']
    channel : str
        string of 1 formation channel, either 'CE', 'CHE', 'GC' etc.
    Returns
    ----------
    popsynth_outputs: pandas dataframe
        dataframe of samples from models hdf5 file, of param for each submodel.
    """
    popsynth_outputs = {}
    #error handling for model keys for Nhyper>1, to set these to tuple type
    for i, channel_smdl_name in enumerate(channel_smdl_names):
        if smdl_indxs_combos[i].ndim > 0:
            dict_key = tuple(smdl_indxs_combos[i])
        else:
            dict_key = smdl_indxs_combos[i]
        popsynth_outputs[dict_key]=pd.read_hdf(path, key=channel_smdl_name)
    return(popsynth_outputs)


def get_models(file_path, channels, param_dict, use_flows, full_hyperparam_dict, sensitivity=None, normalize=False, detectable=False, device='cpu', flow_path=None, **kwargs):
    """
    Call this to get all the models and submodels, as well
    as KDEs of these models, packed inside of dictionaries labelled in the
    dict structure models[channel][smdl]. Will first look for :params: as
    series in the dataframe. If they are not present, it will try to construct
    these parameters if the valid transformations are present in transforms.py.

    If chieff is one of the :params: for inference and spin magnitudes are not
    provided, this function will first check if :spin_distr: is provided and
    if so, will generate spin magnitudes and calculate chieff using these
    spins and the m1/m2 specified in the dataframes.

    Parameters
    ----------
    file_path : str
        filepath to models_reduced.hdf5
    channels : list of str or None
        which channels to load models of, from CE, CHE, SMT, GC and NSC
    params : list of str
        which binary parameters to read from file, from mchirp, q, chieff, and z.
        fed to likelihood model
    use_flows : bool
        flag for whether to use KDEs or flows in inference

    Returns
    ----------
    deepest_models : list of str
        list of submodels to get likelihood models from, in format 'CE/chi00/alpha02'
    kde_models : dictionary of KDEs
        dictionary of KDE models for each submodel
    OR
    flow_models : dictionary of flows
        for each formation channel
    """

    # all models should be saved in 'file_path' in a hierarchical structure, with the channel being the top group
    f = h5py.File(file_path, "r")

    # find all the deepest models to set up dictionary for KDE models
    deepest_models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            deepest_models.append(name.rsplit('/', 1)[0])
    f.visititems(find_submodels)
    f.close()
    deepest_models = sorted(list(set(deepest_models)))
    
    # if only using specific formation channels, remove other models
    if channels:
        deepest_models_cut = []
        for chnl in channels:
            for mdl in deepest_models:
                if chnl+'/' in mdl:
                    deepest_models_cut.append(mdl)
        deepest_models = deepest_models_cut

    #find hyperparameters in models
    hyperparam_dict  = {}
    hyperidx=0
    deepest_models.sort()
    #list of model keys
    hyperparams = sorted(list(set([x.split('/', 1)[1] for x in deepest_models])))
    #total number of hyperparameters = maximum number of hyperparameters in any model
    Nhyper = np.max([len(x.split('/')) for x in hyperparams])
    
    #construct hyperparam dict, with keys [0,..,Nhyper] and values the points in hyperparameter space with simulations
    while hyperidx < Nhyper:
        hyperidx_with_Nhyper = np.argwhere(np.asarray([len(x.split('/')) for x in hyperparams])>hyperidx).flatten()
        hyperparams_at_level = sorted(set([x.split('/')[hyperidx] for x in np.asarray(hyperparams)[hyperidx_with_Nhyper]]))
        hyperparam_dict[hyperidx] = hyperparams_at_level
        hyperidx += 1
    #length of the hyperparam dict for each dimension
    hyperparam_pts_per_dim = [len(hyperparam_dict[x]) for x in range(Nhyper)]

    #KDE case: reads in submodel for each of the deepest model and sends to KDEModel
    #Flow case: reads in samples from all channels and sends to FlowModel
    if use_flows==True:
        flow_models = {}
        for i, chnl in enumerate(tqdm(channels)):
            #find submodel keys, and indices they should correspond to in the input samples dict to the FlowModel
            channel_smdls = [x for x in deepest_models if chnl+'/' in x]
            channel_smdls_split = np.array([x.split('/')[1:] for x in deepest_models if chnl+'/' in x])
            smdl_indices = [list(np.arange(hyperparam_pts_per_dim[i])) for i in range(channel_smdls_split.shape[1])]
            smdl_indxs_combos = np.squeeze(list(product(*smdl_indices)))

            #currently list of lists of hyperparam model strings
            #channel_hyperparams = [hyperparam_dict[i] for i in range(channel_smdls_split.shape[1])]
            #instead want this, assuming hyperparam_dict contains values and keys:
            channel_hyperparams = {}
            for hp in full_hyperparam_dict:
                #TO CHANGE with MR #12 if full_hyperparam_dict[hp] in channels dict parameters:
                if chnl in full_hyperparam_dict[hp]['channels']:
                    channel_hyperparams[hp] = full_hyperparam_dict[hp]

            popsynth_outputs = read_hdf5(file_path, chnl, channel_smdls, smdl_indxs_combos)
            flow_models[chnl] = FlowModel.from_samples(chnl, popsynth_outputs, param_dict, channel_hyperparams, smdl_indxs_combos, sensitivity=sensitivity)
        return deepest_models, flow_models
    else:
        kde_models = {}
        #tqdm shows progress meter
        for smdl in tqdm(deepest_models):
            smdl_list = smdl.split('/')
            current_level = kde_models
            for part in smdl_list:
                if part not in current_level:
                    if part == smdl_list[-1]:
                        # if we are on the last level, read in data and store kdes
                        df = pd.read_hdf(file_path, key=smdl)
                        label = '/'.join(smdl_list)
                        mdl = KDEModel.from_samples(label, df, list(param_dict.keys()), sensitivity=sensitivity, normalize=normalize, detectable=detectable, **kwargs)
                        current_level[part] = mdl
                    else:
                        current_level[part] = {}

                current_level = current_level[part]
        return deepest_models, kde_models
            

