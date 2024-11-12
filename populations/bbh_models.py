import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from . import *
from .utils.transform import _DEFAULT_TRANSFORMS, _to_chieff, \
_uniform_spinmag, _isotropic_spinmag
from .Flowsclass_dev import FlowModel



_VALID_SPIN_DISTR = {
    # Uniform - |a| ~ uniform distribution in 0, 1
    "uniform": _uniform_spinmag,
    # Isotropic - |a| ~ a^2 distribution in 0, 1
    "isotropic": _isotropic_spinmag
}


def get_model_keys(path, channel):
    all_models = []
    models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_models.append(name.rsplit('/', 1)[0])
            
    f = h5py.File(path, 'r')
    f.visititems(find_submodels)
    # get all unique models
    all_models = sorted(list(set(all_models)))
    f.close()

    # use only models with given alpha value
    for model in all_models:
        if channel in model:
            models.append('/'+model)
    return(np.array(models))

def get_model_keys_CE(path):
    all_models = []
    models = []
    def find_submodels(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_models.append(name.rsplit('/', 1)[0])
            
    f = h5py.File(path, 'r')
    f.visititems(find_submodels)
    # get all unique models
    all_models = sorted(list(set(all_models)))
    f.close()

    # use only models with given alpha value
    for model in all_models:
        if 'CE' in model:
            models.append('/'+model)
    return(np.split(np.array(models), 4))

def read_hdf5(path, channel):
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
        dataframe of samples from models hdf5 file, of param for each submodel in One channel (function input)
    """
    if channel=='CE':
        popsynth_outputs = {}
        models = np.asarray(get_model_keys_CE(path))
        for i in range(models.shape[0]):
            for j in range(models.shape[1]):
                popsynth_outputs[i,j]=pd.read_hdf(path, key=models[i,j])
    else:
        popsynth_outputs = {}
        models = np.asarray(get_model_keys(path, channel))
        for i in range(len(models)):
            popsynth_outputs[i]=pd.read_hdf(path, key=models[i])
    return(popsynth_outputs)


def get_models(file_path, channels, params, use_flows, sensitivity=None, normalize=False, detectable=False, device='cpu', flow_path=None, **kwargs):
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

    #KDE case: reads in submodel for each of the deepest model and sends to KDEModel
    #Flow case: reads in samples from all channels and sends to FlowModel
    if use_flows==True:
        flow_models = {}
        for i, chnl in enumerate(tqdm(channels)):
            popsynth_outputs = read_hdf5(file_path, chnl)
            flow_models[chnl] = FlowModel.from_samples(chnl, popsynth_outputs, params, device=device, sensitivity=sensitivity, flow_path=flow_path)
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
                        mdl = KDEModel.from_samples(label, df, params, sensitivity=sensitivity, normalize=normalize, detectable=detectable, **kwargs)
                        current_level[part] = mdl
                    else:
                        current_level[part] = {}

                current_level = current_level[part]
        return deepest_models, kde_models
            

