import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import astropy.units as u
from astropy import cosmology
from astropy.cosmology import z_at_value
cosmo = cosmology.Planck18

from . import *

"""
Function for using GW observations for generating the observations in model 
selection. Events should be stored as dataframes in hdf5 files 
('GWXXXXXX*.hdf5') with the parameters being series in these dataframes. The 
key containing the posterior samples should be consistent with the naming 
scheme of GWTC-3.
"""

# can exclude a subset of events to use by specifying in the list below
_events_to_exclude=['GW190521']

# specify the hdf5 key for the approximant being used
_posterior_key = "combined"

# conversion function
def _gwtc_to_mchirp(gw):
    m1 = gw['m1_detector_frame_Msun']
    m2 = gw['m2_detector_frame_Msun']
    return (m1*m2)**(3./5) / (m1+m2)**(1./5)
def _gwtc_to_q(gw):
    m1 = np.asarray(gw['m1_detector_frame_Msun'])
    m2 = np.asarray(gw['m2_detector_frame_Msun'])
    q = m2/m1
    pos_idxs = np.argwhere(q > 1)
    q[pos_idxs] = m1[pos_idxs]/m2[pos_idxs]
    return q
def _gwtc_to_chieff(gw):
    m1 = np.asarray(gw['m1_detector_frame_Msun'])
    m2 = np.asarray(gw['m2_detector_frame_Msun'])
    a1 = np.asarray(gw['spin1'])
    a2 = np.asarray(gw['spin2'])
    cost1 = np.asarray(gw['costilt1'])
    cost2 = np.asarray(gw['costilt2'])
    return (m1*a1*cost1 + m2*a2*cost2) / (m1+m2)
def _gwtc_to_redshift(gw):
    # This takes time and should be done in preprocessing!
    print('converting luminosity distances to redshift...')
    redz = []
    dL = np.asarray(gw['luminosity_distance_Mpc'])
    for val in tqdm(dL):
        redz.append(z_at_value(cosmo.luminosity_distance, val*u.Mpc))
    return np.asarray(redz)

_conversion_dict = {'z': 'redshift'}
_parameter_transforms = {'mchirp': _gwtc_to_mchirp, 'q': _gwtc_to_q, \
                   'chieff': _gwtc_to_chieff, 'z': _gwtc_to_redshift}


def read_observations(params, Nsamps, obs_path, events_to_exclude=None, prior_key=None):

    event_files = []
    event_names = []
    for f in os.listdir(obs_path):
        event_name = f.split('.')[0]
        if events_to_exclude is not None:
            if event_name in events_to_exclude:
                continue
        event_files.append(f)
        event_names.append(event_name)

    # Set up samples for the specified uncertainty
    obs=np.zeros((len(event_files), len(params)))
    samples_shape = (len(event_files), Nsamps, len(params))
    samples=np.zeros(samples_shape)

    # If prior key is set, set up empty array for prior weights p(theta)
    #   otherwise, assume equal prior weights for all samples
    if prior_key is not None:
        p_theta = np.zeros((samples.shape[0],samples.shape[1]))
    else:
        p_theta = np.ones((samples.shape[0],samples.shape[1]))

    # Now, get the samples for each event
    for idx, f in enumerate(event_files):
        df = pd.read_hdf(os.path.join(obs_path,f), key=_posterior_key)
        # Check to see if the necessary parameters are in the files or the 
        #   transformations provided, else raise error
        for pidx, p in enumerate(params):
            # first, see if parameter is in the keys already
            if p in df.columns:
                continue
            elif p in _conversion_dict.keys():
                df = df.rename({_conversion_dict[p]:p}, axis='columns')
            elif p in _parameter_transforms.keys():
                df[p] = _parameter_transforms[p](df)
            else:
                raise KeyError("Parameter {0:s} not found in the observational data, and no transformations exist to generate it from the data provided!".format(p))

        # get the median observations (need to return for later)
        obs[idx, :] = np.median(df[params], axis=0)

        # see if the specified prior key is in the data
        if prior_key is not None and prior_key not in df.columns:
            raise KeyError("Prior key {0:s} is not in the GW data for file {1:s}!".format(prior_key,f))

        # randomly choose posterior samples to draw, with special treatment for cases,
        #   where there are less samples than specified number of observations
        if len(df) < Nsamps:
            Ndraws = 0
            sample_idxs = np.zeros(Nsamps, dtype=int)
            while Ndraws*len(df) < Nsamps-len(df):
                sample_idxs[Ndraws*len(df):(Ndraws+1)*len(df)] = np.arange(len(df))
                Ndraws+=1
            sample_idxs[Ndraws*len(df):Nsamps] = np.random.choice(np.arange(len(df)), size=Nsamps-(Ndraws*len(df)), replace=False)
        else:
            sample_idxs = np.random.choice(np.arange(len(df)), size=Nsamps, replace=False)
        samples[idx, :, :] = df[params].iloc[sample_idxs]

        if prior_key is not None:
            p_theta[idx, :] = df[prior_key].iloc[sample_idxs]

    return obs, samples, p_theta, event_names
