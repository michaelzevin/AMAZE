#!/usr/bin/env python

"""
Functions for mock selection effects
"""

import numpy as np
from scipy.interpolate import interp1d


def projection_factor_Dominik2015_interp(alpha=1.0, a2=0.374222, a4=2.04216, a8=-2.63948):
    # Interpolation of the projection factor 'w' from Dominik+2015
    def w_cdf(w, alpha=alpha, a2=a2, a4=a4, a8=a8):
        # Note that their PDF is P(w>w'), what we really want is P(w<w') so we do 1-P(w) at the end
        term1 = a2*((1-w/alpha)**2)
        term2 = a4*((1-w/alpha)**4)
        term3 = a8*((1-w/alpha)**8)
        term4 = (1-a2-a4-a8)*((1-w/alpha)**10)
        return 1-(term1+term2+term3+term4)

    w_pts = np.linspace(0,1,10000)
    interp_func = interp1d(w_cdf(w_pts), w_pts)
    return interp_func


_PSD_defaults = {
    "ligo_psd": "LIGO_P1200087.dat",
    "virgo_psd": "Virgo_P1200087.dat",
    "midhighlatelow": {"H1":"midhighlatelow"},
    "midhighlatelow_network": {"H1":"midhighlatelow",
            "L1":"midhighlatelow",
            "V1":"midhighlatelow"},
    "design": {"H1":"design"},
    "design_network": {"H1":"design",
            "L1":"design",
            "V1":"design"},
    "snr_single": 8,
    "snr_network": 10}
