import sys
import argparse
import h5py
import warnings
import operator
import multiprocessing
from copy import deepcopy
import pdb

import numpy as np
import pandas as pd
import scipy.stats

import 1D_flow_plots

argp = argparse.ArgumentParser()
argp.add_argument("--file-path", type=str, required=True, help="")
argp.add_argument("--flow-path", type=str, help="")
argp.add_argument("--channels", nargs="+", type=str, required=True, help="")
argp.add_argument("--flow-bins", nargs="+", type=int, required=True, help="")
argp.add_argument("--use-unityweights", action="store_true", help="")
argp.add_argument("--use-reg", action="store_true", help="")

args = argp.parse_args()

for i, channel in enumerate(args.channels):
    if channel != 'CE':
        flow_plots.plot1Dsamps_nonCE(args.file_path, args.flow_path, channel, args.flow_bins[i], args.use_unityweights)
    else:
        flow_plots.plot1Dsamps_CE(args.file_path, args.flow_path, args.flow_bins[i], args.use_unityweights)
