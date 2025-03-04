import argparse
from plot_functions import *
import glob

"""
Plots results for Nflow inference with AMAZE infrastructure.

Will also plot model corner plots, discrete and continuous result, and dataspace if args supplied.
"""

argp = argparse.ArgumentParser()
argp.add_argument("--flow-path", type=str, default=None, help="Directory from where to load flow models. Default=None.")

argp.add_argument("--plot-discrete-result", action="store_true", help="True if plotting discrete result. Default=False.")
argp.add_argument("--plot-cont-result", action="store_true", help="True if plotting continuous result. Default=False.")
argp.add_argument("--plot-flow-corner", action="store_true", help="True if plotting flow model corner plots. Default=False.")
argp.add_argument("--plot-llh-ratio", action="store_true", help="True if plotting flow KDE log likelihood ratio plot. Default=False.")
argp.add_argument("--plot-dataspace-result", action="store_true", help="True if plotting flow KDE log likelihood ratio plot. Default=False.")
argp.add_argument("--save-det-betas", action="store_true", help="True if saving hdf files of converted branching fractions. Default=None.")
argp.add_argument("--discrete-result-path", type=str, default=None, help="Directory from where to load discrete result files. Default=None.")
argp.add_argument("--cont-result-path", type=str, default=None, help="Directory from where to load continuous inference result files. Default=None.")
argp.add_argument("--KDE-result-path", type=str, default=None, help="Directory from where to load discrete KDE result files. Default=None.")
argp.add_argument("--outdir", type=str, default=None, help="Directory from where to save output files and plots. Default=None.")
argp.add_argument("--name", type=str, default="", help="Name to save corner samples files by.")

argp.add_argument("--hyperparam-idxs", nargs="+", type=int, default=None, help="")
argp.add_argument("--channel-label", type=str, nargs="+", default="CE", help="")
argp.add_argument("--conditional", type=float,  nargs="+", help="")
argp.add_argument("--plot-KDE",  action="store_true", help="If true, adds KDE samples to corner plot")
argp.add_argument("--justplot",  action="store_true", help="If false, draws samples for population corner plots")
argp.add_argument("--testCE",  action="store_true", help="If true, formats the corner plot for the test CE figure")


args = argp.parse_args()
flow_dir = args.flow_path
outdir = args.outdir
discrete_result_path = args.discrete_result_path
discrete_result_path_KDE = args.KDE_result_path
channel_label = args.channel_label
hyperparam_idxs = args.hyperparam_idxs
conditional = np.array(args.conditional)
justplot = args.justplot
name = args.name
cont_result_path = args.cont_result_path

#make corner plots of specified population
if args.plot_flow_corner:
    for channel in channel_label:
        make_pop_corner(channel, hyperparam_idxs, justplot=justplot, flow_dir=flow_dir, conditional=conditional, outdir=outdir, plot_KDE=args.plot_KDE, testCE=args.testCE)

if args.plot_llh_ratio:
    plot_llh_ratio_CE(flow_dir, outdir=outdir, justplot=justplot)

if args.plot_discrete_result:
    discrete_result_files = glob.glob(f'{discrete_result_path}/*.hdf5')
    try:
        KDE_result_files = glob.glob(f'{discrete_result_path_KDE}/*.hdf5')
        make_1D_result_discrete(discrete_result_files, second_files=KDE_result_files, labels = [' flow', ' KDE'], figure_name='Discrete_allKDE', outdir=outdir)
    except FileNotFoundError():
        make_1D_result_discrete(discrete_result_files, second_files=None, labels = [' flow', None], figure_name='Discrete', outdir=outdir)

if args.plot_cont_result:
    if args.save_det_betas:
        save_detectable_betas(glob.glob(f'{cont_result_path}/*.hdf5'), 'cont_retrainedCE', outdir=outdir)
    filenames_det = f'{outdir}/data/cont_retrainedCE_detectable_betas.hdf5'
    cont_result_files = glob.glob(f'{cont_result_path}/*.hdf5')
    make_1D_result_continuous(cont_result_files, filenames_det=filenames_det, outdir=outdir, detectable=True)

if args.plot_dataspace_result:
    #plot dataspace result
    cont_result_files = glob.glob(f'{cont_result_path}/*.hdf5')
    plot_samps_dataspace(cont_result_files, flow_dir, outdir, justplot)