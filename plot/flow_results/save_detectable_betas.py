from plot_functions import *
import glob

result_path = '/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_040924_trainlogalpha'
result_files = glob.glob(f'{result_path}/*.hdf5')

save_detectable_betas(result_files, analysis_name='test')