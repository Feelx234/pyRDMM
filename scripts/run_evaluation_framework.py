import warnings
#warnings.filterwarnings("error")
import sys
from RDMM.evaluation_framework import EvaluationFramework, mine_pair_parameters
from pathlib import Path
import numpy as np
import pysubgroup as ps
from tqdm import tqdm
day_path = Path('06_04')

folder= Path.home()/Path('experiments')/day_path
#folder= "L:"/Path('experiments')/day_path

if __name__ == '__main__':
    print(folder)
    frame = EvaluationFramework(folder)
    #frame.override=True
    createDataset = True
    regression = True
    transition = False
    covariance = True
    mine_pair=True
    exhaustive=True
    processes=6
    mine_pair_params=mine_pair_parameters(50,100,2,'mine_pair',[ps.MinSupportConstraint(50)])
    exhaustive_params=mine_pair_parameters(2000,1000000,2,'exhaustive',[ps.MinSupportConstraint(50)])
    if regression:
        n_reg_frames=10
        #try:
        if createDataset:
            print("Generating regression datasets...")
            out = frame.create_linear_regression_datasets(n_classes=10, n_noise=10, n_dataframes=n_reg_frames, hide_depth=2, show_progress=tqdm)
            print("Generating regression datasets: Done")
        if mine_pair:
            print("Runnning regression mine pair")
            frame.execute_regression_tests(mine_pair_params, n_dataframes=n_reg_frames, processes=processes)
        if exhaustive:
            print("Runnning regression exhaustive")
            frame.execute_regression_tests(exhaustive_params, n_dataframes=n_reg_frames, processes=processes)
    if transition:
        n_states=5
        n_trans_frames=10
        if False:
            print("Generating datasets...")
            out = frame.create_transition_datasets(n_classes=10, n_noise=10, n_dataframes=n_trans_frames, hide_depth=2, n_states=n_states)
            print("Generating datasets: Done")

        if True:
            mine_pair_params=mine_pair_params._replace(task_name='mine_pair'+str(n_states))
            exhaustive_params=exhaustive_params._replace(task_name='exhaustive'+str(n_states))
            frame.execute_transition_tests(exhaustive_params, n_dataframes=n_trans_frames, processes=4, n_states=n_states)

    if covariance:
        n_states=5
        n_cov_frames=10
        mine_pair_params=mine_pair_params._replace(task_name='mine_pair'+str(n_states))
        exhaustive_params=exhaustive_params._replace(task_name='exhaustive'+str(n_states))
        if createDataset:
            print("Generating cov datasets...")
            out = frame.create_cov_datasets(n_classes=10, n_noise=10, n_dataframes=n_cov_frames, hide_depth=2, n_states=n_states, show_progress=tqdm)
            print("Generating cov datasets: Done")

        if mine_pair:
            print("Runnning cov mine pair")
            frame.execute_cov_tests(mine_pair_params, n_dataframes=n_cov_frames, processes=processes, n_states=n_states)
        if exhaustive:
            print("Runnning cov exhautive")
            frame.execute_cov_tests(exhaustive_params, n_dataframes=n_cov_frames, processes=processes, n_states=n_states)


    