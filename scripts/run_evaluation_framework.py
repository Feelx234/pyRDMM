import warnings
#warnings.filterwarnings("error")
import sys
sys.path.append('pysubgroup')
from RDMM.evaluation_framework import EvaluationFramework, mine_pair_parameters
from pathlib import Path
import numpy as np
folder= Path.home()/Path('experiments')/Path('05_05')

if __name__ == '__main__':
    print(folder)
    frame = EvaluationFramework(folder)
    #frame.override=True
    createDataset = False
    regression = True
    transition = False
    mine_pair_params=mine_pair_parameters(50,100,2,'mine_pair',[])
    exhaustive_params=mine_pair_parameters(2000,1000000,2,'exhaustive',[])
    if regression:
        params=mine_pair_params
        n_reg_frames=10
        #try:
        if createDataset:
            out = frame.create_linear_regression_datasets(n_classes=10, n_noise=10, n_dataframes=n_reg_frames, hide_depth=2)
        if True:
            frame.execute_regression_tests(params, n_dataframes=n_reg_frames, processes=4)
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


    