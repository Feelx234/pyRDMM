import numpy as np
import pickle
from tqdm import tqdm
from collections import namedtuple
from .CreateDataSets import generate_two_regression_dataframes, hide, generate_two_transition_dataframes
from .QualityFunctions import *
from .model_algorithm import beam_search_through_candidates, getExceptionalSGs, to_dataframe, find_model_through_heuristic
from . model_target import PolyRegression_ModelClass, Transition_ModelClass
from pathlib import Path
from itertools import product
import pysubgroup as ps
import functools
import os
import time
from collections import defaultdict
from multiprocessing import Pool
import functools
import pandas as pd
import timeit
dataset_tpl = namedtuple('dataset_tpl',['df1','df2','parameters','sizes1','sizes2'])
regression_parameters = namedtuple('regression_parameters',['result_size','total_result_size','depth','constraints'])
mine_pair_parameters = namedtuple('mine_pair_parameters',['result_size','total_result_size','depth','task_name','constraints'])
result_parameters = namedtuple('result_parameters',['alpha','beta','gamma','df_result', 'ex_qf_name','sim_qf_name','dataset_tpl', 'parameters'])
mine_pair_result_parameters = namedtuple('mine_pair_result_parameters',['alpha','beta','gamma','df_result', 'ex_qf_name','sim_qf_name','dataset_tpl', 'parameters','total_runtime','mine_runtime'])
def getDoubleCooks():
    return DoubleCooksSimilarity(PolyRegression_ModelClass(), PolyRegression_ModelClass())

def getLikelihoodSim():
    return LikelihoodSimilarity(PolyRegression_ModelClass(), PolyRegression_ModelClass())

def get_LOG_LikelihoodSim():
    return LikelihoodSimilarity(PolyRegression_ModelClass(), PolyRegression_ModelClass(), use_log=True)

def getCooksExeptionality(gamma):
    return SizeWrapper(EMM_quality_Cook_poly(degree=1), gamma)

def getLikelihoodExceptionality(gamma):
    return SizeWrapper(EMM_LikelihoodGain(PolyRegression_ModelClass()), gamma)

def get_LOG_LikelihoodExceptionality(gamma):
    return SizeWrapper(EMM_LikelihoodGain(PolyRegression_ModelClass(), use_log=True), gamma)

def final_qf(alpha, beta, size, ex, sim):
    if np.isnan(ex):
        return np.nan

    val = size ** alpha * abs(ex) ** beta * sim
    if beta==0:
        return val

    if ex >= 0:
        return val
    else:
        return - val

def getLikelihoodExceptionality_transition(n_states, gamma):
    '''LikelihoodExceptionality'''
    return SizeWrapper(EMM_LikelihoodGain(Transition_ModelClass(n_states, 10)), gamma)

def getLikelihoodSimilarity_transition(n_states):
    '''LikelihoodSimilarity'''
    return LikelihoodSimilarity(Transition_ModelClass(n_states, 10), Transition_ModelClass(n_states, 10))

def getTotalVariationSimilarity(n_states):
    '''TotalVariationSimilarity'''
    return TotalVariationSimilarity(Transition_ModelClass(n_states, 10), Transition_ModelClass(n_states, 10))

def getTotalVariationExceptionality(n_states, gamma):
    '''TotalVariationExceptionality'''
    return SizeWrapper(EMM_TotalVariation(n_states), gamma)

def get_pmatrix(tpl):
    return tpl.p_matrix
def getParameterDiff_trans(n_states, gamma):
    return SizeWrapper(EMM_Exceptionality_ParameterDiff(Transition_ModelClass(n_states, 10), get_pmatrix, exponent=1), gamma)
def getParameterDiff_sim_trans(n_states):
    return ParameterDiff_Similarity(Transition_ModelClass(n_states, 10), Transition_ModelClass(n_states, 10), get_pmatrix)


def get_beta(tpl):
    return tpl.beta
def getParameterDiff_reg(gamma):
    return SizeWrapper(EMM_Exceptionality_ParameterDiff(PolyRegression_ModelClass(), get_beta, exponent=1), gamma)
def getParameterDiff_sim_reg():
    return ParameterDiff_Similarity(PolyRegression_ModelClass(), PolyRegression_ModelClass(), get_beta)


task_tpl=namedtuple('task_tpl',['prefix', 'df_index', 'ex_qf', 'sim_qf', 'final_qf', 'alpha', 'beta', 'gamma', 'parameters', 'task_id', 'model_name', 'ignore_columns'])
def load_run_save_beamsearch_task(params ):
    (ex_qf, ex_name)=params.ex_qf
    (sim_qf, sim_name)=params.sim_qf
    gamma=params.gamma
    framework = EvaluationFramework(params.prefix)
    framework.override = True
    path_prefix=Path(params.model_name)/Path(params.parameters.task_name+'_results')
    #if framework.path(path_prefix,  params.task_id).is_file():
    #    return 
    dataset_t = framework.load_dataset(params.model_name, params.df_index)
    start = timeit.default_timer()
    result, mine_time = framework.run_single_task_beamsearch( dataset_t.df1, dataset_t.df2, ex_qf(gamma), ex_qf(gamma), sim_qf(), functools.partial(params.final_qf,params.alpha,params.beta), params.parameters,params.ignore_columns)
    stop = timeit.default_timer()
    df = to_dataframe(result) 
    new_dataset_tuple=dataset_tpl(None,None,*dataset_t[2:])
    complete_result = mine_pair_result_parameters(params.alpha, params.beta, params.gamma, df, ex_name, sim_name, new_dataset_tuple, params.parameters, stop-start, mine_time)
    framework.save_dataset(path_prefix, params.task_id, complete_result)


def load_run_save_oracle_task(tpl_in ):
    prefix, df_counter, (ex_qf, ex_name), (sim_qf, sim_name), qf, alpha, beta, gamma, parameters, n, task_type, ignore = tpl_in
    framework = EvaluationFramework(prefix)
    framework.override = True
    dataset_t = framework.load_dataset(task_type, df_counter)
    result = framework.run_single_task_oracle( dataset_t.df1, dataset_t.df2, ex_qf(gamma), ex_qf(gamma), sim_qf(), functools.partial(qf,alpha,beta), parameters, tpl_in.ignore_columns)
    df = to_dataframe(result) #.drop(['b','b2'],axis='columns')
    new_dataset_tuple=dataset_tpl(None,None,*dataset_t[2:])
    complete_result = result_parameters(alpha, beta, gamma, df, ex_name, sim_name, new_dataset_tuple, parameters)
    framework.save_dataset(Path(task_type)/Path('oracle_results'), n, complete_result)

import gzip
class EvaluationFramework:
    def __init__(self, prefix_folder):
        self.prefix = Path(prefix_folder)
        self.override = False

    def run_linear_regression(self):
        pass


    def path(self, prefix2, counter, mode='create'):
        curr_folder = self.prefix / Path(prefix2)
        curr_file = curr_folder / Path('data_{}.pkl'.format(counter))
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)
            if not os.path.exists(curr_folder):
                raise RuntimeError('Something went wrong while creating folder: '+curr_folder )
        if not curr_file.exists():
            curr_file=Path(str(curr_file)+'.gz')
        if mode=='create' and (not self.override) and curr_file.exists():
            raise RuntimeError('The file you whish to create already exists: '+ str(curr_file))
        return curr_file

    def create_linear_regression_datasets(self, n_classes, n_noise, n_dataframes, hide_depth):
        tpls = []
        for df_counter in range(n_dataframes):
            background_sizes = np.random.randint(1000, 10000+1, 2)
            tpl = dataset_tpl(*generate_two_regression_dataframes(background_sizes, n_classes, n_noise))
            df1 = hide(tpl.df1, hide_depth, int(background_sizes[0]/4)*hide_depth)
            df2 = hide(tpl.df2, hide_depth, int(background_sizes[1]/4)*hide_depth)
            self.save_dataset('regression',df_counter,  dataset_tpl(df1,df2,tpl.parameters,tpl.sizes1,tpl.sizes2))
            tpls.append(tpl)
        return tpls

    def create_transition_datasets(self, n_classes, n_noise, n_dataframes, hide_depth, n_states):
        tpls = []
        for df_counter in range(n_dataframes):
            background_sizes = np.random.randint(10000, 100000+1, 2)
            tpl = dataset_tpl(*generate_two_transition_dataframes(background_sizes, n_classes, n_noise, n_states))
            df1 = hide(tpl.df1, hide_depth, int(background_sizes[0]/4)*hide_depth)
            df2 = hide(tpl.df2, hide_depth, int(background_sizes[1]/4)*hide_depth)
            self.save_dataset('transition', df_counter,  dataset_tpl(df1,df2,tpl.parameters,tpl.sizes1,tpl.sizes2))
            tpls.append(tpl)
        return tpls


    def save_dataset(self, prefix2, counter, to_save):
        path = self.path(prefix2, counter)
        if str(path).endswith('.gz'):
            with gzip.GzipFile(path, "wb") as f:
                pickle.dump(to_save, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(to_save, f)


    def load_dataset(self, prefix2, counter):
        path = self.path(prefix2, counter,mode='load')
        if str(path).endswith('.gz'):
            with gzip.GzipFile(path, "rb") as f:
                return pickle.load( f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)


    def execute_tasks(self, function_to_run, tasks, processes):
        if processes ==1:
            for task in tqdm(tasks):
                function_to_run(task)
        else:
            with  Pool(processes=processes) as pool:
                print('running in parallel')
                for _ in tqdm(pool.imap(function_to_run, tasks),total=len(tasks),smoothing=0):
                    pass
        print('finished')

    def create_tasks(self, ex_qfs,sim_qfs,n_dataframes, model_name, ignore_columns, parameters):
        n = 0
        tasks = []
        for df_counter in range(n_dataframes):
            for alpha, beta, gamma in product([-1, 0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]):
                for ex_qf in ex_qfs:
                    for sim_qf in sim_qfs:
                        tasks.append(task_tpl(self.prefix, df_counter, ex_qf, sim_qf, final_qf, alpha, beta, gamma, parameters, n,model_name,ignore_columns))
                        n = n+1
        return tasks

    def execute_regression_tests(self, parameters, n_dataframes=10, n_classes=10, processes=1,continue_at=0, ):  
        ex_qfs = [(getLikelihoodExceptionality,'Like'), (get_LOG_LikelihoodExceptionality,'Log'), (getCooksExeptionality,'Cooks'),(getParameterDiff_reg,'par')]
        sim_qfs= [(getLikelihoodSim,'Like_sim'), (get_LOG_LikelihoodSim,'Log_sim'), (getDoubleCooks,'Cooks_sim'),(getParameterDiff_sim_reg,'par_sim')]

        tasks=self.create_tasks(ex_qfs, sim_qfs, n_dataframes, model_name='regression', ignore_columns=['x','y','class'], parameters=parameters)
        #tasks=[x for x in tasks if not (x[3][1]=='Cooks_sim')]
        tasks=[x for x in tasks if x.task_id>=continue_at]
        #tasks=[x for x in tasks if x.ex_qf[1]=='Like']
        self.execute_tasks(load_run_save_beamsearch_task, tasks, processes)


    def execute_transition_tests(self, parameters, n_states=10, n_dataframes=10, n_classes=10,  processes=1, continue_at=0):
        
        ex_qf1=functools.partial(getLikelihoodExceptionality_transition, n_states)
        ex_qf1.__doc__='LikelihoodExceptionality_transition'
        ex_qf2=functools.partial(getTotalVariationExceptionality, n_states)
        ex_qf2.__doc__='TotalVariationExceptionality'
        ex_qf3=functools.partial(getParameterDiff_trans,n_states)
        ex_qf3.__doc__='ParameterDiff'
        ex_qfs = [(ex_qf1,ex_qf1.__doc__), (ex_qf3,ex_qf3.__doc__)]
                
        sim_qf1=functools.partial(getLikelihoodSimilarity_transition, n_states)
        sim_qf1.__doc__='LikelihoodSimilarity'
        sim_qf2=functools.partial(getTotalVariationSimilarity, n_states)
        sim_qf2.__doc__='TotalVariationSimilarity'
        sim_qf3=functools.partial(getParameterDiff_sim_trans, n_states)
        sim_qf3.__doc__='ParameterDiff_sim'
        sim_qfs= [(sim_qf1, sim_qf1.__doc__), (sim_qf2, sim_qf2.__doc__), (sim_qf3, sim_qf3.__doc__)]

        tasks=self.create_tasks(ex_qfs, sim_qfs, n_dataframes, model_name='transition', ignore_columns=['in','out','class'], parameters=parameters)
        tasks=[x for x in tasks if x.task_id>=continue_at]
        self.execute_tasks(load_run_save_beamsearch_task, tasks, processes)
    

    def run_single_task_beamsearch(self, df1, df2, Qf_L, Qf_R, similarity_function, total_fun, parameters,exclusions):
        sels_L=ps.create_nominal_selectors(df1,exclusions)
        sels_R=ps.create_nominal_selectors(df2,exclusions)

        task_L = ps.SubgroupDiscoveryTask(df1, None, sels_L, Qf_L, result_set_size = parameters.result_size, depth=parameters.depth, min_quality=np.NINF)
        task_L.algorithm = ps.SimpleSearch(show_progress=False)#ps.DFS(ps.BitSetRepresentation)#parameters.result_size)
        task_R = ps.SubgroupDiscoveryTask(df2, None, sels_R, Qf_R, result_set_size = parameters.result_size, depth=parameters.depth, min_quality=np.NINF)
        task_R.algorithm = ps.SimpleSearch(show_progress=False)#ps.DFS(ps.BitSetRepresentation)#(parameters.result_size)
        run=beam_search_through_candidates(task_L, task_R, parameters.total_result_size, 
                                            parameters.constraints, similarity_function, total_fun, None, None,show_progress=False)
        start = timeit.default_timer()
        tpl_L=next(run)
        stop = timeit.default_timer()
        time1 = stop-start

        start = timeit.default_timer()
        tpl_R=next(run)
        stop = timeit.default_timer()
        time2 = stop-start
        result = next(run)
        return result, time1+time2


    def run_single_task_oracle(self, df1, df2, Qf_L, Qf_R, similarity_function, total_fun, parameters,exclusions):

        sels_L=ps.create_nominal_selectors(df1,exclusions)
        sels_R=ps.create_nominal_selectors(df2,exclusions)
        task_L = ps.SubgroupDiscoveryTask(df1, None, sels_L, Qf_L, result_set_size = parameters.result_size, depth=parameters.depth)
        task_L.algorithm = ps.SimpleSearch(show_progress=False)#ps.DFS(ps.BitSetRepresentation)#parameters.result_size)
        task_R = ps.SubgroupDiscoveryTask(df2, None, sels_R, Qf_R, result_set_size = parameters.result_size, depth=parameters.depth)
        task_R.algorithm = ps.SimpleSearch(show_progress=False)#ps.DFS(ps.BitSetRepresentation)#(parameters.result_size)

        models = (PolyRegression_ModelClass(), PolyRegression_ModelClass())
        num_models=50
        all_model_parameters=[beta_tuple(np.random.rand(2)) for _ in range(num_models)]
        return find_model_through_heuristic(task_L, task_R, parameters.total_result_size, similarity_function, total_fun, all_model_parameters, models)

    def get_ranks(self, df, n_classes, hide_depth=2):
        equal_inds = df["sgd1"].astype(str) == df["sgd2"].astype(str)
        df_equal = df[equal_inds]
        s_repr = df_equal['sgd1'].astype(str)
        inds_expected = np.zeros(len(df_equal), dtype=bool)
        for class_index in range(1, n_classes+1):
            selectors = [ps.NominalSelector(f'class_{class_index}_{depth_index}', True) for depth_index in range(hide_depth)]
            conj = ps.Conjunction(selectors)
            inds_expected |= (s_repr == str(conj))
        ranks = df_equal.index[inds_expected]+1
        if len(ranks) < n_classes:
            ranks = np.hstack([ranks, np.full(n_classes-len(ranks),len(df)+1,dtype=int)]) 
        return np.array(ranks, dtype=int)

    def get_recovery_in_top(self, rank_list, k):
        return np.count_nonzero(rank_list<=k)/len(rank_list)
    
    @staticmethod
    def get_mean_inverse_rank(ranks):
        return 1/np.mean(1/ranks)

    @staticmethod
    def dict_to_df(d):
        keys=[]
        values=[]
        for key, value in d.items():
            keys.append(key)
            values.append(value)
        return pd.DataFrame.from_records(values,index=pd.MultiIndex.from_tuples(keys,names=['alpha','beta','gamma','ex','sim']))

    def evaluate_results(self, prefix, experiment_name, n_results, show_progress=None, allow_omit=False):
        function_dict={'raw':lambda x: x,
                        'mir':self.get_mean_inverse_rank,
                       'top10':functools.partial(self.get_recovery_in_top,k=10),
                       'top25':functools.partial(self.get_recovery_in_top,k=25)}
        dict_params_dict={key:defaultdict(list) for key in function_dict}
        the_range=range(n_results)
        if not show_progress is None:
            the_range=show_progress(the_range)
        for n in the_range:
            try:
                tpl = self.load_dataset(prefix/Path(experiment_name), n)
                fixed_params = (tpl.alpha, tpl.beta,tpl.gamma, tpl.ex_qf_name, tpl.sim_qf_name)
                df=tpl.df_result.sort_values('qual',ascending=False).reset_index(drop=True)
                ranks = self.get_ranks(df, n_classes=10)
                for key, params_dict in dict_params_dict.items():
                    params_dict[fixed_params].append(function_dict[key](ranks))
            except FileNotFoundError as err:
                if not allow_omit:
                    raise err

        result_frames={key:self.dict_to_df(values) for key,values in dict_params_dict.items()}
        return result_frames

        


        
    