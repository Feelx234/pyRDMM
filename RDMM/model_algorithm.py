import numpy as np
import pandas as pd
import pysubgroup as ps
from .model_target import PolyRegression_ModelClass
from heapq import heappop, heappush
from itertools import product, chain
from tqdm import tqdm
import functools
from collections import namedtuple

def getExceptionalSGs(task, Constraints):
    tpls=task.algorithm.execute(task).to_descriptions()
    #beam_search(df,Qf,AtomicGrowthRefinement(sels, ExtensionTypes.AND),3,Constraints,30)
    #Exs=[ex for ex,_ in tpls]
    #Exs=minMaxScale(Exs)
    #tpls_constrained=[(exceptionality,sG) for exceptionality,(_,sG) in zip(Exs,tpls) if all(c(sG,task.data) for c in Constraints)]
    #sGs=[sG for _,sG in tpls_constrained]
    #print(tpls_constrained)
    return tpls #_constrained


def addIfRequired(result, quality, sg, size, check_for_duplicates=True):
    if ( (len(result) > 0) and check_for_duplicates and any(entry[1][0]==sg[0] and entry[1][1]==sg[1] for entry in result)):
        #print("{} is already in the list".format(str(sg)))
        return
    if (len(result) < size):
        heappush(result, (quality, sg))
    else:
        heappop(result)
        heappush(result, (quality, sg))



compare_tuple=namedtuple('compare_tuple',['exceptionality', 'subgroup', 'cover_array', 'rel_size', 'parameters'])

def get_params_for_tpls(get_params_func, data, requires_cover_arr, tpls):
    def get_params_for_tpl(get_params_func, data, requires_cover_arr, tpl):
        quality, subgroup = tpl
        cover_arr=subgroup.covers(data)
        rel_size=np.count_nonzero(cover_arr)/len(cover_arr)
        params = get_params_func(cover_arr, data)
        
        if not requires_cover_arr:
            cover_arr = None
        else:
            cover_arr=np.nonzero(cover_arr)[0]
        return compare_tuple(quality, subgroup, cover_arr, rel_size,params)
    f = functools.partial(get_params_for_tpl, get_params_func, data, requires_cover_arr)
    return list(map(f, tpls))


def get_candidates_and_params(sGs,task,side,sim_QF):
    calc_stats = functools.partial(sim_QF.calculate_statistics,side = side)
    compare_tuples = get_params_for_tpls(calc_stats, task.data, sim_QF.requires_cover_arr, sGs)
    return compare_tuples

def beam_search_through_candidates(task_L, task_R,
                                    resultLen, Constr, sim_QF, combineFun,
                                    candidates_L=None,candidates_R=None, show_progress=False):

    sim_QF.calculate_constant_statistics(task_L, task_R)
    if candidates_L is None:
        tmp_L=getExceptionalSGs(task_L, Constr)
        candidates_L = get_candidates_and_params(tmp_L,task_L, 0,sim_QF)
    yield candidates_L

    if candidates_R is None:
        tmp_R=getExceptionalSGs(task_R, Constr)
        candidates_R = get_candidates_and_params(tmp_R,task_R, 1,sim_QF)
    yield candidates_R

    result=[]
    compare_candidates(result, candidates_L, candidates_R, combineFun, sim_QF, show_progress, resultLen)
    yield result


def compare_candidates(result, candidates_L, candidates_R, combineFun, sim_QF, show_progress, resultLen):
    prod = product(candidates_L, candidates_R)
    if show_progress:
        prod = tqdm(prod, total=len(candidates_L)*len(candidates_R), smoothing=0)
    for (L, R) in prod:
        similarity = sim_QF.evaluate(L.cover_array, R.cover_array, L.parameters, R.parameters)
        d = combineFun(min(L.rel_size, R.rel_size), min(L.exceptionality, R.exceptionality), similarity)

        #addIfRequired(result, d, (L.subgroup, R.subgroup), resultLen)
        #print((L.subgroup, R.subgroup, L.parameters, R.parameters, similarity, L.exceptionality, R.exceptionality))
        addIfRequired(result, d, (L.subgroup, R.subgroup, L.parameters, R.parameters, similarity, L.exceptionality, R.exceptionality), resultLen, False)
    return result



def run_through_single_side(model, parameters, task, gamma):
    original_qf = task.qf
    task.result_set_size=10
    model.calculate_constant_statistics(task)
    task.data['likelihood'] = model.likelihood(parameters, np.ones(len(task.data),dtype=bool))
    task.target = ps.NumericTarget('likelihood')
    task.qf = ps.StandardQFNumeric(gamma, False, 'sum')
    candidates = ps.DFS(ps.BitSetRepresentation).execute(task)
    task.qf = original_qf
    return candidates

def apply_quality_for_candidates(candidates, task):
    task.qf.calculate_constant_statistics(task)
    return [(task.qf.evaluate(subgroup, task.qf.calculate_statistics(subgroup)), subgroup) for _, subgroup in candidates]
   
        

def sg_to_description(result):
    return [(x, y) for x,y in result]


def find_model_through_heuristic(task_L, task_R,  resultLen,sim_QF, combineFun, all_model_parameters, models):
    gamma = 1
    show_progress = False
    result = []
    sim_QF.calculate_constant_statistics(task_L, task_R)
    for model_parameters in all_model_parameters:
        candidates_L = apply_quality_for_candidates( run_through_single_side(models[0], model_parameters, task_L, gamma), task_L)
        candidates_L = get_candidates_and_params(candidates_L, task_L, 0, sim_QF)
        candidates_R = apply_quality_for_candidates( run_through_single_side(models[1], model_parameters, task_R, gamma), task_R)
        candidates_R = get_candidates_and_params(candidates_R, task_R, 1, sim_QF)
        compare_candidates(result, candidates_L, candidates_R, combineFun, sim_QF, show_progress, resultLen)
        #if len(result)>0:
        #    print(list(zip(*result))[1])
    return result


def mine_and_find(task_L, task_R,  resultLen,sim_QF, combineFun, all_model_parameters, models):
    gamma = 1
    show_progress = False
    result = []
    sim_QF.calculate_constant_statistics(task_L, task_R)
    for model_parameters in all_model_parameters:
        candidates_L = get_candidates_and_params(getCandidates(task_L, Constr),task_L, 0,sim_QF)  # pylint: disable=undefined-variable
        candidates_L = get_candidates_and_params(candidates_L, task_L, 0, sim_QF)
        candidates_R = apply_quality_for_candidates( run_through_single_side(models[1], model_parameters, task_R, gamma), task_R)
        candidates_R = get_candidates_and_params(candidates_R, task_R, 1, sim_QF)
        compare_candidates(result, candidates_L, candidates_R, combineFun, sim_QF, show_progress, resultLen)
        #if len(result)>0:
        #    print(list(zip(*result))[1])
    return result



def to_dataframe(result_in):
    result=result_in.copy()
    final_result=[]
    while len(result) > 0:
        quality,(sGL,sGR,params_L,params_R,similarity, ex_L,ex_R) = heappop(result)
        #beta_L,XTX_L,s_2_L,nL,_,_ = params_L
        #beta_R,XTX_R,s_2_R,nR,_,_ = params_R
        params_L=params_L
        params_R=params_R
        final_result.append((quality,similarity,ex_L,ex_R,str(sGL),str(sGR),params_L.size_sg,params_R.size_sg)) 
    df_final=pd.DataFrame.from_records(final_result)
    df_final.columns = ["qual",'sim','eL','eR','sgd1','sgd2','sizeL','sizeR']
    for col in ['eL','eR','sgd1','sgd2','sizeL','sizeR']:
        df_final[col]=df_final[col].astype('category')

    df_final = df_final[::-1].reset_index(drop=True)
    return df_final