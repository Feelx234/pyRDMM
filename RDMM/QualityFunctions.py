import numpy as np
from abc import ABC, abstractmethod
import pysubgroup as ps
from collections import  namedtuple

from .model_target import PolyRegression_ModelClass, Transition_ModelClass
from .generic_quality_measures import parse_task_input

def getX(X_in,deg):
    if len(X_in.shape)==1:
        X_in=X_in[:,None]
    l=[np.power(X_in,i) for i in range(deg,0,-1)]
    l.append(np.ones(X_in.shape))
    return np.hstack(l)



def CooksDistance(beta,beta_All,XTX,s_2):
    assert(len(beta_All)==len(beta))   
    m=len(beta_All)
    return (beta-beta_All).T @ XTX @ (beta-beta_All)/(m*s_2)
    


def getXTXforFit(X_in,Y_in,beta):
    N=len(X_in)
    m=len(beta)
    deg = len(beta)-1
    f=np.poly1d(beta)
    e=Y_in-f(X_in)
    X=getX(X_in,deg)
    
    s_2=np.sum(np.square(e))/(N-m)
    return X.T @ X, s_2


class EMM_quality(ABC):
    def __init__(self,data):
        pass
    
    @abstractmethod
    def get_info(self,sG,data,is_new):
        pass
        
    
    def __call__(self,sG,data):
        #inds=sG.covers(data)
        args1=self.get_info(sG,data,True)
        args2=self.get_info(sG,data,False)
        #print(args1)
        #print(args2)
        return self.calc(*args1,*args2)

    @abstractmethod
    def calc(self,*args):
        pass



class EMM_quality_entire(EMM_quality):
    def __init__(self,data):
        self.info=self.calc_info(None,data,init=True)
    
    def get_info(self,sG,data,is_new):
        if is_new:
            return self.calc_info(sG,data)
        else:
            return self.info

    @abstractmethod
    def calc_info(self,sG,data,init=False):
        pass


class EMM_quality_Cook_poly(ps.AbstractInterestingnessMeasure):
    def __init__(self,x_name='x',y_name='y',degree=1):
        self.model=PolyRegression_ModelClass(x_name=x_name,y_name=y_name,degree=degree)
        self.has_constant_statistics = False
        self.required_stat_attrs = ('beta',)

    def calculate_constant_statistics(self, task, target=None):
        self.model.calculate_constant_statistics(task)
        self.beta_all = self.model.fit(np.ones(len(task.data), dtype=bool), task.data).beta
        self.XTX, self.s_2 = getXTXforFit(self.model.x, self.model.y, self.beta_all)
        # Future: incorporate Z matrix into XTX
        
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        return self.model.fit(cover_arr)

    def evaluate(self, subgroup, statistics = None):
        statistics = self.ensure_statistics(subgroup, None, None, statistics)
        if statistics.size_sg <=0:
            return np.nan
        return CooksDistance(statistics.beta, self.beta_all, self.XTX, self.s_2)

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True





class EMM_Likelihood(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple('EMM_Likelihood',['model_params','subgroup_likelihood','inverse_likelihood'])
    def __init__(self, model):
        self.model = model
        self.has_constant_statistics = False
        self.required_stat_attrs = EMM_Likelihood.tpl._fields

    def calculate_constant_statistics(self, task):
        self.model.calculate_constant_statistics(task)
        self.data_size = len(task.data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        sg_size = np.count_nonzero(cover_arr)
        params = self.model.fit(cover_arr, data)
        
        #numeric stability?
        all_likelihood = self.model.likelihood(params, None)
        sg_likelihood_sum = np.sum(all_likelihood[cover_arr])
        total_likelihood_sum = np.sum(all_likelihood)
        dataset_average = np.nan
        if (self.data_size - sg_size) > 0:
            dataset_average = (total_likelihood_sum - sg_likelihood_sum)/(self.data_size - sg_size)
        sg_average = np.nan
        if sg_size > 0:
            sg_average = sg_likelihood_sum/sg_size
        return EMM_Likelihood.tpl(params, sg_average, dataset_average)

    def evaluate(self, subgroup, statistics = None):
        statistics = self.ensure_statistics(subgroup, None, None, statistics)
        #numeric stability?
        return (statistics.subgroup_likelihood - statistics.inverse_likelihood)

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True



class EMM_LikelihoodGain(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple('EMM_Likelihood',['model_params','subgroup_likelihood','dataset_likelihood'])
    def __init__(self, model, use_log=False):
        self.model = model
        self.has_constant_statistics = False
        self.required_stat_attrs = EMM_LikelihoodGain.tpl._fields
        self.use_log = use_log
        if self.use_log:
            assert(hasattr(self.model,'loglikelihood'))
        else:
            assert(hasattr(self.model,'likelihood'))

    def calculate_constant_statistics(self, task_or_data, target=None):
        task, target = parse_task_input(task_or_data, target)
        self.model.calculate_constant_statistics(task, target)

        data = task.data
        self.data_size = len(data)

        null_parameters = self.model.fit(np.ones(self.data_size, dtype=bool), data)
        if self.use_log:
            self.dataset_likelihood = self.model.loglikelihood(null_parameters, None)
        else:
            self.dataset_likelihood = self.model.likelihood(null_parameters, None)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, self.data_size, data)
        params = self.model.fit(cover_arr, data)

        #numeric stability?
        if self.use_log:
            sg_likelihood = self.model.loglikelihood(params, cover_arr)
        else:
            sg_likelihood = self.model.likelihood(params, cover_arr)

        sg_average = np.nan
        dataset_average = np.nan
        if size_sg > 0:
            sg_average = np.mean(sg_likelihood)
            dataset_average = np.mean(self.dataset_likelihood[cover_arr])
        return EMM_LikelihoodGain.tpl(params, sg_average, dataset_average)

    def evaluate(self, subgroup, statistics = None):
        statistics = self.ensure_statistics(subgroup, statistics=statistics)
        #numeric stability?
        return max(statistics.subgroup_likelihood - statistics.dataset_likelihood, 0)

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True



class SizeWrapper(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple('SizeWrapper',['size_sg','wrapped_tuple'])
    def __init__(self, qf, alpha):
        self.qf = qf
        self.alpha = alpha
        self.has_constant_statistics = False
        self.required_stat_attrs = SizeWrapper.tpl._fields

    def calculate_constant_statistics(self, task_or_data, target=None):
        task, target = parse_task_input(task_or_data, target)
        data = task.data

        self.qf.calculate_constant_statistics(task, target)
        self.data_size = len(data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target=None, data=None):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, self.data_size, data)
        params = self.qf.calculate_statistics(cover_arr, data)
        assert not params is None
        return SizeWrapper.tpl(size_sg, params)

    def evaluate(self, subgroup=None, target=None, data=None, statistics = None):
        if statistics is None:
            statistics = self.ensure_statistics(subgroup, statistics)
        if statistics.wrapped_tuple is None:
            wrapped_tuple = self.qf.ensure_statistics(subgroup, target, data, statistics.wrapped_tuple)
        else:
            wrapped_tuple = statistics.wrapped_tuple
        #print(self.qf.__class__)
        return (statistics.size_sg / self.data_size) ** self.alpha * self.qf.evaluate(subgroup, statistics=wrapped_tuple)

    def optimistic_estimate(self, subgroup, statistics = None):
        return float('inf')

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True



class EMM_quality_resample(ps.AbstractInterestingnessMeasure):
    tpl = namedtuple("Resample_tpl",('distances_mean','distances_std','other_parameters'))
    def __init__(self, other_distance, resamples):
        self.other_distance = other_distance
        self.resamples=resamples
        self.required_stat_attrs = ('distances_mean','distances_std','other_parameters')
        self.has_constant_statistics = True

    def calculate_constant_statistics(self, task):
        self.other_distance.calculate_constant_statistics(task)

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)

        arr = np.zeros(len(cover_arr), dtype=bool)
        arr[0:int(np.count_nonzero(cover_arr))] = True
        params = self.other_distance.calculate_statistics(subgroup, data)
        subgroup.representation = data
        distances = []
        for _ in range(self.resamples):
            np.random.shuffle(arr)
            distances.append(self.other_distance.evaluate(arr, data))
        return EMM_quality_resample.tpl(np.mean(distances),np.std(distances),params)

    def evaluate(self, subgroup, statistics = None):
        statistics = self.ensure_statistics(subgroup, statistics)
        return (self.other_distance.evaluate(subgroup, statistics.other_parameters) - statistics.distances_mean) / statistics.distances_std

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

from numba import njit

@njit()
def sum_where(arr, where):
    out = 0.0
    for j in where:
        out += arr[j]
    return out

from scipy.stats import norm
class LikelihoodSimilarity(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple('similarity_stats',['params','likelihood','size_sg'])
    def __init__(self, model_L, model_R, use_log=False):
        self.model_L = model_L
        self.model_R = model_R
        self.use_log = use_log
        if self.use_log:
            assert(hasattr(self.model_L,'loglikelihood'))
            assert(hasattr(self.model_R,'loglikelihood'))
        else:
            assert(hasattr(self.model_L,'likelihood'))
            assert(hasattr(self.model_R,'likelihood'))

    def calculate_constant_statistics(self, targetL, targetR):
        self.model_L.calculate_constant_statistics(targetL)
        self.model_R.calculate_constant_statistics(targetR)


    def calculate_statistics(self, subgroup, data=None, side=None):
        if side == 0:
            model_1=self.model_L
            model_2=self.model_R
        elif side==1:
            model_1=self.model_R
            model_2=self.model_L
        else:
            raise ValueError
        
        params=model_1.fit(subgroup, data)
        if self.use_log:
            likelihood=model_2.loglikelihood(params, None)
        else:
            likelihood=model_2.likelihood(params, None)
        return LikelihoodSimilarity.tpl(params,likelihood,np.count_nonzero(subgroup))


    @staticmethod
    @njit()
    def _evaluate(like1, like2, rep1, rep2, size1, size2):
        sum1=sum_where(like1,rep2)
        sum2=sum_where(like2,rep1)
        return min(sum1/size1,sum2/size2)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        return self._evaluate(statistics1.likelihood,statistics2.likelihood,subgroup1,subgroup2,statistics1.size_sg,statistics2.size_sg)
            

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    @property
    def requires_cover_arr(self):
        return True


class Dumb_Sim_Wrapper():
    tpl=namedtuple('size_tpl',['size_sg'])
    def __init__(self, other_quality):
        self.qf=other_quality
        self.data_1=None
        self.data_2=None

    def calculate_constant_statistics(self, targetL, targetR):
        self.qf.calculate_constant_statistics(targetL, targetR)
        self.data_1=targetL.data
        self.data_2=targetR.data

    def calculate_statistics(self, subgroup, data=None, side=None):
        return Dumb_Sim_Wrapper.tpl(np.count_nonzero(subgroup))

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        statistics1 = self.qf.calculate_statistics(subgroup1, self.data_1, side=0)
        statistics2 = self.qf.calculate_statistics(subgroup2, self.data_2, side=1)
        return self.qf.evaluate(subgroup1, subgroup2, statistics1, statistics2)      

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    @property
    def requires_cover_arr(self):
        return True


class ParameterDiff_Similarity(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple('tpl_ParameterDiff',['model_params','size_sg'])

    def __init__(self, model_L, model_R, get_params_func):
        self.model_L = model_L
        self.model_R = model_R
        self.get_params_func = get_params_func

    def calculate_constant_statistics(self, targetL, targetR):
        self.model_L.calculate_constant_statistics(targetL)
        self.model_R.calculate_constant_statistics(targetR)

    def calculate_statistics(self, subgroup, data=None, side=None):

        if side == 0:
            fit_result=self.model_L.fit(subgroup, data)
        elif side==1:
            fit_result=self.model_R.fit(subgroup, data)
        else:
            raise ValueError
        return ParameterDiff_Similarity.tpl(self.get_params_func(fit_result ),fit_result.size_sg)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        return  1/np.sum(np.abs(statistics1.model_params - statistics2.model_params))

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    @property
    def requires_cover_arr(self):
        return False



class DoubleCooksSimilarity(ps.AbstractInterestingnessMeasure):
    tpl=namedtuple("likelihood_stats",['beta','size_sg','basis'])

    def __init__(self, model_L, model_R):
        self.model_L = model_L
        self.model_R = model_R
        self.has_constant_statistics = False
        self.required_stat_attrs = DoubleCooksSimilarity.tpl._fields
        self.qf_L = EMM_quality_Cook_poly(x_name=self.model_L._x_name, y_name=self.model_L._y_name, degree=1)
        self.qf_R = EMM_quality_Cook_poly(x_name=self.model_R._x_name, y_name=self.model_R._y_name, degree=1)
        
    def calculate_constant_statistics(self, taskL, taskR):
        self.model_L.calculate_constant_statistics(taskL)
        self.qf_L.calculate_constant_statistics(taskL)
        self.model_R.calculate_constant_statistics(taskR)
        self.qf_R.calculate_constant_statistics(taskR)

    def calculate_statistics(self, subgroup, data=None, side=None):
        if side == 0:
            model = self.model_L
        elif side == 1:
            model = self.model_R
        else:
            raise ValueError

        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)

        fit_result = model.fit(subgroup, data)
        basis=model.gp_get_stats_multiple(cover_arr)
        return DoubleCooksSimilarity.tpl(fit_result.beta, fit_result.size_sg, basis)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        L = statistics1
        R = statistics2
        if (L.size_sg > self.model_L.degree) and (R.size_sg > self.model_R.degree):
            beta_tpl = PolyRegression_ModelClass.gp_get_params( PolyRegression_ModelClass.gp_merge(L.basis, R.basis,inplace=False))
            Cook1 = CooksDistance(L.beta, beta_tpl.beta, self.qf_L.XTX, self.qf_L.s_2)
            Cook2 = CooksDistance(R.beta, beta_tpl.beta, self.qf_R.XTX, self.qf_R.s_2)
            return 1/max(Cook1, Cook2)
        else:
            return np.nan

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    @property
    def requires_cover_arr(self):
        return False


class TotalVariationSimilarity(ps.AbstractInterestingnessMeasure):

    def __init__(self, model_L, model_R):
        self.model_L = model_L
        self.model_R = model_R
        self.has_constant_statistics = False
        self.required_stat_attrs = model_L.tpl._fields
        
    def calculate_constant_statistics(self, taskL, taskR):
        self.model_L.calculate_constant_statistics(taskL)
        self.model_R.calculate_constant_statistics(taskR)

    def calculate_statistics(self, subgroup, data=None, side=None):
        if side is None:
            raise ValueError
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        if side == 0:
            model = self.model_L
        elif side == 1:
            model = self.model_R
        fit_result = model.fit(cover_arr, data)
        return fit_result

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        L = statistics1
        R = statistics2

        value = 0
        for i in range(len(L.count_vector)):
            value += L.count_vector[i] * R.count_vector[i] * np.sum(np.abs(L.transition_matrix[i,:] - R.transition_matrix[i,:]))

        return 1/value

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    @property
    def requires_cover_arr(self):
        return False



class EMM_TotalVariation(ps.AbstractInterestingnessMeasure):
    def __init__(self, n_states):
        self.model = Transition_ModelClass(n_states, 10)
        self.has_constant_statistics = False
        self.required_stat_attrs = Transition_ModelClass.tpl._fields

    def calculate_constant_statistics(self, task):
        self.model.calculate_constant_statistics(task)
        self.dataset = self.model.fit(np.ones(len(task.data),dtype=int), task.data)
        self.data_size = len(task.data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        fit_result = self.model.fit(cover_arr, data)
        return fit_result


    def evaluate(self, subgroup, statistics = None):
        sg_transition_matrix = statistics.transition_matrix
        value = 0
        for i in range(len(statistics.count_vector)):
            value += statistics.count_vector[i] * np.sum(np.abs(sg_transition_matrix[i,:] - self.dataset.transition_matrix[i,:]))

        return value

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True


class EMM_TotalVariation2(ps.AbstractInterestingnessMeasure):
    def __init__(self, n_states):
        self.model = Transition_ModelClass(n_states, 10)
        self.has_constant_statistics = False
        self.required_stat_attrs = Transition_ModelClass.tpl._fields

    def calculate_constant_statistics(self, task):
        self.model.calculate_constant_statistics(task)
        self.dataset = self.model.fit(np.ones(len(task.data),dtype=int), task.data)
        self.data_size = len(task.data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        fit_result = self.model.fit(cover_arr, data)
        return fit_result


    def evaluate(self, subgroup, statistics = None):
        sg_transition_matrix = statistics.transition_matrix
        value = 0
        for i in range(len(statistics.count_vector)):
            value += statistics.count_vector[i] * self.dataset.count_vector[i] * np.sum(np.abs(sg_transition_matrix[i,:] - self.dataset.transition_matrix[i,:]))

        return value

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True
