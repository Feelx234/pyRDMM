import numpy as np
from numba import njit
import pandas as pd
import pysubgroup as ps
from scipy.stats import norm
from collections import namedtuple
beta_tuple = namedtuple('beta_tuple',['beta','size_sg'])
from numba import njit

from .generic_quality_measures import parse_task_input
@njit(fastmath=True)
def norm_pdf(x,y,beta,out):
    norm_frac=1/np.sqrt(np.pi * 2)
    for i in range(len(x)):
        out[i]=np.exp(-np.square(x[i]*beta[0]+beta[1]-y[i])/2)*norm_frac

@njit(fastmath=True)
def norm_log_pdf(x,y,beta,out):
    norm_frac_log=np.log(1/np.sqrt(np.pi * 2))
    for i in range(len(x)):
        out[i]=-np.square(x[i]*beta[0]+beta[1]-y[i])/2 + norm_frac_log

class PolyRegression_ModelClass:
    def __init__(self, x_name = 'x', y_name = 'y', degree = 1):
        self._x_name = x_name
        self._y_name = y_name
        if degree > 1:
            raise ValueError('Currently only degree == 1 is supported')
        self.degree = degree
        self.x = None
        self.y = None
        self.use_adaptive_sigma=False
        self.has_constant_statistics = True
        super().__init__()

    def calculate_constant_statistics(self, task_or_data, target=None):
        task, target = parse_task_input(task_or_data, target)
        data = task.data
        assert isinstance(data, pd.DataFrame), str(data)
        self.x = data[self._x_name].to_numpy()
        self.y = data[self._y_name].to_numpy()
        self.has_constant_statistics = True

    @staticmethod
    def gp_merge(u, v, inplace=True):
        if inplace:
            x = u
        else:
            x = u.copy()
        del u
        v0 = v[0]
        x0 = x[0]
        if v0 == 0 or x0 == 0:
            d = 0
        else:
            d = v0 * x0/(v0 + x0)*(v[1]/v0 - x[1]/x0)*(v[2]/v0 - x[2]/x0)
        x += v
        x[3] += d
        if not inplace:
            return x

    def gp_get_null_vector(self):
        return np.zeros(5)

    def gp_get_stats(self, row_index):
        x = self.x[row_index]
        return np.array([1, x, self.y[row_index], 0, x*x])

    def gp_get_stats_multiple(self, indices):
        x_arr=self.x[indices]
        y_arr=self.y[indices]
        n=np.count_nonzero(indices)
        x=np.sum(x_arr)
        y=np.sum(y_arr)
        x_square=np.sum(np.square(x_arr))
        cov=np.cov([x_arr,y_arr],ddof=0)
        return np.array([n, x, y, cov[1,0]*n, x_square])
    
    @staticmethod
    def gp_get_params(v):
        size = v[0]
        if size <= 3:
            return beta_tuple(np.full(2, np.nan), size)
        v1 = v[1]
        slope = v[0] * v[3] / (v[0]*v[4] - v1 * v1)
        intersept = v[2]/v[0] - slope * v[1]/v[0]
        return beta_tuple(np.array([slope, intersept]), v[0])

    def fit(self, subgroup, data=None):
        cover_arr, size = ps.get_cover_array_and_size(subgroup, len(self.y), data)
        if size <= self.degree:
            return beta_tuple(np.full(self.degree + 1, np.nan), size)
        return beta_tuple(np.polyfit(self.x[cover_arr], self.y[cover_arr], deg=self.degree), size)

    def likelihood(self, stats, sg):
        if sg is None and not self.use_adaptive_sigma:
            return self.simple_likelihood(stats.beta)
        else:
            return self.do_likelihood(stats, sg, False)

    def loglikelihood(self, stats, sg):
        if sg is None and not self.use_adaptive_sigma:
            return self.simple_log_likelihood(stats.beta)
        else:
            return self.do_likelihood(stats, sg, True)
    
    
    def simple_likelihood(self, beta):
        out=np.empty(len(self.y))
        norm_pdf(self.x, self.y, beta, out)
        return out


    def simple_log_likelihood(self, beta):
        out=np.empty(len(self.y))
        norm_log_pdf(self.x, self.y, beta, out)
        return out

    def do_likelihood(self, stats, sg, use_log):
        if any(np.isnan(stats.beta)):
            return np.full(self.x[sg].shape, np.nan)
       
        scale = 1.0
        if self.use_adaptive_sigma:
            y = np.polyval(stats.beta, self.x[sg])
            scale = np.mean(np.square(y - self.y[sg]))
            if use_log:
                return norm.logpdf(y - self.y[sg], scale=scale)
            else:
                return norm.pdf(y - self.y[sg], scale=scale)
        else:
            if use_log:
                y = np.polyval(stats.beta, self.x[sg])
                return norm.logpdf(y - self.y[sg], scale=scale)
            else:
                y_sg = self.y[sg]
                out=np.empty(len(y_sg))
                norm_pdf(self.x[sg], y_sg, stats.beta,out)
                return out






class Transition_ModelClass:
    tpl = namedtuple('transition_tuple',['transition_matrix', 'count_vector',  'size', 'p_matrix'])
    def __init__(self,n_states, min_count, in_name='in', out_name='out'):
        self.min_count = min_count
        self.n_states = n_states
        self.size = (n_states, n_states)
        self.in_name=in_name
        self.out_name=out_name
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, task, target=None):
        if target is None:
            data = task.data
        else:
            data = task
        self.in_arr = data[self.in_name].to_numpy()
        self.out_arr = data[self.out_name].to_numpy()
        self.has_constant_statistics = True

    def gp_merge(self, u, v):
        u+=v

    def gp_null_vector(self):
        return np.zeros(self.size,dtype=int)

    def gp_get_stats(self, row_index):
        arr = np.array(self.size, dtype=int)
        arr[self.in_arr[row_index], self.out_arr[row_index]] = 1
        return arr
    @staticmethod
    def gp_get_params(count_matrix):
        counts, t_matrix = Transition_ModelClass.normalize(count_matrix)
        size = np.sum(counts)
        return Transition_ModelClass.tpl(t_matrix, counts, size, count_matrix/size)


    def fit(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__"):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        sg_size = np.count_nonzero(cover_arr)
        if sg_size < self.min_count:
            return Transition_ModelClass.tpl(np.full(self.size, np.nan),np.full(self.size[0], np.nan), sg_size,np.full(self.size, np.nan))
        
        in_sg = self.in_arr[cover_arr]
        out_sg = self.out_arr[cover_arr]
        count_matrix = np.zeros(self.size, dtype=int)
        self.count_fast(count_matrix, in_sg, out_sg)
        
        counts, t_matrix = self.normalize(count_matrix)

        return Transition_ModelClass.tpl(t_matrix, counts, sg_size, count_matrix/sg_size)
    @staticmethod
    @njit
    def count_fast(count_matrix, in_sg, out_sg):
        for i, o in zip(in_sg, out_sg):
            count_matrix[i, o] += 1
    @staticmethod
    def normalize(count_matrix):
        t_matrix = np.array(count_matrix,dtype=float)
        counts = np.sum(count_matrix, axis=1)
        for i in range(count_matrix.shape[0]): # normalize
            if counts[i] > 0:
                t_matrix[i,:] /=counts[i]
            else:
                pass
        return counts, t_matrix


    def likelihood(self, stats, sg):
        if sg is None:
            in_sg = self.in_arr
            out_sg = self.out_arr
        else:
            in_sg = self.in_arr[sg]
            out_sg = self.out_arr[sg]
        out = np.empty(len(in_sg))
        self.likelihood_compiled(out, in_sg, out_sg,stats.p_matrix)
        return out
    @staticmethod
    @njit
    def likelihood_compiled(out, in_sg, out_sg,pmatrix):
        for idx,(in_var,out_var) in enumerate(zip(in_sg, out_sg)):
                out[idx] = pmatrix[in_var, out_var]

 