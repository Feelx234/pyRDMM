from collections import namedtuple
from functools import wraps
import numpy as np
import pandas as pd
import pysubgroup as ps

class Ex_Distance:
    def __init__(self, model, distance, invert=False, epsilon=10**-10):
        self.model = model
        self.distance=distance
        self.invert=invert
        self.epsilon = epsilon
        self.has_constant_statistics = False

        if hasattr(distance, 'fit') :
            self.fit_func = distance.fit
        else:
            self.fit_func = self.model.fit

    def calculate_constant_statistics(self, task_or_data, target=None):
        if hasattr(target, "data"):
            task = task_or_data
        else:
            wrapper = namedtuple("task_wrapper", ["data", "target"])
            task = wrapper(task_or_data, target)
            print("wrapping")

        self.model.calculate_constant_statistics(task)
        self.dataset_fit = self.fit_func(slice(None), task.data)
        self.distance.calculate_dataset_statistics(task, self.dataset_fit, self.model)
        self.has_constant_statistics = True
        self.data_len = len(task.data)
        

    def calculate_statistics(self, subgroup, target=None, data=None):
        cover_arr, _ = ps.get_cover_array_and_size(subgroup, self.data_len, data)
        return self.fit_func(cover_arr)

    def evaluate(self, subgroup, target=None, data=None, statistics=None):
        if isinstance(statistics, pd.DataFrame):
            statistics = self.calculate_statistics(subgroup, statistics)
        if statistics.size_sg == 0:
            return np.nan
        if self.invert:
            return 1/(self.epsilon + self.distance.compare(subgroup, None, statistics, self.dataset_fit))
        else:
            return self.distance.compare(subgroup, None, statistics, self.dataset_fit)



class Sim_Direct_Distance:
    def __init__(self, model_L, model_R, distance, epsilon = 10**-10):
        self.model_L = model_L
        self.model_R = model_R
        self.distance = distance
        self.epsilon = epsilon

        if distance.is_symmetric == False:
            raise ValueError("For asymmetric distance functions Sim_Direct_Distance is not applicable")
            
        if hasattr(distance, 'fit') :
            self.fit_func = distance.fit
        else:
            self.fit_func = model_L.__class__.fit


    def calculate_constant_statistics(self, taskL, taskR):
        self.model_L.calculate_constant_statistics(taskL)
        self.model_R.calculate_constant_statistics(taskR)
        #self.distance.calculate_constant_statistics(taskL, taskR)

    def calculate_statistics(self, subgroup, data=None, side=None):
        if side is None:
            raise ValueError
        if hasattr(subgroup, "__array_interface__") or subgroup is None:
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        if side == 0:
            model = self.model_L
        elif side == 1:
            model = self.model_R
        return self.fit_func(model, cover_arr, data)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        return 1/(self.epsilon + self.distance.compare(subgroup1, subgroup2, statistics1, statistics2))

    def supports_weights(self):
        return False
    
    @property
    def requires_cover_arr(self):
        return self.distance.requires_cover_arr

    def is_applicable(self, _):
        return True



class Sim_Asym_Direct_Distance:
    def __init__(self, model_L, model_R, distance,agg=min, epsilon = 10**-10):
        self.model_L = model_L
        self.model_R = model_R
        self.distance = distance
        self.epsilon = epsilon
        self.agg = agg
        if distance.is_symmetric == True:
            raise RuntimeWarning("For symmetric distance functions Sim_Direct_Distance is more suited")

        if hasattr(distance, 'fit') :
            self.fit_func=distance.fit
        else:
            self.fit_func = self.model_L.__class__.fit

    def calculate_constant_statistics(self, taskL, taskR):
        self.model_L.calculate_constant_statistics(taskL)
        self.model_R.calculate_constant_statistics(taskR)
        #self.distance.calculate_constant_statistics(taskL, taskR)

    def calculate_statistics(self, subgroup, data=None, side=None):
        if side is None:
            raise ValueError
        if hasattr(subgroup, "__array_interface__") or subgroup is None:
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        if side == 0:
            model = self.model_L
        elif side == 1:
            model = self.model_R
        return self.fit_func(model, cover_arr, data)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        d1=self.distance.compare(subgroup1, subgroup2, statistics1, statistics2)
        d2=self.distance.compare(subgroup2, subgroup1, statistics2, statistics1)
        return 1/(self.epsilon + self.agg(d1, d2))

    def supports_weights(self):
        return False
    
    @property
    def requires_cover_arr(self):
        return self.distance.requires_cover_arr

    def is_applicable(self, _):
        return True



class Sim_Common_Distance:
    def __init__(self, model_L, model_R, distance_L, distance_R, epsilon = 10**-10):
        self.model_L = model_L
        self.model_R = model_R
        self.distance_L = distance_L
        self.distance_R = distance_R
        self.epsilon = epsilon
        
        assert hasattr(self.distance_L.__class__, 'merge_statistics')


        if hasattr(distance_L, 'fit') and hasattr(distance_R, 'fit'):
            self.fit_L=distance_L.fit
            self.fit_R=distance_R.fit
        else:
            self.fit_L = self.model_L.fit
            self.fit_R = self.model_R.fit


    def calculate_constant_statistics(self, task):
        self.model_L.calculate_constant_statistics(task)
        self.model_R.calculate_constant_statistics(task)
        self.distance_L.calculate_constant_statistics(task, None, self.model_L)
        self.distance_R.calculate_constant_statistics(task, None, self.model_R)


    def calculate_statistics(self, subgroup, data=None, side=None):
        if side is None:
            raise ValueError
        if hasattr(subgroup, "__array_interface__") or subgroup is None:
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        if side == 0:
            fit_func = self.fit_L
        elif side == 1:
            fit_func = self.fit_R
        return fit_func(cover_arr, data)


    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        statistics_merged = self.distance_L.__class__.merge_statistics(subgroup1, subgroup2, statistics1, statistics2)
        return 1/(self.epsilon + self.distance_L.compare(subgroup1, subgroup2, statistics1, statistics_merged) +
                                 self.distance_R.compare(subgroup1, subgroup2, statistics1, statistics_merged))

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True



class ParameterDistance:
    """
    This class implements a generic L_n distance over parameters vectors.
    The parameters are fetched through the get_params_func
    """
    def __init__(self,  get_params_func, order=1):
        self.get_params_func = get_params_func
        self.order = order

    def calculate_dataset_statistics(self, task, dataset_fit, model):
        pass


    def compare(self, subgroup1, subgroup2, statistics1, statistics2):
        x1 = self.get_params_func(statistics1).ravel()
        x2 = self.get_params_func(statistics2).ravel()
        return np.linalg.norm(x1-x2, ord=self.order)

    @property
    def requires_cover_arr(self):
        return False

    @property
    def is_symmetric(self):
        return True



class Likelihood_Unlikely_Distance:
    """
    This class returns the inverted (likelihood of subgroup1 given the model fitted to subgroup2)
    """

    def __init__(self, epsilon = 10**-10):
        self.epsilon = epsilon

    def calculate_dataset_statistics(self, task, dataset_fit, model):
        assert hasattr(model, 'likelihood')
        
    
    def fit(self, model, cover_arr, data):
        fit = model.fit(cover_arr, data)
        if not hasattr(self, tuple):
            self.tuple = namedtuple('likelihood_tpl', fit.tpl._fields + ['likelihood'])
        like = model.likelihood(fit.beta, slice(None))
        return self.tuple(*fit, like)


    def compare(self, subgroup1, subgroup2, statistics1, statistics2):
        p1_give_2 = np.mean(statistics2.likelihood[subgroup1])
        return 1/(self.epsilon + p1_give_2)

    @property
    def requires_cover_arr(self):
        return True

    @property
    def is_symmetric(self):
        return False