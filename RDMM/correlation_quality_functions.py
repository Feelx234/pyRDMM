import numpy as np
import pysubgroup as ps
from collections import namedtuple

correlation_model_tpl = namedtuple('correlation_model_tpl', ['size_sg', 'correlation_matrix'])

class CorrelationModel:
    def __init__(self, columns):
        self.columns = tuple(columns)

    def calculate_constant_statistics(self, task):
        l=[]
        for col in self.columns:
            if task.data[col].dtype.name=='category':
                print(f'using codes for {col}')
                l.append(task.data[col].cat.codes.to_numpy())
            else:
                l.append(task.data[col].to_numpy())
        self.arrs = np.array(l)


    def fit(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__") or isinstance(subgroup, slice):
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        tmp_arrs = self.arrs[:, cover_arr]
        size = tmp_arrs.shape[1]
        return correlation_model_tpl(size, np.corrcoef(tmp_arrs, rowvar=True))



class Correlation_L_Distance:
    def __init__(self, order):
        self.order = order

    def calculate_dataset_statistics(self, task, dataset_fit, model):
        pass

    def compare(self, subgroup1, subgroup2, statistics1, statistics2):
        return np.linalg.norm(statistics1.correlation_matrix.flatten() - statistics2.correlation_matrix.flatten(), ord = self.order)

    @property
    def requires_cover_arr(self):
        return False

    @property
    def is_symmetric(self):
        return True