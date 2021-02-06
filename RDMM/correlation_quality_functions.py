import numpy as np
import pysubgroup as ps
from collections import namedtuple

correlation_model_tpl = namedtuple('correlation_model_tpl', ['size_sg', 'correlation_matrix'])

class CorrelationModel:
    def __init__(self, columns):
        self.columns = tuple(columns)

    def calculate_constant_statistics(self, task, target=None):
        
        if target is None:
            data = task.data
        else:
            data = task


        l=[]
        for col in self.columns:
            if data[col].dtype.name=='category':
                print(f'using codes for {col}')
                l.append(data[col].cat.codes.to_numpy())
            else:
                l.append(data[col].to_numpy())
        self.arrs = np.array(l)


    def fit(self, subgroup, data=None):
        cover_arr, size = ps.get_cover_array_and_size(subgroup, self.arrs.shape[1], data)
        if size > 0:
            tmp_arrs = self.arrs[:, cover_arr]
            size = tmp_arrs.shape[1]
            return correlation_model_tpl(size, np.corrcoef(tmp_arrs, rowvar=True))
        else:
            return correlation_model_tpl(size, np.full((self.arrs.shape[0],self.arrs.shape[0]), np.nan))



class Correlation_L_Distance:
    def __init__(self, order):
        self.order = order

    def calculate_dataset_statistics(self, task, dataset_fit, model):
        pass

    def compare(self, subgroup1, subgroup2, statistics1, statistics2):
        return np.linalg.norm(statistics1.correlation_matrix.ravel() - statistics2.correlation_matrix.ravel(), ord = self.order)

    @property
    def requires_cover_arr(self):
        return False

    @property
    def is_symmetric(self):
        return True