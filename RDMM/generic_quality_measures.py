import numpy as np

class Ex_Distance:
    def __init__(self, model, distance):
        self.model=model
        self.distance=distance

    def calculate_constant_statistics(self, task):
        self.model.calculate_constant_statistics(task)
        self.dataset_fit = self.model.fit(None)
        self.distance.calculate_dataset_statistics(task, self.dataset_fit, self.model)
        

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "__array_interface__") or subgroup is None:
            cover_arr = subgroup
        else:
            cover_arr = subgroup.covers(data)
        return self.model.fit(cover_arr)

    def evaluate(self, subgroup, statistics=None):
        if statistics.size == 0:
            return np.nan
        return self.distance.compare(subgroup, None, statistics, self.dataset_fit)



class Sim_Direct_Distance:
    def __init__(self, model_L, model_R, distance, epsilon = 10**-10):
        self.model_L = model_L
        self.model_R = model_R
        self.distance = distance
        self.epsilon = epsilon


    def calculate_constant_statistics(self, task):
        self.model_L.calculate_constant_statistics(task)
        self.model_R.calculate_constant_statistics(task)
        self.distance.calculate_constant_statistics(task)

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
        return model.fit(cover_arr, data)

    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
        return 1/(self.epsilon + self.distance.compare(subgroup1, subgroup2, statistics1, statistics2))

    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True



class Sim_Common_Distance:
    def __init__(self, model_L, model_R, distance_L, distance_R, epsilon = 10**-10):
        self.model_L = model_L
        self.model_R = model_R
        self.distance_L = distance_L
        self.distance_R = distance_R
        self.epsilon = epsilon

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