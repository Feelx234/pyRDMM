import numpy as np
from numba import njit
from collections import namedtuple

histogram_tuple = namedtuple("histogram_tuple",["histogram"])



class HistogramQuality:
    def calculate_statistics(self, subgroup, data=None, side=None):
        raise NotImplementedError()



class HistogramBaseClass():
    def __init__(self, histogram_class):
        self.histogram_class = histogram_class

    def calculate_constant_statistics(self, taskL, taskR):
        self.histogram_class.calculate_constant_statistics(taskL, taskR)

    def calculate_statistics(self, subgroup, data=None, side=None):
        return self.histogram_class.calculate_statistics(subgroup, data=data, side=side)



class HistogramPearsonCorrelation(HistogramBaseClass):
    """ Computes pearson correlation of two histograms
    
    See e.g.
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
    """


    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
       hist1=statistics1.histogram
       hist2=statistics2.histogram
       return np.corrcoef(hist1, hist2)[0,1]



class HistogramIntersection(HistogramBaseClass):

    """ Computes intersection of two histogram
    
    See e.g.
    https://stats.stackexchange.com/questions/7400/how-to-assess-the-similarity-of-two-histograms
    """


    def evaluate(self, subgroup1, subgroup2, statistics1, statistics2):
       hist1=statistics1.histogram
       hist2=statistics2.histogram
       return 1 - np.sum(np.minimum(hist1, hist2))



class HighDHistogramQuality:
    @staticmethod
    def create_column_from_selectors(selectors, data, length):
        """ uses a list of selectors to create numeric column
        """

        new_column = np.zeros(length, dtype=int)
        for i, sel in enumerate(selectors):
            new_column[sel.covers(data)] += 2**i
        return new_column

    def __init__(self, columns, df_L, df_R):
        """
            columns is a tuple of two list of column names for both sides
        """

        assert(len(columns[0]) == len(columns[1]))

        self.columns=columns
        self.num_dimensions = len(columns[0])
        maxes_L=[df_L[column_L].max() for column_L in columns[0]]
        maxes_R=[df_R[column_R].max() for column_R in columns[1]]
        self.dimension_sizes=tuple([max(l,r)+1 for l,r in zip(maxes_L, maxes_R)])
        
        self.flattened_length = np.prod(self.dimension_sizes)
        self.column_arrays = ()
    
    def calculate_constant_statistics(self, taskL, taskR):
        arrs_L = [taskL.data[col].to_numpy() for col in self.columns[0]]
        arrs_R = [taskR.data[col].to_numpy() for col in self.columns[1]]
        self.column_arrays=(arrs_L, arrs_R)


    def _calculate_statistics(self, cov_arr, side=None):
        multi_index = tuple(col[cov_arr] for col in self.column_arrays[side])
        indices = np.ravel_multi_index(multi_index, self.dimension_sizes)

        #out_arr=np.zeros(self.flattened_length,dtype=int)
        #HighDHistogramQuality._count(indices, out_arr)
        return np.bincount(indices, minlength=self.flattened_length)


    @staticmethod
    #@njit
    def _count(indices, arr):
        for i in indices:
            arr[i]+=1
        

    def calculate_statistics(self, subgroup, data=None, side=None):
        if side == 0 or side == 1:
            return histogram_tuple(self._calculate_statistics(subgroup, side))
        else:
            raise ValueError
