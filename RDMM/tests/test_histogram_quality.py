from RDMM.histogram_quality_functions import HistogramPearsonCorrelation, HistogramIntersection, HighDHistogramQuality
import unittest
import numpy as np
import pandas as pd
from collections import namedtuple

histogram_tuple = namedtuple('histogram_tuple',['histogram'])
task_dummy = namedtuple('task_dummy', ['data'])


class dummySelector():
    def __init__(self, cov):
        self.cov = cov
    
    def covers(self, data):
        return self.cov

class TestHistogram(unittest.TestCase):

    def test_Pearson(self):
        qf = HistogramPearsonCorrelation(None)
        arr1=np.array([0.5,0.1,0.4])
        arr2=np.array([0.5,0.4,0.1])
        h1 = histogram_tuple(arr1)
        h2 = histogram_tuple(arr2)

        nominator = np.sum([(x-1/3)*(y-1/3) for x,y in zip(arr1, arr2)])
        denominator = np.sqrt(np.sum(np.square(arr1-1/3))*np.sum(np.square(arr1-1/3)))
        self.assertAlmostEqual(nominator/denominator, qf.evaluate(None,None,h1,h2))

    def test_Intersection(self):
        qf = HistogramIntersection(None)
        arr1=np.array([0.5,0.1,0.4])
        arr2=np.array([0.5,0.4,0.1])
        h1 = histogram_tuple(arr1)
        h2 = histogram_tuple(arr2)

        self.assertAlmostEqual(1-0.7, qf.evaluate(None,None,h1,h2))

    def test_create_column_from_selectors(self):
        col1 = np.array([3,2,1,0,5], dtype=int)
        sel1=dummySelector(np.array(col1 & 1,dtype=bool))
        sel2=dummySelector(np.array(col1 & 2,dtype=bool))
        sel3=dummySelector(np.array(col1 & 4,dtype=bool))
        np.testing.assert_array_equal(sel1.cov, np.array([1,0,1,0,1],dtype=bool))
        np.testing.assert_array_equal(sel2.cov, np.array([1,1,0,0,0],dtype=bool))
        np.testing.assert_array_equal(sel3.cov, np.array([0,0,0,0,1],dtype=bool))

        col2=HighDHistogramQuality.create_column_from_selectors([sel1,sel2,sel3],None,5)
        np.testing.assert_array_equal(col1, col2)

    def test_HighDHistogramQuality(self):
        col1_L=[1,1,0,0,0]
        col2_L=[1,2,3,1,0]

        col1_R=[0,0,1,1,0,0]
        col2_R=[3,3,2,1,0,0]
        df_L = pd.DataFrame.from_dict({'L1':col1_L,'L2':col2_L})
        df_R = pd.DataFrame.from_dict({'R1':col1_R,'R2':col2_R})
        qf=HighDHistogramQuality((['L1','L2'],['R1','R2']),df_L,df_R)
        sg1_L=np.array([1,1,0,0,0],dtype=bool)
        sg1_R=np.array([0,0,0,1,1,1],dtype=bool)
        qf.calculate_constant_statistics(task_dummy(df_L), task_dummy(df_R))

        other_arr = np.zeros((2,4),dtype=int)
        other_arr[1,1]=1
        other_arr[1,2]=1
        np.testing.assert_array_equal(other_arr.ravel(), qf.calculate_statistics(sg1_L,None,0).histogram)

        other_arr = np.zeros((2,4),dtype=int)
        other_arr[0,0]=2
        other_arr[1,1]=1
        np.testing.assert_array_equal(other_arr.ravel(), qf.calculate_statistics(sg1_R,None,1).histogram)



if __name__ == '__main__':
    unittest.main()
