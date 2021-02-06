import unittest
import numpy as np
from RDMM.CreateDataSets import create_transition_sample, generate_two_transition_dataframes, create_random_transition_matrix

def transitions_to_matrix(x_arr,y_arr,shape):
    M=np.zeros(shape,dtype=int)
    for x,y in zip(x_arr,y_arr):
        M[x,y]+=1
    return M

def norm_matrix(M):
    return M/np.sum(M)

def get_matrix_for_classes(df_in,states):
    result = []
    for cls in np.unique(df_in['class']):
        df=df_in[df_in['class']==cls]
        mat = np.array(transitions_to_matrix(df['in'], df['out'],(states,states)),dtype=float)
        mat/=np.sum(mat)
        #starting_ps=np.sum(mat, axis=1)[:,None]
        #mat/=starting_ps
        result.append(mat)
    return result

class TestAlgorithms(unittest.TestCase):
    def test_TransitionMatrixEqualsInput(self):
        M=np.array([[0.3,0.01,0.1], [0.02,0.3,0.1], [0.05,0.06,0.06]])
        M_generated = norm_matrix(transitions_to_matrix(*create_transition_sample(3,10000,M),(3,3)))
        np.testing.assert_almost_equal(M_generated, M, 2)

    def test_CreateTwoTransitionFrames(self):
        n_states = 3
        df1,df2,matrices,_,_=generate_two_transition_dataframes([100000,110000], 5, num_noise_attributes=1, n_states=n_states)
        matrices1 = get_matrix_for_classes(df1, n_states)
        matrices2 = get_matrix_for_classes(df2, n_states)
        for m,m1,m2 in zip(matrices, matrices1, matrices2):
            np.testing.assert_almost_equal(m,m1,2)
            np.testing.assert_almost_equal(m,m2,2)
    
    def test_RandomGeneration(self):
        n_states=5
        mat=create_random_transition_matrix(n_states)
        print(mat)
        print(np.sum(mat))
        assert(np.isclose(np.sum(mat),1))
        v=np.sum(mat,axis=1)
        Trans=np.empty((n_states,n_states))
        for i in range(n_states):
            Trans[i,:]=mat[i,:]/v[i]
        print('check sum along axis 1 is eigenvec')
        print(Trans.T @ v)
        print(v)
        v1=v
        for i in range(10):
            v1=Trans.T @ v1
            assert(np.all(np.isclose(v1,v)))

if __name__ == '__main__':
    unittest.main()
    #suite1 = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms)
    #suite2 = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms2)
    #suite3 = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms3)
    #complete_suite = unittest.TestSuite([suite1, suite2, suite3])
    #unittest.TextTestRunner(verbosity=2).run(complete_suite)