from RDMM.model_target import *
import pandas as pd
import pysubgroup as ps
import numpy as np
#ps.SubgroupDiscoveryTask(,None,)
import unittest

class TestRegression(unittest.TestCase):
    def create_dataset(self, a, b):
        x=np.linspace(-1,1,25)
        y= a*x + b
        df = pd.DataFrame.from_dict({'x':np.hstack([x, np.random.rand(25)]),'y': np.hstack([y, np.random.rand(25)])})
        return df
    def test_PolyRegression(self,):
        df = self.create_dataset(-2,1)
        x=df['x']
        y=df['y']
        model=PolyRegression_ModelClass()
        model.calculate_constant_statistics(ps.SubgroupDiscoveryTask(df,None,None,None))
        indices=np.hstack([np.ones(25,dtype=bool),np.zeros(25,dtype=bool)])

        beta_poly = np.polyfit(x[indices],y[indices], 1)


        # obtain gp-basis from multiple (faster)
        basis_mult=model.gp_get_stats_multiple(indices)
        beta_mult = model.gp_get_params(basis_mult)

        # obtain gp-basis incrementally
        basis=model.gp_get_null_vector()
        for i in range(25):
            model.gp_merge(basis, model.gp_get_stats(i))
        beta_incre = model.gp_get_params(basis)

        # Assert fits are similar
        np.testing.assert_almost_equal(beta_mult.beta, beta_poly)
        np.testing.assert_almost_equal(beta_incre.beta, beta_poly)

        # Assert basis is similar
        np.testing.assert_almost_equal(basis_mult, basis)



if __name__ == '__main__':
    unittest.main()