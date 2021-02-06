import unittest
import numpy as np
from RDMM.CreateDataSets import generate_two_transition_dataframes, generate_two_regression_dataframes, hide, validate_hide


class TestHiding(unittest.TestCase):

    def test_HideTransitionFrames(self):
        n_states = 3
        df1,df2,_,_,_=generate_two_transition_dataframes([10000,11000], 5, num_noise_attributes=1, n_states=n_states)
        validate_hide(hide(df1,2, 1000))
        validate_hide(hide(df2,3, 3000))

    def test_HideRegressionFrames(self):
        num_classes=5
        df1,df2,_,_,_=generate_two_regression_dataframes([10000,11000], num_classes, num_noise_attributes=1)
        hide1=hide(df1,2, 1000)
        self.assertEqual(len(hide1), len(df1)) # no change in number of instances
        np.testing.assert_array_equal(hide1['class'], df1['class']) # no change in 'class' column
        np.testing.assert_array_equal(hide1['x'], df1['x']) # no change in 'x' column
        np.testing.assert_array_equal(hide1['y'], df1['y']) # no change in 'y' column
        self.assertEqual(len(df1.columns)+2*num_classes, len(hide1.columns)) # added two additional columns for each class

        # validate intersections
        validate_hide(hide1)
        validate_hide(hide(df2,3, 3000))


if __name__ == '__main__':
    unittest.main()