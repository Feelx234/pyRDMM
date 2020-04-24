from RDMM.correlation_quality_functions import CorrelationModel
import unittest
import numpy as np
import pandas as pd
from collections import namedtuple

from collections import namedtuple
task_dummy = namedtuple('task_dummy', ['data'])

class TestHistogram(unittest.TestCase):

    def test_CorrelationModel(self):
        df = pd.DataFrame.from_dict({'A' : np.linspace(-1,1), 'B' : np.linspace(-1,1) + 0.1 * np.random.rand(50)})
        model = CorrelationModel(['A', 'B'])


        model.calculate_constant_statistics(task_dummy(df))

        fit_result = model.fit(slice(None))
        self.assertEqual(fit_result.size, 50)
        np.testing.assert_array_almost_equal(fit_result.correlation_matrix, np.array([[1,1], [1,1]]), decimal=2)


        fit_result = model.fit(np.hstack([np.ones(25, dtype=bool),np.zeros(25, dtype=bool)]))
        self.assertEqual(fit_result.size, 25)
        np.testing.assert_array_almost_equal(fit_result.correlation_matrix, np.array([[1,1], [1,1]]), decimal=2)




if __name__ == '__main__':
    unittest.main()