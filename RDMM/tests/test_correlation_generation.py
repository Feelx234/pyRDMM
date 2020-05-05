import unittest
from RDMM.generate_correlation import *


class TestCorrelationGeneration(unittest.TestCase):
    def test_corr_samples(self):
        # Fails about 1 out 10
        n = 4

        sample_sizes = [100,200,10000,30000]
        
        params = generate_all_cov_parameters(3, n)
        self.assertEqual(len(params),n)
        
        samples = create_corr_samples(sample_sizes, params)
        
        self.assertEqual(len(samples), n)
        self.assertEqual(list(map(len, samples)), sample_sizes)
        
        np.testing.assert_allclose(np.corrcoef(samples[0].T), cov_to_corr(params[0].cov), rtol=0.3, atol=0.3)
        np.testing.assert_allclose(np.cov(samples[0].T), params[0].cov, rtol=0.8, atol=0.8)
        
        np.testing.assert_allclose(np.corrcoef(samples[1].T), cov_to_corr(params[1].cov), rtol=0.3, atol=0.3)
        np.testing.assert_allclose(np.cov(samples[1].T), params[1].cov, rtol=0.7, atol=0.7)
        
        np.testing.assert_allclose(np.corrcoef(samples[2].T), cov_to_corr(params[2].cov), rtol=0.2, atol=0.2)
        np.testing.assert_allclose(np.cov(samples[2].T), params[2].cov, rtol=0.1, atol=0.1)
        
        np.testing.assert_allclose(np.corrcoef(samples[3].T), cov_to_corr(params[3].cov), rtol=0.1, atol=0.1)
        np.testing.assert_allclose(np.cov(samples[3].T), params[3].cov, rtol=0.1, atol=0.1)

if __name__ == '__main__':
    unittest.main()