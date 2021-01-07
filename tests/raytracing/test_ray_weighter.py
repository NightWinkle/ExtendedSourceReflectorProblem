import torch
from math import pi
from reflector_problem.raytracing.ray_weighter import NaiveRayWeighter
from reflector_problem.raytracing import reflector_utils, utils
from reflector_problem.source_description import SourceDescription
import unittest

class TestRayTracingRayWeighter(unittest.TestCase):
    def test_naive_rw(self):
        """Tests the NaiveRayWeighter class
           Class: reflector_problem.raytracing.ray_weighter.NaiveRayWeighter
        """
        gaussian = torch.distributions.Normal(loc = 3*pi/2, scale=pi/10)
        source_description = SourceDescription(lambda x: gaussian.log_prob(x).exp(), gaussian.cdf)
        ray_weighter = NaiveRayWeighter(source_description)
        
        N = 2048
        angles = torch.linspace(pi, 2*pi, N)

        weights = ray_weighter.compute_weights(utils.to_unit_vector(angles))

        self.assertEqual(weights.shape, (N,))
        self.assertEqual(weights.dtype, angles.dtype)
        self.assertTrue(torch.allclose(torch.sum(weights), torch.Tensor([1.])))

if __name__=="__main__":
    unittest.main()