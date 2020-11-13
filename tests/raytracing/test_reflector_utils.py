import torch
from reflector_problem.raytracing import reflector_utils, utils
import unittest
from math import pi

class TestRayTracingReflectorUtils(unittest.TestCase):
    def test_compute_reflector(self):
        """Tests the compute_reflector function
           Function: reflector_problem.raytracing.reflector_utils.compute_reflector(unit_vector_support, potential)
        """
        N_discr = 128
        angles = torch.linspace(0, 2*pi, N_discr)
        
        N_heights = 16
        heights = torch.linspace(1, 5, N_heights)

        unit_vectors = utils.to_unit_vector(angles).view(-1, 1, 2)

        potential = torch.ones((N_discr, 1, 1), dtype=angles.dtype)
        potentials = potential * torch.log(heights.view(1, -1, 1))

        reflector_points = reflector_utils.compute_reflector(unit_vectors, potentials)

        self.assertEqual(reflector_points.shape, (N_discr, N_heights, 2))
        self.assertEqual(reflector_points.dtype, unit_vectors.dtype)
        self.assertTrue(torch.allclose(reflector_points.norm(dim = -1), heights.type_as(unit_vectors)))

if __name__=="__main__":
    unittest.main()