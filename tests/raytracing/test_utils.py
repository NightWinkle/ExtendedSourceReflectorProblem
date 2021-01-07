import torch
from reflector_problem.raytracing import utils
import unittest
from math import pi

class TestRayTracingUtils(unittest.TestCase):
    def test_normalize_vector(self):
        """Tests the normalize_vector function
           Function: reflector_problem.raytracing.utils.normalize_vector(vectors)
        """
        B1, B2, B3, B4 = 5, 4, 6, 4
        vectors = torch.arange(1, B1*B2*B3*B4*2 + 1, dtype=torch.float32).view(B1, B2, B3, B4, 2)

        normalized_vectors = utils.normalize_vector(vectors)

        self.assertEqual(normalized_vectors.shape, (B1, B2, B3, B4, 2))
        self.assertEqual(vectors.dtype, normalized_vectors.dtype)
        self.assertTrue(torch.allclose(normalized_vectors.norm(dim = -1), torch.Tensor([1.]).type_as(vectors)))
        self.assertTrue(torch.allclose(vectors[..., 0] / vectors[..., 1], normalized_vectors[..., 0] / normalized_vectors[..., 1]))

    def test_to_unit_vector_norm(self):
        """Verify the vectors returned by the to_unit_vector() function are unit vectors.
           Function: reflector_problem.raytracing.utils.to_unit_vector(angles)
        """
        angles = torch.linspace(0, 2*pi, 1000)

        unit_vectors = utils.to_unit_vector(angles)

        self.assertTrue(torch.allclose(unit_vectors.norm(dim = -1), torch.Tensor([1.])))
        self.assertEqual(unit_vectors.shape, (1000, 2))
        self.assertEqual(angles.dtype, unit_vectors.dtype)

    def test_to_unit_vector_norm_batch(self):
        """Verify the vectors returned by the to_unit_vector() function are unit vectors for input with multiple batch dimensions.
           Function: reflector_problem.raytracing.utils.to_unit_vector(angles)
        """
        B1, B2, B3, B4 = 4, 5, 6, 7 # Batch dimensions
        angles = torch.linspace(0, 2*pi, B1*B2*B3*B4).view(B1, B2, B3, B4)

        unit_vectors = utils.to_unit_vector(angles)

        self.assertTrue(torch.allclose(unit_vectors.norm(dim = -1), torch.Tensor([1.])))
        self.assertEqual(unit_vectors.shape, (B1, B2, B3, B4, 2))
        self.assertEqual(angles.dtype, unit_vectors.dtype)

    def test_to_angle(self):
        """Verify that the output of the to_angle(unit_vectors) function is correct on common inputs.
           Function: reflector_problem.raytracing.utils.to_angle(unit_vectors)
        """
        unit_vectors = torch.Tensor([[1., 0.], 
                                     [3**(1/2)/2, 1/2],
                                     [2**(1/2)/2, 2**(1/2)/2],
                                     [1/2, 3**(1/2)/2],
                                     [0., 1.],
                                     [-1/2, 3**(1/2)/2],
                                     [-2**(1/2)/2, 2**(1/2)/2],
                                     [-3**(1/2)/2, 1/2],
                                     [-1., 0.],
                                     [-3**(1/2)/2, -1/2],
                                     [-2**(1/2)/2, -2**(1/2)/2],
                                     [-1/2, -3**(1/2)/2],
                                     [0., -1.],
                                     [1/2, -3**(1/2)/2],
                                     [2**(1/2)/2, -2**(1/2)/2],
                                     [3**(1/2)/2, -1/2],
                                   ])
        angles = torch.Tensor([0., pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6, pi, 
                               7*pi/6, 5*pi/4, 4*pi/3, 3*pi/2, 5*pi/3, 7*pi/4, 11*pi/6])

        output = utils.to_angle(unit_vectors)

        self.assertTrue(torch.allclose(output, angles))
        self.assertEqual(angles.dtype, output.dtype)

    def test_to_angle_batch(self):
        """Verify the shapes of outputs of the function to_angle() function with multiple batch size.
           Function: reflector_problem.raytracing.utils.to_angle(unit_vectors)
        """
        B1, B2, B3, B4 = 4, 5, 6, 7 # Batch dimensions
        xs = torch.linspace(-1, 1, B1*B2*B3*B4).view(B1, B2, B3, B4)
        ys = torch.sqrt(1 - xs ** 2)
        unit_vectors = torch.stack([xs, ys], dim=-1)

        angles = utils.to_angle(unit_vectors)

        self.assertEqual(angles.shape, (B1, B2, B3, B4))
        self.assertEqual(angles.dtype, unit_vectors.dtype)

    def test_gradient_to_normal(self):
        """Tests the function gradient_to_normal
           Function: reflector_problem.raytracing.utils.gradient_to_normal(gradients)
        """
        B1, B2, B3, B4 = 4, 5, 6, 7 # Batch dimensions
        xs = torch.linspace(-1, 1, B1*B2*B3*B4).view(B1, B2, B3, B4)
        ys = torch.sqrt(1 - xs ** 2)

        gradients = torch.stack([xs, ys], dim=-1)
        normals_true = torch.stack([-ys, xs], dim=-1)
        
        normals = utils.gradient_to_normal(gradients)

        self.assertEqual(normals.shape, (B1, B2, B3, B4, 2))
        self.assertEqual(gradients.dtype, normals.dtype)
        self.assertTrue(torch.allclose(normals, normals_true))

if __name__=="__main__":
    unittest.main()