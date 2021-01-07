import torch
from reflector_problem.raytracing.utils import to_unit_vector, gradient_to_normal, normalize_vector
from reflector_problem.raytracing.reflection_laws import specular_reflection
import unittest
from math import pi

class TestSpecularReflection(unittest.TestCase):
    def test_flat_broadcasting_1d(self):
        """Tests specular reflection for a flat reflector
           Function: reflector_problem.raytracing.reflections_laws.
                        specular_reflection(incident_rays, normals)
        """
        N = 128
        incident_rays_angles = torch.linspace(0, pi, N)
        incident_rays = to_unit_vector(incident_rays_angles)

        flat_normals = torch.Tensor([0., 1.]).view(1, 2)
        
        reflected_rays = specular_reflection(incident_rays, flat_normals)

        self.assertEqual(incident_rays.shape, reflected_rays.shape)
        self.assertEqual(incident_rays.dtype, reflected_rays.dtype)
        self.assertTrue(torch.allclose(incident_rays[..., 0],  reflected_rays[..., 0]))
        self.assertTrue(torch.allclose(incident_rays[..., 1], -reflected_rays[..., 1]))

    def test_flat_weirdshape(self):
        """Tests specular reflection for a flat reflector, with weird shapes (to check batch and broadcasting)
           Function: reflector_problem.raytracing.reflections_laws.
                        specular_reflection(incident_rays, normals)
        """
        B1, B2, B3, B4 = 5, 2, 3, 7

        incident_rays_angles = torch.linspace(0, pi, B1 * 1 * B3 * 1).view(B1, 1, B3, 1)
        incident_rays = to_unit_vector(incident_rays_angles)

        flat_normals = torch.Tensor([0., 1.]).view(1, 1, 1, 1, 2).repeat(1, B2, 1, B4, 1)
        
        reflected_rays = specular_reflection(incident_rays, flat_normals)

        self.assertEqual(reflected_rays.shape, (B1, B2, B3, B4, 2))
        self.assertEqual(incident_rays.dtype, reflected_rays.dtype)
        self.assertTrue(torch.allclose(incident_rays[..., 0],  reflected_rays[..., 0]))
        self.assertTrue(torch.allclose(incident_rays[..., 1], -reflected_rays[..., 1]))

    def test_parabola_1d(self):
        """Tests specular reflection for a parabolic reflector.
           Function: reflector_problem.raytracing.reflections_laws.
                        specular_reflection(incident_rays, normals)
        """
        N = 128
        incident_rays_angles = torch.linspace(0, pi, N)
        incident_rays = to_unit_vector(incident_rays_angles)

        direction = to_unit_vector(torch.Tensor([5*pi/3]))
        C = 1

        parabola_gradients = C / (1 - torch.sum(incident_rays * direction.view(1, 2), dim=-1, keepdim=True))**2 *\
            (gradient_to_normal(incident_rays - direction.view(1, 2)))
        
        parabola_gradients_normalized = normalize_vector(parabola_gradients)

        parabola_normals = gradient_to_normal(parabola_gradients_normalized)
        
        reflected_rays = specular_reflection(incident_rays, parabola_normals)

        self.assertEqual(reflected_rays.shape, (N, 2))
        self.assertEqual(incident_rays.dtype, reflected_rays.dtype)
        self.assertTrue(torch.allclose(reflected_rays, direction.view(-1, 2)))

    def test_multi_parabola_1d(self):
        """Tests specular reflection for a multiple parabolic reflectors with broadcasting.
           Function: reflector_problem.raytracing.reflections_laws.
                        specular_reflection(incident_rays, normals)
        """
        N = 128
        incident_rays_angles = torch.linspace(0, pi, N)
        incident_rays = to_unit_vector(incident_rays_angles)

        N_parabola = 10
        directions = to_unit_vector(torch.linspace(9*pi/8, 15*pi/8, N_parabola))
        C = 1

        parabola_gradients = C / (1 - torch.sum(incident_rays.view(1, N, 2) * directions.view(N_parabola, 1, 2), dim=-1, keepdim=True))**2 *\
            (gradient_to_normal(incident_rays.view(1, N, 2) - directions.view(N_parabola, 1, 2)))
        
        parabola_gradients_normalized = normalize_vector(parabola_gradients)

        parabola_normals = gradient_to_normal(parabola_gradients_normalized)
        
        reflected_rays = specular_reflection(incident_rays, parabola_normals)

        self.assertEqual(reflected_rays.shape, (N_parabola, N, 2))
        self.assertEqual(incident_rays.dtype, reflected_rays.dtype)
        self.assertTrue(torch.allclose(reflected_rays, directions.view(N_parabola, 1, 2)))
