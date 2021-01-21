import torch
from .reflector_utils import compute_reflector, compute_reflector_normals
from ..reflector import interpolate_potentials
from .utils import *
from .reflection_laws import specular_reflection

class BackwardRaytracer:
    def __init__(self, source_description, ray_weighter, source_spatial_discretization, source_angular_support):
        self.source_definition = source_description
        self.ray_weighter = ray_weighter
        self.source = source_spatial_discretization
        self.source_angular_support = source_angular_support

    def raytrace_reflector(self, 
                           sinkhorn_result):
        self.source_angular_support.requires_grad_(True)
        potential = interpolate_potentials(sinkhorn_result, self.source_angular_support)
        potential_gradients = torch.autograd.grad(potential,
                                                  self.source_angular_support,
                                                  grad_outputs=torch.ones_like(potential).to(potential.device),
                                                  create_graph=True,
                                                  retain_graph=True)[0]

        reflectors = compute_reflector(self.source_angular_support, potential).view(1, -1, 2)

        incident_rays = reflectors[:, :, None, :] - self.source[None, None, :, :]
        incident_rays = normalize_vector(incident_rays)

        normals = compute_reflector_normals(to_unit_vector(self.source_angular_support), potential, potential_gradients)
        normals = normalize_vector(normals)

        reflected_rays = specular_reflection(incident_rays, normals)

        rays_weights = self.ray_weighter.compute_weights(incident_rays)

        return reflected_rays, rays_weights