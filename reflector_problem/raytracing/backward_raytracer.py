import torch
from .reflector_utils import compute_reflector, compute_reflector_normals
from ..reflector import interpolate_potentials, cubic_coeffs, interpolate_cubic, interpolate_cubic_derivative
from .utils import *
from .reflection_laws import specular_reflection
from math import pi

class BackwardRaytracer:
    def __init__(self, source_description, ray_weighter, source_spatial_discretization, source_angular_support, reflector_height):
        self.source_definition = source_description
        self.ray_weighter = ray_weighter
        self.source = source_spatial_discretization
        self.source_angular_support = source_angular_support
        self.reflector_height = reflector_height

    def __str__(self):
        return f"BackwardRaytracer(reflector_height = {self.reflector_height}, ray_weighter = {self.ray_weighter})"

    def raytrace_reflector(self, 
                           sinkhorn_result):
        self.source_angular_support.requires_grad_(True)
        potential = interpolate_potentials(sinkhorn_result, self.source_angular_support) - interpolate_potentials(sinkhorn_result, torch.Tensor([[pi/2]]).to(self.source_angular_support.device)) + torch.log(torch.Tensor([self.reflector_height])).to(self.source_angular_support.device)
        potential_gradients = torch.autograd.grad(potential,
                                                  self.source_angular_support,
                                                  grad_outputs=torch.ones_like(potential).to(potential.device),
                                                  create_graph=True,
                                                  retain_graph=True)[0]
        potential = potential.view(-1)
        potential_gradients = potential_gradients.view(-1)
        reflectors = compute_reflector(to_unit_vector(self.source_angular_support.view(-1)), potential).view(1, -1, 2)

        incident_rays = reflectors[:, :, None, :] - self.source[None, None, :, :]
        incident_rays = normalize_vector(incident_rays)

        normals = compute_reflector_normals(to_unit_vector(self.source_angular_support.view(-1)), potential, potential_gradients)
        normals = normalize_vector(normals)

        reflected_rays = specular_reflection(incident_rays, normals[None, :, None, :])

        rays_weights = self.ray_weighter.compute_weights(incident_rays)

        return reflected_rays, rays_weights


    def raytrace_reflector_raw(self,
                               reflector_points,
                               reflector_potential,
                               reflector_potential_gradients):        
        spline_coeffs = cubic_coeffs(reflector_potential, reflector_potential_gradients)

        potential = interpolate_cubic(spline_coeffs, reflector_points.view(-1), self.source_angular_support.view(-1)) - interpolate_cubic(spline_coeffs, reflector_points.view(-1), torch.Tensor([[pi/2]]).to(self.source_angular_support.device)) + torch.log(torch.Tensor([self.reflector_height])).to(self.source_angular_support.device)
        potential_gradients = interpolate_cubic_derivative(spline_coeffs, reflector_points.view(-1), self.source_angular_support.view(-1))
        potential = potential.view(-1)
        potential_gradients = potential_gradients.view(-1)
        reflectors = compute_reflector(to_unit_vector(self.source_angular_support.view(-1)), potential).view(1, -1, 2)

        incident_rays = reflectors[:, :, None, :] - self.source[None, None, :, :]
        incident_rays = normalize_vector(incident_rays)

        normals = compute_reflector_normals(to_unit_vector(self.source_angular_support.view(-1)), potential, potential_gradients)
        normals = normalize_vector(normals)

        reflected_rays = specular_reflection(incident_rays, normals[None, :, None, :])

        rays_weights = self.ray_weighter.compute_weights(incident_rays)

        return reflected_rays, rays_weights