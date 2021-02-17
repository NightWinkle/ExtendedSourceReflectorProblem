import torch
from .reflector_utils import compute_reflector, compute_reflector_normals
from ..reflector import interpolate_potentials
from .utils import *
from .reflection_laws import specular_reflection
from math import pi

def batch_jacobian(func, x):
    x.requires_grad_(True)
    y = func(x)

    full_jac = []
    for i in range(x.shape[-1]):
      grad_outputs = torch.zeros_like(y)
      grad_outputs[..., i] = 1

      jac_i = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True)[0]
      full_jac.append(jac_i)

    return torch.stack(full_jac, dim=1)


def newton_solve(func, x0, args, niters):
    if not torch.is_tensor(x0):
        raise TypeError("x0 must be a torch.Tensor")
    if niters < 1:
        raise ValueError("niters must be > 1")

    x = x0
    
    for _ in range(niters):
        Jx = batch_jacobian((lambda x: func(x, *args)), x)
        b, n, _  = Jx.shape
        u, _ = torch.solve(-func(x, *args).view(b, n, 1), Jx)
        x = u.view(b, -1) + x.view(b, -1)
    return x

def inverse_cdf(pdf, cdf, x0, y, niters=5):
    if not torch.is_tensor(x0):
        raise TypeError("x0 must be a torch.Tensor")
    if niters < 1:
        raise ValueError("niters must be > 1")

    x = x0
    
    for _ in range(niters):
        u = (cdf(x) - y)/pdf(x)
        x = x.view(-1) - u.view(-1)
    return x

def intersection_condition(variables, source_points, incident_rays, reflector_function):
    ray_lenghts = variables[:,0:1].contiguous()
    reflector_parameters = variables[:,1:].contiguous()

    lhs = source_points + ray_lenghts * to_unit_vector(incident_rays).view(-1, 2)
    rhs = reflector_function(reflector_parameters.view(-1, 1))

    return lhs - rhs

def sampler(n_rays, source_spread, source_description, niters=10):
    x0 = torch.Tensor([pi/2]).repeat(n_rays).cuda()

    engine = torch.quasirandom.SobolEngine(2, scramble=True)
    yfull = engine.draw(n_rays).cuda()

    angles = inverse_cdf(source_description.pdf, source_description.cdf, x0, yfull[:,0], niters=niters)
    xpos = yfull[:,1].view(-1,1)

    pos = torch.Tensor([1., 0.]).view(1, 2).cuda() * (2*source_spread*(xpos - 0.5))
    return angles, pos

def find_angular_params(reflector_function, angles, source_pos):
    alphas = torch.norm(reflector_function(angles.view(-1, 1)), dim=-1, keepdim=True)
    x0 = torch.cat([alphas.view(-1, 1), angles.view(-1, 1)], dim=1)
    angular_params = newton_solve(intersection_condition, x0.cuda(), (source_pos.cuda(), angles.view(-1, 1).cuda(), reflector_function), 3)
    return angular_params

class ForwardRaytracer:
    def __init__(self, source_description, source_spread, source_angular_support, reflector_height, n_rays):
        self.source_definition = source_description
        self.source_spread = source_spread
        self.source_angular_support = source_angular_support
        self.reflector_height = reflector_height
        self.n_rays = n_rays

    def raytrace_reflector(self, 
                           sinkhorn_result):
        # Sampling rays
        angles, source_pos = sampler(self.n_rays, self.source_spread, self.source_definition, 15)
        
        # Building potential and reflector functions
        potential_function = lambda x: interpolate_potentials(sinkhorn_result, x) - interpolate_potentials(sinkhorn_result, torch.Tensor([[pi/2]]).to(self.source_angular_support.device)) + torch.log(torch.Tensor([self.reflector_height])).to(self.source_angular_support.device)
        reflector_function = lambda x: compute_reflector(to_unit_vector(x.view(-1)), potential_function(x))

        # Computing reflector parameters for traced rays
        angular_params = find_angular_params(reflector_function, angles, source_pos)

        # Computing potential and potential gradient for traced rays
        angles_dir = angular_params[:,1].contiguous()
        angles_dir.requires_grad_(True)

        potential = potential_function(angles_dir.view(-1, 1))
        potential_gradients = torch.autograd.grad(potential,
                                                  angles_dir,
                                                  grad_outputs=torch.ones_like(potential).to(potential.device),
                                                  create_graph=True,
                                                  retain_graph=True)[0]
        potential = potential.view(-1)
        potential_gradients = potential_gradients.view(-1)

        incident_rays = to_unit_vector(angles)
        normals = compute_reflector_normals(to_unit_vector(angles_dir), potential.view(-1), potential_gradients.view(-1))
        normals = normalize_vector(normals)

        # Computing specular reflection of traced rays
        reflected_rays = specular_reflection(incident_rays, normals)

        return reflected_rays, torch.ones(reflected_rays.shape[:-1]).to(reflected_rays.device)/self.n_rays