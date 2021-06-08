from reflector_problem.point_source import compute_point_source_reflector
from reflector_problem.raytracing.utils import to_angle
import torch

def design_reflector_gd_direct(
        extended_source_target,
        extended_angular_support,

        initial_reflector_potential, 
        initial_reflector_potential_gradients,
        initial_reflector_angular_support,

        raytracer,
        loss,
        optimizer,
        history,
        cost_normalization=True,
        n_steps=20,
        lr=1.,
        lr_multiplier=1.):
    history.save_vars(optimization = "gradient_descent")
    history.save_vars(raytracer = str(raytracer))
    history.save_vars(loss = str(loss))
    
    modified_potential = initial_reflector_potential.clone()
    modified_potential_gradients = initial_reflector_potential_gradients.clone()
    modified_angular_support = initial_reflector_angular_support.clone()

    modified_potential.requires_grad_(True)
    modified_potential_gradients.requires_grad_(True)

    optim = optimizer([modified_potential, modified_potential_gradients],
                      lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lambda step:lr_multiplier)

    cost_normalizer = 1.
 
    if cost_normalization:
        rays, weights = raytracer.raytrace_reflector_raw(modified_angular_support, modified_potential, modified_potential_gradients)
        cost_normalizer = loss(weights, to_angle(rays), extended_source_target, extended_angular_support)
        cost_normalizer = cost_normalizer.detach()

    for i in range(n_steps):
        optim.zero_grad()
        rays, weights = raytracer.raytrace_reflector_raw(modified_angular_support, modified_potential, modified_potential_gradients)

        cost = loss(weights, to_angle(rays),
                    extended_source_target, extended_angular_support)
        cost = cost / cost_normalizer

        cost.backward()

        optim.step()

        history.save_step(i,
                    modified_potential=modified_potential.detach().cpu().clone(),
                    modified_potential_gradients=modified_potential_gradients.detach().cpu().clone(),
                    rays=rays.detach().cpu().clone(),
                    weights=weights.detach().cpu().clone(),
                    cost=cost.detach().cpu().clone(),
                    lr=scheduler.get_lr())

        scheduler.step()


    return modified_potential, modified_potential_gradients, modified_angular_support, history
