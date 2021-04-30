from reflector_problem.refl2d.point_source import compute_point_source_reflector
from reflector_problem.raytracing.utils import to_angle
import torch

def design_reflector_gd(
        extended_source_target,
        extended_angular_support,
        initial_target,
        initial_angular_support,
        input_measure_vector,
        input_angular_support,
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
    
    modified_target = initial_target.clone()
    modified_angular_support = initial_angular_support.clone()

    modified_target_log = modified_target.log(
    ) + modified_target.logsumexp(dim=-1, keepdim=False)
    modified_target_log.requires_grad_(True)
    optim = optimizer([modified_target_log],
                      lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lambda step:lr_multiplier)

    input_angular_support = input_angular_support.to(input_measure_vector.device)
    modified_target_log = modified_target_log.to(input_angular_support.device)
    modified_angular_support = modified_angular_support.to(input_angular_support.device)

    cost_normalizer = 1.
 
    if cost_normalization:
        sinkhorn_result = compute_point_source_reflector(
            input_measure_vector.view(-1).to(input_measure_vector.device),
            input_angular_support.view(-1, 1),
            modified_target_log.softmax(dim=-1).view(-1).to(input_measure_vector.device),
            modified_angular_support.view(-1, 1)
        )
 
        rays, weights = raytracer.raytrace_reflector(sinkhorn_result)
        cost_normalizer = loss(weights, to_angle(rays), extended_source_target, extended_angular_support)
        cost_normalizer = cost_normalizer.detach()

    for i in range(n_steps):
        optim.zero_grad()
        sinkhorn_result = compute_point_source_reflector(
            input_measure_vector.view(-1).to(input_measure_vector.device),
            input_angular_support.view(-1, 1),
            modified_target_log.softmax(
                dim=-1).view(-1).to(input_measure_vector.device),
            modified_angular_support.view(-1, 1)
        )

        rays, weights = raytracer.raytrace_reflector(sinkhorn_result)

        cost = loss(weights, to_angle(rays),
                    extended_source_target, extended_angular_support)
        cost = cost / cost_normalizer

        cost.backward()

        optim.step()

        history.save_step(i,
                    modified_target=modified_target_log.softmax(dim=1).detach().cpu().clone(),
                    modified_angular_support=modified_angular_support.detach().cpu().clone(),
                    rays=rays.detach().cpu().clone(),
                    weights=weights.detach().cpu().clone(),
                    cost=cost.detach().cpu().clone(),
                    lr=scheduler.get_lr())

        scheduler.step()


    return modified_target_log.softmax(dim=-1), modified_angular_support, history
