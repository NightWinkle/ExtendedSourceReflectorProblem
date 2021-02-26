from reflector_problem.point_source import compute_point_source_reflector
from reflector_problem.raytracing.utils import to_angle


def design_reflector_golds(
        extended_source_target,
        extended_angular_support,
        initial_target,
        initial_angular_support,
        input_measure_vector,
        input_angular_support,
        raytracer,
        binning,
        history,
        n_steps=20,
        lr=1.,
        lr_multiplier=1.):
    modified_target = initial_target.clone()
    modified_angular_support = initial_angular_support.clone()

    input_measure_vector = input_measure_vector / input_measure_vector.sum()
    input_angular_support = input_angular_support.to(
        input_measure_vector.device)
    modified_target = modified_target.to(input_angular_support.device)
    modified_angular_support = modified_angular_support.to(
        input_angular_support.device)

    for i in range(n_steps):
        sinkhorn_result = compute_point_source_reflector(
            input_measure_vector.view(-1).to(input_measure_vector.device),
            input_angular_support.view(-1, 1),
            modified_target.view(-1).to(input_measure_vector.device),
            modified_angular_support.view(-1, 1)
        )

        rays, weights = raytracer.raytrace_reflector(sinkhorn_result)
        centers, dist = binning(to_angle(rays), weights)

        modified_target = modified_target * \
            (extended_source_target /
             (dist.view(*modified_target.shape)))**(lr)

        modified_target = modified_target/modified_target.sum(dim=-1)

        history.save_step(i,
                          modified_target=modified_target.clone().detach(),
                          modified_angular_support=modified_angular_support.clone().detach(),
                          rays=rays.clone().detach(),
                          weights=weights.clone().detach())

        lr = lr * lr_multiplier

    return modified_target, modified_angular_support, history
