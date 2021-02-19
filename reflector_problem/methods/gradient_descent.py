from reflector_problem.point_source import compute_point_source_reflector
from reflector_problem.raytracing.utils import to_angle

def design_reflector_gd(
        extended_source_target,
        initial_target,
        input_angular_support,
        initial_angular_support,
        source_description,
        raytracer,
        loss,
        optimizer,
        n_steps=20,
        lr=1.):
    modified_target = initial_target.clone()
    modified_angular_support = initial_angular_support.clone()

    modified_target_log = modified_target.log() + modified_target.logsumexp(dim=-1, keepdim=False)
    modified_target_log.requires_grad_(True)
    optim = optimizer([modified_target_log],
                      lr = lr)

    input_measure_vector = source_description.pdf(
        input_angular_support.detach()).squeeze(-1).cuda()

    input_measure_vector = input_measure_vector / input_measure_vector.sum()
    input_angular_support = input_angular_support.to(input_measure_vector.device)
    modified_target_log = modified_target_log.to(input_angular_support.device)
    modified_angular_support = modified_angular_support.to(input_angular_support.device)
    
    for i in range(n_steps):
        optim.zero_grad()
        sinkhorn_result = compute_point_source_reflector(
            input_measure_vector.view(-1).to(input_measure_vector.device),
            input_angular_support.view(-1, 1),
            modified_target_log.softmax(dim=-1).view(-1).to(input_measure_vector.device),
            modified_angular_support.view(-1, 1)
        )
        
        rays, weights = raytracer.raytrace_reflector(sinkhorn_result)

        cost = loss(weights, to_angle(rays), extended_source_target, modified_angular_support)
        
        cost.backward()
        optim.step()

    return modified_target_log.softmax(dim=-1)
