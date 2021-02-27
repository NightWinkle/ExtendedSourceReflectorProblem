from reflector_problem.point_source import compute_point_source_reflector
from reflector_problem.raytracing.utils import to_angle
import torch

def design_reflector_lbfgs(
        extended_source_target,
        extended_angular_support,
        initial_target,
        initial_angular_support,
        input_measure_vector,
        input_angular_support,
        raytracer,
        loss,
        history,
        cost_normalization=True,
        n_steps=20,
        n_eval_steps=20,
        line_search="strong_wolfe",
        lr=1.):
    modified_target = initial_target.clone()
    modified_angular_support = initial_angular_support.clone()

    modified_target_log = modified_target.log(
    ) + modified_target.logsumexp(dim=-1, keepdim=False)
    modified_target_log.requires_grad_(True)
    optim = torch.optim.LBFGS([modified_target_log],
                              lr=lr,
                              max_iter=n_steps,
                              max_eval=n_eval_steps,
                              tolerance_change=1e-13,
                              tolerance_grad=1e-13,
                              line_search_fn=line_search)

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

    def optim_closure():
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
                    extended_source_target, modified_angular_support)
        cost = cost / cost_normalizer

        cost.backward()

        # Ensures the steps is not already saved in the history - in the case of evaluations steps for LBFGS
        if not optim.state[optim._params[0]]["n_iter"] in history.step_numbers:
            history.save_step(optim.state[optim._params[0]]["n_iter"],
                modified_target=modified_target_log.softmax(dim=1).clone().detach(),
                modified_angular_support=modified_angular_support.clone().detach(),
                rays=rays.clone().detach(),
                weights=weights.clone().detach(),
                cost=cost.clone().detach())

        return cost

    optim.step(optim_closure)

    return modified_target_log.softmax(dim=-1), modified_angular_support, history
