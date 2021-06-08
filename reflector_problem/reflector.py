from geomloss.sinkhorn_samples import clusterize
import torch

def interpolate_potentials(sinkhorn_result, new_angular_support, new_ranges=None):
    softmin = sinkhorn_result.softmin
    if len(sinkhorn_result.C_xy) > 2:
        if new_ranges == None:
            C_xy = (new_angular_support,
                    sinkhorn_result.C_xy[1].detach(), None, None, None)
            C_xx = (new_angular_support, sinkhorn_result.identity.forward(
                sinkhorn_result.C_xx[0].detach()), None, None, None)
        else:
            new_ranges_xy = (new_ranges,) + sinkhorn_result.C_xy[4][1:]
            new_ranges_xx = (new_ranges,) + sinkhorn_result.C_xx[4][1:]
            C_xy = (new_angular_support,
                    sinkhorn_result.C_xy[1].detach(), None, None, new_ranges_xy)
            C_xx = (new_angular_support, sinkhorn_result.identity.forward(
                sinkhorn_result.C_xx[0].detach()), None, None, new_ranges_xx)
    else:
        C_xy = (new_angular_support, sinkhorn_result.C_xy[1].detach())
        C_xx = (new_angular_support, sinkhorn_result.identity.forward(
            sinkhorn_result.C_xx[0].detach()))
    eps = sinkhorn_result.epsilon
    interpolated_potential = softmin(eps, C_xy, sinkhorn_result.β_log + sinkhorn_result.a_y/eps) \
        - softmin(eps, C_xx, sinkhorn_result.α_log +
                  sinkhorn_result.a_x/eps)
    return interpolated_potential

def cubic_coeffs(potential, potential_gradients):
    anchors = torch.stack([potential_gradients[...,1:],
                           potential[...,1:],
                           potential_gradients[...,:-1], 
                           potential[...,:-1]], dim=-1).to(potential.device)
    M = torch.Tensor([[  1., -2.,  1.,  2.],
                      [ -1.,  3., -2., -3.],
                      [  0.,  0.,  1.,  0.],
                      [  0.,  0.,  0.,  1.]]).type(potential.dtype).to(potential.device)
    return M @ anchors.T

def interpolate_cubic(coeffs, points, new_points):
    index = torch.bucketize(new_points, points) - 1
    index = index.clamp(0, coeffs.shape[-1]-1)
    t = (new_points - points[index]) / (points[index+1] - points[index])
    ret = coeffs[0, index] * t + coeffs[1, index]
    ret = ret * t + coeffs[2, index]
    ret = ret * t + coeffs[3, index]
    return ret

def interpolate_cubic_derivative(coeffs, points, new_points):
    index = torch.bucketize(new_points, points) - 1
    index = index.clamp(0, coeffs.shape[-1]-1)
    t = (new_points - points[index]) / (points[index+1] - points[index])
    ret = 3*coeffs[0, index] * t + 2*coeffs[1, index]
    ret = ret * t + coeffs[2, index]
    return ret