from geomloss.sinkhorn_samples import clusterize

def interpolate_potentials(sinkhorn_result, new_angular_support):
    softmin = sinkhorn_result.softmin
    if len(sinkhorn_result.C_xy) > 2:
      C_xy = (new_angular_support, sinkhorn_result.C_xy[1].detach(), None, None, None)
      C_xx = (new_angular_support, sinkhorn_result.identity.forward(sinkhorn_result.C_xx[0].detach()), None, None, None)
    else:
      C_xy = (new_angular_support, sinkhorn_result.C_xy[1].detach())
      C_xx = (new_angular_support, sinkhorn_result.identity.forward(sinkhorn_result.C_xx[0].detach()))
    eps = sinkhorn_result.epsilon
    interpolated_potential = softmin(eps, C_xy, (sinkhorn_result.β_log + sinkhorn_result.a_y/eps).detach() ) \
        - softmin(eps, C_xx, (sinkhorn_result.α_log + sinkhorn_result.a_x/eps).detach() )
    return interpolated_potential