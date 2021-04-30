from geomloss.sinkhorn_samples import clusterize


def interpolate_potentials(sinkhorn_result, new_spatial_support, new_ranges=None):
    softmin = sinkhorn_result.softmin
    if len(sinkhorn_result.C_xy) > 2:
        if new_ranges == None:
            C_xy = (new_spatial_support,
                    sinkhorn_result.C_xy[1].detach(), None, None, None)
            C_xx = (new_spatial_support, sinkhorn_result.identity.forward(
                sinkhorn_result.C_xx[0].detach()), None, None, None)
        else:
            new_ranges_xy = (new_ranges,) + sinkhorn_result.C_xy[4][1:]
            new_ranges_xx = (new_ranges,) + sinkhorn_result.C_xx[4][1:]
            C_xy = (new_spatial_support,
                    sinkhorn_result.C_xy[1].detach(), None, None, new_ranges_xy)
            C_xx = (new_spatial_support, sinkhorn_result.identity.forward(
                sinkhorn_result.C_xx[0].detach()), None, None, new_ranges_xx)
    else:
        C_xy = (new_spatial_support, sinkhorn_result.C_xy[1].detach())
        C_xx = (new_spatial_support, sinkhorn_result.identity.forward(
            sinkhorn_result.C_xx[0].detach()))
    eps = sinkhorn_result.epsilon
    interpolated_potential = softmin(eps, C_xy, sinkhorn_result.β_log + sinkhorn_result.a_y/eps) \
        - softmin(eps, C_xx, sinkhorn_result.α_log +
                  sinkhorn_result.a_x/eps)
    return interpolated_potential
