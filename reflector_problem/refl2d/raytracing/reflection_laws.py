import torch

def specular_reflection(incident_rays: torch.Tensor,
                        normals: torch.Tensor) -> torch.Tensor:
    """Compute the specular reflection of incident rays on a reflector of which normals are known where the rays hit.
    Both input tensors must have same number of dimensions. This function supports broadcasting.

    Args:
        incident_rays (torch.Tensor): Tensor of incident rays (unit vectors)
        normals (torch.Tensor): Tensor of normals (unit vectors) at the points where rays hit

    Returns:
        torch.Tensor: reflected rays (unit vectors)
    """

    incident_rays_normal_dotproduct = (incident_rays * normals).sum(dim=-1, keepdim=True)

    reflected_rays = incident_rays -\
        2 * (incident_rays_normal_dotproduct * normals)

    return reflected_rays