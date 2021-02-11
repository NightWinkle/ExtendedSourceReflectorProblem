import torch
from .utils import gradient_to_normal

def compute_reflector(unit_vector_support: torch.Tensor,
                      potential: torch.Tensor) -> torch.Tensor:
    """This function computes the reflectors point from a tensor of unit vectors
    and a tensor of potential values associated with these unit vectors.

    Args:
        unit_vector_support (torch.Tensor): unit vectors supporting the potential
        potential (torch.Tensor): Kantorovich potential solution to the OT reflector problem

    Returns:
        torch.Tensor: reflector points
    """
    return unit_vector_support * torch.exp(potential[..., None])

def compute_reflector_normals(unit_vector_support: torch.Tensor,
                              potential: torch.Tensor,
                              potential_gradients) -> torch.Tensor:
    """This function computes the reflectors normals from a tensor of unit vectors,
    a tensor of potential values associated with these unit vectors and a tensor of
    potential gradients associated with these unit vectors.

    Args:
        unit_vector_support (torch.Tensor): unit vectors supporting the potential
        potential (torch.Tensor): Kantorovich potential solution to the OT reflector problem
        potential_gradients (torch.Tensor): Kantorovich potential gradients

    Returns:
        torch.Tensor: reflector normals
    """
    normals = (unit_vector_support - gradient_to_normal(unit_vector_support)*potential_gradients[..., None]) * torch.exp(potential[..., None])
    return normals