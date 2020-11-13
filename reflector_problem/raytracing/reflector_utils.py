import torch

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
    return unit_vector_support * torch.exp(potential)