import torch
from math import pi

def normalize_vector(vectors: torch.Tensor) -> torch.Tensor:
    """This function normalizes vectors

    Args:
        vectors (torch.Tensor): input vectors to be normalized

    Returns:
        torch.Tensor: normalized vectors
    """
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)

def to_unit_vector(angles: torch.Tensor) -> torch.Tensor:
    """Convert a Tensor of angles into a Tensor of corresponding unit vectors.

    Args:
        angles (torch.Tensor): Tensor of angles to be converted, of shape [B_1, B_2, ..., B_n]

    Returns:
        torch.Tensor: Corresponding unit vectors, of shape [B1, B2, ..., B_n, 2]
    """
    return torch.stack([torch.cos(angles),
                        torch.sin(angles)], axis=-1)

def to_angle(unit_vectors: torch.Tensor) -> torch.Tensor:
    """Convert a Tensor of unit vectors into a Tensor of corresponding angles.

    Args:
        angles (torch.Tensor): Tensor of unit vectors to be converted, of shape [B1, B2, ..., B_n, 2]

    Returns:
        torch.Tensor: Corresponding angles, of shape [B_1, B_2, ..., B_n]
    """
    mask = (unit_vectors[..., 1] >= 0.).type_as(unit_vectors)
    return mask*unit_vectors[..., 0].acos() +\
           (1. - mask)*(2*pi - unit_vectors[..., 0].acos())

def gradient_to_normal(gradients: torch.Tensor) -> torch.Tensor:
    """Convert a Tensor of gradients into a Tensor of corresponding normals by applying a Pi/2 rotation.

    Args:
        gradients (torch.Tensor): Tensor of gradients to be rotated

    Returns:
        torch.Tensor: Corresponding normals
    """
    rotation_matrix = torch.Tensor([[0., -1.],
                                    [1.,  0.]]).type_as(gradients)
    return gradients@rotation_matrix.T