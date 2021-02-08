from abc import ABC, abstractmethod
from .utils import to_angle
import torch

class RayWeighter(ABC):
    """Class to compute weights of given rays
    """
    @abstractmethod
    def compute_weights(self, 
                        rays: torch.Tensor) -> torch.Tensor:
        """Computes weights of a family of rays

        Args:
            rays (torch.Tensor): Rays to compute weights of

        Returns:
            torch.Tensor: Resulting weights
        """

class NaiveRayWeighter(RayWeighter):
    """RayWeighter that computes rays using naïve integration method that
    doesn't rely on difference of covered angles.
    """
    def __init__(self, source_description):
        self.source_description = source_description

    def compute_weights(self, rays):
        """Computes weights of a family of rays

        Args:
            rays (torch.Tensor): Rays to compute weights of

        Returns:
            torch.Tensor: Resulting weights
        """
        rays_angle = to_angle(rays)
        weights = self.source_description.pdf(rays_angle)
        return weights / weights.sum()
