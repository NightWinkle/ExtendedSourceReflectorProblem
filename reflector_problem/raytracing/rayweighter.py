from abc import ABC, abstractmethod
from .utils import to_angle
import torch
from math import pi


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


class RiemannRayWeighter(RayWeighter):
    """RayWeighter that computes rays using Riemann integration.
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
        batch_size = rays.shape[0]
        source_size = rays.shape[2]
        rays_angle = to_angle(rays)
        separating_semi_angles = (
            rays_angle[:, 1:, :] + rays_angle[:, :-1, :]) / 2.
        weights = ((torch.cat([separating_semi_angles,
                               torch.Tensor([pi]).cuda().view(1, 1, 1).repeat(batch_size, 1, source_size)], axis=1)
                    -
                    torch.cat([torch.Tensor([0.]).cuda().view(1, 1, 1).repeat(batch_size, 1, source_size),
                               separating_semi_angles], axis=1)) / np.pi) * self.source_description.pdf(rays_angle).cuda()

        return weights / weights.sum()
