import torch
from pykeops.torch import LazyTensor
from reflector_problem.raytracing.utils import to_angle
from math import pi


class SmoothBinning:
    def __init__(self, bins_centers=None, n_bins=None, sigma=0.1, eps=1e-12):
        self.eps = eps
        if n_bins is None and bins_centers is None:
            raise Exception("Either n_bins or bins_centers must be specified")
        elif n_bins is not None:
            self.delta = 15*pi/8 - 9*pi/8
            centers = torch.linspace(9*pi/8, 15*pi/8, n_bins+1)
            centers = (centers[1:] + centers[:-1])/2
            self.centers = centers
            self.n_bins = n_bins
        else:
            self.delta = bins_centers.max() - bins_centers.min()
            self.centers = bins_centers
            self.n_bins = self.centers.shape[-1]
        self.sigma = sigma

    def __str__(self):
        return f"SmoothBinning(n_bins = {self.n_bins}, sigma = {self.sigma})"

    def __call__(self, rays_angles, weights):
        raydiffs = (LazyTensor(rays_angles.view(-1)[:, None, None]) -
                    LazyTensor(self.centers[None, :, None].type(rays_angles.dtype).to(rays_angles.device))).abs()

        x = (-0.5*(raydiffs/self.sigma)**2).exp() / \
            (self.sigma * (pi*2)**(1/2)) * self.delta

        x_weighted = x * LazyTensor(weights.view(-1)
                                    [:, None, None].type(rays_angles.dtype).to(rays_angles.device))

        dist = x_weighted.sum(dim=0)

        dist = dist/dist.sum() + self.eps

        return self.centers, dist

class Binning:
    def __init__(self, bins_centers=None, n_bins=None, eps=1e-12):
        self.eps = eps
        if n_bins is None and bins_centers is None:
            raise Exception("Either n_bins or bins_centers must be specified")
        elif n_bins is not None:
            self.delta = 15*pi/8 - 9*pi/8
            centers = torch.linspace(9*pi/8, 15*pi/8, n_bins+1)
            centers = (centers[1:] + centers[:-1])/2
            self.centers = centers
            self.n_bins = n_bins
        else:
            self.delta = bins_centers.max() - bins_centers.min()
            self.centers = bins_centers
            self.n_bins = self.centers.shape[-1]

    def __str__(self):
        return f"Binning(n_bins = {self.n_bins})"

    def __call__(self, rays_angles, weights):
        raydiffs = (LazyTensor(rays_angles.view(-1)[:, None, None]) -
                    LazyTensor(self.centers[None, :, None].type(rays_angles.dtype).to(rays_angles.device))).abs()
        rays_bins = raydiffs.argmin(dim=1)

        dist = torch.zeros_like(self.centers).to(rays_angles.device).scatter_add(
            dim=0, index=rays_bins.view(-1), src=weights.view(-1))

        dist = dist/dist.sum() + self.eps

        return self.centers, dist
