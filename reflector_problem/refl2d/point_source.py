import geomloss
import torch
from math import pi

keops_reflector_cost = "Minus(Log(IntCst(1) - (X | Y)))"


def torch_reflector_cost(x, y):
    return -torch.log(1 - (x[:, None, :] * y[None, :, :]).sum(dim=-1, keepdim=True)).squeeze(-1)


cost = (keops_reflector_cost, torch_reflector_cost)


class ReflectorIdentity:
    @staticmethod
    def forward(x):
        return -x

    @staticmethod
    def backward(x):
        return -x


def compute_point_source_reflector(input_measure_vector,
                                   input_spatial_support,
                                   target_measure_vector,
                                   target_spatial_support,
                                   debias=True,
                                   blur=0.00001,
                                   scaling=0.95):
    return geomloss.samples_loss.sinkhorn_multiscale(
        input_measure_vector.view(-1),
        input_spatial_support.view(-1, 3),
        target_measure_vector.view(-1),
        target_spatial_support.view(-1, 3),
        p=1,
        blur=blur,
        scaling=scaling,
        identity=ReflectorIdentity,
        cost=cost,
        debias=debias,
        full_result=True)
