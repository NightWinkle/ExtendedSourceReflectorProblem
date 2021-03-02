import geomloss
import torch
from math import pi

keops_reflector_cost = "Minus(Log(IntCst(1) - Cos(X - Y)))"


def torch_reflector_cost(x, y):
    return -torch.log(1 - torch.cos(x[:, None, :] - y[None, :, :])).squeeze(-1)


cost = (keops_reflector_cost, torch_reflector_cost)


class ReflectorIdentity:
    @staticmethod
    def forward(x):
        return x+pi

    @staticmethod
    def backward(x):
        return x-pi


def compute_point_source_reflector(input_measure_vector,
                                   input_angular_support,
                                   target_measure_vector,
                                   target_angular_support,
                                   debias=True,
                                   blur=0.00001,
                                   scaling=0.95):
    return geomloss.samples_loss.sinkhorn_multiscale(
        input_measure_vector.view(-1),
        input_angular_support.view(-1, 1),
        target_measure_vector.view(-1),
        target_angular_support.view(-1, 1),
        blur=blur,
        scaling=scaling,
        identity=ReflectorIdentity,
        cost=cost,
        debias=debias,
        full_result=True)
