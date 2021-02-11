import torch


def dkl(first_dist, second_dist):
    return (first_dist * torch.log(first_dist/second_dist)).sum()


class KLLoss:
    def __init__(self, binning):
        self.binning = binning

    def __call__(self,
                 target_measure_vector,
                 target_angular_support,
                 traced_measure_vector,
                 traced_angular_support):
        traced_angular_support, traced_measure_vector = self.binning(
            traced_angular_support, traced_measure_vector)
        return dkl(traced_measure_vector.view(-1), target_measure_vector.view(-1))


class KLSymLoss:
    def __init__(self, binning):
        self.binning = binning

    def __call__(self,
                 target_measure_vector,
                 target_angular_support,
                 traced_measure_vector,
                 traced_angular_support):
        traced_angular_support, traced_measure_vector = self.binning(
            traced_angular_support, traced_measure_vector)
        return (1/2)*(dkl(traced_measure_vector.view(-1), target_measure_vector.view(-1)) + 
                      dkl(target_measure_vector.view(-1), traced_measure_vector.view(-1)))
