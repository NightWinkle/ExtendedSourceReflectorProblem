import geomloss

class WassersteinLoss:
    def __init__(self, blur=0.001, scaling=0.7, p=1, debias=True, binning=None):
        self.blur = blur
        self.scaling = scaling
        self.p = p
        self.debias = debias
        self.binning=None

    def __call__(self,
                 target_measure_vector,
                 target_angular_support,
                 traced_measure_vector,
                 traced_angular_support):
        if self.binning is not None:
            traced_angular_support, traced_measure_vector = self.binning(traced_angular_support, traced_measure_vector)
        return geomloss.samples_loss.sinkhorn_multiscale(
            target_measure_vector.view(-1),
            target_angular_support.view(-1, 1),
            traced_measure_vector.view(-1),
            traced_angular_support.view(-1, 1),
            blur=self.blur,
            scaling=self.scaling,
            p=self.p,
            debias=self.debias)
