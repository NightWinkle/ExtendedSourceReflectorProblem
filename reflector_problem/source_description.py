import torch
from typing import Callable, NamedTuple

class SourceDescription(NamedTuple):
    """This class holds a source information
    """
    pdf: Callable[[torch.Tensor], torch.Tensor]
    cdf: Callable[[torch.Tensor], torch.Tensor]