from ......utilities.warningsuppressors import unused_variables
from .interfaces import DistributionFunction

from typing import Dict

import torch


class Normalizer(DistributionFunction):

    def __init__(self):
        pass

    def to(self, device: torch.device) -> None:
        unused_variables(device)

    def cumulative(self, points: torch.Tensor) -> torch.Tensor:
        return (points - torch.sin(points)) / torch.pi

    def density(self, points: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.cos(points)) / torch.pi

    def at(self, time: torch.Tensor):
        unused_variables(time)
        return self

    def support(self) -> Dict[str, float]:
        return {
            'lower_bound': 0.0,
            'upper_bound': torch.pi
        }
