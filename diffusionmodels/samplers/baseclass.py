"""
diffusionmodels.samplers.baseclass
==================================

This module defines the abstract base classes for samplers

Classes
-------
"""


import torch
from abc import ABC, abstractmethod

from ..differentialequations import StochasticDifferentialEquation


class DataRecorder(ABC):

    def __init__(self):
        pass


class TimePropagator(ABC):

    def __init__(self, start_time, end_time, interval):
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval


    @abstractmethod
    def get_samples(
        self,
        sde: StochasticDifferentialEquation,
        initial_condition: torch.Tensor
    ):
        raise NotImplementedError("Subclasses must implement this method")


class SolutionSampler(ABC):

    def __init__(self):
        pass
