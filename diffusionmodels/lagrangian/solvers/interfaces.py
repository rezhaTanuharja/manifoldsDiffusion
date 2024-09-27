"""
lagrangian.solvers.interfaces
=============================

This module provides the interface for all solvers in this package

Classes
-------
Solver
    The interface for all solvers in this package
"""


import torch
from abc import ABC, abstractmethod
from typing import Dict

from ..differentialequations import StochasticDifferentialEquation
from .timeintegrators import TimeIntegrator
from .recorders import DataRecorder


class Solver(ABC):
    """
    The interface for all solvers in this package

    Methods
    -------
    `to(device)`
        Moves any tensor attributes to device

    `get_samples(initial_value, stochastic_de)`
        Solve an SDE with the given initial value

    Private Attributes
    ----------
    `_time_integrator : TimeIntegrator`
        A time integrator to solve SDEs

    `_data_recorder : DataRecorder`
        A recorder that can store SDE solutions

    `_num_points : int`
        Number of time steps to solve

    `_grid_size : float`
        The resolution in time
    """


    @abstractmethod
    def __init__(
        self,
        time_integrator: TimeIntegrator,
        data_recorder: DataRecorder,
        num_points: int,
        grid_size: float
    ) -> None:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def to(self, device: torch.device):
        """
        Move any tensor attributes to device

        Parameters
        ----------
        `device: torch.device`
            A device object from PyTorch
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def set_num_points(self, num_points: int) -> None:
        """
        Change the number of timesteps

        Parameters
        ----------
        `num_points : int`
            The new number of timesteps
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def set_grid_size(self, grid_size: float) -> None:
        """
        Change the time resolution

        Parameters
        ----------
        `grid_size : float`
            The new time resolution
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def solve(
        self,
        initial_value: torch.Tensor,
        stochastic_de: StochasticDifferentialEquation
    ) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Extract samples by solving SDEs

        Parameters
        ----------
        `initial_value : torch.Tensor`
            The initial value of the problem

        `stochastic_de : StochasticDifferentialEquation`
            The stochastic differential equation to solve

        Returns
        -------
        `Dict['time': torch.Tensor, 'data': torch.Tensor]`
            A dictionary of SDE solution and time
        """
        raise NotImplementedError("Subclasses must implement this method")
