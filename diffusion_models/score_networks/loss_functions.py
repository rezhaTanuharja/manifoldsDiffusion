"""
diffusion_models.score_networks.loss_functions
==============================================

This module provides functionalities to define loss functions for score_networks
"""


from abc import ABC, abstractmethod


def loss_function(ABC):


    def __init__(self):
        pass


    @abstractmethod
    def loss(self, target, prediction):
        raise NotImplementedError("Subclasses must implement this method")
