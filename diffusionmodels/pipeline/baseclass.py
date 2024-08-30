from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError("Subclasses must implement this method")
