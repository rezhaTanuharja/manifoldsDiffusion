import torch
from abc import ABC, abstractmethod



class NaiveMLP(torch.nn.Module):

    def __init__(self):
        pass


class TimeEncoder(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, times: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def output_length(self) -> int:
        pass
    



class SinusoidEncoder(TimeEncoder):

    def __init__(self, num_waves: int) -> None:
        self._num_waves = num_waves


    def __call__(self, times: torch.Tensor) -> torch.Tensor:

        #TODO: Create a function to encode with Fourier series
        output = times
        return output

    def output_length(self) -> int:
        return self._num_waves


class NonNaiveMLP(torch.nn.Module):

    def __init__(self, time_encoder: TimeEncoder):

        super().__init__()

        self._time_encoder = time_encoder

        self._imgs_layer = torch.nn.Linear(1000, 256)
        self._time_layer = torch.nn.Linear(time_encoder.output_length(), 256)

        self._fully_connected_layer = torch.nn.Linear(256, 256)
        self._final_layer = torch.nn.Linear(256, 3)

        self._activation_layer = torch.nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, images, time_stamps):

        output = self._activation_layer(
            self._imgs_layer(images)
            +
            self._time_layer(
                self._time_encoder(time_stamps)
            )
        )

        output = self._activation_layer(self._fully_connected_layer(output))
        output = self._final_layer(output)

        return output
