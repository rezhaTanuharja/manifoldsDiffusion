import torch


class NaiveMLP(torch.nn.Module):

    def __init__(
        self,
        num_image_features: int,
        num_time_features: int,
    ) -> None:

        super().__init__()

        self._image_layer = torch.nn.Linear(num_image_features, 256)
        self._time_layer = torch.nn.Linear(num_time_features, 256)
        self._rotation_layer = torch.nn.Linear(9, 256)

        self._fully_connected_layer = torch.nn.Linear(256, 256)
        self._final_layer = torch.nn.Linear(256, 3)

        self._activation_layer = torch.nn.LeakyReLU(negative_slope = 0.01)

    def forward(
        self,
        images: torch.Tensor,
        time_stamps: torch.Tensor,
        rotations: torch.Tensor
    ) -> torch.Tensor:

        output = self._activation_layer(
            self._image_layer(images)
            +
            self._time_layer(time_stamps)
            +
            self._rotation_layer(rotations)
        )

        output = self._activation_layer(self._fully_connected_layer(output))
        output = self._final_layer(output)

        return output
