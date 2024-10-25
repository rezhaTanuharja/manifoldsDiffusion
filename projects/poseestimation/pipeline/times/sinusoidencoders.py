import diffusionmodels.dataprocessing as dataprocessing
import torch


def create_time_pipeline(
    num_samples: int,
    num_sample_duplicates: int,
    num_wave_numbers: int,
    device: torch.device
) -> dataprocessing.Transform:

    wave_numbers = torch.arange(start = 0.0, end = 2.0, step = 2.0 / num_wave_numbers)
    wave_numbers = wave_numbers.view(1, wave_numbers.numel())
    wave_numbers.to(device)

    time_pipeline = dataprocessing.Pipeline(
        transforms = [

            lambda times: times.view(times.numel(), 1),

            lambda times: torch.cos(times * wave_numbers),

            lambda times: times.unsqueeze(1).expand(
                times.shape[0], num_samples * num_sample_duplicates, times.shape[1]
            ).flatten(0, 1),

        ]
    )


    return time_pipeline
