import torch


def tangent_to_point(tangent: torch.Tensor, manifold):

    tangent = tangent.unflatten(-1, (-1, 3))
    identity = torch.eye(3, device = tangent.device).view(1, 1, 3, 3).expand(*tangent.shape[:2], 3, 3)

    tangent = manifold.exp(identity, tangent)

    return tangent

def extract_points_from_amass(problems):

    return [
        (tangent_to_point(problem[0], problem[1].manifold()), problem[1])
        for problem in problems
    ]
