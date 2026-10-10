import torch

from refusal_lab.config import LabConfig
from refusal_lab.direction import difference_in_means, select_direction


def test_difference_in_means_recovers_planted_direction():
    torch.manual_seed(0)
    n, layers, pos, d = 200, 6, 2, 16
    planted = torch.zeros(d)
    planted[3] = 1.0
    harmless = torch.randn(n, layers, pos, d)
    harmful = torch.randn(n, layers, pos, d)
    harmful[:, 3] += 4 * planted

    directions = difference_in_means(harmful, harmless)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(layers, pos), atol=1e-5)
    assert directions[3, 0] @ planted > 0.95

    cfg = LabConfig(positions=(-2, -1), layer_fraction_range=(0.0, 1.0), device="cpu")
    best, _, scores = select_direction(harmful, harmless, cfg)
    assert best.layer == 3
    assert scores.shape == (layers, pos)
