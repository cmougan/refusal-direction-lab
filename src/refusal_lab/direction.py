from __future__ import annotations

from dataclasses import dataclass

import torch

from refusal_lab.config import LabConfig


@dataclass
class CandidateDirection:
    """A unit vector in the residual stream, tagged with where it was read."""

    layer: int
    position: int
    vector: torch.Tensor
    score: float = float("nan")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CandidateDirection(layer={self.layer}, position={self.position}, "
            f"score={self.score:.4f})"
        )


def difference_in_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Per (layer, position) difference of mean activations, L2-normalised.

    Both inputs are ``[n_prompts, n_layers + 1, n_positions, d_model]``.
    """
    if harmful.shape[1:] != harmless.shape[1:]:
        raise ValueError("activation tensors must agree on (layers, positions, d_model)")
    diff = harmful.mean(dim=0) - harmless.mean(dim=0)
    return diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)


def separation_scores(
    directions: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor
) -> torch.Tensor:
    """Cohen's d of the projection onto each candidate direction.

    A large value means the direction cleanly separates harmful from harmless
    prompts at that (layer, position) -- a cheap proxy for "this is the
    refusal direction" that needs no generation.
    """
    proj_harmful = torch.einsum("nlpd,lpd->nlp", harmful, directions)
    proj_harmless = torch.einsum("nlpd,lpd->nlp", harmless, directions)
    pooled = torch.sqrt((proj_harmful.var(dim=0) + proj_harmless.var(dim=0)) / 2) + 1e-8
    return (proj_harmful.mean(dim=0) - proj_harmless.mean(dim=0)) / pooled


def select_direction(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    cfg: LabConfig,
    positions: tuple[int, ...] | None = None,
) -> tuple[CandidateDirection, torch.Tensor, torch.Tensor]:
    """Pick the best-separating candidate from the middle layers.

    Returns the winner plus the full ``[n_layers + 1, n_positions, d_model]``
    direction grid and its score grid, so you can plot the whole landscape.
    """
    positions = positions or cfg.positions
    directions = difference_in_means(harmful, harmless)
    scores = separation_scores(directions, harmful, harmless)

    n_layers = directions.shape[0] - 1
    lo = max(1, int(cfg.layer_fraction_range[0] * n_layers))
    hi = max(lo + 1, int(cfg.layer_fraction_range[1] * n_layers))
    masked = torch.full_like(scores, float("-inf"))
    masked[lo:hi] = scores[lo:hi]

    flat = int(torch.argmax(masked))
    layer, pos_idx = divmod(flat, scores.shape[1])
    best = CandidateDirection(
        layer=layer,
        position=positions[pos_idx],
        vector=directions[layer, pos_idx].clone(),
        score=float(scores[layer, pos_idx]),
    )
    return best, directions, scores
