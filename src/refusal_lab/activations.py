from __future__ import annotations

import torch

from refusal_lab.config import LabConfig
from refusal_lab.model import format_prompts


@torch.no_grad()
def collect_residuals(
    model,
    tokenizer,
    instructions: list[str],
    cfg: LabConfig,
    positions: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Mean-free snapshot of the residual stream for each prompt.

    Returns a tensor of shape ``[n_prompts, n_layers + 1, n_positions, d_model]``.
    Layer 0 is the embedding output, layer i the output of block i.
    """
    positions = positions or cfg.positions
    prompts = format_prompts(tokenizer, instructions)
    chunks = []
    for start in range(0, len(prompts), cfg.batch_size):
        batch = prompts[start : start + cfg.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(cfg.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        # [n_layers + 1, batch, seq, d_model]; left padding keeps the final
        # positions aligned across prompts of different lengths.
        stacked = torch.stack(out.hidden_states, dim=0)
        selected = stacked[:, :, list(positions), :]
        chunks.append(selected.permute(1, 0, 2, 3).float().cpu())
    return torch.cat(chunks, dim=0)
