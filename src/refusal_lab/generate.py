from __future__ import annotations

import torch

from refusal_lab.config import LabConfig
from refusal_lab.model import format_prompts


@torch.no_grad()
def generate(model, tokenizer, instructions: list[str], cfg: LabConfig) -> list[str]:
    """Greedy completions for a list of instructions, in batches."""
    prompts = format_prompts(tokenizer, instructions)
    completions: list[str] = []
    for start in range(0, len(prompts), cfg.batch_size):
        batch = prompts[start : start + cfg.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(cfg.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        completions.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return completions
