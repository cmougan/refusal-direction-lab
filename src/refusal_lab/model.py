from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_lab.config import LabConfig


def load_model(cfg: LabConfig):
    """Load a chat model and its tokenizer, configured for left-padded batching."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=cfg.torch_dtype())
    model.to(cfg.device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def format_prompts(tokenizer, instructions: list[str], system: str | None = None) -> list[str]:
    """Apply the model's chat template so activations are read in the chat format."""
    formatted = []
    for instruction in instructions:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": instruction}
        ]
        formatted.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return formatted


def decoder_layers(model) -> torch.nn.ModuleList:
    """Return the list of transformer blocks for common HF architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(f"Could not locate decoder layers on {type(model).__name__}")


def n_layers(model) -> int:
    return len(decoder_layers(model))
