"""Two ways to remove the refusal direction.

``ablate_direction`` is an inference-time hook: reversible, nothing on disk
changes, best for experimenting. ``orthogonalize_weights`` writes the same
projection into the weights, which is what "abliterated" model releases do --
it is included so you can see that the two are mathematically the same edit.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from refusal_lab.model import decoder_layers


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / (direction.norm() + 1e-8)


def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``x`` along ``direction``."""
    d = _unit(direction).to(x.dtype).to(x.device)
    return x - (x @ d).unsqueeze(-1) * d


def _wrap(fn):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (fn(output[0]),) + output[1:]
        return fn(output)

    return hook


@contextmanager
def ablate_direction(model, direction: torch.Tensor, layers: list[int] | None = None):
    """Project the direction out of the residual stream at every layer.

    Ablating at *all* layers (the default) is what Arditi et al. do: the
    direction is written by many layers, so removing it at one is not enough.
    """
    blocks = decoder_layers(model)
    targets = range(len(blocks)) if layers is None else layers
    ablate = _wrap(lambda x: project_out(x, direction))
    handles = []
    for i in targets:
        # Hook the attention and MLP outputs as well as the block output, so the
        # residual stream between attention and MLP is clean too. That is what
        # makes this equivalent to orthogonalize_weights.
        modules = [blocks[i]] + [
            getattr(blocks[i], name)
            for name in ("self_attn", "attn", "mlp")
            if hasattr(blocks[i], name)
        ]
        handles.extend(m.register_forward_hook(ablate) for m in modules)
    # The embedding output feeds layer 0, so it needs the same treatment.
    handles.append(model.get_input_embeddings().register_forward_hook(ablate))
    try:
        yield model
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def add_direction(model, direction: torch.Tensor, layer: int, coeff: float = 1.0):
    """Add the direction into one layer -- the control experiment.

    If this makes the model refuse harmless prompts, the direction really does
    mediate refusal rather than merely correlating with it.
    """
    block = decoder_layers(model)[layer]
    d = _unit(direction)

    def shift(x: torch.Tensor) -> torch.Tensor:
        return x + coeff * d.to(x.dtype).to(x.device)

    handle = block.register_forward_hook(_wrap(shift))
    try:
        yield model
    finally:
        handle.remove()


def _orthogonalize_matrix(weight: torch.Tensor, direction: torch.Tensor, rows: bool) -> None:
    d = _unit(direction).to(weight.dtype).to(weight.device)
    if rows:  # weight is [n, d_model]: each row lives in the residual stream
        weight -= torch.outer(weight @ d, d)
    else:  # weight is [d_model, n]: each column writes to the residual stream
        weight -= torch.outer(d, d @ weight)


@torch.no_grad()
def orthogonalize_weights(model, direction: torch.Tensor) -> int:
    """Bake the ablation into every weight that writes to the residual stream.

    Returns the number of matrices edited. This mutates the model in place.
    """
    edited = 0
    embed = model.get_input_embeddings().weight
    _orthogonalize_matrix(embed, direction, rows=True)
    edited += 1

    for block in decoder_layers(model):
        for name in ("self_attn.o_proj", "attn.c_proj", "mlp.down_proj", "mlp.c_proj"):
            module = block
            for part in name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not hasattr(module, "weight"):
                continue
            # nn.Linear stores [out_features, in_features]; out_features is the
            # residual stream, so the direction lives along the columns.
            _orthogonalize_matrix(module.weight, direction, rows=False)
            edited += 1
    return edited
