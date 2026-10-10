from __future__ import annotations

from dataclasses import dataclass, field


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LabConfig:
    """Everything the pipeline needs, in one place."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = field(default_factory=_default_device)
    dtype: str = "float32"

    n_train: int = 64
    n_val: int = 32
    batch_size: int = 8
    max_new_tokens: int = 48
    seed: int = 0

    # Positions (counted from the end of the formatted prompt) whose residual
    # stream we read. The last few post-instruction tokens carry most of the
    # refusal signal.
    positions: tuple[int, ...] = (-5, -4, -3, -2, -1)

    # Candidate layers are restricted to the middle of the network; the first
    # and last layers rarely give a clean, transferable direction.
    layer_fraction_range: tuple[float, float] = (0.2, 0.8)

    output_dir: str = "results"

    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)
