"""Prompt sets.

Harmful instructions come from the public jailbreak benchmarks that safety
research is evaluated on (AdvBench / HarmBench); harmless instructions come
from Alpaca. Nothing harmful is vendored into this repository -- the datasets
are downloaded on demand, and a small neutral stand-in is used offline so the
pipeline and the tests still run without network access.
"""

from __future__ import annotations

import random

HARMFUL_DATASETS = {
    "advbench": ("walledai/AdvBench", "train", "prompt"),
    "harmbench": ("walledai/HarmBench", "train", "prompt"),
}
HARMLESS_DATASET = ("tatsu-lab/alpaca", "train", "instruction")

# Offline stand-ins: deliberately bland, so the code path can be exercised
# without the real benchmark. They will not actually trigger refusals.
_OFFLINE_HARMFUL = [f"Placeholder disallowed request number {i}." for i in range(128)]
_HARMLESS_TOPICS = [
    "the sea",
    "autumn leaves",
    "a city at night",
    "a lighthouse",
    "mountains",
    "coffee",
    "a rainy day",
    "the moon",
    "friendship",
    "a garden",
    "trains",
    "the desert",
    "winter mornings",
    "a library",
    "rivers",
    "the stars",
]
_HARMLESS_TEMPLATES = [
    "Write a short poem about {}.",
    "Give me three fun facts about {}.",
    "Describe {} in two sentences.",
    "Suggest a title for a story about {}.",
    "Explain why people enjoy {}.",
    "List three words associated with {}.",
    "Write a haiku about {}.",
    "Summarise what makes {} interesting.",
]
_OFFLINE_HARMLESS = [t.format(topic) for t in _HARMLESS_TEMPLATES for topic in _HARMLESS_TOPICS]


def _load_hf(repo: str, split: str, column: str) -> list[str]:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        ds = load_dataset(repo, split=split)
    except DatasetNotFoundError as err:
        raise RuntimeError(
            f"Could not load {repo}. It may be gated: accept its terms on the Hugging Face Hub "
            "and run `huggingface-cli login` (or set HF_TOKEN), or use --offline."
        ) from err
    return [row[column] for row in ds if row[column]]


def load_harmful(name: str = "advbench", offline: bool = False) -> list[str]:
    if offline:
        return list(_OFFLINE_HARMFUL)
    repo, split, column = HARMFUL_DATASETS[name]
    return _load_hf(repo, split, column)


def load_harmless(offline: bool = False, limit: int = 2048) -> list[str]:
    if offline:
        return list(_OFFLINE_HARMLESS)
    repo, split, column = HARMLESS_DATASET
    rows = _load_hf(repo, split, column)
    return rows[:limit]


def split_prompts(
    prompts: list[str], n_train: int, n_val: int, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Disjoint train/validation samples: the direction must generalise."""
    pool = list(prompts)
    random.Random(seed).shuffle(pool)
    if len(pool) < n_train + n_val:
        raise ValueError(f"need {n_train + n_val} prompts, dataset has {len(pool)}")
    return pool[:n_train], pool[n_train : n_train + n_val]
