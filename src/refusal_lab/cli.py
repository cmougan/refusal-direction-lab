"""End-to-end experiment: find the direction, ablate it, report aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from refusal_lab.activations import collect_residuals
from refusal_lab.config import LabConfig
from refusal_lab.data import load_harmful, load_harmless, split_prompts
from refusal_lab.direction import select_direction
from refusal_lab.evaluate import mean_kl, next_token_logprobs, refusal_rate
from refusal_lab.generate import generate
from refusal_lab.intervene import ablate_direction, add_direction, orthogonalize_weights
from refusal_lab.model import load_model


def plot_scores(scores: torch.Tensor, positions: tuple[int, ...], best_layer: int, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, pos in enumerate(positions):
        ax.plot(scores[:, i].numpy(), label=f"pos {pos}")
    ax.axvline(best_layer, color="k", ls="--", lw=1, label="selected")
    ax.set_xlabel("layer")
    ax.set_ylabel("separation (Cohen's d)")
    ax.set_title("How well each layer separates harmful vs harmless prompts")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(cfg: LabConfig, offline: bool = False, check_weights: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg)
    harmful_train, harmful_val = split_prompts(
        load_harmful(offline=offline), cfg.n_train, cfg.n_val, cfg.seed
    )
    harmless_train, harmless_val = split_prompts(
        load_harmless(offline=offline), cfg.n_train, cfg.n_val, cfg.seed
    )

    acts_harmful = collect_residuals(model, tokenizer, harmful_train, cfg)
    acts_harmless = collect_residuals(model, tokenizer, harmless_train, cfg)
    best, _, scores = select_direction(acts_harmful, acts_harmless, cfg)
    torch.save(
        {"layer": best.layer, "position": best.position, "vector": best.vector},
        out_dir / "direction.pt",
    )
    plot_scores(scores, cfg.positions, best.layer, out_dir / "layer_scores.png")

    base_lp = next_token_logprobs(model, tokenizer, harmless_val, cfg)
    metrics = {
        "model": cfg.model_name,
        "direction": {"layer": best.layer, "position": best.position, "score": best.score},
        "baseline": {
            "harmful_refusal_rate": refusal_rate(generate(model, tokenizer, harmful_val, cfg)),
            "harmless_refusal_rate": refusal_rate(generate(model, tokenizer, harmless_val, cfg)),
        },
    }

    with ablate_direction(model, best.vector):
        metrics["ablated"] = {
            "harmful_refusal_rate": refusal_rate(generate(model, tokenizer, harmful_val, cfg)),
            "harmless_kl": mean_kl(
                base_lp, next_token_logprobs(model, tokenizer, harmless_val, cfg)
            ),
        }

    # Control: adding the direction should induce refusals on harmless prompts.
    mean_proj = float((acts_harmful[:, best.layer, -1] @ best.vector).mean())
    with add_direction(model, best.vector, best.layer, coeff=mean_proj):
        metrics["added"] = {
            "harmless_refusal_rate": refusal_rate(generate(model, tokenizer, harmless_val, cfg)),
        }

    if check_weights:
        # The weight edit should reproduce the hook-based ablation. The edited
        # model only exists in memory for this comparison and is not saved.
        orthogonalize_weights(model, best.vector)
        ortho_lp = next_token_logprobs(model, tokenizer, harmless_val, cfg)
        with ablate_direction(model, best.vector):
            hook_lp = next_token_logprobs(model, tokenizer, harmless_val, cfg)
        metrics["weights_vs_hook_max_abs_diff"] = float((ortho_lp - hook_lp).abs().max())

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="refusal-lab", description=__doc__)
    p.add_argument("--model", default=LabConfig.model_name)
    p.add_argument("--n-train", type=int, default=LabConfig.n_train)
    p.add_argument("--n-val", type=int, default=LabConfig.n_val)
    p.add_argument("--batch-size", type=int, default=LabConfig.batch_size)
    p.add_argument("--max-new-tokens", type=int, default=LabConfig.max_new_tokens)
    p.add_argument("--output-dir", default=LabConfig.output_dir)
    p.add_argument("--device", default=None)
    p.add_argument("--offline", action="store_true", help="use neutral stand-in prompts")
    p.add_argument("--skip-weight-check", action="store_true")
    args = p.parse_args(argv)

    cfg = LabConfig(
        model_name=args.model,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )
    if args.device:
        cfg.device = args.device
    print(
        json.dumps(
            run(cfg, offline=args.offline, check_weights=not args.skip_weight_check), indent=2
        )
    )


if __name__ == "__main__":
    main()
