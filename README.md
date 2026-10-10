# refusal-direction-lab

A small, readable replication of **"Refusal in Language Models Is Mediated by a Single
Direction"** (Arditi et al., 2024) for learning how safety fine-tuning is represented
inside open-weight chat models, and how fragile it is.

The pipeline:

1. **Collect activations.** Read the residual stream at the last few prompt tokens for
   harmful instructions (AdvBench, downloaded on demand) and harmless ones (Alpaca).
2. **Find the direction.** Take the difference of the two means at every (layer, position)
   and score each candidate by how well it separates the two sets (Cohen's d).
3. **Intervene.**
   - `ablate_direction`: an inference-time hook that projects the direction out of every
     layer. Reversible.
   - `add_direction`: the control experiment. Adding the direction should make the model
     refuse harmless prompts.
   - `orthogonalize_weights`: the same projection written into the weights, to show the two
     edits are equivalent.
4. **Evaluate.** Refusal rate before and after, KL divergence on harmless prompts (how much
   normal behaviour changed), and a check that the weight edit matches the hook.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # unit tests on a tiny random Llama, no downloads
refusal-lab --n-train 32 --n-val 16      # Qwen2.5-0.5B-Instruct, runs on CPU in minutes
```

AdvBench and HarmBench are gated on the Hugging Face Hub. Accept their terms and run
`huggingface-cli login` before the first real run.

The output in `results/` is `metrics.json`, `direction.pt` and `layer_scores.png`. Use
`--offline` to exercise the code path with neutral placeholder prompts and no dataset
download.

## Code map

| module | what it does |
|---|---|
| `data.py` | loads prompt sets and makes disjoint train/val splits |
| `model.py` | loads the model, applies the chat template, finds decoder blocks |
| `activations.py` | collects residual-stream activations |
| `direction.py` | difference-in-means, separation scores, layer selection |
| `intervene.py` | ablation hook, addition hook, weight orthogonalization |
| `evaluate.py` | refusal detection, KL divergence |
| `cli.py` | the end-to-end experiment |

## Scope and limits

This is a research and learning tool. Be clear about what it does:

- The ablation produces a model that refuses far less often. That is the result being
  studied, and anyone who runs the code gets that model in memory.
- The package's guardrails are small. It writes only aggregate metrics for harmful prompts
  (no completions), and it has no command that saves or uploads edited weights. Nothing
  stops a user from calling `model.save_pretrained` themselves. Treat any edited model as a
  research artefact, keep it local, and don't deploy or redistribute it.
- The keyword refusal detector is crude. For serious evaluation, use a classifier such as
  HarmBench's or LlamaGuard.

## References

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, 2024. arXiv:2406.11717
- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* (AdvBench), 2023
