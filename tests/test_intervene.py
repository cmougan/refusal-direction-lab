import torch

from refusal_lab.intervene import (
    ablate_direction,
    add_direction,
    orthogonalize_weights,
    project_out,
)


def test_project_out_removes_component():
    d = torch.randn(8)
    x = torch.randn(5, 8)
    assert torch.allclose(project_out(x, d) @ (d / d.norm()), torch.zeros(5), atol=1e-5)


def test_ablation_hook_zeroes_projection_at_every_layer(tiny_model):
    direction = torch.randn(32)
    ids = torch.randint(0, 64, (2, 10))
    with ablate_direction(tiny_model, direction):
        hidden = tiny_model(ids, output_hidden_states=True).hidden_states
    unit = direction / direction.norm()
    # Every hooked block output (all but the final, post-norm state) is clean.
    for h in hidden[:-1]:
        assert (h @ unit).abs().max() < 1e-4


def test_hooks_are_removed(tiny_model):
    ids = torch.randint(0, 64, (1, 6))
    before = tiny_model(ids).logits
    with ablate_direction(tiny_model, torch.randn(32)):
        pass
    with add_direction(tiny_model, torch.randn(32), layer=1, coeff=3.0):
        pass
    assert torch.allclose(before, tiny_model(ids).logits)


def test_weight_orthogonalization_matches_hook(tiny_model):
    direction = torch.randn(32)
    ids = torch.randint(0, 64, (2, 12))
    with ablate_direction(tiny_model, direction):
        hooked = tiny_model(ids).logits
    edited = orthogonalize_weights(tiny_model, direction)
    assert edited == 1 + 2 * 4
    assert torch.allclose(hooked, tiny_model(ids).logits, atol=1e-4)
