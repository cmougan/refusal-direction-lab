import torch

from refusal_lab.data import split_prompts
from refusal_lab.evaluate import is_refusal, mean_kl, refusal_rate


def test_refusal_detection():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert not is_refusal("Sure! Here is a poem about the sea.")
    assert refusal_rate(["I cannot do that.", "Here you go."]) == 0.5


def test_kl_zero_for_identical():
    lp = torch.log_softmax(torch.randn(4, 10), dim=-1)
    assert abs(mean_kl(lp, lp)) < 1e-6


def test_split_is_disjoint():
    train, val = split_prompts([str(i) for i in range(20)], 10, 5)
    assert not set(train) & set(val)
