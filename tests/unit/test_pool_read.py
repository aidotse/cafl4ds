"""Unit tests for the P1.1.0 tight competence read (``cafl4ds.pool_read``)."""

from __future__ import annotations

import torch
from torch import nn

from cafl4ds.data.streams import EvalSet
from cafl4ds.pool_read import balanced_accuracy, balanced_subsample, frozen, probe_read


class _Flatten(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A parameter-light encoder exposing ``embed`` (with a dropout so eval mode is observable)."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.5)

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.drop(imgs.flatten(1))


def test_balanced_accuracy_is_mean_per_class_recall() -> None:
    """A majority-class guess scores chance under balanced accuracy, however imbalanced the labels."""
    truth = [0] * 8 + [1] * 2
    assert balanced_accuracy([0] * 10, truth) == 0.5
    assert balanced_accuracy(truth, truth) == 1.0


def test_balanced_subsample_caps_every_class_without_replacement() -> None:
    """Each class contributes at most ``per_class`` distinct images; a scarce class gives all it has."""
    labels = torch.tensor([0] * 10 + [1] * 3)
    pool = EvalSet(torch.arange(13, dtype=torch.float32).view(13, 1, 1, 1), labels)
    fit = balanced_subsample(pool, per_class=5, seed=0)
    assert torch.bincount(fit.labels).tolist() == [5, 3]
    assert fit.images.flatten().unique().numel() == 8
    assert torch.equal(fit.images, balanced_subsample(pool, per_class=5, seed=0).images)


def test_frozen_reads_in_eval_mode_without_grad_and_restores_training() -> None:
    """The frozen read is deterministic (dropout off), gradient-free, and leaves the encoder training."""
    encoder = _Flatten().train()
    imgs = torch.ones(600, 1, 2, 2, requires_grad=True)
    z = frozen(encoder)(imgs)
    assert z.shape == (600, 4) and not z.requires_grad
    assert torch.equal(z, torch.ones(600, 4)) and encoder.training


def test_probe_read_separates_separable_classes() -> None:
    """Two well-separated clusters read perfectly on both probes."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.tensor([0, 1] * 20)
    images = torch.eye(2)[labels].view(-1, 2, 1, 1) + 0.05 * torch.randn(40, 2, 1, 1, generator=generator)
    scores = probe_read(frozen(_Flatten()), EvalSet(images[:20], labels[:20]), EvalSet(images[20:], labels[20:]), 3)
    assert scores == {"knn_bacc": 1.0, "linear_bacc": 1.0}
