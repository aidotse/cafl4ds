"""Unit tests for :mod:`cafl4ds.final_eval` — the end-of-run test-split probes."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cafl4ds.data.sources import SyntheticSource
from cafl4ds.data.streams import EvalSet
from cafl4ds.final_eval import FinalEvaluator, encode_batched, subsample_per_class
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.ssl.factory import build_simsiam

_CLASSES = 4


class _StubMethod(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    """A minimal stand-in exposing ``encode``: a flattened view of the image, or a constant."""

    def __init__(self, useful: bool) -> None:
        super().__init__()
        self.useful = useful
        self.scale = nn.Parameter(torch.ones(()))  # gives the module a device

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        flat = imgs.flatten(1) * self.scale
        return flat if self.useful else torch.ones_like(flat[:, :8])


def _separable(n_per_class: int, seed: int) -> EvalSet:
    """Images whose class sets one channel bright: trivially separable by a linear map."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(_CLASSES).repeat_interleave(n_per_class)
    images = 0.1 * torch.rand(len(labels), _CLASSES, 4, 4, generator=generator)
    images[torch.arange(len(labels)), labels] += 1.0
    return EvalSet(images, labels)


def test_encode_batched_matches_one_pass_and_restores_the_training_flag() -> None:
    """Batching is an implementation detail: same embeddings, and the caller's mode survives."""
    torch.manual_seed(0)
    method = build_simsiam(TinyViTEncoder(img_size=16, patch_size=8, in_chans=3, embed_dim=32, depth=2, num_heads=2))
    images = torch.rand(10, 3, 16, 16)
    method.train()
    batched = encode_batched(method, images, batch_size=3)
    assert method.training
    method.eval()
    assert torch.allclose(batched, method.encode(images), atol=1e-6)


def test_subsample_per_class_caps_each_class_reproducibly() -> None:
    """Each class keeps at most ``per_class``; the draw is seeded; ``None`` keeps everything."""
    labels = torch.tensor([0] * 10 + [1] * 3 + [2] * 6)
    picked = subsample_per_class(labels, per_class=4, seed=0)
    assert torch.bincount(labels[picked]).tolist() == [4, 3, 4]  # a short class keeps all it has
    assert torch.equal(picked, subsample_per_class(labels, per_class=4, seed=0))
    assert torch.equal(subsample_per_class(labels, None), torch.arange(len(labels)))


def test_probes_separate_a_useful_representation_from_a_useless_one() -> None:
    """The protocol must discriminate: near-perfect on separable features, chance on a constant."""
    evaluator = FinalEvaluator(_separable(30, seed=0), _separable(30, seed=1), knn_k=5)
    useful = evaluator.evaluate(_StubMethod(useful=True))
    useless = evaluator.evaluate(_StubMethod(useful=False))
    assert useful["knn_acc"] > 0.95 and useful["linear_acc"] > 0.95
    assert useless["knn_acc"] < 0.5 and useless["linear_acc"] < 0.5  # chance is 1 / 4


def test_from_sources_uses_the_train_source_for_support_and_the_test_source_for_scoring() -> None:
    """Support is subsampled from the train source; the test source is used whole."""
    train = SyntheticSource(num_classes=_CLASSES, per_class=20, img_size=16, seed=0)
    test = SyntheticSource(num_classes=_CLASSES, per_class=10, img_size=16, seed=1)
    evaluator = FinalEvaluator.from_sources(train, test, support_per_class=5)
    assert len(evaluator.support.labels) == 5 * _CLASSES
    assert torch.equal(evaluator.test.images, test.load()[0])


def test_write_report_records_the_gain(tmp_path: Path) -> None:
    """The report carries start, final, their difference, and the set sizes."""
    evaluator = FinalEvaluator(_separable(5, seed=0), _separable(5, seed=1))
    start, final = {"knn_acc": 0.25, "linear_acc": 0.30}, {"knn_acc": 0.40, "linear_acc": 0.50}
    evaluator.write_report(start, final, tmp_path / "final_eval.json")
    report = json.loads((tmp_path / "final_eval.json").read_text())
    assert report["gain"] == pytest.approx({"knn_acc": 0.15, "linear_acc": 0.20})
    assert report["support_size"] == report["test_size"] == 5 * _CLASSES
