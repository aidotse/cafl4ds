"""Tests for the P1.2 readout ladder and adapted-vs-frozen contrast (:mod:`cafl4ds.adaptation`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from cafl4ds import adaptation
from cafl4ds.adaptation import FineTuneRecipe, HeldOutPool


def _pool(n: int = 12) -> HeldOutPool:
    cats = [{"car": 2, "pedestrian": 1} if i % 3 == 0 else {"car": 1} for i in range(n)]
    return HeldOutPool(
        images=torch.zeros(n, 3, 4, 4),
        scene=torch.arange(n) % 2,
        stratum=torch.arange(n) % 3,
        regime=torch.arange(n) % 4,
        categories=cats,
    )


def test_presence_labels_mark_frames_with_any_box_of_the_category() -> None:
    """Presence labels mark frames with any box of the category."""
    labels = adaptation.task_labels(_pool(), "presence:pedestrian")
    assert labels.tolist() == [1, 0, 0] * 4
    assert adaptation.task_labels(_pool(), "scene").tolist() == [0, 1] * 6


def test_presence_needs_categories_and_unknown_tasks_raise() -> None:
    """Presence needs categories and unknown tasks raise."""
    pool = _pool()
    pool.categories = None
    with pytest.raises(ValueError, match="categories"):
        adaptation.task_labels(pool, "presence:car")
    with pytest.raises(ValueError, match="unknown task"):
        adaptation.task_labels(_pool(), "weather")


def test_split_pool_is_disjoint_exhaustive_and_seed_fixed() -> None:
    """Split pool is disjoint exhaustive and seed fixed."""
    fit, query = adaptation.split_pool(100, 0.4, seed=3)
    assert fit.size == 40 and query.size == 60
    assert np.intersect1d(fit, query).size == 0
    assert np.array_equal(np.sort(np.concatenate([fit, query])), np.arange(100))
    again, _ = adaptation.split_pool(100, 0.4, seed=3)
    assert np.array_equal(fit, again)
    with pytest.raises(ValueError):
        adaptation.split_pool(10, 1.0, seed=0)


def test_balanced_indices_caps_each_class_within_the_candidates() -> None:
    """Balanced indices caps each class within the candidates."""
    labels = np.array([0] * 10 + [1] * 3)
    candidates = np.arange(1, 13)  # frame 0 is excluded
    keep = adaptation.balanced_indices(labels, candidates, per_class=4, seed=0)
    assert np.isin(keep, candidates).all()
    assert (labels[keep] == 0).sum() == 4 and (labels[keep] == 1).sum() == 3


def test_balanced_accuracy_is_not_inflated_by_the_majority_class() -> None:
    """Balanced accuracy is not inflated by the majority class."""
    labels = np.array([0] * 9 + [1])
    always_zero = (labels == 0).astype(float)  # correct on every 0, wrong on the 1
    assert adaptation.balanced_accuracy(always_zero, labels) == pytest.approx(0.5)


def test_bootstrap_delta_is_zero_and_covers_zero_for_identical_encoders() -> None:
    """Bootstrap delta is zero and covers zero for identical encoders."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 600)
    ref = (rng.random(600) < 0.6).astype(float)
    out = adaptation.bootstrap_delta(np.stack([ref] * 5), ref, labels, n_boot=500)
    assert out["delta"] == pytest.approx(0.0)
    assert out["lo"] <= 0.0 <= out["hi"]


def test_bootstrap_delta_recovers_a_clear_gain_with_a_small_p() -> None:
    """Bootstrap delta recovers a clear gain with a small p."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 2000)
    ref = (rng.random(2000) < 0.55).astype(float)
    adapted = np.stack([(rng.random(2000) < 0.65).astype(float) for _ in range(5)])
    out = adaptation.bootstrap_delta(adapted, ref, labels, n_boot=1000)
    assert out["delta"] == pytest.approx(0.10, abs=0.03)
    assert out["lo"] > 0.0 and out["p_one_sided"] < 0.01


def test_bootstrap_delta_widens_with_seed_spread() -> None:
    """Seeds that disagree widen the interval: the seed level is resampled, not only the images."""
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 3000)
    ref = (rng.random(3000) < 0.6).astype(float)
    tight = np.stack([(rng.random(3000) < 0.62).astype(float) for _ in range(5)])
    spread = np.stack([(rng.random(3000) < p).astype(float) for p in (0.52, 0.57, 0.62, 0.67, 0.72)])
    w_tight = np.subtract(*[adaptation.bootstrap_delta(tight, ref, labels, n_boot=800)[k] for k in ("hi", "lo")])
    w_spread = np.subtract(*[adaptation.bootstrap_delta(spread, ref, labels, n_boot=800)[k] for k in ("hi", "lo")])
    assert w_spread > 1.5 * w_tight


def test_bootstrap_delta_holds_its_level_under_the_null() -> None:
    """With adapted and reference drawn from one accuracy, a one-sided 5% test rejects about 5% of the time."""
    rng = np.random.default_rng(3)
    rejections = 0
    trials = 200
    for t in range(trials):
        labels = rng.integers(0, 3, 400)
        ref = (rng.random(400) < 0.6).astype(float)
        adapted = np.stack([(rng.random(400) < 0.6).astype(float) for _ in range(5)])
        rejections += adaptation.bootstrap_delta(adapted, ref, labels, n_boot=300, seed=t)["p_one_sided"] <= 0.05
    assert rejections / trials < 0.10


def test_bootstrap_delta_rejects_misaligned_inputs() -> None:
    """Bootstrap delta rejects misaligned inputs."""
    with pytest.raises(ValueError):
        adaptation.bootstrap_delta(np.ones((2, 5)), np.ones(4), np.zeros(5, dtype=int))


def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    """Holm steps down and stops at the first failure."""
    assert adaptation.holm({"a": 0.01, "b": 0.04}, alpha=0.05) == {"a": True, "b": True}
    assert adaptation.holm({"a": 0.03, "b": 0.04}, alpha=0.05) == {"a": False, "b": False}
    assert adaptation.holm({"a": 0.02, "b": 0.30}, alpha=0.05) == {"a": True, "b": False}


def test_scene_overlap_reads_a_task_the_context_fully_determines() -> None:
    """Scene overlap reads a task the context fully determines."""
    rng = np.random.default_rng(4)
    context = np.stack([rng.integers(0, 3, 300), rng.integers(0, 2, 300)], axis=1)
    labels = (context[:, 0] == 1).astype(int)
    fit, query = adaptation.split_pool(300, 0.5, seed=0)
    assert adaptation.scene_overlap(context, labels, fit, query) == pytest.approx(1.0)
    noise = rng.integers(0, 2, 300)
    assert adaptation.scene_overlap(context, noise, fit, query) < 0.65


class _Embed(nn.Module):  # type: ignore[misc]  # nn.Module is Any without torch stubs (mypy hook env)
    embed_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def embed(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.proj(imgs.mean(dim=(2, 3)))


def test_fine_tune_learns_a_separable_task_and_leaves_the_encoder_untouched() -> None:
    """Fine tune learns a separable task and leaves the encoder untouched."""
    torch.manual_seed(0)
    labels = torch.arange(64) % 2
    images = torch.rand(64, 3, 4, 4) * 0.2 + labels[:, None, None, None].float() * 0.8
    encoder = _Embed()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    pred = adaptation.fine_tune_predict(
        encoder,
        images[:32],
        labels[:32],
        images[32:],
        FineTuneRecipe(lr=1e-2, steps=60, batch_size=8),
        seed=0,
        device="cpu",
    )
    assert (pred == labels[32:].numpy()).mean() == 1.0
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in before.items())


def test_frozen_correctness_returns_per_query_rows_for_both_probes() -> None:
    """Frozen correctness returns per query rows for both probes."""
    rng = np.random.default_rng(5)
    labels = np.repeat([0, 1], 50)
    features = rng.normal(size=(100, 8)).astype(np.float32) + labels[:, None] * 3.0
    fit, query = adaptation.split_pool(100, 0.5, seed=1)
    out = adaptation.frozen_correctness(features, labels, fit, query, knn_k=5)
    assert set(out) == {"knn", "linear"}
    assert out["linear"].shape == (query.size,) and out["linear"].mean() > 0.95
