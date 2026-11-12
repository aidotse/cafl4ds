"""Unit tests for P1.1.2's layer-wise lenses (``cafl4ds.lenses``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cafl4ds.lenses import LENSES, fires, lens_values, signal_name, twin_effect
from cafl4ds.reading_protocol import fit_bands


def _features(seed: int, shift: float = 0.0) -> dict[str, torch.Tensor]:
    """Two layers of synthetic features; ``shift`` mixes fresh noise into the second."""
    g = torch.Generator().manual_seed(0)
    base = {"block01": torch.randn(60, 12, generator=g), "backbone": torch.randn(60, 8, generator=g)}
    noise = torch.randn(60, 8, generator=torch.Generator().manual_seed(seed))
    return {"block01": base["block01"], "backbone": base["backbone"] + shift * noise}


def test_lens_values_cover_every_lens_and_layer_and_localize_a_change() -> None:
    """One value per lens and layer; a change confined to one layer moves only that layer's drift."""
    anchor = _features(1)
    values = lens_values(_features(2, shift=1.0), anchor)
    assert set(values) == {signal_name(lens, layer) for lens in LENSES for layer in anchor}
    assert values["dcka_drift@block01"] == pytest.approx(0.0, abs=1e-6)
    assert values["procrustes_drift@block01"] == pytest.approx(0.0, abs=1e-6)
    assert values["dcka_drift@backbone"] > 0.1 and values["procrustes_drift@backbone"] > 0.1


def test_twin_effect_is_harm_signed_in_null_spread_units() -> None:
    """A drift rise reads positive; a RankMe rise reads negative (not harm); two-sided keeps the raw sign."""
    rng = np.random.default_rng(0)
    null = 5.0 + rng.normal(size=(5, 6))
    rank = fit_bands(null, "rankme@block03", span=1, threshold=3.0)
    sd = float(rank.sd[2])
    assert twin_effect(5.0 + 4 * sd, 5.0, rank, 2) == pytest.approx(-4.0)
    assert not fires(-4.0, rank) and fires(4.0, rank)
    twonn = fit_bands(null, "twonn@block03", span=1, threshold=3.0)
    assert twin_effect(5.0 + 4 * float(twonn.sd[2]), 5.0, twonn, 2) == pytest.approx(4.0)
    assert fires(-4.0, twonn) and not fires(2.0, twonn)
    drift = fit_bands(np.abs(null) / 10, "dcka_drift@block03", span=1, threshold=3.0)
    assert drift.transform == "log" and twin_effect(0.9, 0.3, drift, 2) > 0
    assert not fires(math.nan, drift)
