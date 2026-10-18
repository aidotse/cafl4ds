"""Unit tests for P1.1.3's CKA drift reads (``cafl4ds.cka_reads``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cafl4ds.cka_reads import gap_of_gaps, jolt, panel_read, sign_flip_max_t, stratum_signal, stratum_values
from cafl4ds.reading_protocol import direction, fit_bands


def test_stratum_values_localize_a_change_to_its_stratum() -> None:
    """Rewriting one stratum's rows moves that stratum's drift and leaves the others at zero."""
    g = torch.Generator().manual_seed(0)
    anchor = {"backbone": torch.randn(90, 16, generator=g)}
    groups = torch.arange(90) % 3
    moved = anchor["backbone"].clone()
    moved[groups == 2] = torch.randn(30, 16, generator=g)
    names = {0: "daytime", 1: "dawn/dusk", 2: "night", 3: "absent"}
    values = stratum_values({"backbone": moved}, anchor, groups, names)
    assert set(values) == {
        stratum_signal(lens, "backbone", s)
        for lens in ("cka_drift", "dcka_drift")
        for s in names.values()
        if s != "absent"
    }
    assert values["dcka_drift@backbone:daytime"] == pytest.approx(0.0, abs=1e-6)
    assert values["cka_drift@backbone:dawn/dusk"] == pytest.approx(0.0, abs=1e-6)
    assert values["dcka_drift@backbone:night"] > 0.5
    assert direction("dcka_drift@backbone:night") == 1  # a per-stratum read inherits drift's harmful sign


def test_jolt_counts_the_run_from_the_first_reading_and_the_settled_floor() -> None:
    """A skipped reading neither counts nor breaks the run; the settled floor is read past ``settle`` steps."""
    gap = np.array([6.0, math.nan, 4.5, 2.5, 1.0, 3.0, 0.5, -0.5, 1.5])
    steps = [65, 66, 67, 68, 69, 70, 75, 76, 77]
    out = jolt(gap, steps, threshold=3.8, settle=10)
    assert out["peak"] == 6.0 and out["peak_step"] == 65
    assert out["over_threshold"] == 2 and out["over_2sd"] == 3
    assert out["settled_abs_median"] == pytest.approx(0.5)


def test_gap_of_gaps_is_on_the_log_scale_in_null_spread_units() -> None:
    """Drift is read on the log scale: equal ratios cancel, and a doubled walk gap reads log 2 / sd."""
    null = np.exp(np.random.default_rng(0).normal(size=(5, 4)) * 0.3) * 0.01
    bands = fit_bands(null, "cka_drift", span=1, threshold=3.0)
    sd = float(bands.sd[3])
    assert gap_of_gaps(0.04, 0.02, 0.02, 0.01, bands, 3) == pytest.approx(0.0, abs=1e-2)
    assert gap_of_gaps(0.04, 0.01, 0.02, 0.01, bands, 3) == pytest.approx(math.log(2) / sd, rel=1e-2)


def test_sign_flip_max_t_reaches_its_floor_only_when_every_seed_agrees() -> None:
    """Five consistent seeds at one reader give p = 1/32 despite a noise reader; mixed signs do not."""
    rng = np.random.default_rng(1)
    clean = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    noise = rng.normal(size=5)
    out = sign_flip_max_t(np.column_stack([noise, clean]), ["noise", "clean"])
    assert out["reader"] == "clean" and out["n_positive"] == 5
    assert out["p"] == pytest.approx(1 / 32) and out["p_min"] == pytest.approx(1 / 32)
    mixed = sign_flip_max_t(np.column_stack([noise, clean * [1, 1, -1, 1, 1]]), ["noise", "mixed"])
    assert mixed["p"] > 1 / 32


def test_panel_read_fires_on_any_member_and_names_the_largest() -> None:
    """A state far above one layer's null fires the panel; at the null mean it stays quiet."""
    rng = np.random.default_rng(2)
    null = np.exp(rng.normal(size=(5, 3)) * 0.2) * 0.01
    rules = {s: fit_bands(null, s, span=1, threshold=3.0) for s in ("cka_drift@block04", "cka_drift@block05")}
    quiet = {s: float(np.exp(b.mean[1])) for s, b in rules.items()}
    assert not panel_read(quiet, rules, 1)["fired"]
    loud = {**quiet, "cka_drift@block05": 0.5}
    out = panel_read(loud, rules, 1)
    assert out["fired"] and out["signal"] == "cka_drift@block05" and out["max_z"] > 3.0
