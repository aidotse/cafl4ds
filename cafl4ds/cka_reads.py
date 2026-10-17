"""P1.1.3's reads of CKA drift — can it tell information loss from benign adaptation?

P1.1.2 left CKA drift against the well as the one label-free MAE signal that tracks damage without
reading the lever, but it also fires on healthy walks and on optimizer restarts. The reads here:

* **per stratum** — drift on one time-of-day stratum's probe rows only (:func:`stratum_values`), so
  a monitor can read the regimes a walk has left behind, where forgetting should live. Named
  ``<lens>@<layer>:<stratum>``, so the reading protocol treats each as an instrument inheriting the
  base lens's harmful direction and scale;
* **the restart jolt** — how big and how long a fresh optimizer's kick to drift is (:func:`jolt`);
* **a gap of gaps** — the walk's extra drift over IID at one learning rate against another, in null
  seed-spread units (:func:`gap_of_gaps`), for JE's naturally degrading lr-1e-4 walk;
* **a paired sign-flip test** — does any lens, at any layer, separate a forgetting arm from its
  healthy twin across seeds, with the search over lenses and layers paid for inside the test
  (:func:`sign_flip_max_t`).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import torch

from cafl4ds import measurements
from cafl4ds.lenses import signal_name
from cafl4ds.reading_protocol import Bands, apply_transform


def stratum_signal(lens: str, layer: str, stratum: str) -> str:
    """The instrument name of ``lens`` read at ``layer`` on one stratum's rows (``dcka_drift@backbone:night``)."""
    return signal_name(lens, f"{layer}:{stratum}")


def stratum_values(
    features: dict[str, torch.Tensor],
    anchor: dict[str, torch.Tensor],
    groups: torch.Tensor,
    names: dict[int, str],
    layers: tuple[str, ...] = ("backbone",),
) -> dict[str, float]:
    """Plain and debiased CKA drift against ``anchor`` on each stratum's rows, at each of ``layers``.

    Args:
        features: ``layer -> [N, d]`` for the checkpoint read.
        anchor: ``layer -> [N, d]`` for the reference, on the same rows.
        groups: ``[N]`` stratum id per row.
        names: Stratum id -> name (ids absent from ``groups`` are skipped).
        layers: The layers to read.

    Returns:
        ``<lens>@<layer>:<stratum> -> value`` for ``cka_drift`` and ``dcka_drift``.
    """
    out: dict[str, float] = {}
    for layer in layers:
        for gid, name in names.items():
            rows = groups == gid
            if not bool(rows.any()):
                continue
            ref, z = anchor[layer][rows], features[layer][rows]
            out[stratum_signal("cka_drift", layer, name)] = measurements.cka_drift(ref, z)
            out[stratum_signal("dcka_drift", layer, name)] = 1.0 - measurements.debiased_linear_cka(ref, z)
    return out


def _run(gap: np.ndarray, level: float) -> int:
    """Consecutive readings from the first above ``level`` (a skipped, nan reading neither counts nor breaks)."""
    n = 0
    for g in gap:
        if not np.isfinite(g):
            continue
        if g <= level:
            break
        n += 1
    return n


def jolt(gap: np.ndarray, steps: list[int], threshold: float, settle: int = 10) -> dict[str, Any]:
    """Size and duration of a jolt, from a harm-signed gap trajectory starting at the jolt.

    Args:
        gap: Per reading, the jolted run minus its reference, in null seed-spread units (nan = skipped).
        steps: The step of each reading.
        threshold: The panel threshold the gap is also counted against.
        settle: Readings from this many steps after the first on count as settled.

    Returns:
        ``peak`` and ``peak_step``; ``over_threshold`` / ``over_2sd`` — consecutive readings from the
        first above each level; ``settled_abs_median`` — the median ``|gap|`` once settled, the floor
        two runs of the same seed differ by.
    """
    g = np.asarray(gap, dtype=float)
    i = int(np.nanargmax(g))
    tail = np.abs(g[np.asarray(steps) >= steps[0] + settle])
    return {
        "peak": float(g[i]),
        "peak_step": int(steps[i]),
        "over_threshold": _run(g, threshold),
        "over_2sd": _run(g, 2.0),
        "settled_abs_median": float(np.nanmedian(tail)) if np.isfinite(tail).any() else math.nan,
    }


def gap_of_gaps(walk_hi: float, iid_hi: float, walk_lo: float, iid_lo: float, bands: Bands, checkpoint: int) -> float:
    """(walk − IID) at one setting minus (walk − IID) at another, on the bands' scale, in null seed-spread units.

    Harm-signed by the bands' direction, so positive means the first setting's walk is worse relative to
    its IID control than the second's.
    """
    x = apply_transform(np.array([walk_hi, iid_hi, walk_lo, iid_lo]), bands.transform)
    sd = float(bands.sd[checkpoint])
    if not sd > 0:
        return math.nan
    return float(((x[0] - x[1]) - (x[2] - x[3])) / sd) * (bands.direction or 1)


def _paired_t(d: np.ndarray) -> np.ndarray:
    """Paired t per column of ``[seeds, readers]`` (nan where the differences do not vary)."""
    n = d.shape[0]
    sd = d.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, d.mean(axis=0) / (sd / math.sqrt(n)), np.nan)


def sign_flip_max_t(diffs: np.ndarray, readers: list[str]) -> dict[str, Any]:
    """Exact paired sign-flip test of the maximum paired t over readers (the search paid for inside).

    Under the null (no arm effect) each seed's difference is as likely flipped as not, so every one of
    the ``2^n`` sign patterns is equally likely. The statistic is the largest paired t over readers;
    its p-value is the share of patterns whose largest t reaches the observed one. Taking the maximum
    inside each pattern prices the search over lenses and layers into the p-value.

    Args:
        diffs: ``[seeds, readers]`` — forgetting arm minus healthy arm, harm-signed (positive = forgetting worse).
        readers: The reader name of each column.

    Returns:
        ``t_max`` and the reader attaining it, ``p`` (smallest possible ``1 / 2^n``), the reader's
        per-seed differences and how many are positive.
    """
    d = np.asarray(diffs, dtype=float)
    observed = _paired_t(d)
    best = int(np.nanargmax(observed))
    t_max = float(observed[best])
    n = d.shape[0]
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        t = _paired_t(d * np.asarray(signs)[:, None])
        hits += int(np.nanmax(t) >= t_max - 1e-12)
    return {
        "t_max": t_max,
        "reader": readers[best],
        "p": hits / 2**n,
        "p_min": 1 / 2**n,
        "diffs": d[:, best].tolist(),
        "n_positive": int((d[:, best] > 0).sum()),
    }
