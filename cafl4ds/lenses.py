"""P1.1.2's layer-wise lenses — candidate label-free readers, read per encoder block.

The monitor's scalar instruments read one pooled embedding: they say *that* the representation
moved, not *where*. A lens reads every layer :func:`cafl4ds.block_features.block_features` captured,
under a name ``<lens>@<layer>`` the reading protocol (:mod:`cafl4ds.reading_protocol`) treats as an
instrument of its own — inheriting the base lens's harmful direction and reading scale:

* ``cka_drift`` — ``1 −`` linear CKA against the well (the monitor's drift, per layer);
* ``dcka_drift`` — the same with the debiased estimator (no high-dimension floor), one-sided;
* ``procrustes_drift`` — orthogonal Procrustes distance against the well (sees low-variance change);
* ``rankme`` — effective rank of the layer (a fall is harm, as for the monitor's RankMe);
* ``twonn`` — intrinsic dimension of the layer (TwoNN), read two-sided: no harmful sign is known.

A **twin effect** is how far a damaged arm sits from its undamaged twin at one checkpoint, in units
of the healthy null's seed spread there. The twin removes the branch's own confounds (the optimizer
restart, the walk's compression); the null's spread is the one a monitor without a twin would meet,
so the effect is conservative — a same-seed twin differs far less than two healthy seeds do.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cafl4ds import measurements
from cafl4ds.reading_protocol import Bands, apply_transform

LENSES = ("cka_drift", "dcka_drift", "procrustes_drift", "rankme", "twonn")


def signal_name(lens: str, layer: str) -> str:
    """The instrument name of ``lens`` read at ``layer`` (``dcka_drift@block07``)."""
    return f"{lens}@{layer}"


def lens_values(features: dict[str, torch.Tensor], anchor: dict[str, torch.Tensor]) -> dict[str, float]:
    """Every lens at every layer of one checkpoint, the drift lenses against ``anchor`` (the well).

    Args:
        features: ``layer -> [N, d]`` for the checkpoint read.
        anchor: ``layer -> [N, d]`` for the reference, on the same probe rows.

    Returns:
        ``<lens>@<layer> -> value`` for every lens in :data:`LENSES` and every layer of ``features``.

    Raises:
        KeyError: If ``anchor`` lacks one of the layers.
    """
    out: dict[str, float] = {}
    for layer, z in features.items():
        ref = anchor[layer]
        out[signal_name("cka_drift", layer)] = measurements.cka_drift(ref, z)
        out[signal_name("dcka_drift", layer)] = 1.0 - measurements.debiased_linear_cka(ref, z)
        out[signal_name("procrustes_drift", layer)] = measurements.procrustes_distance(ref, z)
        out[signal_name("rankme", layer)] = measurements.rankme(z)
        out[signal_name("twonn", layer)] = measurements.twonn_id(z)
    return out


def twin_effect(value: float, twin: float, bands: Bands, checkpoint: int) -> float:
    """The arm's distance from its twin at ``checkpoint``, in null seed-spread units, harm-signed.

    Read on the bands' scale (log for drift). Positive means worse than the twin for a one-sided
    instrument; a two-sided instrument keeps the raw sign (arm above twin = positive), and is called
    on its magnitude (:func:`fires`).

    Args:
        value: The arm's reading.
        twin: The twin's reading at the same checkpoint.
        bands: The instrument's null bands (its scale, spread and direction).
        checkpoint: Index of the checkpoint on the bands' grid.

    Returns:
        The signed effect (``nan`` if the null spread there is not positive).
    """
    x = apply_transform(np.array([value, twin]), bands.transform)
    sd = float(bands.sd[checkpoint])
    if not sd > 0:
        return math.nan
    return float((x[0] - x[1]) / sd) * (bands.direction or 1)


def fires(effect: float, bands: Bands) -> bool:
    """Whether a twin effect clears the panel threshold (on its magnitude for a two-sided instrument)."""
    size = abs(effect) if bands.direction == 0 else effect
    return bool(np.isfinite(size) and size > bands.threshold)
