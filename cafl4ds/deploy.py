"""The P1.0.2 deployment-prototype harness — multivariate, trust-annotated health logging.

Where the P1.0.0 harness (:mod:`cafl4ds.harness`) logs a *directional* smoke on toy data, the
deployment prototype logs the **full multivariate health vector** of one config-selected backbone on
the BDD regime diet, partitioned into two self-describing channels — a **label-free** vector (what a
real deployment would see) and a **labelled canary** (kNN / linear probes, ground truth available
because we control the eval set) — with every signal annotated by its Phase-0 calibrated status
(:mod:`cafl4ds.health_trust`) so the corpus is safe to mine without the code.

It reads **no degradation**. The acceptance bar is a **two-tier validation**: Tier A (the anyone-can-
run wiring check) asserts the pipeline is wired — every configured signal present at every checkpoint,
both channels populated, the trust map complete, nothing non-finite, memory flat; Tier B (our
sanity-of-readings pass) confirms the readings are plausible. This module owns the report the harness
writes; the arm execution reuses :mod:`cafl4ds.harness` untouched.
"""

from __future__ import annotations

from typing import Any

from cafl4ds.harness import Arm, all_finite
from cafl4ds.health_trust import CANARY_SIGNALS, Backbone, annotate

# The deploy-corpus schema version (independent of the P1.0.0 comparison schema).
SCHEMA_VERSION = 1

# Health-record bookkeeping fields that are not instrument signals.
_BOOKKEEPING = frozenset({"step", "era", "loss", "run", "series"})


def emitted_signals(health: list[dict[str, Any]]) -> list[str]:
    """The instrument signals present across a health series (bookkeeping fields excluded).

    Args:
        health: The arm's health records (one per checkpoint).

    Returns:
        The sorted union of signal keys logged over the checkpoints.
    """
    keys: set[str] = set()
    for record in health:
        keys |= {k for k in record if k not in _BOOKKEEPING}
    return sorted(keys)


def split_channels(signals: list[str]) -> dict[str, list[str]]:
    """Partition signals into the label-free vector and the labelled canary channel.

    Args:
        signals: The emitted signal keys.

    Returns:
        ``{"label_free": [...], "canary": [...]}`` — the canary holds the labelled probes
        (``knn_acc`` / ``linear_acc``), the label-free vector holds the rest.
    """
    canary = sorted(s for s in signals if s in CANARY_SIGNALS)
    label_free = sorted(s for s in signals if s not in CANARY_SIGNALS)
    return {"label_free": label_free, "canary": canary}


def project_health(health: list[dict[str, Any]], keep: list[str] | None) -> list[dict[str, Any]]:
    """Restrict each health record to the configured signals (plus bookkeeping), or pass through.

    This is the log/un-log lever: ``keep=None`` logs everything the monitor emitted; a list keeps
    only those signals (the bookkeeping fields are always retained).

    Args:
        health: The arm's health records.
        keep: The signals to keep, or ``None`` to keep all.

    Returns:
        The (possibly projected) health records.
    """
    if keep is None:
        return health
    allowed = _BOOKKEEPING | set(keep)
    return [{k: v for k, v in record.items() if k in allowed} for record in health]


def channel_completeness(health: list[dict[str, Any]], expected: list[str]) -> dict[str, Any]:
    """Check every expected signal is present (non-``None``) at every checkpoint.

    Args:
        health: The (projected) health records.
        expected: The signals that must appear at every checkpoint.

    Returns:
        The completeness report: the ``complete`` verdict, the expected set, and any per-step gaps.
    """
    missing: dict[str, list[str]] = {}
    for record in health:
        absent = [s for s in expected if record.get(s) is None]
        if absent:
            missing[str(int(record["step"]))] = absent
    return {
        "complete": not missing,
        "expected": sorted(expected),
        "missing": missing,
        "num_checkpoints": len(health),
    }


def build_deploy_report(
    *,
    config_header: dict[str, Any],
    family: Backbone,
    live: Arm,
    expected_signals: list[str],
    pc: Arm | None = None,
    b5: Arm | None = None,
    leak: dict[str, Any] | None = None,
    log_signals: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the deploy corpus — the config, the trust-annotated channels, the arms, the validation.

    Args:
        config_header: The run's factor levels (backbone, diet, seed, device, …).
        family: The backbone family (drives the trust annotation).
        live: The live arm (the system under test — its health series is the corpus).
        expected_signals: The signals Tier-A completeness requires at every checkpoint.
        pc: Optional manufactured-pathology gate arm.
        b5: Optional frozen-floor gate arm.
        leak: Optional long-horizon leak report (adds the flat-memory check to the verdict).
        log_signals: Optional restriction of the logged signals (``None`` logs all emitted).

    Returns:
        The deploy-corpus dict, ready to serialize.
    """
    arms = [a for a in (live, pc, b5) if a is not None]
    projected = {a.role: project_health(a.health, log_signals) for a in arms}
    live_health = projected[live.role]

    signals = emitted_signals(live_health)
    channels = split_channels(signals)
    trust = annotate(signals, family)
    completeness = channel_completeness(live_health, expected_signals)

    finite = all_finite(*arms)
    channels_present = bool(channels["label_free"] and channels["canary"])
    trust_complete = all(any(v["status"] != "uncalibrated" for v in trust.get(s, [])) for s in expected_signals)

    checks = [finite, channels_present, completeness["complete"], trust_complete]
    if leak is not None:
        checks.append(bool(leak["flat"]))
    validation = {
        "finite": finite,
        "channels_present": channels_present,
        "channel_complete": completeness["complete"],
        "trust_complete": trust_complete,
        "memory_flat": (bool(leak["flat"]) if leak is not None else None),
        "passed": all(checks),
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": "P1.0.2",
        "config": config_header,
        "backbone_family": family.value,
        "channels": channels,
        "trust": trust,
        "arms": {
            role: {"run_name": next(a.name for a in arms if a.role == role), "health": h}
            for role, h in projected.items()
        },
        "completeness": completeness,
        "validation": validation,
    }
    if leak is not None:
        report["leak_check"] = leak
    return report
