"""Integration regression test for the FORGETTING positive control (audit P0.3 §E, E1 + E4).

The unit tests validate the forgetting *pieces* — BWT/FM known-answers (``test_eval.py``), the
MAE recon path (``test_models.py``), the probes/CKA (``test_measurements.py``) — but the
*behavioural wiring* of the forgetting harness (``scripts/positive_control_forgetting.py``) had no
regression guard. ``test_positive_control.py`` covers the *collapse* harness; this is its forgetting
analogue. The P0.3 code-validity audit (``docs/experiments/audits/P0.3.md`` §E) flagged that a future
edit silently mis-wiring an arm (a swapped ``replay`` slot, a broken paired seed, a mis-signed recon
gap) would still pass CI.

This drives the **real harness** on the fast synthetic source and asserts the fingerprint a mis-wire
would break — *without* re-running the calibration (toy MAE does not forget; that needs the escalated
HPU regime). The load-bearing invariants checked here:

* **paired-seed phase A is bit-identical across arms** — both arms reset the seed and train an
  identical phase A, so post-A task-A accuracy (``R00``), the init probe, and the post-A recon are
  equal across the PC and healthy arms (a broken seed reset would desync them);
* **arm-slot toggles are correct** — ``replay=False`` is the PC, ``replay=True`` is healthy;
* **the metric wiring is right** — ``BWT = R10 − R00`` on the recorded matrix, and the reported recon
  gap is ``pc_recon_a_rise − healthy_recon_a_rise`` (a mis-sign would flip it);
* **the supervised vehicle runs** — the same invariants through ``_run_arm_supervised`` /
  ``SupervisedMethod`` (otherwise 0 %-covered), with the MAE-native recon readouts reported as ``NaN``.

E4: the ``_guard_from_scratch_encoder`` footgun guard raises when a checkpoint-loading encoder would be
silently kept under a from-scratch arm, and is inert otherwise.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"

# Fast, network-free, deterministic: the synthetic 4-class source split into two 2-class tasks, a
# tiny ViT + tiny MAE decoder, a short horizon, and a small kNN probe. This is a *wiring* guard, so
# the regime need only run — not fire.
_BASE = [
    "data=synthetic",
    "data.per_class=16",
    "img_size=16",
    "encoder.embed_dim=32",
    "encoder.depth=1",
    "encoder.num_heads=2",
    "ssl.decoder_dim=16",
    "ssl.decoder_depth=1",
    "ssl.decoder_heads=2",
    "epochs_a=2",
    "epochs_b=2",
    "batch_size=8",
    "seed=0",
    "task_a_classes=[0,1]",
    "task_b_classes=[2,3]",
    "support_per_class=4",
    "query_per_class=4",
    "probe=knn",
    "knn_k=3",
    "recon_masks=2",
]


def _load_harness() -> ModuleType:
    """Import ``scripts/positive_control_forgetting.py`` as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "positive_control_forgetting.py"
    spec = importlib.util.spec_from_file_location("positive_control_forgetting_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(overrides: list[str]) -> DictConfig:
    """Compose the ``positive_control_forgetting`` config with the given overrides."""
    GlobalHydra.instance().clear()  # isolate from any other hydra-using test
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="positive_control_forgetting", overrides=overrides)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    """The imported forgetting-harness module (shared across the test module)."""
    return _load_harness()


@pytest.fixture(scope="module")
def mae_arms(harness: ModuleType) -> dict[str, Any]:
    """Run both MAE arms once (healthy then PC, as ``main`` does) and share the records."""
    config = _compose(_BASE)
    split = harness._task_split(config)
    healthy, _ = harness._run_arm(config, split, replay=True, run_name="mae_healthy")
    pc, _ = harness._run_arm(config, split, replay=False, run_name="mae_pc")
    gate = harness._evaluate_forgetting_gate(config, pc, healthy)
    return {"pc": pc, "healthy": healthy, "gate": gate}


def test_arm_slots_and_paired_seed_phase_a(mae_arms: dict[str, Any]) -> None:
    """The replay slots are correct and phase A is bit-identical across arms (paired seed)."""
    pc, healthy = mae_arms["pc"], mae_arms["healthy"]
    assert pc["replay"] is False and healthy["replay"] is True, "replay slots swapped"
    # Post-A task-A accuracy, init probe, and post-A recon must match — a broken seed reset desyncs them.
    assert pc["matrix"]["0"]["0"] == pytest.approx(healthy["matrix"]["0"]["0"], abs=1e-6), "R00 differs across arms"
    assert pc["task_a_init"] == pytest.approx(healthy["task_a_init"], abs=1e-6), "init probe differs across arms"
    assert pc["recon_a_after_a"] == pytest.approx(healthy["recon_a_after_a"], abs=1e-6), "post-A recon differs"


def test_backward_transfer_matches_the_matrix(mae_arms: dict[str, Any]) -> None:
    """BWT is R10 − R00 on the recorded accuracy matrix (metric-wiring guard)."""
    pc = mae_arms["pc"]
    expected = pc["matrix"]["1"]["0"] - pc["matrix"]["0"]["0"]
    assert pc["backward_transfer"] == pytest.approx(expected, abs=1e-9)
    assert pc["forgetting_measure"] == pytest.approx(-expected, abs=1e-9)  # FM = R00 − R10 = −BWT


def test_gate_structure_and_recon_gap_sign(mae_arms: dict[str, Any]) -> None:
    """The gate reports the right shape and the recon gap is signed pc − healthy."""
    pc, healthy, gate = mae_arms["pc"], mae_arms["healthy"], mae_arms["gate"]
    assert gate["mode"] == "forgetting"
    assert set(gate["checks"]) == {"pc_learned_A", "pc_bwt_fires", "pc_fm_fires", "healthy_holds", "contrast"}
    assert gate["passed"] == all(gate["checks"].values())
    expected_gap = pc["recon_a_rise"] - healthy["recon_a_rise"]
    assert gate["reported"]["recon_forget_gap"] == pytest.approx(expected_gap, abs=1e-9)


def test_phase_b_grad_norm_is_captured(mae_arms: dict[str, Any]) -> None:
    """The pre-clip phase-B grad norm is captured (the P0.4 divergence instrument, cross-mode specificity, audit C3)."""
    for arm in (mae_arms["pc"], mae_arms["healthy"]):
        assert arm["phase_b_grad_finite"] is True, "phase-B grad went non-finite on a sane-LR forgetting run"
        assert arm["phase_b_max_grad_norm"] > 0.0, "phase-B grad norm not captured"


def test_supervised_vehicle_runs_with_paired_seed(harness: ModuleType) -> None:
    """The supervised vehicle drives ``_run_arm_supervised``/``SupervisedMethod`` with the same wiring."""
    config = _compose([*_BASE, "training_mode=supervised", "supervised_augment=false"])
    split = harness._task_split(config)
    healthy, _ = harness._run_arm_supervised(config, split, replay=True, run_name="sup_healthy")
    pc, _ = harness._run_arm_supervised(config, split, replay=False, run_name="sup_pc")
    assert pc["replay"] is False and healthy["replay"] is True
    assert pc["matrix"]["0"]["0"] == pytest.approx(healthy["matrix"]["0"]["0"], abs=1e-6), "R00 differs across arms"
    assert pc["backward_transfer"] is not None and healthy["backward_transfer"] is not None
    # The MAE-native recon readouts do not apply to the supervised vehicle → reported as NaN.
    assert math.isnan(pc["recon_a_rise"]) and math.isnan(pc["recon_a_after_a"])


def test_from_scratch_guard_rejects_a_checkpoint_encoder(harness: ModuleType) -> None:
    """E4: a checkpoint-loading encoder under a from-scratch arm raises before the checkpoint loads."""
    ckpt_cfg = _compose([*_BASE, "encoder=vit_b16_mae"])  # declares a state_dict_path
    # Fires under a from-scratch arm (keep_weights False) — and before any instantiate, so no file needed.
    with pytest.raises(ValueError, match="state_dict_path"):
        harness._guard_from_scratch_encoder(ckpt_cfg, keep_weights=False)
    # Inert when the weights are intentionally kept, and for a from-scratch encoder (no state_dict_path).
    harness._guard_from_scratch_encoder(ckpt_cfg, keep_weights=True)
    harness._guard_from_scratch_encoder(_compose(_BASE), keep_weights=False)


def test_from_scratch_guard_is_wired_into_run_arm(harness: ModuleType) -> None:
    """E4: the guard is actually invoked by ``_run_arm`` (end-to-end, no checkpoint file required)."""
    config = _compose([*_BASE, "encoder=vit_b16_mae"])  # from-scratch arm + checkpoint encoder
    split = harness._task_split(config)
    with pytest.raises(ValueError, match="state_dict_path"):
        harness._run_arm(config, split, replay=False, run_name="should_raise")


def test_freeze_encoder_phase_a_pins_the_well(harness: ModuleType) -> None:
    """D1: with ``freeze_encoder_phase_a`` the encoder does not move in phase A, so R00 == the init probe.

    Freezing warms only the decoder, so the frozen encoder reads task A identically before and after
    phase A — post-A ``R00`` is bit-identical to the pre-A transfer probe (``task_a_init``). If phase A
    leaked into the encoder (e.g. weight decay on a frozen param), the two probes would diverge. Uses
    ``pretrained_encoder=true`` (the tiny synthetic encoder as a stand-in well) so ``keep_weights`` holds.
    """
    config = _compose([*_BASE, "pretrained_encoder=true", "freeze_encoder_phase_a=true"])
    split = harness._task_split(config)
    pc, _ = harness._run_arm(config, split, replay=False, run_name="frozen_pc")
    assert pc["matrix"]["0"]["0"] == pytest.approx(pc["task_a_init"], abs=1e-6), "encoder moved during frozen phase A"


def test_freeze_encoder_phase_a_requires_a_pretrained_well(harness: ModuleType) -> None:
    """D1: freezing on a from-scratch arm (no pretrained well) raises — nothing would be learned."""
    config = _compose([*_BASE, "freeze_encoder_phase_a=true"])  # from-scratch (keep_weights False)
    split = harness._task_split(config)
    with pytest.raises(ValueError, match="pretrained encoder well"):
        harness._run_arm(config, split, replay=False, run_name="should_raise")


def test_decoder_warmup_full_freeze_pins_the_well(harness: ModuleType) -> None:
    """D1 redo: decoder_warmup_epochs >= epochs_a is the full-freeze endpoint — R00 == the init probe.

    Frozen through all of phase A, the encoder never moves, so the post-A probe is bit-identical to the
    pre-A transfer probe (the same property the ``freeze_encoder_phase_a`` alias asserts).
    """
    config = _compose([*_BASE, "pretrained_encoder=true", "decoder_warmup_epochs=99"])  # >= epochs_a
    split = harness._task_split(config)
    pc, _ = harness._run_arm(config, split, replay=False, run_name="warmup_full")
    assert pc["matrix"]["0"]["0"] == pytest.approx(pc["task_a_init"], abs=1e-6), "encoder moved under full freeze"


def test_freeze_flag_is_the_full_warmup_alias(harness: ModuleType) -> None:
    """D1 redo: ``freeze_encoder_phase_a=true`` is exactly ``decoder_warmup_epochs = epochs_a`` (same record)."""
    flag = _compose([*_BASE, "pretrained_encoder=true", "freeze_encoder_phase_a=true"])
    knob = _compose([*_BASE, "pretrained_encoder=true", "decoder_warmup_epochs=2"])  # _BASE has epochs_a=2
    split = harness._task_split(flag)
    pc_flag, _ = harness._run_arm(flag, split, replay=False, run_name="flag")
    pc_knob, _ = harness._run_arm(knob, split, replay=False, run_name="knob")
    assert pc_flag["matrix"]["0"]["0"] == pytest.approx(pc_knob["matrix"]["0"]["0"], abs=1e-9)
    assert pc_flag["backward_transfer"] == pytest.approx(pc_knob["backward_transfer"], abs=1e-9)


def test_decoder_warmup_requires_a_pretrained_well(harness: ModuleType) -> None:
    """D1 redo: a decoder warm-up on a from-scratch arm raises (same guard as the freeze alias)."""
    config = _compose([*_BASE, "decoder_warmup_epochs=1"])  # from-scratch (keep_weights False)
    split = harness._task_split(config)
    with pytest.raises(ValueError, match="pretrained encoder well"):
        harness._run_arm(config, split, replay=False, run_name="should_raise")


def test_task_a_trajectory_logging_is_correct_and_rng_neutral(harness: ModuleType) -> None:
    """D1 redo: per-epoch logging is correctly shaped, respects the freeze schedule, and is RNG-neutral.

    The probe is eval-mode + sklearn, so it consumes no training RNG. With ``decoder_warmup_epochs=1`` and
    ``epochs_a=2``, phase-A epoch 0 warms the decoder alone (encoder frozen) so the first logged probe
    equals the init probe, and epoch 1 is joint. The matrix must be bit-identical to a non-logged run.
    """
    base = [*_BASE, "pretrained_encoder=true", "decoder_warmup_epochs=1"]
    logged = _compose([*base, "log_task_a_trajectory=true"])
    plain = _compose(base)
    split = harness._task_split(logged)
    pc_log, _ = harness._run_arm(logged, split, replay=False, run_name="logged")
    pc_plain, _ = harness._run_arm(plain, split, replay=False, run_name="plain")
    # Shape: one probe per phase-A epoch and per phase-B epoch.
    assert len(pc_log["task_a_traj_a"]) == 2 and len(pc_log["task_a_traj_b"]) == 2
    # Freeze schedule: epoch 0 is decoder-only (encoder pinned), so the first trajectory probe == init.
    assert pc_log["task_a_traj_a"][0] == pytest.approx(pc_log["task_a_init"], abs=1e-6)
    # The last phase-A probe is R00; the last phase-B probe is R10.
    assert pc_log["task_a_traj_a"][-1] == pytest.approx(pc_log["matrix"]["0"]["0"], abs=1e-6)
    assert pc_log["task_a_traj_b"][-1] == pytest.approx(pc_log["matrix"]["1"]["0"], abs=1e-6)
    # RNG-neutral: logging changed nothing about the training path.
    assert pc_log["matrix"]["0"]["0"] == pytest.approx(pc_plain["matrix"]["0"]["0"], abs=1e-9)
    assert pc_log["backward_transfer"] == pytest.approx(pc_plain["backward_transfer"], abs=1e-9)
    # Non-logged runs carry empty trajectories (bulky series stay opt-in).
    assert pc_plain["task_a_traj_a"] == [] and pc_plain["task_a_traj_b"] == []
