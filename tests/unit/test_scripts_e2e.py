"""End-to-end smoke tests for the Phase-1 entry-point scripts (audit P1.0 C1).

Each script's Hydra-free core is composed on the network-free synthetic source and run once into a
tmp dir, asserting the expected artifact lands with the right shape. This guards the glue the library
unit tests don't reach — the seed loop + manifest assembly (``run_deploy_harness``), the warm-well
save (``warm_well``), the divergence report (``measure_divergence``), and the canonical comparison
(``run_harness``) — from regressions the promoted one-shot runs would not catch.
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "cafl4ds" / "configs"

# Fast, network-free synthetic overrides shared by the class-axis scripts (divergence + harness):
# a tiny image, a short horizon, small reservations, probes off (they add wall-clock for no wiring
# value here).
_SYNTHETIC_CLASS = [
    "data=synthetic",
    "img_size=16",
    "eval_every=3",
    "max_per_class=40",
    "max_train_per_class=8",
    "stream.support_per_class=6",
    "stream.query_per_class=6",
    "stream.era_eval_per_class=4",
    "monitor.run_knn=false",
    "monitor.run_linear=false",
]


def _load_script(name: str) -> ModuleType:
    """Import a ``scripts/<name>.py`` entry point as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    """Compose a script config with fast-smoke overrides (isolated from other hydra-using tests)."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_run_deploy_harness_writes_the_ensemble_corpus(tmp_path: Path) -> None:
    """The deploy script drives a 2-seed ensemble and lands the four-part corpus + per-seed reports."""
    script = _load_script("run_deploy_harness")
    config = _compose(
        "deploy",
        [
            "data=attr_synthetic",
            "img_size=16",
            "eval_every=3",
            "stream.support_per_canary=6",
            "stream.query_per_canary=6",
            "monitor.knn_k=3",
            "seeds=[0,1]",
        ],
    )
    paths = script.run_deploy(config, tmp_path)
    for part in ("health_csv", "segments", "trust", "manifest"):
        assert paths[part].is_file(), part
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["n_seeds"] == 2
    reports = list((tmp_path / "reports").glob("*.json"))
    assert len(reports) == 2
    # both the Tier-A wiring verdict and the Tier-B sanity block are present per seed
    first = json.loads(reports[0].read_text())
    assert first["validation"]["passed"] is True
    assert "tier_b" in first


def test_warm_well_saves_a_resumable_well(tmp_path: Path) -> None:
    """The warm-well script plateaus on the stationary diet and writes a resumable well."""
    script = _load_script("warm_well")
    well_out = tmp_path / "je_well.pt"
    config = _compose(
        "warm_well",
        [
            "data=attr_synthetic",
            "img_size=16",
            "stream.support_per_canary=6",
            "stream.query_per_canary=6",
            "monitor.knn_k=3",
            "warmup.window=2",
            "warmup.min_steps=2",
            "warmup.max_steps=6",
            "warmup.max_passes=2",
            f"well_out={well_out}",
        ],
    )
    assert script.warm(config) == well_out
    assert well_out.is_file()


# The P1.1.0 diet knobs, shared by the well and the drive so both reserve the same probes / well slice.
_P11_DIET = [
    "data=attr_synthetic",
    "img_size=16",
    "data.long_tail=false",
    "stream.support_per_canary=3",
    "stream.query_per_canary=3",
    "stream.stratify_canary=true",
    "stream.well_fraction=0.3",
    "monitor.knn_k=3",
]


def test_p11_null_control_drives_walk_and_iid_from_a_disjoint_anchored_well(tmp_path: Path) -> None:
    """P1.1.0 end to end: warm a disjoint well, then drive walk + IID anchored to it, stratified, checkpointed."""
    well_out = tmp_path / "je_well.pt"
    warm_cfg = _compose(
        "warm_well",
        [*_P11_DIET, "warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=4", f"well_out={well_out}"],
    )
    _load_script("warm_well").warm(warm_cfg)

    script = _load_script("run_deploy_harness")
    for order in ("walk", "iid"):
        out = tmp_path / order
        config = _compose(
            "deploy",
            [*_P11_DIET, "eval_every=2", f"stream.order={order}", f"well={well_out}", "checkpoint_legs=true"],
        )
        script.run_deploy(config, out)
        report = json.loads(next((out / "reports").glob("*.json")).read_text())
        assert report["validation"]["passed"] is True
        assert report["config"]["order"] == order
        assert report["config"]["anchor"] == str(well_out)
        assert report["config"]["familiarity"] == {"well_side": 0.0, "drive_side": 0.0}  # a disjoint well
        counts = report["config"]["n_probe_query"]
        assert counts["pooled"] == counts["early"] + counts["late"]
        live = report["arms"]["live"]["health"]
        # stratified canary slices + grad norm at every checkpoint
        assert {"knn_acc_early", "knn_acc_late", "linear_acc_early", "linear_acc_late"} <= set(live[0])
        assert "knn_acc_late" in report["channels"]["canary"]
        assert all("grad_norm" in h for h in live)
        # anchored to the well the drive resumed: the first read sits at (near) zero drift
        assert live[0]["cka_drift"] < 0.05
        index = json.loads((out / "checkpoints" / "s0" / "index.json").read_text())
        assert index[-1]["file"] == "final.pt" and len(index) >= 2


def test_p111_branches_resume_damage_and_recover_from_a_leg_checkpoint(tmp_path: Path) -> None:
    """P1.1.1 end to end: resume a drive from a leg checkpoint (dose 0 and damaged), then recover from its end."""
    well_out = tmp_path / "je_well.pt"
    warm_cfg = _compose(
        "warm_well",
        [*_P11_DIET, "warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=4", f"well_out={well_out}"],
    )
    _load_script("warm_well").warm(warm_cfg)
    script = _load_script("run_deploy_harness")
    base = [*_P11_DIET, "eval_every=1", f"well={well_out}"]
    script.run_deploy(_compose("deploy", [*base, "checkpoint_legs=true"]), tmp_path / "drive")
    index = json.loads((tmp_path / "drive" / "checkpoints" / "s0" / "index.json").read_text())
    start = index[1]["batches_seen"]
    ckpt = tmp_path / "drive" / "checkpoints" / "s{seed}" / f"step{start:05d}.pt"

    reports = {}
    for name, damage in (("dose0", []), ("reinit", ["damage.mode=reinit", "damage.k=2"])):
        out = tmp_path / name
        overrides = [*base, f"resume.checkpoint='{ckpt}'", f"resume.step={start}", "checkpoint_legs=final", *damage]
        script.run_deploy(_compose("deploy", overrides), out)
        reports[name] = json.loads(next((out / "reports").glob("*.json")).read_text())
        saved = json.loads((out / "checkpoints" / "s0" / "index.json").read_text())
        assert [e["file"] for e in saved] == ["final.pt"]  # checkpoint_legs=final: the end state only
    for name, report in reports.items():
        assert report["validation"]["passed"] is True, name
        branch = report["config"]["branch"]
        assert branch["phase"] == "drive" and branch["start_step"] == start
        assert report["arms"]["live"]["health"][0]["step"] == start  # the resumed arm sits on the drive's grid
    dose0, reinit = reports["dose0"]["config"]["branch"], reports["reinit"]["config"]["branch"]
    assert dose0["damage"]["rel_change"] == 0.0 and dose0["pre"]["rankme"] == dose0["post"]["rankme"]
    assert reinit["damage"]["rel_change"] > 0.0 and reinit["pre"]["rankme"] == dose0["pre"]["rankme"]
    assert reinit["post"]["rankme"] != reinit["pre"]["rankme"]  # the dent is on record

    final = tmp_path / "reinit" / "checkpoints" / "s{seed}" / "final.pt"
    out = tmp_path / "recover"
    script.run_deploy(_compose("deploy", [*base, f"resume.checkpoint='{final}'", "recovery_steps=5"]), out)
    report = json.loads(next((out / "reports").glob("*.json")).read_text())
    assert report["config"]["branch"]["phase"] == "recover"
    assert [h["step"] for h in report["arms"]["live"]["health"]] == list(range(5))


def test_p111_recovery_needs_a_checkpoint_to_recover_from(tmp_path: Path) -> None:
    """P111 recovery needs a checkpoint to recover from."""
    config = _compose("deploy", [*_P11_DIET, "recovery_steps=3"])
    with pytest.raises(ValueError, match="resume.checkpoint"):
        _load_script("run_deploy_harness").run_deploy(config, tmp_path)


def test_p11_scout_warms_a_budgeted_well_with_a_selection_trace(tmp_path: Path) -> None:
    """The competence scout: a fixed budget, a selection read every N steps, the trace beside the well."""
    well_out = tmp_path / "je_well.pt"
    select = ["stream.select_support_per_canary=2", "stream.select_query_per_canary=2"]
    warm_cfg = _compose(
        "warm_well",
        [
            *_P11_DIET,
            *select,
            "warmup.stop=budget",
            "warmup.max_steps=5",
            "warmup.select_every=2",
            "warmup.drive_read_per_class=4",
            f"well_out={well_out}",
        ],
    )
    _load_script("warm_well").warm(warm_cfg)
    trace = json.loads(well_out.with_suffix(".trace.json").read_text())
    assert trace["steps"] == 5
    assert [r["step"] for r in trace["competence_trace"]] == [0.0, 2.0, 4.0, 5.0]
    # the drive-pool read rides along every selection reading
    assert all(
        0.0 <= r["drive_knn_bacc"] <= 1.0 and 0.0 <= r["drive_linear_bacc"] <= 1.0 for r in trace["competence_trace"]
    )
    assert set(trace["measurement"]) == {"knn_acc", "linear_acc"}
    # the drive carries the same selection reservation, so it accepts the well
    config = _compose("deploy", [*_P11_DIET, *select, "eval_every=4", f"well={well_out}"])
    _load_script("run_deploy_harness").run_deploy(config, tmp_path / "drive")


def test_p11_drive_read_switches_off_when_the_well_shares_every_drive_frame(tmp_path: Path) -> None:
    """At overlap 1 no well frame is held out from the drive, so the drive-pool read logs nothing."""
    well_out = tmp_path / "je_well.pt"
    overrides = ["stream.select_support_per_canary=2", "stream.select_query_per_canary=2", "stream.well_overlap=1.0"]
    warm_cfg = _compose(
        "warm_well",
        [
            *_P11_DIET,
            *overrides,
            "warmup.stop=budget",
            "warmup.max_steps=2",
            "warmup.select_every=2",
            "warmup.drive_read_per_class=4",
            f"well_out={well_out}",
        ],
    )
    _load_script("warm_well").warm(warm_cfg)
    trace = json.loads(well_out.with_suffix(".trace.json").read_text())
    assert trace["competence_trace"] and not any("drive_knn_bacc" in r for r in trace["competence_trace"])


def test_p11_drive_refuses_a_well_reserved_with_another_split(tmp_path: Path) -> None:
    """A drive whose familiarity level differs from its well's is refused before it runs."""
    well_out = tmp_path / "je_well.pt"
    warm_cfg = _compose(
        "warm_well",
        [*_P11_DIET, "warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=4", f"well_out={well_out}"],
    )
    _load_script("warm_well").warm(warm_cfg)
    config = _compose("deploy", [*_P11_DIET, "stream.well_overlap=0.5", f"well={well_out}", "study=P1.1.0"])
    with pytest.raises(ValueError, match="well_overlap"):
        _load_script("run_deploy_harness").run_deploy(config, tmp_path / "drive")


def test_measure_divergence_certifies_same_seed(tmp_path: Path) -> None:
    """The divergence script runs its four arms and certifies the same-seed rebuild bit-identically."""
    script = _load_script("measure_divergence")
    report = script._measure_divergence(_compose("divergence", _SYNTHETIC_CLASS), tmp_path)
    assert report["study"] == "P1.0.1"
    assert report["data_identity"]["bit_identical"] is True
    assert set(report["attribution"]) == {"stream_construction", "in_arm_rng_matched", "in_arm_rng_desynced"}


def test_run_harness_writes_the_canonical_comparison(tmp_path: Path) -> None:
    """The P1.0.0 harness co-logs the three arms into the canonical comparison schema."""
    script = _load_script("run_harness")
    comparison = script._run_smoke(_compose("harness", _SYNTHETIC_CLASS), tmp_path)
    assert comparison["schema_version"] == 1
    assert set(comparison["arms"]) == {"live", "pc", "b5"}
    assert "gate" in comparison and "directional" in comparison


def test_analyze_null_reads_a_walk_iid_pair_into_the_report(tmp_path: Path) -> None:
    """The null analysis pairs a walk with its matched IID corpus and reports gaps, movement and power."""
    deploy = _load_script("run_deploy_harness")
    for order in ("walk", "iid"):
        config = _compose(
            "deploy",
            [*_P11_DIET, "eval_every=2", f"stream.order={order}", "seeds=[0,1,2]", "study=P1.1.0"],
        )
        deploy.run_deploy(config, tmp_path / "warm" / f"je_o0.0_{order}")
        manifest = json.loads((tmp_path / "warm" / f"je_o0.0_{order}" / "corpus" / "manifest.json").read_text())
        assert manifest["study"] == "P1.1.0"
    config = _compose(
        "null_envelope",
        [
            f"root={tmp_path}",
            "backbones=[je]",
            "overlaps=[0.0]",
            "cold_backbones=[]",
        ],
    )
    report = _load_script("analyze_null").analyze(config)
    pair = report["warm"]["je"]["o=0.0"]
    assert pair["n_seeds"] == 3
    assert {"rankme", "grad_norm", "knn_acc_late"} <= set(pair["signals"])
    assert set(pair["signals"]["rankme"]) == {"iid_movement", "walk_movement", "gap"}
    assert {"knn_acc", "knn_acc_early", "linear_acc"} <= set(pair["canary_power"])
    # the query counts come from the run itself, not the config
    counts = json.loads((tmp_path / "warm" / "je_o0.0_iid" / "corpus" / "manifest.json").read_text())["config"][
        "n_probe_query"
    ]
    assert pair["canary_power"]["knn_acc"]["n_query"] == counts["pooled"]
    assert pair["canary_power"]["knn_acc_early"]["n_query"] == counts["early"]
    assert (tmp_path / "null_report.json").is_file()


def test_analyze_null_falls_back_to_configured_counts_for_a_legacy_manifest(tmp_path: Path) -> None:
    """A corpus written before ``n_probe_query`` existed reads its counts from the config instead."""
    script = _load_script("analyze_null")
    (tmp_path / "run" / "corpus").mkdir(parents=True)
    (tmp_path / "run" / "corpus" / "manifest.json").write_text(json.dumps({"config": {}}))
    config = _compose("null_envelope", [f"root={tmp_path}"])
    counts = script._query_counts(tmp_path, "run", config)
    assert script._n_query("knn_acc", counts) == config.n_query_pooled
    assert script._n_query("linear_acc_night", counts) == config.n_query_stratum


def test_supervised_ceiling_trains_and_reports_against_its_references(tmp_path: Path) -> None:
    """The supervised ceiling trains on the well pool and scores the drive pool beside its references."""
    script = _load_script("supervised_ceiling")
    config = _compose(
        "supervised_ceiling",
        [
            "data=attr_synthetic",
            "img_size=16",
            "data.long_tail=false",
            "stream.support_per_canary=2",
            "stream.query_per_canary=2",
            "stream.select_support_per_canary=2",
            "stream.select_query_per_canary=2",
            "epochs=2",
            "batch_size=16",
            "probe_per_class=8",
            "knn_k=3",
            "pixel_grid=4",
        ],
    )
    report = script.run(config, tmp_path)
    assert [r["epoch"] for r in report["trace"]] == [1.0, 2.0]
    assert {"train_bacc", "drive_bacc", "selection_bacc"} <= set(report["final"])
    assert set(report["references"]) == {"encoder_at_init", "pixels"}
    assert 0.0 <= report["best_drive_bacc"] <= 1.0
    assert (tmp_path / "supervised_ceiling.json").is_file()
    # epochs=0 is the probe-only read of the encoder as built
    config.epochs = 0
    probe_only = script.run(config, tmp_path)
    assert probe_only["trace"] == [] and probe_only["best_drive_bacc"] is None


def test_supervised_ceiling_reads_a_saved_ssl_well(tmp_path: Path) -> None:
    """With ``well=<path>`` the read loads the SSL well's encoder weights, not a fresh encoder."""
    well_out = tmp_path / "je_well.pt"
    warm = _compose("warm_well", [*_P11_DIET, "warmup.max_steps=2", f"well_out={well_out}"])
    _load_script("warm_well").warm(warm)
    script = _load_script("supervised_ceiling")
    overrides = [*_P11_DIET, "stream.select_support_per_canary=0", "stream.select_query_per_canary=0", "epochs=0"]
    encoder = script._encoder(_compose("supervised_ceiling", [*overrides, f"well={well_out}"]))
    saved = torch.load(well_out)
    assert all(torch.equal(v, saved[f"encoder.{k}"]) for k, v in encoder.state_dict().items())
    report = script.run(
        _compose("supervised_ceiling", [*overrides, f"well={well_out}", "probe_per_class=8", "pixel_grid=4"]), tmp_path
    )
    assert report["config"]["well"] == str(well_out)


def test_bdd_legacy_labels_merges_per_image_jsons_into_the_legacy_format(tmp_path: Path) -> None:
    """Per-image 2018 labels of one split become legacy records: boxes only, legacy category names."""
    script = _load_script("bdd_legacy_labels")
    attrs = {"weather": "clear", "scene": "highway", "timeofday": "daytime"}
    image = {
        "name": "a",
        "attributes": attrs,
        "frames": [{"objects": [{"category": "bike", "box2d": {}}, {"category": "lane", "poly2d": []}]}],
    }
    archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("100k/train/a.json", json.dumps(image))
        z.writestr("100k/val/b.json", json.dumps({**image, "name": "b"}))
    records = script.legacy_records(archive, "train")
    assert records == [{"name": "a.jpg", "attributes": attrs, "labels": [{"category": "bicycle", "box2d": {}}]}]
    check = script.compare(records, [{"name": "a.jpg", "attributes": attrs, "labels": [{"category": "bicycle"}]}])
    assert check["missing"] == check["attribute_mismatches"] == check["box_count_mismatches"] == 0
