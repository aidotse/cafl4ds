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

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cafl4ds.measurements import rankme

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


def test_p112_extracts_block_features_that_reproduce_the_branch_record(tmp_path: Path) -> None:
    """P1.1.2: the extracted leg checkpoint and rebuilt post-damage state read as the branch recorded them."""
    well_out = tmp_path / "je_well.pt"
    warm_cfg = _compose(
        "warm_well",
        [*_P11_DIET, "warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=4", f"well_out={well_out}"],
    )
    _load_script("warm_well").warm(warm_cfg)
    harness = _load_script("run_deploy_harness")
    base = [*_P11_DIET, "eval_every=1", f"well={well_out}"]
    harness.run_deploy(_compose("deploy", [*base, "checkpoint_legs=true"]), tmp_path / "drive")
    index = json.loads((tmp_path / "drive" / "checkpoints" / "s0" / "index.json").read_text())
    start = index[1]["batches_seen"]
    ckpt = tmp_path / "drive" / "checkpoints" / "s{seed}" / f"step{start:05d}.pt"
    branch_dir = tmp_path / "noise"
    branch_cfg = _compose(
        "deploy",
        [*base, f"resume.checkpoint='{ckpt}'", f"resume.step={start}", "checkpoint_legs=final"]
        + ["damage.mode=noise", "damage.k=2", "damage.scale=0.5"],
    )
    harness.run_deploy(branch_cfg, branch_dir)
    (branch_dir / ".hydra").mkdir()
    OmegaConf.save(branch_cfg, branch_dir / ".hydra" / "config.yaml")  # what a Hydra launch leaves behind
    record = json.loads(next((branch_dir / "reports").glob("*.json")).read_text())["config"]["branch"]

    out = tmp_path / "lens"
    cfg = _compose("deploy", [*base, f"lens.drives=[{tmp_path / 'drive'}]", f"lens.branches=[{branch_dir}]"])
    entries = _load_script("extract_block_features").extract(cfg, out)
    assert len(entries) == 1 + len(index) + 2  # the well, every leg checkpoint, final + post-damage
    probe = torch.load(out / "probe.pt")
    query = probe["is_query"]
    pre = torch.load(out / "features" / "drive" / "s0" / f"step{start:05d}.pt")
    post = torch.load(out / "features" / "noise" / "s0" / "post_damage.pt")
    assert list(post)[-2:] == ["backbone", "proj"] and post["block01"].shape[0] == query.shape[0]
    assert rankme(pre["backbone"][query]) == pytest.approx(record["pre"]["rankme"], rel=1e-4)
    assert rankme(post["backbone"][query]) == pytest.approx(record["post"]["rankme"], rel=1e-4)
    assert [e for e in entries if "damage" in e][0]["damage"]["rel_change"] == pytest.approx(
        record["damage"]["rel_change"]
    )
    # P1.1.3: the probe record carries both halves' strata, and a rewrite must describe the same rows.
    assert probe["support_groups"].shape[0] == int((~query).sum())
    torch.save({**probe, "query_groups": probe["query_groups"].flip(0) + 1}, out / "probe.pt")
    with pytest.raises(ValueError, match="query_groups"):
        _load_script("extract_block_features")._save_probe(cfg, out)


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


def test_warm_well_freeze_encoder_trains_only_the_heads_then_resume_unfreezes(tmp_path: Path) -> None:
    """P1.2's settled-head start: a frozen-encoder warm-up leaves the encoder bit-identical.

    Resuming it with the encoder trainable then moves the encoder from exactly that state.
    """
    script = _load_script("warm_well")
    heads = tmp_path / "heads.pt"
    steps = ["warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=4", "warmup.stop=budget"]
    script.warm(_compose("warm_well", [*_P11_DIET, *steps, "warmup.freeze_encoder=true", f"well_out={heads}"]))
    trace = json.loads(heads.with_suffix(".trace.json").read_text())
    assert trace["encoder_unchanged"] is True
    torch.manual_seed(0)  # warm() seeds from config.seed=0, so a fresh build reproduces the start
    fresh = _compose("warm_well", [*_P11_DIET, f"well_out={tmp_path / 'x.pt'}"])
    start = instantiate(fresh.ssl, encoder=instantiate(fresh.encoder)).state_dict()
    settled = torch.load(heads)
    enc = [k for k in start if k.startswith("encoder.")]
    head = [k for k in start if not k.startswith("encoder.") and start[k].is_floating_point()]
    assert all(torch.equal(start[k], settled[k]) for k in enc)
    assert any(not torch.equal(start[k], settled[k]) for k in head)

    resumed = tmp_path / "resumed.pt"
    script.warm(_compose("warm_well", [*_P11_DIET, *steps, f"resume_well={heads}", f"well_out={resumed}"]))
    moved = torch.load(resumed)
    assert any(not torch.equal(settled[k], moved[k]) for k in enc)


def test_read_adaptation_reads_the_ladder_and_saves_per_image_correctness(tmp_path: Path) -> None:
    """The reader scores B5, a random init and a saved well on frozen + few-shot rungs, on both pools."""
    well_out = tmp_path / "je_well.pt"
    _load_script("warm_well").warm(_compose("warm_well", [*_P11_DIET, "warmup.max_steps=2", f"well_out={well_out}"]))
    script = _load_script("read_adaptation")
    encoders = [
        "+encoders=[{name:b5,kind:pretrained},{name:rand,kind:random},"
        f"{{name:well,kind:well,path:'{well_out}',fewshot:false}}]"
    ]
    common = [
        *_P11_DIET,
        "probe_per_class=6",
        "fewshot.enabled=true",
        "fewshot.ks=[2]",
        "fewshot.draws=2",
        "fewshot.steps=3",
        "threads=1",
        "~encoders",
        *encoders,
    ]
    summary = script.run(_compose("read_adaptation", [*common, "pool.fit_fraction=0.5"]), tmp_path / "heldout")
    assert set(summary["encoders"]) == {"b5", "rand", "well"}
    b5 = np.load(tmp_path / "heldout" / "b5.npz")
    pool = np.load(tmp_path / "heldout" / "pool.npz")
    assert b5["scene|linear"].shape == (1, pool["query_index"].size)
    assert b5["scene|ft2"].shape == (2, pool["query_index"].size)
    assert "scene|ft2" not in np.load(tmp_path / "heldout" / "well.npz")
    canary = script.run(_compose("read_adaptation", [*common, "pool.kind=canary"]), tmp_path / "canary")
    assert canary["pool"]["n_fit"] == canary["pool"]["n_query"]

    lr = script.run(
        _compose(
            "read_adaptation",
            [
                *common,
                "mode=select_lr",
                "fewshot.lr_grid=[1e-4,1e-3]",
                "fewshot.select_k=2",
                "fewshot.val_per_class=2",
                "pool.fit_fraction=0.8",
            ],
        ),
        tmp_path / "lr",
    )
    assert lr["best_lr"] in (1e-4, 1e-3)


def test_p121_drive_from_a_settled_head_well_saves_chosen_steps_and_reads_on_the_ladder(tmp_path: Path) -> None:
    """P1.2.1's wiring: a settled-head well, a drive checkpointing only the chosen steps, the reader on them."""
    heads = tmp_path / "heads.pt"
    warm = [*_P11_DIET, "warmup.window=2", "warmup.min_steps=2", "warmup.max_steps=3", "warmup.freeze_encoder=true"]
    _load_script("warm_well").warm(_compose("warm_well", [*warm, f"well_out={heads}"]))
    drive = [*_P11_DIET, "eval_every=4", f"well={heads}", "checkpoint_legs=steps", "checkpoint_steps=[2,0.5]"]
    _load_script("run_deploy_harness").run_deploy(_compose("deploy", drive), tmp_path / "drive")
    index = json.loads((tmp_path / "drive" / "checkpoints" / "s0" / "index.json").read_text())
    files = [e["file"] for e in index]  # step 2, the drive's halfway batch, and the end
    assert files[0] == "step00002.pt" and files[-1] == "final.pt" and len(files) == 3
    assert index[1]["batches_seen"] > 2

    ckpt = tmp_path / "drive" / "checkpoints" / "s0"
    encoders = (
        f"+encoders=[{{name:b5,kind:well,path:'{heads}'}},{{name:end,kind:checkpoint,path:'{ckpt / 'final.pt'}'}}]"
    )
    summary = _load_script("read_adaptation").run(
        _compose("read_adaptation", [*_P11_DIET, "probe_per_class=6", "threads=1", "~encoders", encoders]),
        tmp_path / "read",
    )
    assert set(summary["encoders"]) == {"b5", "end"}


def _fake_ladder(path: Path, names: list[str], gain: dict[str, float], n: int = 300) -> None:
    """A reader output dir: pool metadata plus per-encoder correctness at accuracy 0.6 + ``gain[name]``."""
    rng = np.random.default_rng(len(str(path)))
    path.mkdir(parents=True)
    labels = rng.integers(0, 3, n)
    np.savez(
        path / "pool.npz",
        query_index=np.arange(n),
        stratum=rng.integers(0, 2, n),
        regime=rng.integers(0, 2, n),
        **{"labels|scene": labels},
    )
    summary = {"stratum_names": {"0": "day", "1": "night"}, "regime_names": {"0": "day·clear", "1": "night·clear"}}
    (path / "summary.json").write_text(json.dumps(summary))
    for name in names:
        p = 0.6 + gain.get(name, 0.0)
        frozen = {f"scene|{r}": (rng.random((1, n)) < p).astype(np.float32) for r in ("knn", "linear")}
        ft = {f"scene|ft{k}": (rng.random((3, n)) < p).astype(np.float32) for k in (10, 100)}
        np.savez(path / f"{name}.npz", **frozen, **({} if name.endswith("_mid") else ft))


def test_analyze_adaptation_reads_p120_contrasts_and_the_p121_gate(tmp_path: Path) -> None:
    """The analysis pairs encoders on one query set, reports both framings, and calls the gate."""
    script = _load_script("analyze_adaptation")
    drives = [f"{bb}_{o}_s{s}_{pt}" for o in ("walk", "iid") for s in (0, 1) for pt in ("mid", "end") for bb in ("je",)]
    _fake_ladder(tmp_path / "p120" / "ladder" / "je", ["b5", "random", "je_well", *drives], {"je_well": 0.1})
    p120 = script.run(
        OmegaConf.create(
            {
                "mode": "p120",
                "root": str(tmp_path / "p120"),
                "n_boot": 200,
                "seed": 0,
                "threads": 1,
                "n_seeds": 2,
                "backbones": ["je"],
                "pools": ["ladder", "canary"],
            }
        )
    )
    linear = p120["ladder"]["je"]["scene|linear"]
    assert linear["vs_b5"]["well"]["delta"] > 0.05 and linear["vs_well"]["walk_end"]["delta"] < -0.05
    assert "ft100" not in str(p120["ladder"]["je"].get("scene|ft100", {}).get("vs_b5", {}).get("walk_mid"))

    names = ["b5", *[f"walk_s{s}_{t}" for s in range(3) for t in ("t00100", "end")], "iid_s0_end", "iid_s1_end"]
    _fake_ladder(tmp_path / "p121" / "ladder" / "je", names, {f"walk_s{s}_end": 0.12 for s in range(3)}, n=2000)
    corpus = tmp_path / "p121" / "drive" / "je_walk" / "corpus"
    corpus.mkdir(parents=True)
    legs = [
        {"era_index": 0, "era_name": "day·clear", "start_step": 0, "end_step": 149},
        {"era_index": 1, "era_name": "night·clear", "start_step": 150, "end_step": 299},
    ]
    (corpus / "segments.json").write_text(json.dumps({"segments": legs}))
    p121 = script.run(
        OmegaConf.create(
            {
                "mode": "p121",
                "root": str(tmp_path / "p121"),
                "n_boot": 300,
                "seed": 0,
                "threads": 1,
                "n_seeds": 3,
                "n_iid_seeds": 2,
                "backbones": ["je"],
                "tasks": ["scene"],
                "gate_tasks": ["scene"],
                "alpha": 0.025,
                "rungs": ["knn", "linear", "ft10", "ft100"],
                "primary": {"je": "linear"},
                "delta_floor": 0.02,
                "b5_name": "b5",
            }
        )
    )
    assert p121["je"]["gate"]["verdict"] == "PASS"
    assert set(p121["je"]["trajectory"]) == {"end", "t00100"}
    secondary = p121["je"]["secondary"]
    assert secondary["walk_vs_iid"]["scene|linear"]["delta"] > 0.05  # the walk ends 0.12 up, the IID seeds flat
    by_regime = secondary["by_regime"]["scene|linear"]
    assert by_regime["t00100"]["era"] == 0 and by_regime["t00100"]["left"] is None  # nothing left yet at step 100
    assert (
        by_regime["end"]["era"] == 1 and by_regime["end"]["left"]["delta"] > 0.05 and by_regime["end"]["ahead"] is None
    )


def test_read_adaptation_screen_reads_reference_and_candidates_and_applies_the_rule(tmp_path: Path) -> None:
    """The screen scores the reference and each candidate on B5 and applies overlap + headroom bars."""
    script = _load_script("read_adaptation")
    base = [
        *_P11_DIET,
        "mode=screen",
        "probe_per_class=6",
        "threads=1",
        "fewshot.select_k=2",
        "fewshot.draws=1",
        "fewshot.steps=2",
        "screen.ceiling_per_class=4",
        "screen.ceiling_steps=2",
        "screen.reference=scene",
        "screen.candidates=[scene]",
    ]
    open_bar = script.run(
        _compose("read_adaptation", [*base, "screen.max_overlap=1.0", "screen.min_headroom=-1.0"]), tmp_path / "open"
    )
    assert set(open_bar["tasks"]) == {"scene"} and open_bar["choice"] == "scene"
    assert {"overlap", "b5_linear", "b5_ft2", "ceiling", "headroom"} <= set(open_bar["tasks"]["scene"])
    closed = script.run(_compose("read_adaptation", [*base, "screen.min_headroom=2.0"]), tmp_path / "closed")
    assert closed["choice"] is None


def test_read_and_analyze_align_a_cell_balanced_task_to_its_own_query_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subset task (presence_cell) is read on its own frames, and the gate analysis pairs it correctly."""
    from cafl4ds import adaptation  # noqa: PLC0415 — local: only this test builds a pool by hand

    rng = np.random.default_rng(0)
    n = 240
    scene = torch.from_numpy(rng.integers(0, 3, n))
    present = rng.random(n) < np.where(scene.numpy() == 0, 0.7, 0.3)
    images = torch.rand(n, 3, 16, 16) + torch.from_numpy(present)[:, None, None, None].float()
    pool = adaptation.HeldOutPool(
        images=images.clamp(0, 1),
        scene=scene,
        stratum=torch.from_numpy(rng.integers(0, 2, n)),
        regime=torch.zeros(n, dtype=torch.long),
        categories=[{"car": 1} if p else {} for p in present],
    )
    script = _load_script("read_adaptation")
    fit, query = adaptation.split_pool(n, 0.5, seed=0)
    monkeypatch.setattr(script, "build_pool", lambda config: (pool, fit, query))
    names = "[{name:b5,kind:pretrained},{name:walk_s0_end,kind:pretrained},{name:walk_s1_end,kind:pretrained}]"
    overrides = [
        "data=attr_synthetic",
        "img_size=16",
        "probe_per_class=20",
        "threads=1",
        "~encoders",
        f"+encoders={names}",
        'tasks=[scene,"presence_cell:car"]',
    ]
    summary = script.run(_compose("read_adaptation", overrides), tmp_path / "ladder" / "je")
    cell = adaptation.task_indices(pool, "presence_cell:car", query, seed=0)
    assert (
        summary["n_query_by_task"] == {"scene": query.size, "presence_cell:car": cell.size} and cell.size < query.size
    )
    assert np.load(tmp_path / "ladder" / "je" / "b5.npz")["presence_cell:car|linear"].shape == (1, cell.size)

    report = _load_script("analyze_adaptation").run(
        OmegaConf.create(
            {
                "mode": "p121",
                "root": str(tmp_path),
                "n_boot": 100,
                "seed": 0,
                "threads": 1,
                "n_seeds": 2,
                "backbones": ["je"],
                "tasks": ["scene", "presence_cell:car"],
                "gate_tasks": ["scene", "presence_cell:car"],
                "rungs": ["knn", "linear"],
                "primary": {"je": "linear"},
                "delta_floor": 0.02,
                "alpha": 0.05,
                "b5_name": "b5",
            }
        )
    )
    contrast = report["je"]["contrasts"]["presence_cell:car"]["linear"]
    assert contrast["n_images"] == cell.size and contrast["delta"] == pytest.approx(0.0)
