"""End-to-end pairing, leakage, image-keyed randomness, and exact checkpoint continuation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from cafl4ds.routing_analysis import analyze
from cafl4ds.routing_experiment import (
    epoch_order,
    keyed_seed,
    paired_views,
    reserve_panels,
    run_arm,
    run_experiment,
)
from cafl4ds.routing_order_analysis import compare_ordering


def _config() -> DictConfig:
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "cafl4ds/configs"), version_base=None
    ):
        return compose(
            config_name="routing_experiment",
            overrides=[
                "routing_source@source=synthetic",
                "device=cpu",
                "threads=1",
                "img_size=16",
                "seeds=[0,1]",
                "batch_size=8",
                "epochs=2",
                "max_steps=6",
                "eval_every=2",
                "mask_draws=1",
                "support_per_class=3",
                "query_per_class=4",
                "natural_query_size=8",
                "encoder.embed_dim=16",
                "encoder.depth=3",
                "encoder.num_heads=2",
                "ssl.decoder_dim=16",
                "ssl.decoder_depth=1",
                "ssl.decoder_heads=2",
            ],
        )


def test_panels_and_epoch_order_have_no_leakage() -> None:
    """Held-out identities never reach the training permutation; epochs independently reshuffle."""
    cfg = _config()
    corpus = instantiate(cfg.source)
    panels = reserve_panels(corpus, cfg)
    sets = [set(v) for v in panels.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == len(corpus.images)
    first, second = epoch_order(panels["train"], 0, 0), epoch_order(panels["train"], 0, 1)
    assert not torch.equal(first, second)
    assert set(first.tolist()) == set(second.tolist()) == set(panels["train"])
    assert torch.equal(first, epoch_order(panels["train"], 0, 0))
    assert keyed_seed("data", 0) != keyed_seed("mask", 0)


def test_shared_images_keep_views_and_masks_despite_different_selection_positions() -> None:
    """Image-keyed random draws pair overlapping samples independently of admission and batch position."""
    cfg = _config()
    corpus = instantiate(cfg.source)
    method = instantiate(cfg.ssl, encoder=instantiate(cfg.encoder))
    ids = torch.tensor([5, 6, 9, 12])
    original = torch.get_rng_state().clone()
    views, noise = paired_views(method, corpus.images[ids], ids, 0, 2)
    other, other_noise = paired_views(method, corpus.images[ids[[3, 1]]], ids[[3, 1]], 0, 2)
    torch.testing.assert_close(views[[3, 1]], other, rtol=0, atol=0)
    torch.testing.assert_close(noise[[3, 1]], other_noise, rtol=0, atol=0)
    assert torch.equal(original, torch.get_rng_state())


@pytest.mark.parametrize(
    "device,stream_order",
    [
        ("cpu", "iid"),
        ("cpu", "regime_blocked"),
        pytest.param(
            "mps", "iid", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
        ),
    ],
)
def test_exact_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path, device: str, stream_order: str) -> None:
    """Restoring selection and optimizer state reproduces learned weights and scientific endpoint metrics."""
    cfg = _config()
    cfg.device = device
    cfg.stream_order = stream_order
    cfg.threads = 4 if device == "cpu" else 1
    torch.set_num_threads(int(cfg.threads))
    corpus = instantiate(cfg.source)
    panels = reserve_panels(corpus, cfg)
    full = run_arm(cfg, corpus, panels, 0, "high", tmp_path / "full", "test-protocol")
    interrupted = run_arm(cfg, corpus, panels, 0, "high", tmp_path / "resume", "test-protocol", stop_after=2)
    assert interrupted["step"] == 2
    # Simulate a crash after logs were appended beyond the last durable checkpoint.
    with (tmp_path / "resume/training.jsonl").open("a") as stream:
        stream.write('{"uncommitted":true}\n')
    resumed = run_arm(cfg, corpus, panels, 0, "high", tmp_path / "resume", "test-protocol")
    a = torch.load(tmp_path / "full/latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "resume/latest.pt", map_location="cpu", weights_only=False)
    for key, value in a["method"].items():
        torch.testing.assert_close(value, b["method"][key], rtol=0, atol=0)
    assert full["final"] == resumed["final"]
    assert (tmp_path / "full/selection.jsonl").read_text() == (tmp_path / "resume/selection.jsonl").read_text()
    assert (tmp_path / "full/training.jsonl").read_text() == (tmp_path / "resume/training.jsonl").read_text()


@pytest.mark.parametrize("stream_order", ["iid", "regime_blocked"])
def test_three_arms_share_candidate_order_and_only_train_accepted_images(tmp_path: Path, stream_order: str) -> None:
    """All requested arms write complete trajectories and use exactly half the label-free candidates."""
    cfg = _config()
    cfg.stream_order = stream_order
    comparison = run_experiment(cfg, tmp_path)
    if stream_order == "regime_blocked":
        manifest = json.loads((tmp_path / "stream_manifest.json").read_text())
        assert manifest["order"] == "regime_blocked" and len(manifest["seeds"]) == 2
    assert len(comparison["runs"]) == 6
    for seed in (0, 1):
        orders, hashes = [], []
        for arm in ("random", "high", "low"):
            root = tmp_path / f"seed_{seed}" / arm
            training = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
            selections = [json.loads(line) for line in (root / "selection.jsonl").read_text().splitlines()]
            assert len(training) == len(selections) == 6
            for row, decision in zip(training, selections, strict=True):
                assert row["candidate_count"] == 8 and row["accepted_count"] == 4
                # 16px / 8px patches with 75% masking: exactly one visible patch per accepted image.
                assert all(sum(counts) == 4 for counts in row["train_counts"].values())
                scores = torch.tensor(decision["entropy"])
                chosen = decision["accepted_indices"]
                rejected = [i for i in range(8) if i not in chosen]
                if arm == "high":
                    assert scores[chosen].min() >= scores[rejected].max()
                if arm == "low":
                    assert scores[chosen].max() <= scores[rejected].min()
            orders.append([row["candidate_order_hash"] for row in training])
            hashes.append(json.loads((root / "complete.json").read_text())["initial_hash"])
        assert orders[0] == orders[1] == orders[2]
        assert len(set(hashes)) == 1
    assert (tmp_path / "trajectories.png").exists()
    assert (tmp_path / "comparison.json").exists()
    analysis = analyze(tmp_path)
    assert analysis["replicate_unit"] == "training seed"
    assert all(row["heldout_mse"]["n_seeds"] == 2 for row in analysis["arms"].values())
    for name in (
        "seed_summary.csv",
        "acceptance_rates.csv",
        "selection_bias.png",
        "selection_entropy.png",
        "l1_expert_loads.png",
        "load_balance.png",
        "load_statistics.csv",
        "load_statistics.json",
    ):
        assert (tmp_path / "analysis" / name).is_file()


def test_regime_vs_iid_comparison_checks_actual_membership_and_renders(tmp_path: Path) -> None:
    """Two complete, matched small ensembles exercise the saved-data comparison without a BDD dependency."""
    cfg = _config()
    cfg.seeds = [0]
    cfg.stream_order = "iid"
    run_experiment(cfg, tmp_path / "iid")
    cfg.stream_order = "regime_blocked"
    run_experiment(cfg, tmp_path / "blocked")
    result = compare_ordering(tmp_path / "blocked", tmp_path / "iid")
    assert result["verification"]["matched_actual_candidate_membership_each_epoch"]
    for name in (
        "order_comparison.json",
        "order_comparison.csv",
        "order_comparison.png",
        "stream_timeline.png",
        "regime_acceptance.csv",
    ):
        assert (tmp_path / "blocked" / "analysis" / name).is_file()
    with (tmp_path / "blocked" / "analysis" / "regime_acceptance.csv").open() as stream:
        admissions = list(csv.DictReader(stream))
    for arm in ("random", "high", "low"):
        group = [row for row in admissions if row["arm"] == arm]
        assert sum(int(row["accepted"]) for row in group) * 2 == sum(int(row["candidate"]) for row in group)
        assert all(0 <= float(row["rate"]) <= 1 for row in group)
    path = tmp_path / "blocked" / "seed_0" / "random" / "selection.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sample_ids"][0] = -1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="stream manifest"):
        compare_ordering(tmp_path / "blocked", tmp_path / "iid")
