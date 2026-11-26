"""P1.1.4 — the referee power, the split adjudication and the damage branches' engineering.

Part 4's referee-power core, Part 1's false-alarm count and split adjudication, and Part 2's damage on
DINO's method (student and teacher alike) and its resumed momentum ramp.
"""

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cafl4ds import referee_power as rp
from cafl4ds.branch import damage_method, damage_top_blocks, top_blocks
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.null_envelope import drop_profile, gap, gap_fast, injected_mdd, injected_power, split_masks
from cafl4ds.reading_protocol import adjudicate_split, direction, finetune_drop, finetune_drop_consecutive
from cafl4ds.ssl.dino import DINO, build_dino, momentum_at
from scripts.analyze_p114 import (
    LadderSet,
    _open_ladders,
    _own_name,
    _perm_p,
    _rows_any,
    _welch_ci,
    three_panels,
    wilson_interval,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_p125 import _DINO_SMALL, _tiny_vit  # noqa: E402


def test_change_noise_point_is_the_seed_mean_bacc_fall() -> None:
    """The point loss is each seed's balanced-accuracy fall from t0, averaged."""
    labels = np.array([0, 0, 1, 1])
    start = np.array([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=float)  # bacc 0.75, 0.75
    later = [np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=float)]  # bacc 0.25, 0.5
    out = rp.change_noise(start, later, labels, n_boot=50)
    assert out["loss"] == pytest.approx([0.375])
    assert out["seed_noise"].shape == (50, 1)
    assert out["image_noise"].shape == (50, 1)


def test_change_noise_components_are_centred_and_sized() -> None:
    """No change gives no noise; a random 10% flip gives the binomial image SD."""
    rng = np.random.default_rng(0)
    n, s = 4000, 5
    labels = np.repeat([0, 1], n // 2)
    start = rng.integers(0, 2, size=(s, n)).astype(float)
    later = [start.copy(), np.where(rng.random((s, n)) < 0.1, 1 - start, start)]
    out = rp.change_noise(start, later, labels, n_boot=2000)
    # no change at the first later checkpoint: no noise at all
    assert np.allclose(out["image_noise"][:, 0], 0) and np.allclose(out["seed_noise"][:, 0], 0)
    # 10% of images flip per seed: the per-seed change has mean ~0 and image SD ~ sqrt(0.1 / n / s)
    sd = out["image_noise"][:, 1].std()
    assert sd == pytest.approx(math.sqrt(0.1 / n / s), rel=0.15)
    assert abs(out["image_noise"][:, 1].mean()) < sd / 5


def test_mdd_matches_the_gaussian_formula_at_one_checkpoint() -> None:
    """One checkpoint, Gaussian noise: MDD = (z_0.95 + z_0.80) sd."""
    noise = np.random.default_rng(1).normal(0, 0.01, size=(200_000, 1))
    out = rp.mdd(noise, np.ones(1), alpha=0.05, power=0.8)
    assert out["mdd"] == pytest.approx((1.6449 + 0.8416) * 0.01, rel=0.02)


def test_mdd_grows_with_the_max_over_checkpoints_and_shrinks_for_a_sustained_loss() -> None:
    """The max over checkpoints costs power; a loss held at every one buys it back."""
    noise = np.random.default_rng(2).normal(0, 0.01, size=(100_000, 4))
    one = rp.mdd(noise[:, :1], np.ones(1), alpha=0.05)["mdd"]
    single = rp.mdd(noise, np.array([0, 0, 0, 1.0]), alpha=0.05)["mdd"]
    sustained = rp.mdd(noise, np.ones(4), alpha=0.05)["mdd"]
    assert single > one  # the max over four independent readings needs a larger loss at one of them
    assert sustained < single


def test_combined_noise_scales_each_component() -> None:
    """The image part scales as 1/sqrt(pool), the seed part as 1/sqrt(seeds)."""
    rng = np.random.default_rng(3)
    seed_noise, image_noise = rng.normal(0, 1, (50_000, 1)), rng.normal(0, 1, (50_000, 1))
    assert rp.combined_noise(seed_noise, image_noise).std() == pytest.approx(math.sqrt(2), rel=0.02)
    assert rp.combined_noise(seed_noise, image_noise, pool=4, seeds=4).std() == pytest.approx(math.sqrt(0.5), rel=0.02)


def test_pool_for_is_infinite_when_the_seed_part_alone_misses_the_target() -> None:
    """A seed part above the target forbids any pool; without one, the pool follows 1/sqrt(n)."""
    rng = np.random.default_rng(4)
    seed_noise, image_noise = rng.normal(0, 0.02, (20_000, 1)), rng.normal(0, 0.02, (20_000, 1))
    assert rp.pool_for(seed_noise, image_noise, np.ones(1), 0.02, alpha=0.05) == math.inf
    # with a negligible seed part, the pool needed follows 1/sqrt(n): target half the image MDD -> ~4x
    pool = rp.pool_for(seed_noise * 1e-6, image_noise, np.ones(1), 0.5 * 2.4865 * 0.02, alpha=0.05)
    assert pool == pytest.approx(4.0, rel=0.1)


def test_dense_grid_mdd_is_bonferroni() -> None:
    """The dense-grid bound is Bonferroni over the checkpoints."""
    assert rp.dense_grid_mdd(0.01, 1, alpha=0.05) == pytest.approx((1.6449 + 0.8416) * 0.01, rel=1e-3)
    assert rp.dense_grid_mdd(0.01, 10, alpha=0.05) > rp.dense_grid_mdd(0.01, 1, alpha=0.05)


def test_wilson_interval() -> None:
    """The Wilson interval matches its tabled value, and rejects k > n."""
    lo, hi = wilson_interval(1, 15)
    assert lo == pytest.approx(0.0119, abs=1e-3) and hi == pytest.approx(0.2983, abs=1e-3)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


# ---- the split adjudication (Part 1) -------------------------------------------------------------------------


def test_adjudicate_split_reads_information_and_geometry_apart() -> None:
    """Linear canary = degradation; kNN canary = geometry unless a fine-tune rung drops with it."""
    assert adjudicate_split({}) == "quiet"
    assert adjudicate_split({"label_free": True}) == "movement"
    assert adjudicate_split({"label_free": True, "canary_knn": True}) == "geometry"
    assert adjudicate_split({"canary_knn": True}, finetune=None) == "geometry"  # corroboration unread
    assert adjudicate_split({"canary_knn": True}, finetune=False) == "geometry"
    assert adjudicate_split({"canary_knn": True}, finetune=True) == "degradation"
    assert adjudicate_split({"canary_linear": True}, finetune=False) == "degradation"


def test_finetune_drop_counts_only_checkpoints_at_or_after_the_fire_with_a_healthy_read() -> None:
    """Below every healthy seed at an eligible checkpoint drops; earlier checkpoints and unread ones do not count."""
    steps = {"t00100": 100.0, "t00846": 846.0, "end": math.inf}
    healthy = {"t00100": [0.60, 0.62], "end": [0.60, 0.61]}
    assert finetune_drop({"t00100": 0.50, "end": 0.605}, healthy, steps, after=500) is False  # the drop came before
    assert finetune_drop({"t00100": 0.50, "end": 0.59}, healthy, steps, after=500) is True
    assert finetune_drop({"t00846": 0.10}, healthy, steps, after=500) is None  # no healthy read at 846
    assert finetune_drop({"t00100": 0.50}, healthy, steps, after=0) is True


def test_three_panels_split_the_canary_by_rung() -> None:
    """The label-free signals stay one panel; the canary channels split into linear and kNN."""
    rows = [{"rankme": 1.0, "knn_acc": 0.5, "knn_acc_night": 0.5, "linear_acc": 0.6}]
    out = three_panels(rows, ["rankme", "uniformity"], ["knn_acc", "linear_acc", "knn_acc_night", "linear_acc_night"])
    assert out == {
        "label_free": ["rankme"],
        "canary_linear": ["linear_acc"],
        "canary_knn": ["knn_acc", "knn_acc_night"],
    }


def test_welch_interval_and_exact_permutation() -> None:
    """The Welch interval covers the difference; the permutation p of a complete separation is 2 / C(10, 5)."""
    a, b = np.array([1.0, 1.1, 1.2, 1.3, 1.4]), np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    diff, lo, hi = _welch_ci(a, b)
    assert diff == pytest.approx(1.0) and lo < 1.0 < hi and lo > 0
    assert _perm_p(a, b) == pytest.approx(2 / 252)
    assert _perm_p(a, a.copy()) == 1.0


# ---- damage on DINO's method and the resumed momentum (Part 2's engineering) ---------------------------------


def _tiny_dino() -> DINO:
    method = build_dino(_tiny_vit(), **_DINO_SMALL)
    with torch.no_grad():  # a trained student, apart from the teacher
        for p in method.student.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return method


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


@pytest.mark.parametrize(("mode", "k", "scale"), [("none", 2, 1.0), ("reinit", 0, 0.0), ("noise", 2, 0.0)])
def test_damage_method_at_dose_zero_is_a_no_op_on_the_whole_method(mode: str, k: int, scale: float) -> None:
    """Dose 0 leaves teacher, student, heads and centre bit for bit."""
    method = _tiny_dino()
    before = _snapshot(method)
    out = damage_method(method, mode, k, scale=scale)
    assert out["rel_change"] == 0.0 and out["twin_rel_change"] == 0.0
    assert all(torch.equal(before[n], v) for n, v in method.state_dict().items())


@pytest.mark.parametrize(("mode", "scale"), [("reinit", 0.0), ("noise", 0.5)])
def test_damage_method_hits_teacher_and_student_alike_and_nothing_else(mode: str, scale: float) -> None:
    """The same damage reaches both backbones: equal blocks stay equal; heads and centre stay as they were."""
    method = _tiny_dino()
    with torch.no_grad():  # make the damaged block equal in both, to test "equal stays equal"
        top_blocks(method.student, 1)[0].load_state_dict(top_blocks(method.encoder, 1)[0].state_dict())
    before = _snapshot(method)
    out = damage_method(method, mode, 1, scale=scale, seed=3)
    assert out["rel_change"] > 0 and out["twin_rel_change"] > 0
    teacher, student = top_blocks(method.encoder, 1)[0].state_dict(), top_blocks(method.student, 1)[0].state_dict()
    assert all(torch.equal(teacher[n], student[n]) for n in teacher)
    after = method.state_dict()
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    last = f"blocks.{len(method.encoder.blocks) - 1}."
    assert changed and all(n.startswith((f"encoder.{last}", f"student.{last}")) for n in changed)


def test_damage_method_noise_keeps_the_teacher_student_difference() -> None:
    """Under noise the two get the same delta, so the moving average's lag behind the student is unchanged."""
    method = _tiny_dino()
    t, s = top_blocks(method.encoder, 2), top_blocks(method.student, 2)
    gap = [(a.mlp.fc1.weight - b.mlp.fc1.weight).clone() for a, b in zip(t, s, strict=True)]
    damage_method(method, "noise", 2, scale=0.7, seed=1)
    for g, a, b in zip(gap, t, s, strict=True):
        assert torch.allclose(a.mlp.fc1.weight - b.mlp.fc1.weight, g, atol=1e-6)


def test_damage_without_a_twin_is_p111s_draw() -> None:
    """A method with no student (SimSiam, MAE) is damaged exactly as P1.1.1 damaged it."""
    torch.manual_seed(0)
    enc = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=16, depth=4, num_heads=2)
    holder = torch.nn.Module()
    holder.encoder = copy.deepcopy(enc)
    out = damage_method(holder, "noise", 2, scale=0.3, seed=4)
    damage_top_blocks(enc, "noise", 2, scale=0.3, seed=4)
    assert "twin_rel_change" not in out
    assert all(
        torch.equal(a, b) for a, b in zip(enc.state_dict().values(), holder.encoder.state_dict().values(), strict=False)
    )


def test_damage_rejects_a_mismatched_twin() -> None:
    """A twin whose blocks differ in shape is refused."""
    a = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=16, depth=4, num_heads=2)
    b = TinyViTEncoder(img_size=8, patch_size=4, embed_dim=8, depth=4, num_heads=2)
    with pytest.raises(ValueError, match="twin"):
        damage_top_blocks(a, "noise", 2, scale=0.1, twin=b)


def test_resumed_dino_continues_the_drives_momentum_ramp() -> None:
    """set_horizon(total, start=s): the next update uses the momentum the drive would have used at step s + 1."""
    method = _tiny_dino()
    method.set_horizon(10, start=6)
    before = method.encoder.norm.weight.clone()
    method.after_step()
    m = momentum_at(7, 10, 0.996)
    assert m > momentum_at(1, 10, 0.996)
    assert torch.allclose(method.encoder.norm.weight, m * before + (1 - m) * method.student.norm.weight)
    with pytest.raises(ValueError, match="start"):
        method.set_horizon(10, start=11)


def test_gap_fast_matches_the_loop_with_skips_and_tied_checkpoints() -> None:
    """The vectorized null reproduces P1.1.0's loop, skipped readings and a shared start included."""
    rng = np.random.default_rng(3)
    for trial in range(12):
        walk = rng.normal(size=(5, 9)) * 0.02 + 0.6
        iid = rng.normal(size=(6, 9)) * 0.02 + 0.6 + 0.01 * trial
        walk[rng.random(walk.shape) < 0.1] = np.nan
        iid[rng.random(iid.shape) < 0.1] = np.nan
        walk[:, 0] = iid[:, 0] = 0.5  # every drive reads the same start
        a, b = gap(walk, iid), gap_fast(walk, iid)
        for key in a:
            assert math.isclose(a[key], b[key], rel_tol=1e-9, abs_tol=1e-12), (trial, key)


def test_gap_fast_reads_coinciding_unrepresentable_readings_as_no_gap() -> None:
    """Readings that coincide across arms read as no gap, even off the binary grid."""
    walk, iid = np.full((5, 3), 0.1), np.full((5, 3), 0.1)
    walk[:, 1] += np.linspace(0, 0.01, 5)
    iid[:, 1] += np.linspace(0, 0.01, 5)
    assert gap_fast(walk, iid)["max_abs_d"] == pytest.approx(0.0, abs=1e-9)


def test_split_masks_enumerate_every_split_once() -> None:
    """Every split of the seeds appears exactly once."""
    masks = split_masks(6, 2)
    assert masks.shape == (15, 6)
    assert (masks.sum(axis=1) == 2).all()
    assert len({tuple(r) for r in masks}) == 15


def test_drop_profile_shapes_end_at_the_last_usable_reading() -> None:
    """Each drop shape covers its share of the drive, ending at the last usable reading."""
    assert drop_profile(8, "sustained").tolist() == [1.0] * 8
    assert drop_profile(8, "last_quarter").tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    assert drop_profile(8, "single", last=5).tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert drop_profile(8, "sustained", first=3).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="unknown"):
        drop_profile(8, "ramp")


def test_injected_power_rises_with_the_drop_and_holds_the_level_at_zero() -> None:
    """No drop is rarely caught; a large one always is."""
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(10, 6)) * 0.01 + 0.6
    masks = split_masks(10, 3)
    groups = masks[rng.choice(len(masks), 40, replace=False)]
    profile = drop_profile(6, "sustained")
    at_zero = injected_power(healthy, profile, 0.0, groups, masks)
    small, large = (injected_power(healthy, profile, d, groups, masks) for d in (0.005, 0.03))
    assert at_zero <= 0.2
    assert small <= large
    assert large == 1.0


def test_injected_mdd_is_larger_for_a_single_reading_than_a_sustained_drop() -> None:
    """A drop held over the drive is caught at a smaller size than one at a single reading."""
    rng = np.random.default_rng(1)
    healthy = rng.normal(size=(10, 8)) * 0.01 + 0.6
    sustained = injected_mdd(healthy, "sustained", n_inject=3, n_groups=30)
    single = injected_mdd(healthy, "single", n_inject=3, n_groups=30)
    assert sustained["mdd"] < single["mdd"]
    assert sustained["power_at_mdd"] >= 0.8


def _ladder_dir(path: Path, reads: dict[str, np.ndarray], query: np.ndarray) -> None:
    """A minimal ladder dir: a 4-frame scene pool and one kNN read per encoder."""
    path.mkdir(parents=True)
    labels = np.array([0, 1, 0, 1])
    np.savez(path / "pool.npz", query_index=query, stratum=np.zeros(4), regime=np.zeros(4), **{"labels|scene": labels})
    (path / "summary.json").write_text("{}", encoding="utf-8")
    for name, correct in reads.items():
        np.savez(path / f"{name}.npz", **{"scene|knn": correct[None, :]})


def test_ladder_set_maps_index_and_drive_seed_naming(tmp_path: Path) -> None:
    """P1.2.5's index naming and P1.1.4's drive-seed naming read the same drive; a mismatched query is refused."""
    query = np.arange(4)
    _ladder_dir(tmp_path / "old", {"walk_s0_end": np.array([1.0, 1, 0, 0])}, query)
    _ladder_dir(tmp_path / "new", {"iid_s6_end": np.array([1.0, 1, 1, 1])}, query)
    ladders = LadderSet(tmp_path, ["old", {"dir": "new", "names": "seed"}])
    assert ladders.read("walk", 1, "end", "scene|knn") == pytest.approx(0.5)
    assert ladders.read("iid", 6, "end", "scene|knn") == pytest.approx(1.0)
    assert math.isnan(ladders.read("iid", 1, "end", "scene|knn"))
    _ladder_dir(tmp_path / "other", {}, np.arange(1, 5))
    with pytest.raises(ValueError, match="val query"):
        LadderSet(tmp_path, ["old", "other"])


def test_power_rows_read_seed_named_dirs_and_the_floor(tmp_path: Path) -> None:
    """Parts 4 and 4b ask by index (``walk_s0``); a seed-named dir answers with its ``walk_s1``, the floor as is."""
    query = np.arange(4)
    _ladder_dir(tmp_path / "mae", {"walk_s1_end": np.array([1.0, 0, 0, 0]), "floor": np.ones(4)}, query)
    _ladder_dir(tmp_path / "old", {"walk_s0_end": np.array([0.0, 0, 0, 0])}, query)
    ladders = _open_ladders(tmp_path, [{"dir": "mae", "names": "seed"}, "old"])
    assert _own_name("walk_s0_t00846", "seed") == "walk_s1_t00846" and _own_name("floor", "seed") == "floor"
    rows = _rows_any(ladders, ["walk_s0_end"], "scene|knn")
    assert rows is not None and rows.tolist() == [[1.0, 0, 0, 0]]
    assert _rows_any(ladders, ["floor"], "scene|knn") is not None
    assert _rows_any(ladders[1:], ["walk_s0_end"], "scene|knn").tolist() == [[0.0, 0, 0, 0]]  # type: ignore[union-attr]
    _ladder_dir(tmp_path / "other", {}, np.arange(1, 5))
    with pytest.raises(ValueError, match="val query"):
        _open_ladders(tmp_path, ["old", "other"])


def test_finetune_drop_consecutive_needs_a_run_of_drops() -> None:
    """One checkpoint below every healthy seed is not enough; two in a row are."""
    steps = {"a": 100.0, "b": 200.0, "c": 300.0, "end": math.inf}
    healthy = {c: [0.6, 0.62] for c in steps}
    single = {"a": 0.5, "b": 0.61, "c": 0.5, "end": 0.61}
    run = {"a": 0.61, "b": 0.5, "c": 0.5, "end": 0.61}
    assert finetune_drop_consecutive(single, healthy, steps, 0.0) is False
    assert finetune_drop_consecutive(run, healthy, steps, 0.0) is True
    assert finetune_drop_consecutive(run, healthy, steps, 250.0) is False
    assert finetune_drop_consecutive(run, healthy, steps, 1e9) is None


def test_finetune_reads_are_harmful_when_they_fall() -> None:
    """A fine-tune read is a one-sided panel: a fall is harm."""
    assert direction("finetune_acc_scene|ft100") == -1
