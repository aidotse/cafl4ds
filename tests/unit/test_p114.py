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
from cafl4ds.reading_protocol import adjudicate_split, finetune_drop
from cafl4ds.ssl.dino import DINO, build_dino, momentum_at
from scripts.analyze_p114 import _perm_p, _welch_ci, three_panels, wilson_interval

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
