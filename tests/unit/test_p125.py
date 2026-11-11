"""Unit tests for P1.2.5's engineering: geometry labels, the new tasks and readouts, and the seed-level tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from cafl4ds import adaptation
from cafl4ds.adaptation import HeldOutPool
from cafl4ds.data.attributes import BDD100KSource, SyntheticAttributeSource
from cafl4ds.data.offdomain import COCOSource
from cafl4ds.data.regime import RegimeStream
from cafl4ds.health_trust import Backbone, backbone_family
from cafl4ds.models.vit import TinyViTEncoder
from cafl4ds.optim_groups import param_groups
from cafl4ds.ssl.dino import DINOHead, build_dino, momentum_at

sys.path.insert(0, str(Path(__file__).parent))
from test_offdomain import _coco  # noqa: E402
from test_scripts_e2e import _P11_DIET, _compose, _load_script  # noqa: E402

_ATTRS = {"weather": "clear", "scene": "highway", "timeofday": "daytime"}


def _bdd_with_geometry(root: Path) -> Path:
    """Two 32 x 24 frames with boxes, and a drivable map (ids 0 background, 1 direct, 2 alternative) for the first."""
    images = root / "images" / "100k" / "train"
    images.mkdir(parents=True)
    (root / "labels").mkdir()
    records: list[dict[str, Any]] = []
    for name in ("a", "b"):
        Image.new("RGB", (32, 24)).save(images / f"{name}.jpg")
        records.append({"name": f"{name}.jpg", "attributes": _ATTRS})
    records[0]["labels"] = [
        {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 16, "y2": 12}},  # a quarter of the frame
        {"category": "traffic light", "box2d": {"x1": 0, "y1": 0, "x2": 2, "y2": 2}},
    ]
    records[1]["labels"] = [{"category": "bus"}]  # no geometry
    (root / "labels" / "bdd100k_labels_images_train.json").write_text(json.dumps(records))
    maps = root / "drivable" / "train"
    maps.mkdir(parents=True)
    ids = np.zeros((24, 32), dtype=np.uint8)
    ids[12:, :16] = 1  # bottom-left quarter direct ...
    ids[12:, 16:] = 2  # ... bottom-right alternative: the whole bottom half is drivable
    Image.fromarray(ids).save(maps / "a_drivable_id.png")
    return root


def test_bdd_source_reads_box_areas_and_pools_the_drivable_map_to_the_patch_grid(tmp_path: Path) -> None:
    """Box areas are frame shares; drivable ids >= 1 count; a frame without a map is NaN; the cache key moves."""
    root = _bdd_with_geometry(tmp_path)
    plain = BDD100KSource(str(root), img_size=16, cache_dir=str(tmp_path / "cache"))
    source = BDD100KSource(
        str(root),
        img_size=16,
        boxes=True,
        drivable_dir=str(root / "drivable"),
        drivable_grid=2,
        cache_dir=str(tmp_path / "cache"),
    )
    assert plain._cache_path() != source._cache_path()
    data = source.load()
    assert data.object_boxes == [[("car", 0.25), ("traffic light", 4 / 768)], []]
    assert data.object_categories == [{"car": 1, "traffic light": 1}, {"bus": 1}]
    assert data.drivable is not None and data.drivable[0].tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert torch.isnan(data.drivable[1]).all()
    assert plain.load().object_boxes is None and plain.load().drivable is None


def test_bdd_source_refuses_boxes_from_a_file_without_geometry(tmp_path: Path) -> None:
    """``boxes=True`` on the category-only val file fails loudly rather than reading every frame as box-free."""
    root = _bdd_with_geometry(tmp_path)
    labels = root / "labels" / "bdd100k_labels_images_train.json"
    labels.write_text(json.dumps([{**r, "labels": [{"category": "car"}]} for r in json.loads(labels.read_text())]))
    with pytest.raises(ValueError, match="box2d"):
        BDD100KSource(str(root), img_size=16, boxes=True).load()


def test_bdd_legacy_labels_takes_val_boxes_from_a_fiftyone_export_in_the_reference_order(tmp_path: Path) -> None:
    """Relative FiftyOne boxes become pixel box2d; the reference's order and attributes are kept."""
    script = _load_script("bdd_legacy_labels")
    sample = {
        "filepath": "data/b.jpg",
        "metadata": {"width": 100, "height": 50},
        "detections": {"detections": [{"label": "car", "bounding_box": [0.1, 0.2, 0.5, 0.4]}]},
    }
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps({"samples": [{**sample, "filepath": "data/a.jpg", "detections": None}, sample]}))
    reference = [
        {"name": "b.jpg", "attributes": _ATTRS, "labels": [{"category": "car"}]},
        {"name": "a.jpg", "attributes": _ATTRS},
    ]
    records = script.fiftyone_records(samples, reference)
    assert [r["name"] for r in records] == ["b.jpg", "a.jpg"]
    assert [b["category"] for b in records[0]["labels"]] == ["car"]
    assert records[0]["labels"][0]["box2d"] == pytest.approx({"x1": 10.0, "y1": 10.0, "x2": 60.0, "y2": 30.0})
    assert script.compare(records, reference)["box_count_mismatches"] == 0
    with pytest.raises(ValueError, match="missing"):
        script.fiftyone_records(samples, [*reference, {"name": "c.jpg", "attributes": _ATTRS}])


def _pool(n: int = 8) -> HeldOutPool:
    drivable = torch.zeros(n, 2, 2)
    drivable[:, 1] = 0.8
    drivable[-1] = float("nan")
    return HeldOutPool(
        images=torch.rand(n, 3, 8, 8),
        scene=torch.arange(n) % 2,
        stratum=torch.zeros(n, dtype=torch.long),
        regime=torch.zeros(n, dtype=torch.long),
        boxes=[[("car", 0.03)], [("car", 0.01)], [("bus", 0.05)], [], [("truck", 0.02)], [], [("car", 0.019)], []],
        drivable=drivable,
    )


def test_large_cell_reads_box_areas_against_a_threshold_and_is_cell_balanced() -> None:
    """Any listed category at or above the area counts; other categories and small boxes do not."""
    pool = _pool()
    labels = adaptation.task_labels(pool, "large_cell:car,truck:0.02").tolist()
    assert labels == [1, 0, 0, 0, 1, 0, 0, 0]
    kept = adaptation.task_indices(pool, "large_cell:car,truck:0.02", np.arange(8), seed=0)
    assert np.bincount(np.array(labels)[kept]).tolist() == [2, 2]
    with pytest.raises(ValueError, match="min_area"):
        adaptation.task_labels(pool, "large_cell:car")
    with pytest.raises(ValueError, match="box areas"):
        adaptation.task_labels(HeldOutPool(pool.images, pool.scene, pool.stratum, pool.regime), "large_cell:car:0.1")


def test_size_cell_bins_the_largest_box_and_drops_frames_whose_largest_box_is_out_of_bin() -> None:
    """Positive: the category's largest box is in the bin; negative: no box of it; otherwise unused (-1)."""
    pool = _pool()
    # car boxes: 0.03, 0.01, none, none, none, none, 0.019, none (a frame with a bus or truck only is a car negative)
    assert adaptation.task_labels(pool, "size_cell:car:0.005:0.02").tolist() == [-1, 1, 0, 0, 0, 0, 1, 0]
    assert adaptation.task_labels(pool, "size_cell:car:0.02:inf").tolist() == [1, -1, 0, 0, 0, 0, -1, 0]
    # the largest box decides: a small car beside a large one puts the frame in the large bin only
    two = HeldOutPool(
        pool.images[:1], pool.scene[:1], pool.stratum[:1], pool.regime[:1], boxes=[[("car", 0.001), ("car", 0.05)]]
    )
    assert adaptation.task_labels(two, "size_cell:car:0:0.005").tolist() == [-1]
    assert adaptation.task_labels(two, "size_cell:car:0.02:inf").tolist() == [1]
    # a category name with a space parses; cell balancing applies
    spaced = HeldOutPool(
        pool.images[:2], pool.scene[:2], pool.stratum[:2], pool.regime[:2], boxes=[[("traffic sign", 0.001)], []]
    )
    assert adaptation.task_labels(spaced, "size_cell:traffic sign:0:0.005").tolist() == [1, 0]
    kept = adaptation.task_indices(pool, "size_cell:car:0.005:0.02", np.arange(8), seed=0)
    labels = adaptation.task_labels(pool, "size_cell:car:0.005:0.02").numpy()
    assert (labels[kept] >= 0).all() and np.bincount(labels[kept]).tolist() == [2, 2]
    for bad in ("size_cell:car:0.02", "size_cell:car:0.02:0.01"):
        with pytest.raises(ValueError, match="size_cell|min_area"):
            adaptation.task_labels(pool, bad)


def test_bootstrap_combination_reads_within_slice_changes_and_their_differences() -> None:
    """A single term equals bootstrap_delta against zero; a difference of identical slices is exactly zero."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 50)
    early = (rng.random((4, 100)) < 0.6).astype(float)
    late = np.clip(early + (rng.random((4, 100)) < 0.2), 0, 1)
    one = adaptation.bootstrap_combination([(1.0, late - early, labels)], n_boot=500, seed=1)
    ref = adaptation.bootstrap_delta(late - early, np.zeros(100), labels, n_boot=500, seed=1)
    assert one["delta"] == pytest.approx(ref["delta"]) and one["delta"] > 0 and one["lo"] > 0
    assert one["delta"] == pytest.approx(
        np.mean([adaptation.balanced_accuracy(r, labels) for r in late])
        - np.mean([adaptation.balanced_accuracy(r, labels) for r in early])
    )
    same = adaptation.bootstrap_combination(
        [(1.0, late - early, labels), (-1.0, late - early, labels)], n_boot=200, seed=0
    )
    assert same["delta"] == pytest.approx(0.0) and same["n_images"] == 200.0
    with pytest.raises(ValueError, match="same seeds"):
        adaptation.bootstrap_combination([(1.0, late, labels), (1.0, late[:2], labels)])


def test_drivable_labels_are_per_patch_and_frames_without_a_map_are_dropped() -> None:
    """A patch is drivable above half; the per-frame label only marks which frames carry a map."""
    pool = _pool()
    assert adaptation.is_patch_task("drivable") and not adaptation.is_patch_task("scene")
    assert adaptation.task_labels(pool, "drivable").tolist() == [0] * 7 + [-1]
    patches = adaptation.patch_labels(pool)
    assert patches[0].tolist() == [0, 0, 1, 1] and patches[-1].tolist() == [-1] * 4
    assert adaptation.patch_counts(patches.numpy())[:2].tolist() == [[2, 2], [2, 2]]


def test_patch_probes_score_separable_patches_and_the_patch_bootstrap_pairs_frames() -> None:
    """Linearly separable patch tokens read perfectly; the patch bootstrap is zero against itself and positive above."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(40, 9))
    tokens = (labels[..., None] * 4.0 + rng.normal(size=(40, 9, 5))).astype(np.float16)
    counts = adaptation.patch_counts(labels[20:])
    out = adaptation.patch_correctness(tokens, labels, np.arange(20), np.arange(20, 40), 3, 50, 0)
    assert adaptation.patch_balanced_accuracy(out["linear"], counts) > 0.95
    half = counts * 0.5
    same = adaptation.bootstrap_delta_patches(half[None], half, counts, n_boot=200)
    assert same["delta"] == pytest.approx(0.0) and same["lo"] == pytest.approx(0.0)
    better = adaptation.bootstrap_delta_patches(np.stack([counts, counts * 0.9]), half, counts, n_boot=200)
    assert better["delta"] == pytest.approx(0.45) and better["lo"] > 0 and better["p_one_sided"] < 0.01


def test_seed_level_tests_match_student_t() -> None:
    """The t tail reproduces tabled critical values; the one-sample and Welch tests read the seeds' spread."""
    assert adaptation.t_upper_tail(2.776, 4) == pytest.approx(0.025, abs=1e-4)
    assert adaptation.t_upper_tail(-2.015, 5) == pytest.approx(0.95, abs=1e-4)
    one = adaptation.seed_ttest(np.array([0.52, 0.53, 0.55, 0.51, 0.54]), 0.50)
    assert one["df"] == 4 and one["t"] == pytest.approx(4.2426, abs=1e-4) and one["p_one_sided"] < 0.01
    welch = adaptation.seed_welch(np.array([0.52, 0.53, 0.55]), np.array([0.50, 0.49, 0.51, 0.50]))
    assert 2 < welch["df"] < 5 and 0.01 < welch["p_one_sided"] < 0.05
    assert adaptation.seed_ttest(np.array([0.6, 0.6]), 0.5)["p_one_sided"] == 0.0
    with pytest.raises(ValueError):
        adaptation.seed_ttest(np.array([0.6]), 0.5)


def test_readout_reads_the_mean_or_the_class_token() -> None:
    """``mean`` reproduces the cafl4ds ViT's own embed; ``cls`` is the first token; both come from one pass."""
    torch.manual_seed(0)
    encoder = TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=2).eval()
    imgs = torch.rand(3, 3, 16, 16)
    with torch.no_grad():
        tokens = encoder.tokens(imgs)
        assert torch.allclose(adaptation.Readout(encoder, "mean").embed(imgs), encoder.embed(imgs))
        assert torch.allclose(adaptation.Readout(encoder, "cls").embed(imgs), tokens[:, 0])
    both = adaptation.encode_readouts(encoder, imgs, "cpu")
    assert np.allclose(both["cls"], tokens[:, 0].numpy(), atol=1e-6)
    assert adaptation.encode_patches(encoder, imgs, "cpu").shape == (3, 4, 12)
    with pytest.raises(ValueError):
        adaptation.Readout(encoder, "max")


def test_vit_runs_a_smaller_crop_on_interpolated_positions() -> None:
    """A crop smaller than the build size gets bicubically interpolated position embeddings (DINO's local crops)."""
    encoder = TinyViTEncoder(img_size=32, patch_size=8, embed_dim=12, depth=1, num_heads=2)
    tokens, _, _ = encoder.forward_encoder(torch.rand(2, 3, 16, 16), resize=False)
    assert tokens.shape == (2, 1 + 4, 12)


def test_synthetic_geometry_and_the_selection_set_as_an_attributed_pool() -> None:
    """The synthetic source carries boxes and a drivable grid; the stream exposes the selection set's indices."""
    source = SyntheticAttributeSource(num_regimes=2, per_cell=12, img_size=16, long_tail=False, geometry=True)
    data = source.load()
    assert data.object_boxes is not None and data.drivable is not None and data.drivable.shape[1:] == (4, 4)
    assert data.object_categories is not None and data.object_categories[0]["car"] == 1
    stream = RegimeStream(
        source, support_per_canary=2, query_per_canary=2, select_support_per_canary=3, select_query_per_canary=3
    )
    indices = stream.selection_indices
    assert indices is not None and indices[0].numel() == indices[1].numel() == 3 * 4
    assert stream.selection_sets is not None
    assert torch.equal(stream.selection_sets.probe_support.labels, data.canary[indices[0]])
    assert RegimeStream(source, support_per_canary=2, query_per_canary=2).selection_indices is None


def test_offdomain_subset_keeps_a_seed_fixed_share_of_the_cached_draw(tmp_path: Path) -> None:
    """A subset is drawn from the decode cache (so it loads without the archive) and relabels the placeholder canary."""
    archive, annotations = _coco(tmp_path)
    kwargs: dict[str, Any] = {
        "archive": str(archive),
        "annotations": str(annotations),
        "n_images": 5,
        "exclude": ["car", "bus"],
        "crop_aspect": 16 / 9,
        "img_size": 16,
        "cache_dir": str(tmp_path / "cache"),
        "workers": 1,
    }
    full = COCOSource(**kwargs).load()
    archive.unlink()
    sub = COCOSource(**kwargs, subset=3, subset_seed=1).load()
    keep = np.sort(np.random.default_rng(1).choice(5, size=3, replace=False))
    assert torch.equal(sub.images, full.images[torch.from_numpy(keep)])
    assert sub.canary.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="subset"):
        COCOSource(**kwargs, subset=6)


_GEOMETRY_DIET = [
    *_P11_DIET,
    "data.geometry=true",
    "data.drivable_grid=2",
    "stream.select_support_per_canary=6",
    "stream.select_query_per_canary=6",
]


def test_reader_reads_the_panel_on_the_selection_set_with_cls_rungs_and_the_patch_task(tmp_path: Path) -> None:
    """The selection set carries every task; the patch task stores per-frame correct-patch counts and their totals."""
    tasks = 'tasks=[scene,"large_cell:car,bus:0.02",drivable]'
    overrides = [
        *_GEOMETRY_DIET,
        "pool.kind=selection",
        tasks,
        "probe_per_class=4",
        "threads=1",
        "readout=mean",
        "cls_rungs=true",
        "patch.fit_frames=10",
        "patch.per_class=20",
        "patch.query_frames=10",
        "fewshot.enabled=true",
        "fewshot.ks=[2]",
        "fewshot.draws=1",
        "fewshot.steps=2",
        "fewshot.tasks=[scene]",
    ]
    summary = _load_script("read_adaptation").run(_compose("read_adaptation", overrides), tmp_path / "sel")
    keys = set(summary["encoders"]["b5"]["bacc"])
    assert {"scene|linear", "scene|linear@cls", "scene|ft2", "large_cell:car,bus:0.02|knn", "drivable|linear"} <= keys
    assert summary["readout"] == "mean" and summary["n_query_by_task"]["drivable"] == 10
    pool = np.load(tmp_path / "sel" / "pool.npz")
    counts = pool["counts|drivable"]
    arrays = np.load(tmp_path / "sel" / "b5.npz")
    assert arrays["drivable|linear"].shape == (1, 10, 2) and (arrays["drivable|linear"][0] <= counts).all()
    assert 0.0 <= summary["encoders"]["b5"]["bacc"]["drivable|linear"]["pooled"] <= 1.0


def test_screen_reads_the_patch_task_and_returns_a_ranked_panel(tmp_path: Path) -> None:
    """The per-patch candidate gets a patch-level row; the panel keeps up to max_tasks eligible tasks by headroom."""
    overrides = [
        *_GEOMETRY_DIET,
        "mode=screen",
        "probe_per_class=6",
        "threads=1",
        "fewshot.select_k=2",
        "fewshot.draws=1",
        "fewshot.steps=2",
        "screen.ceiling_per_class=4",
        "screen.ceiling_steps=2",
        'screen.candidates=[scene,drivable,"large_cell:car:0.02"]',
        "screen.max_overlap=1.0",
        "screen.min_headroom=-1.0",
        "screen.max_tasks=2",
        "patch.fit_frames=10",
        "patch.per_class=20",
        "patch.query_frames=10",
    ]
    report = _load_script("read_adaptation").run(_compose("read_adaptation", overrides), tmp_path / "screen")
    row = report["tasks"]["drivable"]
    assert {"overlap", "b5_linear", "b5_knn", "ceiling", "headroom"} <= set(row) and row["b5_ft2"] is None
    assert len(report["panel"]) == 2 and report["choice"] == report["panel"][0]
    headroom = [report["tasks"][t]["headroom"] for t in report["panel"]]
    assert headroom == sorted(headroom, reverse=True)


# ---- DINO's objective (cafl4ds/ssl/dino.py) ----

_DINO_SMALL: dict[str, Any] = {
    "out_dim": 32,
    "hidden_dim": 16,
    "bottleneck_dim": 8,
    "global_size": 16,
    "local_size": 8,
    "n_local": 2,
}


def _tiny_vit() -> TinyViTEncoder:
    torch.manual_seed(0)
    return TinyViTEncoder(img_size=16, patch_size=8, embed_dim=12, depth=2, num_heads=2)


def _full_checkpoint(path: Path, encoder: TinyViTEncoder) -> dict[str, torch.Tensor]:
    """An official-layout DINO checkpoint: DDP-prefixed student, plain teacher, timm patch-embed names, args, centre."""

    def backbone(prefix: str, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {f"{prefix}backbone.{k.replace('patch_embed.', 'patch_embed.proj.')}": v for k, v in state.items()}

    own = {k: v for k, v in encoder.state_dict().items() if not k.startswith("_")}
    student = {k: v + 0.5 for k, v in own.items()}
    head = DINOHead(12, 32, hidden_dim=16, bottleneck_dim=8).state_dict()
    ckpt = {
        "student": {**backbone("module.", student), **{f"module.head.{k}": v + 1.0 for k, v in head.items()}},
        "teacher": {**backbone("", own), **{f"head.{k}": v for k, v in head.items()}},
        "dino_loss": {"center": torch.full((1, 32), 0.25)},
        "args": argparse.Namespace(out_dim=32, use_bn_in_head=False, norm_last_layer=True),
    }
    torch.save(ckpt, path)
    return student


def test_dino_resumes_the_full_checkpoint_and_refuses_a_teacher_that_is_not_the_encoder(tmp_path: Path) -> None:
    """Student, heads and centre come from the checkpoint; the teacher must be the configured encoder bit for bit."""
    encoder = _tiny_vit()
    student = _full_checkpoint(tmp_path / "full.pth", encoder)
    method = build_dino(encoder, checkpoint=str(tmp_path / "full.pth"), **_DINO_SMALL)
    assert torch.equal(method.student.blocks[0].attn.qkv.weight, student["blocks.0.attn.qkv.weight"])
    assert torch.equal(method.center, torch.full((1, 32), 0.25)) and method.name == "dino"
    assert not torch.equal(method.student_head.mlp[0].weight, method.teacher_head.mlp[0].weight)
    assert not any(p.requires_grad for p in [*method.encoder.parameters(), *method.teacher_head.parameters()])
    assert not method.student_head.last_layer.weight_g.requires_grad  # norm_last_layer: the gain stays 1
    with pytest.raises(ValueError, match="teacher differs"):
        build_dino(_perturbed(), checkpoint=str(tmp_path / "full.pth"), **_DINO_SMALL)


def _perturbed() -> TinyViTEncoder:
    encoder = _tiny_vit()
    with torch.no_grad():
        encoder.norm.weight.add_(1.0)
    return encoder


def test_dino_step_trains_the_student_and_the_teacher_follows_by_the_scheduled_momentum() -> None:
    """Gradients reach only the student side; after_step moves the teacher toward it; the momentum ramps to 1."""
    assert momentum_at(0, 10, 0.996) == pytest.approx(0.996) and momentum_at(10, 10, 0.996) == pytest.approx(1.0)
    assert 0.996 < momentum_at(5, 10, 0.996) < 1.0
    method = build_dino(_tiny_vit(), **_DINO_SMALL)
    with torch.no_grad():
        for p in method.student.parameters():
            p.add_(0.1)
    method.set_horizon(4)
    loss = method.training_step(torch.rand(4, 3, 16, 16))
    assert torch.isfinite(loss) and float(method.center.abs().sum()) > 0  # the centre moved
    loss.backward()
    assert all(p.grad is not None for p in method.student.parameters())
    assert all(p.grad is None for p in method.encoder.parameters())
    before = method.encoder.norm.weight.clone()
    method.after_step()
    m = momentum_at(1, 4, 0.996)
    assert torch.allclose(method.encoder.norm.weight, m * before + (1 - m) * method.student.norm.weight)


def test_dino_collapse_switches_name_the_method_and_turn_one_mechanism_off() -> None:
    """centering=False keeps the centre still; sharpening=False puts the teacher at the student's temperature."""
    nocenter = build_dino(_tiny_vit(), centering=False, **_DINO_SMALL)
    nocenter.training_step(torch.rand(4, 3, 16, 16))
    assert nocenter.name == "dino_nocenter" and float(nocenter.center.abs().sum()) == 0.0
    nosharpen = build_dino(_tiny_vit(), sharpening=False, **_DINO_SMALL)
    assert nosharpen.name == "dino_nosharpen" and nosharpen.teacher_temp == nosharpen.student_temp
    assert all(backbone_family(n) is Backbone.JE for n in ("dino", "dino_nocenter", "dino_nosharpen"))
    surfaces = nosharpen.embedding_surfaces(torch.rand(3, 3, 16, 16))
    assert surfaces["backbone"].shape == (3, 12) and surfaces["proj"].shape == (3, 8)


def test_layer_decay_counts_the_student_blocks_and_leaves_the_teacher_out() -> None:
    """The protections recipe on DINO: student blocks decay with depth, frozen teacher parameters are never grouped."""
    method = build_dino(_tiny_vit(), **_DINO_SMALL)
    groups = param_groups(method, lr=1.0, weight_decay=0.05, layer_decay=0.5, no_decay=True)
    grouped = {id(p) for g in groups for p in g["params"]}
    assert not grouped & {id(p) for p in method.encoder.parameters()}
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    assert lr_of[id(method.student.blocks[0].attn.qkv.weight)] == pytest.approx(0.25)  # depth 2: 0.5 ** 2
    assert lr_of[id(method.student_head.mlp[0].weight)] == pytest.approx(1.0)


def test_dino_drives_through_the_harness_and_its_teacher_is_read_from_the_checkpoint(tmp_path: Path) -> None:
    """backbone=dino drives the synthetic diet with the protections, checkpoints, and the reader reads the teacher."""
    dino = ["backbone=dino", "ssl.checkpoint=null", *[f"ssl.{k}={v}" for k, v in _DINO_SMALL.items()]]
    drive = [
        *_P11_DIET,
        *dino,
        "eval_every=4",
        "checkpoint_legs=final",
        "param_groups.layer_decay=0.75",
        "param_groups.no_decay=true",
    ]
    _load_script("run_deploy_harness").run_deploy(_compose("deploy", drive), tmp_path / "drive")
    final = tmp_path / "drive" / "checkpoints" / "s0" / "final.pt"
    state = torch.load(final)
    assert any(k.startswith("student.") for k in state) and "center" in state
    assert not torch.equal(state["encoder.norm.weight"], state["student.norm.weight"])
    _load_script("run_deploy_harness").run_deploy(
        _compose("deploy", [*drive, "ssl.centering=false", "max_steps=3"]), tmp_path / "nocenter"
    )
    index = json.loads((tmp_path / "nocenter" / "checkpoints" / "s0" / "index.json").read_text())
    assert index[-1]["batches_seen"] == 3  # Part E: the drive's first steps only
    encoders = f"+encoders=[{{name:start,kind:pretrained}},{{name:end,kind:checkpoint,path:'{final}'}}]"
    read = [*_P11_DIET, *dino, "probe_per_class=6", "threads=1", "~encoders", encoders]
    summary = _load_script("read_adaptation").run(_compose("read_adaptation", read), tmp_path / "read")
    assert summary["encoders"]["end"]["bacc"]["scene|linear"]["pooled"] >= 0.0


# ---- the analysis (scripts/analyze_adaptation.py mode=p125) ----

_GATE = ["scene", "presence_cell:traffic light"]


def _write_ladder(
    path: Path,
    correct_rate: dict[str, float],
    n: int = 400,
    n_patch_frames: int = 50,
    rungs: tuple[str, ...] = ("knn", "linear", "linear@cls"),
) -> None:
    """A fake reader output: correctness drawn at each encoder's rate, on the two gate tasks and the patch task."""
    path.mkdir(parents=True)
    rng = np.random.default_rng(len(str(path)))
    labels = {t: np.arange(n) % 2 for t in _GATE}
    counts = np.full((n_patch_frames, 2), 10.0, dtype=np.float32)
    np.savez_compressed(
        path / "pool.npz",
        query_index=np.arange(n),
        stratum=np.arange(n) % 3 // 2,
        regime=np.zeros(n, dtype=int),
        **{f"labels|{t}": labels[t] for t in _GATE},
        **{f"pos|{t}": np.arange(n) for t in _GATE},
        **{
            "labels|drivable": np.zeros(n, dtype=int),
            "pos|drivable": np.arange(n_patch_frames),
            "counts|drivable": counts,
        },
    )
    encoders: dict[str, Any] = {}
    for name, rate in correct_rate.items():
        arrays = {}
        for t in _GATE:
            for rung in rungs:
                arrays[f"{t}|{rung}"] = (rng.random((1, n)) < rate).astype(np.float32)
        for rung in ("knn", "linear"):
            arrays[f"drivable|{rung}"] = rng.binomial(10, rate, size=(1, n_patch_frames, 2)).astype(np.float32)
        np.savez_compressed(path / f"{name}.npz", **arrays)
        encoders[name] = {"bacc": {}}
    (path / "summary.json").write_text(json.dumps({"encoders": encoders, "stratum_names": {"0": "day", "1": "night"}}))


def _arm(gain: float, *, walk_rate: float | None = None, controls: dict[str, float] | None = None) -> dict[str, float]:
    rate = walk_rate if walk_rate is not None else 0.6 + gain
    names = {"floor": 0.6}
    for s in range(5):
        names |= {f"walk_s{s}_t00100": 0.6, f"walk_s{s}_t00500": rate, f"walk_s{s}_end": rate + 0.003 * s}
    names |= {f"iid_s{s}_end": rate for s in range(3)}
    for control, c_rate in (controls or {}).items():
        names |= {f"{control}_s{s}_end": c_rate for s in range(3)}
    return names


def _analyze(root: Path, extra: list[str] | None = None) -> dict[str, Any]:
    overrides = [
        "mode=p125",
        f"root={root}",
        "n_boot=400",
        "rungs=[knn,linear]",
        "p125.panel=[drivable]",
        "+p125.attribution={mae_je:{coco:3}}",
        "p125.n_attribution_iid=3",
        *(extra or []),
    ]
    report: dict[str, Any] = _load_script("analyze_adaptation").run(_compose("analyze_adaptation", overrides))
    return report


def test_p125_reads_the_gates_curve_breadth_and_attribution_and_breaks_a_tie_for_dino(tmp_path: Path) -> None:
    """Two passing DINO arms that read alike: DINO's own objective carries forward; the claim reads the breadth."""
    _write_ladder(tmp_path / "ladder" / "dino_dino", _arm(0.15))
    _write_ladder(tmp_path / "ladder" / "dino_simsiam", _arm(0.15))
    _write_ladder(tmp_path / "ladder" / "mae_je", _arm(0.15, controls={"coco": 0.6}))
    report = _analyze(tmp_path)
    dino = report["arms"]["dino_dino"]
    assert dino["gate"]["verdict"] == "PASS" and not dino["gate"]["tests"]["disagree"]
    assert dino["gate"]["cls_reverses"] == dict.fromkeys(_GATE, False)
    assert dino["curve"]["scene"]["time_to_payoff"] == "t00500"  # step 100 reads as the floor
    assert dino["breadth"]["gains"] == 1 and dino["breadth"]["most"]
    assert "seed_level" in dino["gate_contrasts"]["scene"]["linear"]
    assert set(dino["walk_vs_iid"]) == {f"{t}|{r}" for t in _GATE for r in ("knn", "linear")}
    assert report["arms"]["mae_je"]["attribution"]["claim"] == "STREAM-SPECIFIC"
    assert report["carry_forward"]["vehicle"] == "dino_dino"
    assert report["claim"] == {"level": "Strong", "backbone": "dino_dino"}


def test_p125_carries_simsiam_forward_only_when_it_is_clearly_ahead_and_falls_back_when_neither_passes(
    tmp_path: Path,
) -> None:
    """SimSiam wins only on a contrast excluding zero in its favour; with no pass the hybrid stays, the claim Weak."""
    _write_ladder(tmp_path / "ahead" / "ladder" / "dino_dino", _arm(0.08))
    _write_ladder(tmp_path / "ahead" / "ladder" / "dino_simsiam", _arm(0.3))
    assert _analyze(tmp_path / "ahead", ["p125.attribution={}"])["carry_forward"]["vehicle"] == "dino_simsiam"
    _write_ladder(tmp_path / "none" / "ladder" / "dino_dino", _arm(0.0))
    _write_ladder(tmp_path / "none" / "ladder" / "dino_simsiam", _arm(0.0))
    report = _analyze(tmp_path / "none", ["p125.attribution={}"])
    assert report["carry_forward"]["vehicle"] == "mae_je" and report["claim"]["level"] == "Weak"


def test_p125g_reads_the_unprotected_arm_its_recipe_contrast_and_the_every_rung_curve(tmp_path: Path) -> None:
    """Part G: the unprotected arm is read like any arm, the contrast pairs checkpoints, the re-read curve has ft."""
    _write_ladder(tmp_path / "ladder" / "mae_je", _arm(0.15))
    _write_ladder(tmp_path / "ladder" / "mae_je_np", _arm(0.0, controls={"coco": 0.6}))
    _write_ladder(tmp_path / "ladder_ft" / "mae_je", _arm(0.15), rungs=("knn", "linear", "ft10"))
    overrides = [
        "mode=p125g",
        f"root={tmp_path}",
        "n_boot=400",
        "rungs=[knn,linear]",
        "p125.panel=[drivable]",
        "p125.n_attribution_iid=3",
        "p125g.arms=[mae_je_np]",
        "+p125g.attribution={mae_je_np:{coco:3}}",
        "+p125g.pairs={mae_je_np:mae_je}",
        "p125g.ft_arms=[mae_je]",
        "report=partG.json",
    ]
    report: dict[str, Any] = _load_script("analyze_adaptation").run(_compose("analyze_adaptation", overrides))
    assert (tmp_path / "partG.json").is_file()
    unprotected = report["arms"]["mae_je_np"]
    assert {"gate", "breadth", "curve", "attribution"} <= set(unprotected)
    recipe = report["recipe"]["mae_je_np"]
    assert recipe["protected"] == "mae_je" and list(recipe["points"]) == ["t00100", "t00500", "end"]
    assert recipe["points"]["end"]["scene"]["linear"]["lo"] > 0 and recipe["tests"]["bootstrap"]["scene"]
    early = recipe["points"]["t00100"]["scene"]["linear"]  # both arms read as the floor there
    assert early["lo"] < 0 < early["hi"]
    assert "linear@cls" in recipe["points"]["end"]["scene"] and "ft10" not in recipe["points"]["end"]["scene"]
    curve = report["curve"]["mae_je"]["scene"]["points"]
    assert curve["end"]["ft10"]["lo"] > 0 and curve["t00100"]["ft10"]["lo"] < 0 and "ft100" not in curve["end"]


def _released(tmp_path: Path, encoder: TinyViTEncoder) -> dict[str, Any]:
    """The released layout: a plain unit-norm ``last_layer.weight``, no ``args``, no centre."""
    _full_checkpoint(tmp_path / "full.pth", encoder)
    ckpt = torch.load(tmp_path / "full.pth", weights_only=False)
    for part, prefix in (("student", "module.head."), ("teacher", "head.")):
        state = ckpt[part]
        v = state.pop(f"{prefix}last_layer.weight_v")
        state.pop(f"{prefix}last_layer.weight_g")
        state[f"{prefix}last_layer.weight"] = v / v.norm(dim=1, keepdim=True)
    del ckpt["args"], ckpt["dino_loss"]
    torch.save(ckpt, tmp_path / "released.pth")
    return dict(ckpt)


def test_dino_loads_the_released_format_without_args_or_centre(tmp_path: Path) -> None:
    """The released checkpoint loads: its plain last layer becomes the weight-normed one; the centre waits at zero."""
    encoder = _tiny_vit()
    ckpt = _released(tmp_path, encoder)
    method = build_dino(encoder, checkpoint=str(tmp_path / "released.pth"), **_DINO_SMALL)
    assert float(method.center.abs().sum()) == 0.0
    weight = method.teacher_head.last_layer
    effective = weight.weight_g * weight.weight_v / weight.weight_v.norm(dim=1, keepdim=True)
    assert torch.allclose(effective, ckpt["teacher"]["head.last_layer.weight"])


def test_dino_estimates_a_missing_centre_before_step_0_without_moving_the_teacher(tmp_path: Path) -> None:
    """prime: the centre becomes the teacher's mean logits on the global crops; nothing read moves; once only."""
    encoder = _tiny_vit()
    _released(tmp_path, encoder)
    method = build_dino(encoder, checkpoint=str(tmp_path / "released.pth"), **_DINO_SMALL)
    before = {k: v.clone() for k, v in method.state_dict().items() if k != "center"}
    batches = [torch.rand(4, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    torch.manual_seed(1)
    with torch.no_grad():
        logits = [
            method.teacher_head(method._cls(method.encoder, torch.cat([t(b) for t in method.crops.globals])))
            for b in batches
        ]
    torch.manual_seed(1)
    method.prime(batches)
    assert torch.allclose(method.center, torch.cat(logits).mean(dim=0, keepdim=True), atol=1e-6)
    assert all(torch.equal(v, method.state_dict()[k]) for k, v in before.items())
    estimated = method.center.clone()
    method.prime([torch.rand(4, 3, 16, 16)])  # known now: a second call leaves it
    assert torch.equal(method.center, estimated)
    resumed = build_dino(encoder, checkpoint=str(tmp_path / "full.pth"), **_DINO_SMALL)
    resumed.prime(batches)  # the checkpoint's own centre is kept
    assert torch.equal(resumed.center, torch.full((1, 32), 0.25))
    off = build_dino(encoder, checkpoint=str(tmp_path / "released.pth"), centering=False, **_DINO_SMALL)
    off.prime(batches)  # no centering, nothing to estimate
    assert float(off.center.abs().sum()) == 0.0


def test_centring_check_reads_a_flat_target_on_alike_frames_and_a_sharp_one_on_varied_frames() -> None:
    """dino_centring_check: alike frames leave a near-uniform centred target; varied ones keep it sharp."""
    check = _load_script("dino_centring_check")
    gen = torch.Generator().manual_seed(0)
    k, n = 64, 50
    base = torch.randn(1, k, generator=gen) * 5
    alike = base + 0.01 * torch.randn(n, k, generator=gen)  # every frame's logits close to the set's mean
    varied = 5 * torch.randn(n, k, generator=gen)
    cls_alike, cls_varied = torch.ones(n, 8) + 0.01 * torch.randn(n, 8, generator=gen), torch.randn(n, 8, generator=gen)
    flat = check.set_stats(alike, cls_alike, alike.mean(dim=0, keepdim=True), temp=0.07)
    sharp = check.set_stats(varied, cls_varied, alike.mean(dim=0, keepdim=True), temp=0.07)
    assert flat["uniform_entropy"] == pytest.approx(np.log(k))
    assert flat["entropy_own_center"] > 0.9 * np.log(k) > sharp["entropy_own_center"]
    assert flat["entropy_no_center"] < 0.5  # uncentred, the shared peak dominates every frame
    assert flat["top_prototypes_used_no_center"] == 1 < sharp["top_prototypes_used"]
    assert flat["cls_mean_pairwise_cosine"] > 0.99 > 0.2 > abs(sharp["cls_mean_pairwise_cosine"])
    z = torch.randn(5, 3, generator=gen)
    pairs = [float(F.cosine_similarity(z[i], z[j], dim=0)) for i in range(5) for j in range(5) if i != j]
    assert check.mean_pairwise_cosine(z) == pytest.approx(np.mean(pairs))


# ---- Part F (scripts/analyze_adaptation.py mode=p125f) ----

_PF_TASK, _PF_RARE = "presence_cell:bus", "size_cell:bus:0:0.01"


def _pf_correct(labels: np.ndarray, masks: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Deterministic correctness: within each mask, the first ``rate`` share of each class's frames are right."""
    out = np.zeros(labels.size, dtype=np.float32)
    for mask, rate in masks:
        for cls in (0, 1):
            members = np.flatnonzero(mask & (labels == cls))
            out[members[: int(round(rate * members.size))]] = 1.0
    return out


def _write_partf_ladder(path: Path) -> None:
    """600 frames: daytime clear / daytime rainy / night clear, 200 each; the stream gains, tracks and forgets."""
    path.mkdir(parents=True)
    n = 600
    labels = np.tile([0, 1], n // 2)
    stratum = np.repeat([0, 0, 2], 200)
    regime = np.repeat([0, 3, 12], 200)
    clear, rainy, night = (regime == 0), (regime == 3), (regime == 12)
    rare = np.full(n, -1)
    rare[:20] = labels[:20]  # 10 positives: below the read rule
    np.savez(
        path / "pool.npz",
        query_index=np.arange(n),
        stratum=stratum,
        regime=regime,
        **{f"labels|{_PF_TASK}": labels, f"pos|{_PF_TASK}": np.arange(n)},
        **{f"labels|{_PF_RARE}": rare[:20], f"pos|{_PF_RARE}": np.arange(20)},
    )
    summary = {
        "stratum_names": {"0": "daytime", "1": "dawn/dusk", "2": "night"},
        "regime_names": {"0": "daytime·clear", "3": "daytime·rainy", "12": "night·clear"},
        "encoders": {},
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rates = {  # (daytime clear, daytime rainy, night clear) per checkpoint
        "floor": (0.6, 0.6, 0.6),
        "t00423": (0.6, 0.6, 0.6),
        "t00846": (0.7, 0.9, 0.6),  # the day gains, the rainy slice most (its era sits in between)
        "end": (0.6, 0.6, 0.9),  # night gains once the stream is there; the day gives its gain back
        "iid": (0.65, 0.65, 0.65),
    }

    def save(name: str, key: str) -> None:
        c, r, nt = rates[key]
        row = _pf_correct(labels, [(clear, c), (rainy, r), (night, nt)])[None]
        rare_row = row[:, :20]
        np.savez(
            path / f"{name}.npz",
            **{f"{_PF_TASK}|{rung}": row for rung in ("knn", "linear")},
            **{f"{_PF_RARE}|{rung}": rare_row for rung in ("knn", "linear")},
        )

    save("floor", "floor")
    for s in range(3):
        for sfx in ("t00423", "t00846", "end"):
            save(f"walk_s{s}_{sfx}", sfx)
        save(f"iid_s{s}_end", "iid")


def test_p125f_reads_tracking_forgetting_and_weather_windows_and_skips_rare_tasks(tmp_path: Path) -> None:
    """The windows recover the planted within-slice changes; a task under the positive floor is not read."""
    _write_partf_ladder(tmp_path / "ladder" / "mae_je")
    train = [  # two highway frames with a bus (one small, one large), two without; a tunnel frame is out of scope
        {
            "attributes": {"scene": "highway", "timeofday": "daytime", "weather": "clear"},
            "labels": [{"category": "bus", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 72}}],
        },  # 0.005 of the frame
        {
            "attributes": {"scene": "highway", "timeofday": "daytime", "weather": "clear"},
            "labels": [{"category": "bus", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}}],
        },
        {"attributes": {"scene": "highway", "timeofday": "night", "weather": "clear"}, "labels": []},
        {"attributes": {"scene": "highway", "timeofday": "night", "weather": "clear"}},
        {
            "attributes": {"scene": "tunnel", "timeofday": "night", "weather": "clear"},
            "labels": [{"category": "bus", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}}],
        },
    ]
    (tmp_path / "train.json").write_text(json.dumps(train), encoding="utf-8")
    overrides = [
        "mode=p125f",
        f"root={tmp_path}",
        "report=partF_ladder.json",
        "n_boot=300",
        "n_seeds=3",
        "n_iid_seeds=3",
        "p125f.arms=[mae_je]",
        'p125f.slices=["tod:daytime","tod:night","regime:daytime·clear","regime:daytime·rainy"]',
        f"p125f.train_labels={tmp_path / 'train.json'}",
        "p125f.scenes=[highway]",
    ]
    _load_script("analyze_adaptation").run(_compose("analyze_adaptation", overrides))
    report = json.loads((tmp_path / "partF_ladder.json").read_text(encoding="utf-8"))
    assert report["read"] == [_PF_TASK] and report["skipped"] == {_PF_RARE: 10}
    arm = report["arms"]["mae_je"][_PF_TASK]["linear"]
    w = arm["windows"]
    assert w["tracking"]["delta"] == pytest.approx(0.3) and w["tracking"]["lo"] > 0
    assert w["forgetting"]["delta"] == pytest.approx(-0.2) and w["forgetting"]["hi"] < 0
    assert w["pooled_late"]["delta"] == pytest.approx((0.3 - 0.1 - 0.3) / 3, abs=1e-6)
    assert w["weather_rainy"]["delta"] == pytest.approx(0.2) and w["weather_rainy"]["terms"][
        "regime:daytime·rainy"
    ] == pytest.approx(0.3)
    assert w["weather_snowy"] is None and w["night_rainy"] is None  # slices absent from the pool
    assert arm["curve"]["tod:night"]["end"]["delta"] == pytest.approx(0.3)
    assert arm["iid"]["tod:night"]["walk_vs_iid"]["delta"] == pytest.approx(0.25)
    assert report["train_frequency"][_PF_TASK] == pytest.approx(0.5) and report["train_frequency"]["n_frames"] == 4.0
    assert (tmp_path / "partF_ladder.json").is_file() and not (tmp_path / "adaptation_report.json").exists()
