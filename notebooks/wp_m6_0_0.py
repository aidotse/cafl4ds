"""Inspect WP-M6.0.0. Run: marimo edit notebooks/wp_m6_0_0.py.

Optional environment overrides: WP_M6_ARTIFACT_DIR and BDD_ROOT.
"""

# mypy: ignore-errors
# marimo generates the reactive cell signatures and their module-valued arguments.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import pickle
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return Image, Path, json, mo, np, os, pickle, plt


@app.cell
def _(Path, json, np, os, pickle):
    artifact_dir = Path(
        os.environ.get("WP_M6_ARTIFACT_DIR", str(Path(__file__).resolve().parents[1] / "outputs/wp_m6_0_0"))
    )
    summary = json.loads((artifact_dir / "summary.json").read_text())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    exemplars = json.loads((artifact_dir / "exemplars.json").read_text())
    bdd_root = Path(os.environ.get("BDD_ROOT", manifest["bdd_root"]))
    encodings = np.load(artifact_dir / "embeddings_64.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(artifact_dir / "labels.npy", allow_pickle=False)
    probabilities = np.load(artifact_dir / "probabilities.npy", allow_pickle=False)
    with (artifact_dir / "pca_gmm.pkl").open("rb") as _stream:
        pipeline = pickle.load(_stream)  # noqa: S301 - locally generated experiment artifact
    return (
        artifact_dir,
        bdd_root,
        encodings,
        exemplars,
        labels,
        manifest,
        pipeline,
        probabilities,
        summary,
    )


@app.cell
def _(mo, summary):
    mo.md(f"""
    # WP-M6.0.0 — offline image clusters

    **{summary["n_images"]:,} images**, including train, validation and test, embedded by frozen
    ImageNet ViT-B/16 using its standard resize/center crop. PCA reduces the 768-D CLS features
    to **{summary["pca_dim"]} dimensions**, retaining **{summary["pca_explained_variance"]:.1%}**
    of variance. A four-component, full-covariance GMM is fitted to all images.

    This is the **all-data look-ahead baseline**. No evaluation split was reserved for fitting.
    Mean normalized posterior entropy: **{summary["mean_normalized_entropy"]:.3f}**
    (0 = hard assignment, 1 = uniform). **{summary["fraction_max_posterior_above_099"]:.1%}**
    of images have a maximum posterior above 0.99.
    """)
    return


@app.cell
def _(mo, summary):
    mo.ui.table(
        [
            {
                "Cluster": _cluster,
                "Most likely members": summary["hard_cluster_counts"][_cluster],
                "Mean probability": round(summary["mean_posterior"][_cluster], 4),
            }
            for _cluster in range(4)
        ],
        selection=None,
    )
    return


@app.cell
def _(artifact_dir, encodings, labels, mo, np, pipeline, plt, summary):
    _fig, _ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    _colors = plt.get_cmap("tab10").colors
    # Shuffle only draw order so one component does not systematically cover another.
    _order = np.random.default_rng(0).permutation(len(labels))
    _ax.scatter(
        encodings[_order, 0],
        encodings[_order, 1],
        c=np.asarray(_colors)[labels[_order]],
        s=2,
        alpha=0.22,
        linewidths=0,
        rasterized=True,
    )
    for _cluster, _mean in enumerate(pipeline.named_steps["gmm"].means_):
        _ax.scatter(*_mean[:2], s=260, marker="*", c=[_colors[_cluster]], edgecolor="black", linewidth=1.3, zorder=5)
        _ax.annotate(str(_cluster), _mean[:2], xytext=(9, 9), textcoords="offset points", weight="bold")
        _ax.scatter([], [], color=_colors[_cluster], label=f"Cluster {_cluster}")
    _ratios = pipeline.named_steps["pca"].explained_variance_ratio_
    _ax.set(
        xlabel=f"PC1 ({_ratios[0]:.1%} variance)",
        ylabel=f"PC2 ({_ratios[1]:.1%} variance)",
        title=f"All {summary['n_images']:,} image embeddings · color = most likely GMM component · stars = means",
    )
    _ax.legend(markerscale=1.5)
    _fig.savefig(artifact_dir / "clusters_pca2.png", dpi=160)
    mo.vstack(
        [
            mo.md(
                f"The first two PCs retain **{summary['pca_2d_explained_variance']:.1%}** of variance. "
                "Stars project the fitted 64-D GMM means onto these same axes. "
                "Overlap here does not imply overlap in all 64 dimensions."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    exemplar_mode = mo.ui.dropdown(
        options={"Closest to each mean (Mahalanobis distance)": "central", "Random assigned members": "random"},
        value="Closest to each mean (Mahalanobis distance)",
        label="Examples",
    )
    exemplar_mode
    return (exemplar_mode,)


@app.cell
def _(
    Image,
    Path,
    artifact_dir,
    bdd_root,
    exemplar_mode,
    exemplars,
    manifest,
    mo,
    plt,
    probabilities,
):
    _fig, _axes = plt.subplots(4, 6, figsize=(18, 8.8), constrained_layout=True)
    for _entry, _row in zip(exemplars, _axes, strict=True):
        _cluster = _entry["cluster"]
        _indices = _entry[exemplar_mode.value]
        for _column, _ax in enumerate(_row):
            _ax.axis("off")
            if _column < len(_indices):
                _index = _indices[_column]
                _record = manifest["images"][_index]
                with Image.open(bdd_root / _record["path"]) as _image:
                    _thumbnail = _image.convert("RGB")
                    _thumbnail.thumbnail((384, 216))
                    _ax.imshow(_thumbnail)
                _ax.set_title(
                    f"Cluster {_cluster} · p={probabilities[_index, _cluster]:.3f}\n{Path(_record['path']).name}",
                    fontsize=8,
                )
    _fig.savefig(artifact_dir / f"exemplars_{exemplar_mode.value}.jpg", dpi=130)
    mo.vstack(
        [
            mo.md(
                "### Exemplars of each cluster\n"
                "Central examples are assigned members closest to their component mean in the full 64-D "
                "Mahalanobis metric. Switch to random members to inspect diversity. "
                "Original frames are shown; the ViT embedding used the standard center crop."
            ),
            _fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
