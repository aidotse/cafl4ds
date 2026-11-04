# WP-M6.0.0 — offline BDD100K clustering

This experiment builds the fixed image-level cluster probabilities for the WP-M6.0 offline routing baseline. It embeds
and fits to **all 100,000 available still images**: 70,000 train, 10,000 validation and 20,000 test. Images are
discovered directly from the three image directories; missing or undefined driving attributes never exclude an image.
Both PCA and the GMM see all splits and the full future stream. This intentional look-ahead makes it an all-data
reference for subsequent online clustering experiments. It is not a held-out clustering evaluation.

The pipeline is independent of the MoE. It does not change training, experts or routing.

## Representation and fit

- Encoder: frozen, evaluation-mode `torchvision.models.vit_b_16`, with explicitly selected
    `ViT_B_16_Weights.IMAGENET1K_V1`. The classification head is replaced by identity, giving the final 768-D CLS token.
- Inputs: original RGB images processed by `weights.transforms()`: resize shorter edge to 256 with bilinear
    interpolation, center crop to 224, rescale to `[0, 1]`, then ImageNet normalization. No random augmentation.
- Cached embeddings: float32, without additional feature scaling or L2 normalization. Extraction uses inference mode.
- PCA: 64 dimensions, centered, without whitening; randomized SVD with seed 0.
- GMM: four components, full covariance, `reg_covar=1e-6`, five k-means initializations, `max_iter=300`, `tol=1e-3`,
    seed 0. The best initialization is retained by scikit-learn. Nonconvergence fails the run.
- Inference: the saved sklearn pipeline accepts raw 768-D ViT embeddings and applies the fitted PCA before
    `predict_proba`. Its four output columns correspond to component/expert IDs 0–3 and already sum to one.

The encoder and preprocessing are separate from the sklearn pipeline: new images must first pass through the same frozen
ViT and standard transform. The exact weights URL, transform, software versions and row-aligned image manifest are saved
alongside the fitted model.

## Reproduce on this Mac

```bash
uv sync --extra cpu --group analysis

TORCH_HOME="$PWD/outputs/wp_m6_0_0/torch-cache" \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
caffeinate -i .venv/bin/python -u -m scripts.build_offline_clusters \
  --bdd-root /Users/williammin/cafl4ds_datasets/BDD \
  --output outputs/wp_m6_0_0 --device mps --batch-size 32 --workers 4
```

The initial short benchmark on this Apple M1 Pro (16 GB) measured approximately 80 images/s on Metal versus 32 images/s
on CPU, excluding decoding and preprocessing. The full run uses Metal with float32 inference. `--device auto` selects
Metal when available and otherwise CPU; `--device cpu` forces CPU. The Mac's PyTorch wheel supports Metal even though
the repository's dependency selector is named `cpu`.

Re-running the same command resumes from the last flushed embedding batch. The cache checks image paths, file
sizes/mtimes, weights, transform and library versions; changed inputs require a new output directory. Image decoding
errors fail extraction rather than silently skipping records. `--fit-only` reuses the completed embedding cache and
refits PCA/GMM without encoding images again.

## Inspect the result

```bash
.venv/bin/marimo edit notebooks/wp_m6_0_0.py
```

The short notebook shows component counts and mean probabilities, posterior softness, a 2-D PCA scatter of every image
colored by its most likely component, projected GMM means, and six exemplars per component. The scatter uses the first
two axes of the same fitted PCA as the 64-D GMM; the mean markers use the corresponding first two mean coordinates. The
GMM is never refitted in 2-D. The figure reports how much variance the two displayed axes retain.

Central exemplars minimize within-component Mahalanobis distance in the full 64-D space, restricted to images whose most
likely component is that cluster. A dropdown switches to seeded random assigned members to inspect diversity. Exemplars
display original frames; the ViT saw the standard center crop. No driving labels enter the fit.

Optional `WP_M6_ARTIFACT_DIR` and `BDD_ROOT` environment variables relocate the artifacts or dataset. For an executed
static copy:

```bash
MPLCONFIGDIR="$PWD/outputs/wp_m6_0_0/matplotlib" \
.venv/bin/marimo export html notebooks/wp_m6_0_0.py \
  -o outputs/wp_m6_0_0/analysis.html
```

The interactive notebook can switch exemplar selection; a static HTML export shows its default central exemplars.

## Artifacts

All large artifacts live in the ignored `outputs/wp_m6_0_0/` directory:

| File | Contents |
| -- | -- |
| `manifest.json` | Ordered relative image paths, split, file identity, encoder/transform provenance and fingerprint |
| `embedding_progress.json` | Committed embedding rows, extraction device, batch size, workers and session timing |
| `embeddings.npy` | Row-aligned original ViT embeddings, shape `[100000, 768]` |
| `embeddings_64.npy` | The fitted PCA transform of every embedding, shape `[100000, 64]` |
| `pca_gmm.pkl` | Fitted sklearn `Pipeline(pca, gmm)` for future `predict_proba` calls |
| `probabilities.npy` | Row-aligned float32 component probabilities, shape `[100000, 4]` |
| `labels.npy` | Most likely component ID per image |
| `exemplars.json` | Central and seeded random exemplar row indices for each component |
| `summary.json` | Counts, fit settings, explained variance, entropy, likelihood and convergence |
| `clusters_pca2.png` | Notebook's all-image scatter and projected means |
| `exemplars_central.jpg` | Notebook's central exemplar grid |
| `exemplars_random.jpg` | Written when the notebook displays random members |
| `analysis.html` | Executed static notebook export |

The saved model uses sklearn's pickle serialization and should be loaded in the recorded sklearn environment. The
artifact files and the image manifest use identical row order; component IDs are local to this fitted model.

## Result — 2026-11-03

All 100,000 images were embedded successfully on Metal in 1,286 seconds (21.4 minutes), averaging 77.8 images/s
including decoding and preprocessing. The 64-D PCA retains **69.18%** of embedding variance; the plotted first two axes
retain **18.67%**. PCA and all five GMM initializations completed in 5.3 seconds. Every initialization converged; the
selected fit converged in 14 EM iterations.

| Component | Most likely members | Share | Pattern suggested by the central exemplars |
| -- | -- | -- | -- |
| 0 | 23,017 | 23.0% | Nighttime, relatively open roads |
| 1 | 17,363 | 17.4% | Nighttime urban streets and intersections |
| 2 | 23,325 | 23.3% | Daytime urban streets and intersections |
| 3 | 36,295 | 36.3% | Daytime, relatively open roads |

These are visual interpretations of six central examples per component, rather than labels assigned to all members. The
notebook's random-member view helps inspect variation beyond these central examples. The PCA plot also suggests a strong
day/night separation along its first axis.

The probabilities are sharp: **87.535%** of images have maximum posterior above 0.99; median maximum posterior is 1.0 at
the saved precision. Mean entropy is **0.04479 nats**, or **0.03231** of the uniform four-component entropy. Thus this
baseline supplies largely hard routing despite retaining the complete four-component probability vectors.

Validation confirmed all image paths are unique and all 100,000 embedding/probability rows are present and finite.
Reloaded pipeline inference exactly reproduced saved probabilities on 519 rows, including split boundaries. Independent
CPU extraction of eight original JPEGs across batch/split boundaries matched the cached Metal embeddings with maximum
absolute difference 2.10e-5. The maximum float32 posterior row-sum error is 7.64e-6. All component covariance matrices
are positive definite, and every central exemplar belongs to its displayed component. Details and the saved model hash
are in `verification.json`.

Three focused tests, strict mypy, marimo validation/execution/HTML export, and the repository's pre-commit hooks passed.
The repository-wide dependency audit still reports five existing undeclared SciPy imports in earlier routing/analysis
modules; the new pipeline and notebook introduce no dependency-audit findings.
