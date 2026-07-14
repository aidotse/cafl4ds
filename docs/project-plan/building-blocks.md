‹ [Project Plan index](index.md)

# Building blocks

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution (see
[index](index.md#novelty-at-a-glance)).*

**SSL backbone `[STD]`:** default **MAE** for the pipeline (stable; reconstruction error is a free per-frame
informativeness signal). **MAE is collapse-resistant** (a constant code can't reconstruct diverse pixels), so the
*collapse* demonstration uses a **joint-embedding method** (SimSiam/BYOL/DINO/SimCLR) — MAE's own degradation mode is
forgetting/overspecialization. We start with **SimSiam** as the minimal joint-embedding variant (predictor +
stop-gradient, no EMA target, no negatives — fewest moving parts, and its stop-gradient is *precisely* the
collapse-avoidance mechanism under study), expanding to BYOL then SimCLR once the dynamics are established. Adapt an
ImageNet-pretrained backbone (data too small for from-scratch); ViTDet-compatible if detection is a target. *Methods
used as-is: MAE (He 2022), SimSiam/BYOL/DINO/SimCLR (Chen & He 2021 / Grill 2020 / Caron 2021 / Chen 2020).*

**Filter families (the knobs):**

- **(F-a) Latent-feature tracking — novelty criterion `[EXT]`.** Encode → PCA to 8–32D metric space → score =
    kNN/Mahalanobis distance to a same-domain reference set (frozen reference encoder; not UMAP). *Extends:*
    embedding-space OOD scoring — kNN-OOD (Sun 2022), Mahalanobis (Lee 2018), Deep SVDD (Ruff 2018) — built to *flag*
    OOD at test time; we repurpose them as a *streaming, label-free, single-pass training-selection* signal.
- **(F-b) Streaming core-set — coverage criterion `[EXT]`.** Maintain a buffer that geometrically covers the seen
    embedding distribution. *Extends:* core-set / k-center (Sener & Savarese 2018), herding (Welling 2009 / iCaRL
    Rebuffi 2017), prototypicality (Sorscher 2022), GSS (Aljundi 2019) — offline or supervised-CL coverage methods; we
    run them as streaming, label-free buffer maintenance over the SSL embedding.
- **(F-c) Health-steerable filter `[NEW]` ([N-G](novelty.md)).** Explicit mix `α·coverage + (1−α)·novelty` + acceptance
    rate, both set **live by the monitor**. The actuator for the closed loop. *Contrasts with:* D2 Pruning (2024) / CCS
    (2023), which have the coverage/difficulty mix but fix it *offline*; here it is a controlled variable. A frozen `α`
    reduces F-c to a hybrid `[EXT]` baseline (= [B-open](experiments.md#baselines)).
- **Temporal-robustness layer `[EXT]`.** Denoise the per-frame signal, separating sustained shifts (true novelty) from
    impulsive spikes (sensor noise). *Extends:* view-disagreement OOD (RSS-MGM) and forgetting/difficulty (FALSE) from
    the label-noise line, plus drift/change-point detection (ADWIN, Page-Hinkley); we transpose to label-free streaming
    SSL and add a sustained-vs-impulsive persistence test.

**Datasets (what we run on, and why):**

- **STL-10 — Phase 0 + Phase 1 pilot.** SSL-purpose-built (100k-image unlabeled split, 96px), small enough for CPU smoke
    tests and fast iteration, clean 10-class labels for linear-probe / kNN eval. Correlation is *synthetic*
    (class-blocked ordering). Role: validate the loop + instruments + positive control cheaply — the harness, not the
    headline.
- **TinyImageNet / ImageNet-100 — optional.** A larger non-driving SSL scale check, if STL-10 proves too small to stress
    the backbone.
- **BDD100K — Phase 1 make-or-break.** Real driving video → *genuine* temporal correlation, the property that makes
    selection meaningful (a make-or-break on shuffled toy data won't convince reviewers). Built-in weather / scene /
    time-of-day attribute labels → ready-made label-efficiency probes and natural era boundaries. GPS/IMU → the optional
    telemetry fused-input ablation. Standard, respected benchmark.
- **ZOD (Zenseact Open Dataset) — Phase 4 confirmation.** In-house partner data; confirms the findings transfer to the
    consortium's own automotive distribution and the use case the project ultimately serves. Multimodal (camera/LiDAR +
    vehicle/GNSS signals).
- **nuScenes — optional alternative.** Use if the telemetry-fusion ablation needs tighter time-synchronized CAN-bus
    signals than BDD provides.

*Ordering rationale:* cheap harness (STL-10) → real-correlation make-or-break (BDD100K) → in-house confirmation (ZOD) —
matching the "instrument & demonstrate first" sequence. Dataset choice is not tagged `[STD]`/`[EXT]`/`[NEW]`: a dataset
isn't a contribution, so the tags (which mark method novelty) don't apply.
