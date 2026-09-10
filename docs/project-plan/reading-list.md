‹ [Project Plan index](index.md)

# Reading list

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution. The `N-x` claims are stated
in full in [novelty.md](novelty.md).*

## Position against (work we must distinguish ourselves from)

*Grouped by the part of our approach each bears on.*

- **The coupling as a live loop (N-A):** DiSF (2025) — diversified *file* selection to combat dimensional collapse in
    **LLM pre-training data**, *offline, submodular, no model in the loop* · SOFed/FedCoCo (Shi 2022) — streaming-SSL
    selection, *single criterion, no loop* (also the closest prior for N-D) · the **streaming-SSL degradation line**
    (below) is the closest prior to the *premise* itself.
- **Budget flip under co-adaptation (N-B):** CCS (Zheng 2023) · Sorscher (2022) · D2 Pruning (Maharana 2024) — *offline,
    fixed model*.
- **Selection-loop (in)stability (N-C):** one-sided-feedback (2020) · recsys/bandit sampling-bias loops · LLM
    self-consuming / model collapse (Shumailov 2023) — *other domains*.
- **Selection × aggregation skew in FL (N-D):** FedU (Zhuang 2021) · Orchestra (Lubana 2022) — *federated SSL without
    active selection bias* · heterogeneous-FL dimensional collapse (2022) — *non-IID → representational collapse, but
    via aggregation, not selection*.
- **Health-monitor → control loop (N-E):** RankMe (Garrido 2023) · LiDAR (Thilak 2024) · dimensional collapse (Jing
    2022\) · ADWIN (Bifet 2007) — *health metrics / drift as offline diagnostics, not live control*.
- **Federation-level health analytics (N-F):** Krum (Blanchard 2017) · trimmed-mean/median (Yin 2018) — *bad-client
    detection by update geometry, not representation health*.
- **Health-steerable filter (N-G):** D2 Pruning (2024) · CCS (2023) — coverage/difficulty mix *fixed offline*, never
    steered live.
- **Collapse prevention (context for N-A/N-E):** VICReg (Bardes 2022) · IConE (2026) · AdaDim (2025) · CMP (2025) — *via
    loss/architecture, not selection, not a loop*.
- **Streaming-SSL degradation premise (Phase 1 / motivates B5, bears on N-A):** Purushwalkam *Challenges of Continuous
    SSL* (2022) — temporal correlation + distribution shift are the two streaming-SSL failure drivers, mitigated by
    **MinRed** (minimum-redundancy replay) · Memory Storyboard (Yang & Ren, CoLLAs 2025) — temporal-segmentation replay
    · Learning from One Continuous Video Stream (CVPR 2024) · Orthogonal Gradients (2025) — optimizer-level fix. *All
    de-correlate the stream via **replay / temporal segmentation / optimizer**; none steer **selection by representation
    health**, live (our N-A / N-G).*
- **Frozen-beats-learned existential null (Phase 1 leg a, motivates B5 / P1.2):** RanDumb (Prabhu et al., NeurIPS 2024)
    — random Fourier features + decorrelation **outperform** online continually-learned representations (single-pass,
    exemplar-free) — but *discriminative* CL, **open whether it holds for SSL**; counterpoint: Memory Storyboard (2025)
    \+ Learning from One Continuous Video Stream (2024) show streaming-SSL adaptation *can* beat frozen. *The leg is
    genuinely contested.*
- **Reading the instruments — degradation vs. benign reorganization (Phase 1 / P1.1):** CKA reliability — Davari et al.
    (ICLR 2023) — CKA is sensitive to outliers / function-preserving shifts and biased in high-dim/low-sample, so use
    the **debiased** estimator + a complementary metric (Procrustes/CCA) · tunnel effect — Masarczyk et al. (NeurIPS
    2023\) — a numerical-rank drop in deep layers is **benign** neural-collapse-style compression for the trained task
    (hurts only OOD), and is **amplified by toy / low-resolution / few-class data** (⇒ STL-10: only 10 classes, small
    scale) · representational-drift formalism — benign drift is **non-mean-reverting change without performance loss**,
    forgetting is change *with* info loss (distinguish via mean-reversion + probe-recovery) · label-free metric
    reliability — RankMe is **necessary-but-not-sufficient**, and a large-scale comparison (Arputharaj et al., **TMLR
    2026** — 260 vision models, six datasets) finds **intrinsic dimension** the most reliable single label-free
    downstream predictor, though its reliability too stays **moderated by architecture class and training objective**;
    the concrete estimator is **IdEst** (Mordacq et al., **ICML 2026** — MST-based, correlates strongly with the linear
    probe) — so read a **suite**, not one needle.
- **MAE / streaming-SSL forgetting (closest prior to the Phase-1 forgetting demo, bears on N-A):** Davari (2022,
    *Probing Representation Forgetting*) · Representational Continuity (Madaan 2021) — *SSL forgets less than
    supervised, but offline & task-boundaried* · Beyond Cosine Decay (CoLLAs 2025) — *continual MAE **pretraining**:
    Experience Replay + infinite-LR schedule drives forgetting ≈ 0 even at low buffer rates, and the LR **schedule**
    itself is a forgetting confound* · CoSMAE (2025) · MedCoSS · MAE-continual-FL (2023) — *continual MAE via
    replay/distillation/mixup under task or coarse-domain shift*. **What they leave open:** all sample randomly or by
    replay under task/coarse-domain boundaries; none couple a **live informativeness filter** to the backbone (our N-A)
    on a **single-pass correlated** stream, nor read health **label-free in real time**. Their standing result — MAE
    resists, replay trivially protects — reframes our forgetting demo: the novel lever is **selection**, not
    replay-vs-none. *Caveat carried:* for MAE, label-free probing can **understate** forgetting (fine-tuning reflects it
    better).
- **Noise-robust selection (temporal-robustness layer, `[EXT]`):** Co-teaching · DivideMix · FALSE · RSS-MGM —
    *supervised label noise; we transpose to label-free sensor corruption*.
- **Generative replay (replay codebook, `[EXT]`):** CAN (2025) · diffusion-as-replay — *federated generative replay; we
    differ by inversion + embedding-space + temporal scheduling*.
- **Automotive FL-SSL (the use case):** federated SSL for AV depth (2023).

## Toolbox we build on (work we use, does not threaten novelty of our work)

- **SSL backbones `[STD]`:** MAE (He 2022) · DINO (Caron 2021) · SwAV (Caron 2020) · SimSiam (Chen & He 2021) ·
    BYOL/SimCLR/MoCo (2020) · CaSSLe (Fini 2022).
- **Selection `[EXT]`:** Core-Set (Sener & Savarese 2018) · Herding/iCaRL (Welling 2009 / Rebuffi 2017) · SemDeDup
    (Abbas 2023) · GSS/MIR (Aljundi 2019) · reservoir (Vitter 1985).
- **OOD scoring (F-a) `[EXT]`:** kNN-OOD (Sun 2022) · Mahalanobis (Lee 2018) · Deep SVDD (Ruff 2018).
- **Health / collapse metrics `[EXT]`:** effective rank (Roy & Vetterli 2007) · RankMe (Garrido 2023) · LiDAR (Thilak
    2024\) · VICReg variance+covariance (Bardes 2022) · alignment & uniformity (Wang & Isola 2020) · dimensional collapse
    (Jing 2022) · intrinsic dimension (IdEst 2026 — label-free downstream proxy) · representation drift via CKA
    (Kornblith 2019; reliability caveats — Davari 2023) · ADWIN (Bifet 2007).
- **CL & evaluation metrics `[STD]`:** linear probe + kNN eval (Wu 2018) · Backward Transfer (Lopez-Paz & Ranzato 2017)
    · Forgetting Measure (Chaudhry 2018).
- **Control / replay `[EXT]`:** Deep Generative Replay (Shin 2017) · Brain-inspired replay (van de Ven 2020) ·
    DeepInversion (Yin 2020).
- **FL `[STD]`:** FedAvg (McMahan 2017) · FedProx (Li 2020) · Marfoq (2023) · Scaleout FedN.
- **Datasets:** STL-10 (Coates 2011) · BDD100K (Yu 2020) · nuScenes (Caesar 2020) · ZOD.
