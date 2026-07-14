‹ [Project Plan index](index.md)

# Reading list

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution. The `N-x` claims are stated
in full in [novelty.md](novelty.md).*

## Position against (work we must distinguish ourselves from)

*Grouped by the part of our approach each bears on.*

- **The coupling as a live loop (N-A):** DiSF (2025) — selection→collapse, *offline* · SOFed/FedCoCo (Shi 2022) —
    streaming-SSL selection, *single criterion, no loop* (also the closest prior for N-D).
- **Budget flip under co-adaptation (N-B):** CCS (Zheng 2023) · Sorscher (2022) · D2 Pruning (Maharana 2024) — *offline,
    fixed model*.
- **Selection-loop (in)stability (N-C):** one-sided-feedback (2020) · recsys/bandit sampling-bias loops · LLM
    self-consuming / model collapse (Shumailov 2023) — *other domains*.
- **Selection × aggregation skew in FL (N-D):** FedU (Zhuang 2021) · Orchestra (Lubana 2022) — *federated SSL without
    active selection bias*.
- **Health-monitor → control loop (N-E):** RankMe (Garrido 2023) · LiDAR (Thilak 2024) · dimensional collapse (Jing
    2022\) · ADWIN (Bifet 2007) — *health metrics / drift as offline diagnostics, not live control*.
- **Federation-level health analytics (N-F):** Krum (Blanchard 2017) · trimmed-mean/median (Yin 2018) — *bad-client
    detection by update geometry, not representation health*.
- **Health-steerable filter (N-G):** D2 Pruning (2024) · CCS (2023) — coverage/difficulty mix *fixed offline*, never
    steered live.
- **Collapse prevention (context for N-A/N-E):** VICReg (Bardes 2022) · IConE (2026) · AdaDim (2025) · CMP (2025) — *via
    loss/architecture, not selection, not a loop*.
- **Online-SSL degradation premise (Phase 1 / motivates B5):** continual-SSL line · RanDumb (2024) — *streaming degrades
    SSL; fixed features can rival online-learned ones*.
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
    (Jing 2022) · representation drift via CKA (Kornblith 2019) · ADWIN (Bifet 2007).
- **CL & evaluation metrics `[STD]`:** linear probe + kNN eval (Wu 2018) · Backward Transfer (Lopez-Paz & Ranzato 2017)
    · Forgetting Measure (Chaudhry 2018).
- **Control / replay `[EXT]`:** Deep Generative Replay (Shin 2017) · Brain-inspired replay (van de Ven 2020) ·
    DeepInversion (Yin 2020).
- **FL `[STD]`:** FedAvg (McMahan 2017) · FedProx (Li 2020) · Marfoq (2023) · Scaleout FedN.
- **Datasets:** STL-10 (Coates 2011) · BDD100K (Yu 2020) · nuScenes (Caesar 2020) · ZOD.
