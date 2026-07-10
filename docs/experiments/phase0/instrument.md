# Instrument — the streaming SSL loop (CPU)

The first Phase-0 sub-study (see the [Phase 0 overview](index.md)). Status: **CPU implementation complete.** The
streaming SSL adaptation loop, the class-blocked STL-10 stream, both SSL backbones, and separate loss/health logging all
run end-to-end on CPU, exercising the full `C × I` matrix under the B-floor knob. The positive control (PC) and the
Phase-0 exit *gate* ("PC collapses on RankMe") are **not** part of this slice — they are separate Phase-0 studies (see
the [overview](index.md) and [What is deliberately out of scope](#what-is-deliberately-out-of-scope)).

This page is the handoff record for the instrument/loop slice. It is enough to resume in a fresh session.

______________________________________________________________________

## Scope of this slice

Per the plan's *Strategic ordering* (build the loop + measurement apparatus first), this slice adds — on top of the
already-verified instruments (`cafl4ds/measurements.py`) — the machinery to drive those instruments over a live,
correlated stream:

- **The class-blocked STL-10 stream** — synthetic correlation (eras = class blocks), with per-era and global held-out
    eval sets. Labels are used *only* to build the ordering and the eval sets, never in the SSL update.
- **Two SSL backbones behind the `C` flag** — **MAE** (default; reconstruction-error signal, collapse-resistant) and
    **SimSiam** (joint-embedding; predictor + stop-gradient, no EMA / no negatives — the collapse-capable backbone; BYOL
    then SimCLR are later drop-ins). Both support the `I` flag: `from_scratch` and `pretrained`.
- **The streaming loop** with the **B-floor** knob only (accept-all, no replay, raw order), calling the instruments
    every *N* steps and logging SSL loss and health metrics to separate series in one run log.

**Only B-floor exists.** No other filters, no positive control, no closed loop.

### The `I=pretrained` warm start

`pretrained` init loads the encoder from a checkpoint produced by an **IID (shuffled)** SSL pre-pass
(`scripts/pretrain.py`, `stream=iid`). This is deliberate: the warm start must be a *well-behaved reference*. If we
pretrained on the class-blocked ordering we would have half-run the experiment and baked stream-induced effects into the
"clean" starting point — correlation must enter **only** in the streaming phase.

______________________________________________________________________

## Architecture (what maps to what)

Everything is Hydra `instantiate`-wired and left with seams for Phases 1–3. The factor letters (`A`, `C`, `F`, `I`) are
from the plan's experiment matrix.

| Component          | Module                                      | Role / factor                                                                        | Future-phase seam                |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------- |
| Data source        | `cafl4ds/data/sources.py`                   | `STL10Source`, `SyntheticSource` (network-free)                                      | BDD100K/ZOD = new source         |
| Stream             | `cafl4ds/data/streams.py`                   | `EraStream` — `order=class_blocked` (**F**=correlated) or `iid`; eras, held-out eval | drift/segment streams            |
| Encoder            | `cafl4ds/models/vit.py`                     | shared `TinyViTEncoder` (MAE masking + pooled `embed`)                               | timm/pretrained encoder at scale |
| Heads              | `cafl4ds/models/heads.py`                   | `MLPHead` (proj/pred), `MAEDecoder`                                                  | BYOL predictor etc.              |
| SSL method (**C**) | `cafl4ds/ssl/{base,mae,simsiam,factory}.py` | `SSLMethod`: `training_step`, `encode`; `apply_encoder_init` (**I**)                 | BYOL/SimCLR, PC                  |
| Filter (**A**)     | `cafl4ds/filters/{base,accept_all}.py`      | `Filter.select`; **B-floor** `AcceptAll`                                             | F-a/F-b/F-c                      |
| Monitor            | `cafl4ds/monitor.py`                        | `HealthMonitor` runs instruments on the fixed probe set                              | slow-edge controller             |
| Run log            | `cafl4ds/run_log.py`                        | one JSONL, **loss** + **health** as separate series; `tabulate`                      | —                                |
| Loop               | `cafl4ds/loop.py`                           | `StreamingLoop` — B-floor, raw order, no replay                                      | knobs plug in here               |

Key invariants (all covered by tests in `tests/unit/`):

- A `StreamBatch` carries **only** images (+ era/step); labels can never reach `training_step`.
- Held-out probe-support / probe-query / per-era eval sets are pairwise disjoint from training.
- The health probe reads the **encoder** embedding (`encode`), never a projector/predictor head.
- Drift is measured against the **first** checkpoint's embeddings of a fixed probe set.

______________________________________________________________________

## How to run

Environment: `uv sync --group dev --extra cpu` (adds `torchvision`, pinned to the same per-backend PyTorch index as
`torch`). STL-10 lives on the shared drive at `/mnt/stl10` (the `data_root` default). To provision a fresh machine,
download it once (writing to `/mnt` needs root; point `data_root` at a local copy otherwise):

```python
from torchvision.datasets import STL10
STL10(root="/mnt/stl10", split="train", download=True)
STL10(root="/mnt/stl10", split="test", download=True)
```

Fastest network-free smoke (synthetic source):

```bash
uv run python scripts/run_loop.py data=synthetic img_size=16 eval_every=3
```

Produce the `I=pretrained` warm-start checkpoints (IID pre-pass, both backbones):

```bash
uv run python scripts/pretrain.py -m ssl=mae,simsiam   # -> outputs/pretrain/<method>.pt
```

The Phase-0 **exit matrix** (`C × I` on class-blocked STL-10):

```bash
uv run python scripts/run_loop.py -m ssl=mae,simsiam init=from_scratch,pretrained
```

Run logs land in each Hydra run/job dir as `run_log.jsonl` (resolved via `HydraConfig`, so multirun jobs never collide).
Render a run's side-by-side table:

```python
from cafl4ds.run_log import read_run, tabulate
recs = read_run("outputs/.../run_log.jsonl")
print(tabulate([r for r in recs if r["series"] == "health"]))
```

Config groups live in `cafl4ds/configs/` (`data/`, `stream/`, `encoder/`, `ssl/`, `filter/`, `monitor/`, `optim/`,
`init/`) composed by `loop.yaml` (and `pretrain.yaml`).

______________________________________________________________________

## Exit-criterion result (CPU)

A short CPU run of all four `C × I` configs completes and the run log shows the SSL loss and the health metrics (rankme,
drift, probe) **side by side over steps**. Settings: tiny ViT (`embed_dim=96, depth=4, patch=8`), `img_size=32`, STL-10
capped to 120 imgs/class, `eval_every=4`. These are *harness-validation* numbers, not a results run — the model is tiny
and the horizon is a handful of batches, so probe accuracy sits near chance by design.

What to look for (and what we see): loss decreases; **RankMe responds** (falls under the correlated stream); **drift
accumulates** (cka/cosine of the fixed probe set rise as the model adapts across class blocks); probes are logged and
functional for both backbones and both inits.

### mae_from_scratch (30 loss steps, 9 health checkpoints)

```
        step           era          loss        rankme     cka_drift  cosine_drift       knn_acc    linear_acc
------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000        1.3193        7.2750        0.0000        0.0000        0.1800        0.2150
      4.0000        1.0000        1.1233        3.9638        0.0186        0.2442        0.1700        0.2100
      8.0000        2.0000        1.0421        3.1623        0.0321        0.3270        0.1750        0.2200
     12.0000        4.0000        0.9501        2.7975        0.0437        0.3494        0.1750        0.2200
     16.0000        5.0000        0.9902        2.5955        0.0546        0.3551        0.1700        0.2050
     20.0000        6.0000        0.8679        2.4686        0.0508        0.3537        0.1750        0.2050
     24.0000        8.0000        1.1812        2.3956        0.0485        0.3588        0.1750        0.2000
     28.0000        9.0000        1.0396        2.3632        0.0432        0.3710        0.1750        0.2000
     29.0000        9.0000        1.0226        2.3574        0.0421        0.3737        0.1750        0.2100
```

### mae_pretrained (30 loss steps, 9 health checkpoints)

```
        step           era          loss        rankme     cka_drift  cosine_drift       knn_acc    linear_acc
------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000        1.2858        5.7616        0.0000        0.0000        0.1050        0.2000
      4.0000        1.0000        1.0926        4.4303        0.0387        0.0650        0.1350        0.2050
      8.0000        2.0000        1.0315        3.9026        0.1685        0.1127        0.1400        0.2400
     12.0000        4.0000        0.9408        3.6880        0.1528        0.1239        0.1350        0.2350
     16.0000        5.0000        0.9912        3.6264        0.1306        0.1204        0.1250        0.2350
     20.0000        6.0000        0.8689        3.5177        0.2974        0.3617        0.1650        0.2400
     24.0000        8.0000        1.1526        3.5668        0.4007        0.4426        0.1650        0.2350
     28.0000        9.0000        1.0254        3.6609        0.5798        0.6825        0.1650        0.2200
     29.0000        9.0000        1.0132        3.6950        0.5896        0.7037        0.1700        0.2250
```

### simsiam_from_scratch (30 loss steps, 9 health checkpoints)

```
        step           era          loss        rankme     cka_drift  cosine_drift       knn_acc    linear_acc
------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000        0.0020        8.5762        0.0000        0.0000        0.2000        0.2350
      4.0000        1.0000       -0.0875        6.4934        0.0483        0.2075        0.1750        0.2150
      8.0000        2.0000       -0.2420        6.2126        0.0760        0.2692        0.1650        0.2250
     12.0000        4.0000       -0.3577        6.1544        0.0739        0.3107        0.1700        0.2400
     16.0000        5.0000       -0.4083        5.4714        0.1140        0.4133        0.1700        0.2400
     20.0000        6.0000       -0.4766        4.4264        0.0972        0.4192        0.1600        0.2350
     24.0000        8.0000       -0.5512        3.7843        0.1097        0.4909        0.1600        0.2200
     28.0000        9.0000       -0.4770        3.1512        0.0678        0.5466        0.1850        0.2100
     29.0000        9.0000       -0.6228        3.0435        0.0616        0.5434        0.1800        0.2050
```

### simsiam_pretrained (30 loss steps, 9 health checkpoints)

```
        step           era          loss        rankme     cka_drift  cosine_drift       knn_acc    linear_acc
------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
      0.0000        0.0000       -0.0437        2.4588        0.0000        0.0000        0.1550        0.1950
      4.0000        1.0000       -0.1647        2.5181        0.0012        0.0138        0.1600        0.2050
      8.0000        2.0000       -0.2811        2.5735        0.0013        0.0351        0.1400        0.2050
     12.0000        4.0000       -0.4525        2.6672        0.0023        0.0480        0.1650        0.2200
     16.0000        5.0000       -0.4960        2.6860        0.0029        0.0616        0.1600        0.2400
     20.0000        6.0000       -0.5717        2.6285        0.0026        0.0609        0.1550        0.2400
     24.0000        8.0000       -0.6557        2.5565        0.0182        0.0851        0.1400        0.2400
     28.0000        9.0000       -0.6276        2.3845        0.0444        0.1104        0.1400        0.2300
     29.0000        9.0000       -0.7547        2.3362        0.0403        0.1135        0.1400        0.2300
```

Reading the numbers (illustrative, not claims): under the class-blocked stream, RankMe drifts downward and
representation drift climbs in every configuration — i.e. the instruments *move* in response to a correlated single-pass
diet, which is exactly the responsiveness Phase 1 will need. `mae_pretrained` shows the largest drift (a good pretrained
basin being pulled around by the correlated stream), while `simsiam_pretrained` starts at a lower RankMe and stays
flattest. None of this is interpreted as degradation yet — that is Phase 1's job, with the PC as the gate.

______________________________________________________________________

## What is deliberately out of scope

- **Positive control + the Phase-0 gate.** The plan's formal Phase-0 exit is "instruments validated **and PC collapses
    on RankMe**." The instruments are validated (`tests/unit/test_measurements.py`) and now demonstrably *drivable* over
    a stream, but the PC (a config known to collapse) is not built here. It is the natural next increment.
- **Any knob other than B-floor** (F-a novelty, F-b coverage, F-c steerable), replay, the monitor→filter controller, and
    FL — all later phases.
- **HPU scaling.** This slice is CPU-first by design; the next step is a scaled model/dataset run on the Gaudi HPUs
    (`docs/developing.md`).

## Suggested next steps

1. Build the **positive control** and confirm it collapses on RankMe (closes the Phase-0 gate).
1. Scale the model/dataset and run on the HPU(s) (`scripts/run_gaudi_dev.sh`); torchvision on Gaudi comes from the
    Habana base image (mirror the torch handling in `docker/gaudi.env.Dockerfile`).
1. Begin **Phase 1a**: sweep `A{B-floor,…} × I × C × P`, instrument both degradation modes.
