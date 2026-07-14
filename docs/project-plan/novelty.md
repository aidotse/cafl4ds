‹ [Project Plan index](index.md)

# Novelty & positioning

*Tags: `[STD]` re-implement · `[EXT]` extend to our setting · `[NEW]` genuine contribution. The one-line version of the
claims below lives in the [index](index.md#novelty-at-a-glance); this file is the full statement + the closest prior
work each `[NEW]` claim stands against.*

## ★ What is `[NEW]`, and the prior work each claim stands against

*The space is crowded along each axis; the assembly is not. Every claim below is paired with the closest prior work on
that axis — what it covers, and what it leaves open — so the mapping is explicit. Terminology: an **edge** is a literal
connection in the loop (model→filter, or monitor→filter); the claims below are the distinct **axes** of contribution,
only two of which (inside N-A) are literal edges. The contribution is the **closed loop as a live system**, not any
single axis.*

**N-A — The coupling, run as a live loop `[NEW]`.** *Claim:* the two loop **edges** operate together and are studied
live — the **fast edge** (model→filter: the filter scores informativeness against the continuously-adapting backbone, so
what counts as novel shifts as the model learns) and the **slow edge** (monitor→filter: the health monitor re-aims
selection based on the degradation regime). *Closest prior:* DiSF (2025) links selection→collapse but *offline, fixed
budget, no model in the loop*; SOFed/FedCoCo (2022) does selection in streaming SSL but with *one fixed criterion and no
health coupling*. *Open:* nobody runs selection ↔ representation-health as a closed feedback loop with a co-adapting
model.

**N-B — Does the budget flip survive co-adaptation? `[NEW]`** *Terms:* **storage budget** = how many frames the buffer
may keep; **coverage** selection = pick frames that represent/span the data distribution; **novelty** selection = pick
hard/atypical frames. The **budget flip**: at small budgets coverage wins, at large budgets novelty wins — the best rule
"flips" coverage→novelty as the budget grows. *Closest prior:* CCS (Zheng 2023) — coverage beats hard-example selection
at high pruning rates (small budgets); Sorscher (2022) — keep prototypical when data is scarce, hard when abundant; D2
Pruning (2024) — balances diversity vs difficulty. All *offline, fixed model, pool-based*. *Open:* whether the flip
holds, moves, or inverts under single-pass causal selection against a model the selection is itself changing.

**N-C — Self-reinforcing selection (in)stability `[NEW]`.** *Claim:* selection drives the model, which changes what is
selected next — a loop that may self-reinforce toward a degenerate diet, or be stabilized. *Closest prior:*
one-sided-feedback learning (2020), recommender/bandit sampling-bias loops, and LLM self-consuming "model collapse"
(Shumailov et al. 2023) — the same pathology, in *other domains*. *Open:* the streaming-SSL-selection instance, its
damping, and how FL averaging amplifies or damps it.

**N-D — Selection × aggregation skew in FL `[NEW]`.** *Claim:* independent clients each over-selecting their local tail
may skew the aggregate's effective training distribution; global-aware selection may correct it. *Closest prior:* SOFed
(2022) notes overlapping client selections in passing; federated SSL — FedU (2021), Orchestra (2022) — aggregates
unsupervised representations but *without* active per-client selection bias. *Open:* characterizing and correcting the
selection-induced aggregate drift.

**N-E — Online health-monitor → control loop `[NEW]`.** *Claim:* a multivariate health signal used as a **live
controller** — modulating selection (the monitor→filter edge) and scheduling replay/rollback — not merely a diagnostic.
*Closest prior:* RankMe (Garrido 2023), LiDAR (Thilak 2024), dimensional collapse (Jing 2022) supply the health
**metrics**, but as *offline diagnostics / checkpoint selection*; ADWIN (Bifet 2007) detects drift but is not tied to
SSL-health control. *Open:* closing the loop live; payoff is conditional — does closed-loop beat open-loop (B-open)?

**N-F — Federation-level health analytics `[NEW]`.** *Claim:* flag a degraded/corrupted **client** via
representation-health diagnostics before its update poisons the global model — the server-side twin of the on-device
monitor. *Closest prior:* robust-FL aggregation — Krum (Blanchard 2017), trimmed-mean/median (Yin 2018) — detects bad
updates via *update geometry*, not via the client model's representation health. *Open:* using the same health analytics
(effective rank, etc.) as the pre-aggregation gate.

**N-G — Health-steerable filter (the actuator) `[NEW]`.** *Claim:* a selection policy whose coverage↔novelty operating
point (mix `α`) **and** acceptance rate are a **live function of the health signal**, so the monitor can steer it — the
actuator that makes the loop closable. *Closest prior:* the static filters we would otherwise steer — reservoir (Vitter
1985), SemDeDup (Abbas 2023), loss-gate (SOFed 2022), core-set (Sener & Savarese 2018) — expose *no* coverage↔novelty
knob; the hybrid filters that *do* have the mix — D2 Pruning (2024), CCS (2023) — fix it *offline*, never driven by a
live signal. *Open:* making the mix (and acceptance rate) a controlled variable set by the monitor. *Validation:* only
via closed-loop vs. open-loop ([B-open](experiments.md#baselines)) on health + downstream — never a static-benchmark
win.

**Summary**

- The loop, not just the leaderboard, is what makes this filter design novel
- This informs our approach: we can implement the loop with existing pieces, and analyze system behavior. This study is
    novel.
- Afterward: work toward novelty along any of the axes above.

*Everything else is `[STD]` or `[EXT]`. If a claim is not in the list above, we are not claiming it as novel. The full
reference list (including the `[STD]`/`[EXT]` toolbox) is in the [reading list](reading-list.md).*

The prior work grouped by the claim it bears on is in the
[reading list → Position against](reading-list.md#position-against-work-we-must-distinguish-ourselves-from).
