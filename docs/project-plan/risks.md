‹ [Project Plan index](index.md)

# Risks

*The `N-x` novelty claims are stated in full in [novelty.md](novelty.md); baselines (PC, B5, B-open, …) in
[experiments.md](experiments.md#baselines).*

Ordered: **validity risks** (could make us conclude the wrong thing) first, then **mechanism risks**, then **honest
nulls** (disappointing but still results), then **execution**.

| Risk | Signal | Response |
| -- | -- | -- |
| **False negative on degradation** (underpowered demo) — collapse *or* forgetting | No degradation seen under stress | Pressure-sweep incl. from-scratch + no-replay + long horizon; collapse on joint-embedding, forgetting on MAE; **PC must fire** first |
| **Label-free signal can't detect in time** (leading-indicator null) | Tier-1 signal doesn't precede Tier-2 forgetting, or no usable lead time | Oracle-driven control result still stands (decoupled); report which signals are/aren't leading indicators — a negative result of independent value; deployability becomes the open question |
| **Coupling is regime-specific** (init) | Dynamics present from-scratch, absent pretrained (or vice versa) | Report the init-dependence as the finding (the regime where adaptation matters); pretrained bounds realism — don't overclaim deployment relevance from a from-scratch-only effect |
| **Synthetic-correlation artifact** (STL-10) | Degradation on 1a pilot but not 1b BDD | STL-10 validates the apparatus only; the claim rests on BDD's real correlation; non-transfer means class-blocking drove it — say so, lean on BDD |
| Filter wins on collapse metric but not downstream | RankMe up, probe flat | Optimized the thermometer, not the patient — validate on downstream + forgetting |
| **Actuator null** — F-c can't be steered | Varying α in open-loop doesn't move health | The controller has no lever → F-c reduces to a static hybrid; **gated (see below)** |
| Adaptation doesn't beat frozen (RanDumb) | B5 ≥ adapted | Existential — reframe to selection-for-efficiency; surface in Phase 1 |
| Closed loop doesn't beat open, *with oracle signal* (N-E/N-G null) | Closed ≈ or < B-open | Still a result (closing this loop is hard/unhelpful); keep the monitor as a passive alarm |
| Closed loop oscillates / destabilizes | Health metrics ring | Slow the controller; hysteresis; bound the α excursion — the stability finding is itself reportable |
| Coupling result just confirms DiSF/CCS offline | No surprise | Hunt loop instability / flip inversion / closed-loop benefit explicitly |
| Reservoir+dedup / SOFed hard to beat | Bake-off flat | Fine — criteria were never the contribution; weight stays on the coupling + control loop |
| FL infra slips / aggregation confounded | FedN not ready; can't separate skew from non-IID | FL science is parallel and deferrable (doesn't block the spine); the centralized reference isolates selection-induced skew from ordinary non-IID |
| Compute / Gaudi bottleneck; Phase-1 grid too large | Habana port slow; init × pressure × method blows up | CPU + small models for the demonstration; isolate Gaudi as a contained task; **trim rule:** full init×pressure sweep on the STL-10 pilot, carry 1–2 init levels to BDD |
| Drift back to the criterion bake-off | Work optimizing filters, not studying the coupling | The Novelty-at-a-glance (N-A…N-G) is the honesty check — if work doesn't map there, it's drift |
