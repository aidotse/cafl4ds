# WP-M0.1: routing entropy filtering

## The value proposition

This is a subexperiment of [WP-M0: shared MoE groundwork](WP-M0.md).

Test whether using live router entropy to admit images changes later routing and representation quality. Separate the
mechanical effect on selected images from model change measured on fixed reference images.

## Results

The full paired comparison is complete. High-entropy admission changes the routing trajectory and increases held-out
reconstruction error relative to random-half. Low-entropy admission's reconstruction difference is uncertain. Neither
filter has a clear scene-probe advantage. Both alter the training diet; endpoint mean per-token router entropy remains
high.

The load analysis shows higher time-averaged concentration of clean full-image assignments under high-entropy admission
in both MoE layers. Endpoint differences are less certain. Actual noisy masked training assignments remain broadly
distributed, with no observed expert receiving zero visible patches in a training update. The load charts therefore show
diagnostic concentration, not established training starvation or semantic specialization.

The apparatus verifies reconstruction gradients through the requested top-1 gate, identity/augmentation/mask pairing,
and exact checkpoint continuation on CPU and MPS. Raw trajectories, compact artifacts and source provenance are saved.

## The one-line takeaway

Routing entropy is a live selection lever in this stationary reference, but this experiment does not establish a
representation-quality benefit, forgetting, or the correlated-stream Phase-1 gate.

See the [method and artifact record](../phase1/WP-M0.1.md) and [design](../moe-routing-entropy-filtering-plan.md).
