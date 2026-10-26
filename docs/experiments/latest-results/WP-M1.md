# WP-M1: routing entropy filtering

## The value proposition

Test whether using live router entropy to admit images changes later routing and representation quality. Separate the
mechanical effect on selected images from model change measured on fixed reference images.

## Results

The full paired comparison is complete. High-entropy admission changes the routing trajectory and increases held-out
reconstruction error relative to random-half. Low-entropy admission's reconstruction difference is uncertain. Neither
filter has a clear scene-probe advantage. Both alter the training diet; endpoint routing entropy remains high.

The apparatus verifies reconstruction gradients through the requested top-1 gate, identity/augmentation/mask pairing,
and exact checkpoint continuation on CPU and MPS. Raw trajectories, compact artifacts and source provenance are saved.

## The one-line takeaway

Routing entropy is a live selection lever in this stationary reference, but this experiment does not establish a
representation-quality benefit, forgetting, or the correlated-stream Phase-1 gate.

See the [method and artifact record](../phase1/WP-M1.md) and [design](../moe-routing-entropy-filtering-plan.md).
