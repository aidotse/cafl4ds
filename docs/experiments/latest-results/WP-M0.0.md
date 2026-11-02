# WP-M0.0: native MAE warm-well comparison

## The value proposition

Check a native MoE-MAE against active- and total-parameter-matched dense references, then verify that the saved methods
run through the deployment harness. This is a subexperiment of [WP-M0](WP-M0.md).

## Results

All planned primary runs completed their fixed budget. Reconstruction parity is supported against the active dense
reference but remains inconclusive against the total-capacity reference. Linear-probe parity is supported against both;
kNN parity remains inconclusive. Most runs hit the budget cap without a certified plateau.

Every expert remains active in the final training window and held-out routing reads. The matched deployment smokes pass
their enabled validation checks, but use endpoint checkpoints where certified plateau wells are unavailable. These
checks establish integration, not full convergence, forgetting protection or completion of WP-M0.

## The one-line takeaway

The native MAE vehicle runs and supports some parity claims; convergence and the remaining parity comparisons are
unresolved.

See the [results and artifacts](../WP-M0-warm-well-mae-bdd-results.md) and
[execution plan](../WP-M0-warm-well-mae-bdd-plan.md).
