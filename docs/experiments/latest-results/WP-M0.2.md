# WP-M0.2: regime-blocked routing entropy filtering

## The value proposition

This is a subexperiment of [WP-M0: shared MoE groundwork](WP-M0.md).

Repeat the shuffled reference with contiguous time-of-day × weather blocks, holding data, initialization, training
budget, filters and evaluation fixed. Test how live admission behaves when available candidates are locally correlated.

## Results

All paired runs completed with the same images, initial weights, training budget and evaluation as the shuffled
reference. Both entropy filters worsen reconstruction relative to random-half and increase concentration in clean expert
loads over training. Neither improves scene probes. Low-entropy admission has a larger reconstruction penalty under
regime ordering than under shuffling; high entropy's corresponding change remains uncertain.

Large regimes contribute approximately half their images under every policy, so these differences persist without
substantial rebalancing of their frequencies. Low-entropy admission also produces transient gaps in actual training
assignments. Those gaps resolve before the final training window, and all experts are active at the endpoint. The study
uses one repeated label-defined regime walk; it does not establish forgetting or semantic expert specialization.

## The one-line takeaway

Regime ordering exposes a stronger disadvantage of low-entropy filtering while both entropy policies concentrate clean
expert loads and underperform random-half on reconstruction.

See the [protocol and artifact record](../phase1/WP-M0.2.md).
