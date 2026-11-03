# WP-M0: an MoE backbone in the loop — shared groundwork

## The value proposition

Build a correct, reproducible MoE vehicle before using it to study specialization or health. The
[work-package goals](https://docs.google.com/document/d/1h1MxKhs0Bz2-21CArZ0lvcU2ZnvEaOoquVsFrt76FzY/edit) set parity
with matched dense references as the bar, rather than superiority.

The experiments completed so far are subexperiments of this groundwork. The
[Phase-1 index](../phase1/index.md#supplementary-moe-groundwork-wp-m0) routes to their methods and artifacts.

## Results

| Subexperiment | What it contributes | Record |
| -- | -- | -- |
| WP-M0.0 — native MAE warm-well comparison | Dense controls, expert-activity checks and a harness smoke; parity is supported on some endpoints but remains inconclusive on others, and most wells are cap-limited. | [Summary](WP-M0.0.md) · [Results](../WP-M0.0-warm-well-mae-bdd-results.md) · [Plan](../WP-M0.0-warm-well-mae-bdd-plan.md) |
| WP-M0.1 — shuffled routing entropy filtering | A paired, resumable live-selection vehicle with routing-gradient and data-identity checks; high-entropy filtering concentrates clean loads and worsens reconstruction. | [Summary](WP-M0.1.md) · [Method](../phase1/WP-M0.1.md) |
| WP-M0.2 — regime-blocked routing entropy filtering | A matched ordering reference; both entropy policies worsen reconstruction, and the low-entropy penalty grows relative to shuffling. | [Summary](WP-M0.2.md) · [Method](../phase1/WP-M0.2.md) |

WP-M0.1 and WP-M0.2 start from random weights, not the WP-M0.0 wells. Their full-softmax top-1 gate differs from the
earlier warm-well gate, so the warm-well numbers are not matched controls for filtering.

**WP-M0 remains incomplete.** The deferred balancing modes and horizons, upcycling comparison, JE cross-view consistency
option, and required warm-JE native/upcycled and dense-reference corpora remain package requirements. The warm-well
results also leave convergence and parts of dense parity unresolved.

The broader package IDs keep the meanings in the goals: **WP-M1** asks whether experts specialize by regime and whether
change localizes; **WP-M2** builds and calibrates routing health signals against positive controls. Neither package is
completed by the filtering references. Clean-load concentration is not evidence of semantic specialization, and logging
routing statistics does not calibrate them as health signals. These results also do not establish forgetting or satisfy
the correlated single-pass Phase-1 gate.

Promoted artifact directories and experiment identifiers use the subexperiment IDs above. Archived raw-run paths and
frozen provenance retain their recorded names and contents. Protocol hashes and scientific results are unchanged.

## The one-line takeaway

The MoE and live-selection apparatus now have useful reference experiments; the full WP-M0 groundwork and the planned
WP-M1/WP-M2 scientific questions remain open.
