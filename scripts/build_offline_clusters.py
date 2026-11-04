"""Extract all BDD100K images and fit the WP-M6.0.0 offline PCA/GMM baseline.

Run from the repository root with ``python -m scripts.build_offline_clusters --bdd-root ...``.
Re-running resumes extraction; ``--fit-only`` reuses a complete embedding cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cafl4ds.offline_clustering import extract_embeddings, fit_clusters


def main() -> None:
    """Run the standalone clustering experiment without constructing an MoE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/wp_m6_0_0"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        parser.error("Batch size and thread count must be positive; workers must be nonnegative.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if not args.fit_only:
        extract_embeddings(
            args.bdd_root.resolve(), args.output, device=args.device, batch_size=args.batch_size, workers=args.workers
        )
    fit_clusters(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
