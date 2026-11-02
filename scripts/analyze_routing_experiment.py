"""Build the scientific tables and figures for a completed routing admission ensemble."""

import argparse
from pathlib import Path

from cafl4ds.routing_analysis import analyze
from cafl4ds.routing_order_analysis import compare_ordering


def main() -> None:
    """Analyze an existing result directory without modifying any training state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/moe_routing_entropy_2026-10-19"))
    parser.add_argument("--compare-with", type=Path, help="Matched IID output directory for a regime-order comparison")
    arguments = parser.parse_args()
    analyze(arguments.output)
    if arguments.compare_with is not None:
        compare_ordering(arguments.output, arguments.compare_with)


if __name__ == "__main__":
    main()
