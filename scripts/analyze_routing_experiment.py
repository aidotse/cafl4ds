"""Build the scientific tables and figures for a completed routing admission ensemble."""

import argparse
from pathlib import Path

from cafl4ds.routing_analysis import analyze


def main() -> None:
    """Analyze an existing result directory without modifying any training state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/moe_routing_entropy_2026-10-19"))
    arguments = parser.parse_args()
    analyze(arguments.output)


if __name__ == "__main__":
    main()
