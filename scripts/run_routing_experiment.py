"""Execute the paired MoE-MAE entropy admission experiment through Hydra."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from cafl4ds.routing_experiment import run_experiment


@hydra.main(version_base=None, config_path="../cafl4ds/configs", config_name="routing_experiment")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Train the three paired selectors and save their measurements and comparison."""
    run_experiment(cfg, Path(cfg.output))


if __name__ == "__main__":
    main()
