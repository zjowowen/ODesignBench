import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline


@hydra.main(config_path="../configs", config_name="config_dna")
def main(cfg: DictConfig):
    run_unified_pipeline(cfg, task_name="dna")


if __name__ == "__main__":
    main()
