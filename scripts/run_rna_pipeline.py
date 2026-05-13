import hydra
from omegaconf import DictConfig

from pipeline_framework import run_unified_pipeline


@hydra.main(config_path="../configs", config_name="config_rna")
def main(cfg: DictConfig):
    run_unified_pipeline(cfg, task_name="nuc")


if __name__ == "__main__":
    main()
