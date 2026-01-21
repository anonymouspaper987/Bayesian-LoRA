import logging
import os
from omegaconf import DictConfig, OmegaConf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")


def clean_dir(dir_path: str) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def setup_logging(cfg: DictConfig):
   
    logger = logging.getLogger("baye_lora")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    if cfg.baye_lora:
        path_prime = os.path.join(LOG_DIR, cfg.dset.alias_name, "baye_lora")
        os.makedirs(path_prime, exist_ok=True)
        log_path = os.path.join(path_prime, f"{cfg.dset.alias_name}_s_{cfg.inducing.inducing_rows}_{cfg.inducing_type}.log")
    else:
        path_prime = os.path.join(LOG_DIR, cfg.dset.alias_name, "lora")
        if cfg.dropout: 
            path_prime = os.path.join(path_prime, "dropout")
        else:
            path_prime = os.path.join(path_prime, "MAP")
        os.makedirs(path_prime, exist_ok=True)
        log_path = os.path.join(path_prime, "pure_lora.log")
   
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    if getattr(cfg, "print_config", False):
        logger.debug(OmegaConf.to_yaml(cfg))

    return logger
