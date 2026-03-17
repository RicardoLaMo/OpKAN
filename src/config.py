"""Loads OpKAN configuration from config/heston_defaults.yaml."""
import os
import yaml
from typing import Any

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "heston_defaults.yaml"
)

def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config file and return as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_heston_params(cfg: dict) -> dict:
    """Extract Heston model parameters from config dict."""
    return dict(cfg["heston"])

def get_training_params(cfg: dict) -> dict:
    """Extract training parameters from config dict."""
    return dict(cfg["training"])

def get_collocation_params(cfg: dict) -> dict:
    """Extract PDE domain parameters from config dict."""
    return dict(cfg["collocation"])
