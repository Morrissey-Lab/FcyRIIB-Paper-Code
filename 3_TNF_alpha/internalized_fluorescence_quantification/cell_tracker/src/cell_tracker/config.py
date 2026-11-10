"""Generic YAML config loading, shared by project pipelines."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Loads a project config.yaml and resolves every entry under `paths:`
    (if present) to an absolute path relative to the config file's own
    directory, so notebooks can be run from a `notebooks/` subfolder without
    path headaches.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_root = config_path.parent
    cfg["_project_root"] = project_root

    if "paths" in cfg:
        resolved = {}
        for key, value in cfg["paths"].items():
            p = Path(value)
            resolved[key] = p if p.is_absolute() else (project_root / p).resolve()
        cfg["paths"] = resolved

    return cfg
