"""Central registry of ilastik "panels" — one trained Pixel Classification
project per marker, reused across every project that images that marker.
Every panel here is trained on a SINGLE marker channel only (e.g. a
cytoplasm/membrane marker) — never a nucleus channel. Nuclei are always
identified separately in pure Python (see cell_tracker.nuclear_segmentation),
not by ilastik. Each panel lives in its own folder under panels_root/
<panel_name>/ (panel_name may itself contain '/' for nested storage, e.g.
"cell tracker/red") containing:

  panel.yaml               channel definition, pixel-class index mapping, tuning
  membrane_segmentation.ilp the trained ilastik project

Training a panel is a one-time cost (see notebooks/00_train_and_test_panel.ipynb).
Every project pipeline just references a panel by name from then on.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_ILP_FILENAME = "membrane_segmentation.ilp"
PANEL_YAML_FILENAME = "panel.yaml"


def list_panels(panels_root: str | Path) -> list[str]:
    """Recursively finds every panel.yaml under panels_root and returns its
    location relative to panels_root (posix-style, '/'-separated) — so
    nested panel names like 'cell tracker/red' are listed correctly, not
    just immediate children of panels_root.
    """
    panels_root = Path(panels_root)
    if not panels_root.exists():
        return []
    return sorted(
        p.parent.relative_to(panels_root).as_posix() for p in panels_root.rglob(PANEL_YAML_FILENAME)
    )


def load_panel(panels_root: str | Path, panel_name: str) -> dict:
    panel_dir = Path(panels_root) / panel_name
    panel_yaml = panel_dir / PANEL_YAML_FILENAME
    if not panel_yaml.exists():
        available = list_panels(panels_root)
        raise FileNotFoundError(
            f"No panel.yaml found for panel '{panel_name}' at {panel_yaml}. "
            f"Available panels in {panels_root}: {available or '(none yet)'}"
        )

    with open(panel_yaml, "r") as f:
        panel = yaml.safe_load(f)

    panel["_panel_dir"] = panel_dir
    ilp_filename = panel.get("ilastik", {}).get("project_filename", DEFAULT_ILP_FILENAME)
    panel["ilastik_project_path"] = panel_dir / ilp_filename

    if not panel["ilastik_project_path"].exists():
        raise FileNotFoundError(
            f"panel.yaml for '{panel_name}' exists but its ilastik project "
            f"({panel['ilastik_project_path']}) doesn't. Train it first — "
            f"see notebooks/00_train_and_test_panel*.ipynb (there's one per "
            f"panel, e.g. 00_train_and_test_panel_{panel_name}.ipynb if it "
            f"exists) or that panel's own README.md."
        )

    return panel


def scaffold_panel(
    panels_root: str | Path,
    panel_name: str,
    channel_name: str,
    wavelength_nm: float | None = None,
    channel_index: int | None = None,
    membrane_class: str = "membrane",
    description: str = "",
    overwrite: bool = False,
) -> Path:
    """Creates panels_root/<panel_name>/panel.yaml for a new SINGLE-channel
    ilastik Pixel Classification panel — the only kind this package trains.
    ilastik is shown exactly one marker channel (e.g. a cytoplasm/membrane
    marker) and classifies it into `membrane_class` vs "background"; nuclei
    are never part of this panel at all (see cell_tracker.nuclear_segmentation
    for that half of segmentation). Does NOT create the .ilp — that's trained
    interactively in the ilastik GUI and saved into this same folder
    afterward. panel_name may contain '/' for nested storage (e.g.
    "cell tracker/red"). Returns the panel directory.
    """
    panel_dir = Path(panels_root) / panel_name
    panel_yaml = panel_dir / PANEL_YAML_FILENAME
    if panel_yaml.exists() and not overwrite:
        raise FileExistsError(f"{panel_yaml} already exists. Pass overwrite=True to replace it.")

    panel_dir.mkdir(parents=True, exist_ok=True)

    panel = {
        "panel_name": panel_name,
        "description": description,
        "channels": {
            membrane_class: {
                "name": channel_name,
                "wavelength_nm": wavelength_nm,
                "index": channel_index,
            }
        },
        "channel_stack_order": [membrane_class],
        "ilastik": {
            "project_filename": DEFAULT_ILP_FILENAME,
            "export_source": "Probabilities",
            "pixel_classes": {membrane_class: 0, "background": 1},
            "membrane_class": membrane_class,
        },
        "segmentation": {
            # Gaussian-smooths BOTH the raw membrane probability map and the
            # nuclear channel (before Otsu-thresholding it for nuclei). 0 = off.
            "smoothing_sigma": 0.0,
            # Absolute cutoff on the (smoothed) membrane probability map used
            # to build the foreground mask that nuclei seeds grow into.
            "probability_threshold": 0.5,
            # Minimum object size (px) -- applied to both the nuclear mask
            # (drops debris before it can seed a spurious cell) and the final
            # watershed cell regions (drops slivers with no real nucleus).
            "min_cell_size_px": 20,
            # Minimum (99.5th percentile - median) intensity gap required in
            # the nuclear channel before Otsu thresholding is even attempted.
            # 0 = off. Raise above 0 to stop images with no real signal from
            # having noise force-split into fake nuclei.
            "min_signal_range": 0.0,
            # Fills any background-classified hole in the mask that's fully
            # enclosed by foreground (fixes ilastik misclassifying interior/
            # cytoplasmic pixels as background). True = on, doesn't touch
            # probability values or true outer boundaries.
            "fill_holes": True,
            # Frame-level exclusion (flags via the manifest, doesn't drop
            # anything): None = off.
            "max_nuclei_per_frame": None,
            "max_membrane_area_fraction": None,
        },
    }

    with open(panel_yaml, "w") as f:
        yaml.safe_dump(panel, f, sort_keys=False)

    return panel_dir
