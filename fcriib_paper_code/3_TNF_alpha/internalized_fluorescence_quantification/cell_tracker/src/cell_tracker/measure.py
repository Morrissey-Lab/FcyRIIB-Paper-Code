"""Generic per-cell measurement helpers, meant to be reused by any
project-specific downstream analysis built on top of the shared segmentation
(target-channel MFI today; nothing stops a project from adding morphology,
multiple channels, ratios, etc. using the same label masks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import measure as skmeasure
from skimage.segmentation import clear_border as sk_clear_border


def load_segmentation(path: str | Path) -> np.ndarray:
    """Loads an instance-labeled segmentation mask (integer labels, 0=background)."""
    label_img = tifffile.imread(path)
    if label_img.ndim > 2:
        label_img = np.squeeze(label_img)
    return label_img.astype(np.int32)


def compute_segmentation_background(label_img: np.ndarray, intensity_img: np.ndarray) -> float:
    """Mean intensity over every pixel NOT covered by any segmented cell
    (label_img == 0) -- i.e. the actual imaged background of this specific
    frame, rather than a generic whole-image statistic like the median (which
    background_subtract's 'median' method uses and which is skewed by
    however much of the frame is covered by cells). Meant to be computed
    once on a manually chosen reference point (e.g. an empty/no-bead frame)
    and reused as a fixed background value for other points -- see
    background_subtract's 'fixed' method.
    """
    label_img = np.asarray(label_img)
    intensity_img = np.asarray(intensity_img)
    background_pixels = intensity_img[label_img == 0]
    if background_pixels.size == 0:
        raise ValueError(
            "No non-cell pixels found in this reference point (the whole frame "
            "is covered by segmented cells) -- pick a different reference point."
        )
    return float(background_pixels.mean())


def background_subtract(img: np.ndarray, method: str = "median", value: float | None = None) -> np.ndarray:
    img = img.astype(np.float64)
    if method == "median":
        bg = np.median(img)
    elif method == "rolling_ball":
        from skimage.restoration import rolling_ball

        bg = rolling_ball(img)
        return np.clip(img - bg, 0, None)
    elif method == "fixed":
        if value is None:
            raise ValueError("background_subtract(method='fixed') requires a value=... to subtract.")
        bg = value
    else:
        raise ValueError(f"Unknown background subtraction method: {method}")
    return np.clip(img - bg, 0, None)


def measure_intensity_per_cell(
    label_img: np.ndarray,
    intensity_img: np.ndarray,
    channel_name: str = "target",
    exclude_border: bool = False,
    **extra_cols,
) -> pd.DataFrame:
    """One row per cell: area + mean/max/integrated intensity of `intensity_img`
    within each labeled cell, plus centroid. `channel_name` prefixes the
    intensity columns (e.g. 'mean_intensity_488'). Any keyword args in
    extra_cols (e.g. source_file=..., replicate=..., condition=...) are
    attached as constant columns for traceability.
    """
    if exclude_border:
        label_img = sk_clear_border(label_img)

    props = skmeasure.regionprops_table(
        label_img,
        intensity_image=intensity_img,
        properties=("label", "area", "mean_intensity", "max_intensity", "centroid"),
    )
    df = pd.DataFrame(props).rename(
        columns={
            "label": "cell_id",
            "area": "area_px",
            "mean_intensity": f"mean_intensity_{channel_name}",
            "max_intensity": f"max_intensity_{channel_name}",
            "centroid-0": "centroid_row",
            "centroid-1": "centroid_col",
        }
    )
    df[f"integrated_intensity_{channel_name}"] = df["area_px"] * df[f"mean_intensity_{channel_name}"]

    for i, (k, v) in enumerate(extra_cols.items()):
        df.insert(i, k, v)

    return df


def aggregate_results(
    per_image_dfs: list[pd.DataFrame],
    results_dir: str | Path,
    group_cols: list[str] = ("source_file",),
    channel_name: str = "target",
):
    """Concatenates per-image per-cell DataFrames, saves a master CSV, and a
    grouped summary (mean/median/std/n) over `group_cols` (e.g.
    ['replicate', 'condition'] or ['source_file']).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    group_cols = list(group_cols)

    master = pd.concat(per_image_dfs, ignore_index=True)
    master_path = results_dir / "per_cell_results.csv"
    master.to_csv(master_path, index=False)

    intensity_col = f"mean_intensity_{channel_name}"
    summary = (
        master.groupby(group_cols)
        .agg(
            n_cells=("cell_id", "count"),
            **{
                f"mean_{intensity_col}": (intensity_col, "mean"),
                f"median_{intensity_col}": (intensity_col, "median"),
                f"std_{intensity_col}": (intensity_col, "std"),
            },
        )
        .reset_index()
    )
    summary_path = results_dir / "summary_by_group.csv"
    summary.to_csv(summary_path, index=False)

    return master, summary, master_path, summary_path
