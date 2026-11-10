"""QC visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.segmentation import mark_boundaries


def _normalize(img: np.ndarray) -> np.ndarray:
    disp = img.astype(np.float64)
    lo, hi = np.percentile(disp, (1, 99.5))
    return np.clip((disp - lo) / max(hi - lo, 1e-9), 0, 1)


def make_qc_overlay(
    label_img: np.ndarray,
    background_img: np.ndarray,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Saves a PNG with cell instance boundaries drawn over a normalized
    background image (typically the channel you're about to measure, or the
    nuclear/cytosolic channel used for segmentation).
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = mark_boundaries(_normalize(background_img), label_img, color=(1, 0, 0))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title or out_path.stem)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_pipeline_qc(
    nuclear_img: np.ndarray,
    nucleus_labels: np.ndarray,
    membrane_img: np.ndarray,
    membrane_mask: np.ndarray,
    instance_labels: np.ndarray,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Saves ONE three-row QC panel (input image | what was identified) that
    covers every step of the segmentation pipeline for a single image:

      1. nuclear channel input -> detected nuclei (Python nuclear segmentation)
      2. membrane channel input -> thresholded membrane probability mask
      3. membrane channel input -> final per-cell instance boundaries

    Meant to be generated for only the FIRST image/point in a batch (see
    `qc_first_only` in cell_tracker.pipeline.run_segmentation_pipeline) —
    enough to sanity-check every step without regenerating QC for every
    single image, which would be slow for large datasets.
    """
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_nuclei = len(np.unique(nucleus_labels)) - 1
    n_cells = len(np.unique(instance_labels)) - 1

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    axes[0, 0].imshow(_normalize(nuclear_img), cmap="gray")
    axes[0, 0].set_title("Nuclear channel (input)")
    axes[0, 1].imshow(label2rgb(nucleus_labels, bg_label=0))
    axes[0, 1].set_title(f"Detected nuclei — Python segmentation (n={n_nuclei})")

    axes[1, 0].imshow(_normalize(membrane_img), cmap="gray")
    axes[1, 0].set_title("Membrane channel (input)")
    axes[1, 1].imshow(membrane_mask, cmap="gray")
    axes[1, 1].set_title("Membrane probability > threshold (ilastik)")

    axes[2, 0].imshow(_normalize(membrane_img), cmap="gray")
    axes[2, 0].set_title("Membrane channel (input)")
    overlay = mark_boundaries(_normalize(membrane_img), instance_labels, color=(1, 0, 0))
    axes[2, 1].imshow(overlay)
    axes[2, 1].set_title(f"Final cell instances — nucleus-seeded watershed (n={n_cells})")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(title or out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_membrane_only_qc(
    membrane_img: np.ndarray,
    membrane_mask: np.ndarray,
    instance_labels: np.ndarray,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Saves a TWO-row QC panel (input image | what was identified) for
    projects with NO nuclear channel — segmentation goes straight from the
    membrane mask to per-cell instances via distance-transform watershed
    (instance_segmentation.cells_from_membrane_mask), so there's no separate
    nuclear-detection row the way make_pipeline_qc has:

      1. membrane channel input -> thresholded membrane probability mask
      2. membrane channel input -> final per-cell instance boundaries

    Meant to be generated for only the FIRST image/point in a batch (see
    `qc_first_only` in cell_tracker.pipeline.run_segmentation_pipeline).
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = len(np.unique(instance_labels)) - 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(_normalize(membrane_img), cmap="gray")
    axes[0, 0].set_title("Membrane channel (input)")
    axes[0, 1].imshow(membrane_mask, cmap="gray")
    axes[0, 1].set_title("Membrane probability > threshold (ilastik)")

    axes[1, 0].imshow(_normalize(membrane_img), cmap="gray")
    axes[1, 0].set_title("Membrane channel (input)")
    overlay = mark_boundaries(_normalize(membrane_img), instance_labels, color=(1, 0, 0))
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Final cell instances — distance-transform watershed (n={n_cells})")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(title or out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_class_probability_overlay(
    prob: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Saves a side-by-side panel of each class's probability map — useful
    while training/tuning a panel to check the pixel classifier is sane
    before trusting the downstream watershed instance split.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = prob.shape[-1]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(prob[..., i], cmap="magma", vmin=0, vmax=1)
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title or out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
