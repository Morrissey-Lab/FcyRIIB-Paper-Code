"""Turn ilastik's single-class membrane probability map into per-cell
instance labels, in one of two ways depending on whether the project has a
nuclear stain:

- cells_from_nucleus_and_membrane: nuclei are found directly in Python
  (nuclear_segmentation.segment_nuclei) and used as watershed seeds. Needs a
  nuclear channel (e.g. Erika TNFa's UV - 405).
- cells_from_membrane_mask: no nuclear channel available at all -- seeds are
  found directly from the membrane mask's shape instead, via the classic
  distance-transform + local-maxima watershed split (same idea skimage uses
  to separate touching coins/cells with no separate marker channel). Used
  for projects imaging only a cell-body marker (e.g. Erika small beads'
  Green - 488 macrophage marker, no nuclear stain).

Either way, ilastik itself is only ever trained on ONE cytoplasm/membrane
marker channel (see panels.py) and never sees a nuclear channel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def smooth_probabilities(prob: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooths a (Y, X, n_classes) probability stack over its
    spatial axes only (never across classes). sigma=0 is a no-op passthrough.
    """
    if sigma <= 0:
        return prob
    return gaussian_filter(prob, sigma=(sigma, sigma, 0))


def load_probabilities(path: str | Path, class_axis: int = -1) -> np.ndarray:
    """Loads an ilastik 'Probabilities' export. Supports both HDF5 (ilastik's
    default 'exported_data' dataset) and TIFF. Returns an array with the
    class/channel dimension moved to `class_axis`'s target position (last,
    by default) — i.e. shape (Y, X, n_classes). For the single-channel
    membrane panel this package trains, n_classes is typically 2
    (membrane, background).
    """
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            key = "exported_data" if "exported_data" in f else list(f.keys())[0]
            arr = f[key][()]
    else:
        import tifffile

        arr = tifffile.imread(path)

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D (Y, X, n_classes) probability array from {path}, got shape {arr.shape}"
        )

    # Move whichever axis is smallest (the class axis) to the end, unless
    # it's already there. ilastik 2D Probabilities exports are typically
    # (y, x, c) already, but this guards against axis-order surprises.
    if class_axis != -1 and class_axis != arr.ndim - 1:
        arr = np.moveaxis(arr, class_axis, -1)
    elif arr.shape[-1] > min(arr.shape):
        # last axis isn't the smallest -> probably not the class axis; move
        # the actual smallest axis to the end instead.
        c_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, c_axis, -1)

    return arr


def cells_from_nucleus_and_membrane(
    membrane_prob: np.ndarray,
    nucleus_labels: np.ndarray,
    membrane_class_idx: int = 0,
    probability_threshold: float = 0.5,
    min_cell_size_px: int = 20,
    smoothing_sigma: float = 0.0,
    fill_holes: bool = True,
) -> np.ndarray:
    """membrane_prob: (Y, X, n_classes) from load_probabilities() — ilastik's
    single-channel membrane panel output. nucleus_labels: (Y, X) int label
    image from nuclear_segmentation.segment_nuclei() — one unique positive
    integer per detected nucleus, used directly as watershed seeds.

    Manual tuning knobs, all read from a panel's panel.yaml `segmentation:`
    block by the caller (cell_tracker.pipeline.run_segmentation_pipeline):

      - smoothing_sigma: Gaussian-smooths the raw membrane probability map
        (spatial axes only) before thresholding/watershed. 0 = off. Raise it
        if the membrane probability QC panel looks speckly/noisy; too high
        washes out real boundaries between touching cells.
      - probability_threshold: absolute cutoff on the (smoothed) membrane
        probability used to build the foreground mask that nucleus seeds are
        allowed to grow into.
      - min_cell_size_px: drops final cell regions smaller than this many
        pixels (debris / spurious splits with no real nucleus backing them).
      - fill_holes: after thresholding, fills any background-classified hole
        that is fully enclosed by foreground (scipy's binary_fill_holes) —
        fixes ilastik misclassifying interior/cytoplasmic pixels as
        background without touching probability values anywhere else, so it
        can't merge separate cells or move a true outer boundary. Default
        True. A nucleus always counts as foreground already (see below), so
        this mainly matters when a hole doesn't overlap the nucleus.

    The watershed elevation surface is the continuous (smoothed) membrane
    probability itself — nuclei always count as valid foreground even right
    at a marginal boundary, so a real cell is never dropped just because its
    membrane signal was weak near the nucleus.

    Returns an int32 label image, one unique positive integer per cell,
    0 = background.
    """
    if membrane_prob.ndim != 3:
        raise ValueError(
            f"Expected (Y, X, n_classes) probability array, got shape {membrane_prob.shape}"
        )

    membrane_prob = smooth_probabilities(membrane_prob, smoothing_sigma)
    cell_prob = membrane_prob[..., membrane_class_idx]

    foreground_mask = cell_prob > probability_threshold
    foreground_mask = foreground_mask | (nucleus_labels > 0)
    if fill_holes:
        foreground_mask = ndi.binary_fill_holes(foreground_mask)

    instance_labels = watershed(-cell_prob, markers=nucleus_labels, mask=foreground_mask)
    instance_labels = remove_small_objects(instance_labels, min_size=min_cell_size_px)
    return instance_labels.astype(np.int32)


def cells_from_membrane_mask(
    membrane_prob: np.ndarray,
    membrane_class_idx: int = 0,
    probability_threshold: float = 0.5,
    min_cell_size_px: int = 20,
    smoothing_sigma: float = 0.0,
    min_seed_distance_px: int = 10,
    fill_holes: bool = True,
    contact_class_idx: int | None = None,
    contact_probability_threshold: float | None = None,
) -> np.ndarray:
    """Splits touching cells directly from the membrane mask's shape, with
    NO separate seed channel (no nuclear stain available). Use this instead
    of cells_from_nucleus_and_membrane when a project has no nuclear
    channel — e.g. Erika small beads, which only images a cell-body marker
    (Green - 488) and a target readout, no nucleus.

    membrane_prob: (Y, X, n_classes) from load_probabilities().

    How it works (the standard "distance transform watershed" split, same
    idea as skimage's touching-coins example): threshold the (smoothed)
    membrane probability into a binary mask, compute each foreground
    pixel's distance to the nearest background pixel, then treat local
    maxima of that distance map (i.e. the "widest" point of each blob,
    typically near a cell's center) as seeds for a watershed that splits
    the mask back apart wherever two blobs' distance peaks are far enough
    apart to be separate cells.

    Seven manual tuning knobs, all read from a panel's panel.yaml
    `segmentation:` block:

      - smoothing_sigma: Gaussian-smooths the raw membrane probability map
        before thresholding. 0 = off.
      - probability_threshold: absolute cutoff on the (smoothed) membrane
        probability used to build the binary foreground mask.
      - min_cell_size_px: drops final cell regions smaller than this many
        pixels.
      - min_seed_distance_px: minimum pixel distance required between two
        distance-map peaks for them to count as separate cells. Lower it if
        tightly-packed/touching cells are being merged into one; raise it
        if single cells are being split into multiple pieces. This is a
        purely geometric fallback — if contact_class_idx is given and doing
        its job, this matters a lot less, since real gaps at contact points
        already give the distance transform separate peaks on their own.
      - fill_holes: after thresholding, fills any background-classified hole
        that is fully enclosed by foreground (scipy's binary_fill_holes) —
        fixes ilastik misclassifying interior pixels as background. This
        matters MORE here than in the nucleus-seeded pathway: an interior
        hole depresses the distance transform right where a cell's true
        center peak should be, which can split one real cell into several
        spurious local maxima (i.e. this is a common cause of a frame
        segmenting into far more "cells" than actually exist). MUST be
        False if contact_class_idx is given — otherwise it fills the
        contact gaps back in (they're fully enclosed by foreground on both
        sides), silently undoing the split those gaps exist to create.
      - contact_class_idx: index of an optional third ilastik pixel class
        trained to recognize the boundary where two touching cells press
        against each other (e.g. a "cell_contact" label, in addition to
        membrane/background). When given, pixels classified as contact are
        carved OUT of the foreground mask before the distance transform —
        i.e. ilastik's learned touching-cell boundary becomes an actual gap
        in the mask, splitting the two cells into separate connected
        regions with their own genuine distance-transform peaks, rather
        than relying on min_seed_distance_px to guess where one cell ends
        and the next begins from geometry alone. None (default) = off, use
        the geometric heuristic only.
      - contact_probability_threshold: cutoff on the (smoothed) contact
        probability used to decide which pixels count as "contact" and get
        carved out. Defaults to probability_threshold if not given
        separately — pass a different value if the contact class needs a
        different confidence bar than the membrane class does.

    Returns an int32 label image, one unique positive integer per cell,
    0 = background.
    """
    if membrane_prob.ndim != 3:
        raise ValueError(
            f"Expected (Y, X, n_classes) probability array, got shape {membrane_prob.shape}"
        )

    membrane_prob = smooth_probabilities(membrane_prob, smoothing_sigma)
    cell_prob = membrane_prob[..., membrane_class_idx]
    mask = cell_prob > probability_threshold

    if contact_class_idx is not None:
        contact_threshold = (
            contact_probability_threshold
            if contact_probability_threshold is not None
            else probability_threshold
        )
        contact_prob = membrane_prob[..., contact_class_idx]
        mask = mask & ~(contact_prob > contact_threshold)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_seed_distance_px, labels=mask)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    seeds = label(seed_mask)

    instance_labels = watershed(-distance, markers=seeds, mask=mask)
    instance_labels = remove_small_objects(instance_labels, min_size=min_cell_size_px)
    return instance_labels.astype(np.int32)
