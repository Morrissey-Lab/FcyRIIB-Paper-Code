"""Pure-Python nuclear segmentation — deliberately NOT ilastik. Operates
directly on the raw nuclear-marker channel (e.g. UV - 405) using classical
image processing: Gaussian smoothing, a minimum-signal check, automatic Otsu
thresholding, small-object removal, then connected-component labeling. The
resulting label image is used as watershed seeds in
instance_segmentation.cells_from_nucleus_and_membrane — ilastik itself never
sees the nuclear channel at all; it's only ever trained on the membrane/
cytoplasm marker (see panels.py).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects


def segment_nuclei(
    nuclear_img: np.ndarray,
    smoothing_sigma: float = 0.0,
    min_size_px: int = 20,
    min_signal_range: float = 0.0,
) -> np.ndarray:
    """nuclear_img: (Y, X) raw nuclear-channel image (already Z-projected).

    Three tuning knobs, all shared with the membrane/watershed side of the
    pipeline (see a panel's panel.yaml `segmentation:` block) so there's one
    consistent vocabulary across the whole segmentation:

      - smoothing_sigma: Gaussian blur (pixels, default 0 = off) applied
        before Otsu thresholding. Raise it (try 0.5, 1, 2) if nuclei look
        speckly/fragmented in the QC overlay; too high merges nearby nuclei
        together into one blob.
      - min_size_px: removes thresholded blobs smaller than this many pixels
        (debris, noise) before connected-component labeling. Lower it if
        real (small) nuclei are being filtered out along with noise.
      - min_signal_range: minimum (99.5th percentile - median) intensity
        gap, in raw camera-count units, required before Otsu thresholding is
        even attempted. Default 0 = off (always threshold). Otsu always
        forces a split into "high"/"low" halves even on an image that's pure
        noise with no real nuclei -- smoothing and min_size_px can suppress
        a lot of that, but never fully, because Otsu has no concept of "no
        signal here at all". Raising min_signal_range above 0 fixes that: if
        the (smoothed) image's brightest pixels don't stand out from its
        background/median by at least this much, the image is treated as
        having no real signal and segmentation returns zero nuclei instead
        of manufacturing fake ones from noise. Tune it by comparing this gap
        on a known-blank image against a known-real one (see the panel
        README for a short script to check).

    Returns an int32 label image, same shape as nuclear_img, 0 = background,
    one unique positive integer per detected nucleus.
    """
    img = nuclear_img.astype(np.float64)
    if smoothing_sigma > 0:
        img = gaussian_filter(img, sigma=smoothing_sigma)

    if img.max() <= img.min():
        # Flat/empty image -- Otsu would raise or return a meaningless split.
        return np.zeros(img.shape, dtype=np.int32)

    if min_signal_range > 0:
        signal_range = np.percentile(img, 99.5) - np.percentile(img, 50)
        if signal_range < min_signal_range:
            # Nothing stands out from background by enough to trust -- skip
            # Otsu entirely rather than let it force a split on pure noise.
            return np.zeros(img.shape, dtype=np.int32)

    mask = img > threshold_otsu(img)
    mask = remove_small_objects(mask, min_size=min_size_px)
    return label(mask).astype(np.int32)
