"""ND2 loading, multi-point handling, and channel resolution/extraction.

ND2 files can carry several independent axes beyond channel/Y/X:
  - Z: a real focal stack of one field of view -> should be projected/collapsed
       into a single plane per channel (see max_project_if_stack).
  - P: multiple stage positions/points (i.e. multiple distinct fields of view,
       e.g. a multi-point acquisition within one well) -> these are DIFFERENT
       images and must NOT be projected together. Each point has to be
       treated as its own sample.
  - T: timepoints -> same deal as P if present (kept generic below).

Getting P/T confused with Z is an easy, silent mistake (both look like "just
another stack dimension" from array shape alone) and it produces meaningless
blurred composites if projected. `load_nd2` returns the array's axis labels
explicitly so callers never have to guess, and `iter_points` does the correct
per-point splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

LOOP_AXES = ("P", "T")  # axes that represent distinct samples, never projected


def load_nd2(path: str | Path):
    """Returns (array, channel_meta, axes):
      - array: the raw ND2 array, axes in whatever order ND2 reports them
      - channel_meta: list of dicts {index, name, emission_nm, excitation_nm}
      - axes: list of axis-label strings (subset of 'P','T','Z','C','Y','X'),
        one per array dimension, in array.shape order

    Does NOT reorder or project anything — use iter_points()/extract_channels()
    for that, so P/T (separate fields of view) never get silently collapsed
    together with a real Z focal stack.
    """
    import nd2

    path = Path(path)
    with nd2.ND2File(path) as f:
        arr = f.asarray()
        sizes = f.sizes  # e.g. {'P': 30, 'C': 3, 'Y': 1152, 'X': 1152}
        axes = list(sizes.keys())

        channel_meta = []
        try:
            channels = f.metadata.channels
            for i, ch in enumerate(channels):
                emission = getattr(ch.channel, "emissionLambdaNm", None)
                excitation = getattr(ch.channel, "excitationLambdaNm", None)
                name = getattr(ch.channel, "name", f"channel_{i}")
                channel_meta.append(
                    {
                        "index": i,
                        "name": name,
                        "emission_nm": emission,
                        "excitation_nm": excitation,
                    }
                )
        except Exception:
            n_channels = sizes.get("C", 1)
            channel_meta = [
                {"index": i, "name": f"channel_{i}", "emission_nm": None, "excitation_nm": None}
                for i in range(n_channels)
            ]

    return arr, channel_meta, axes


def get_point_names(path: str | Path) -> Optional[list[str]]:
    """Returns the ND2 stage position names (e.g. 'B02_0000') for a
    multi-point acquisition's P axis, in P-index order, or None if the file
    has no XYPosLoop / isn't a multi-point acquisition.
    """
    import nd2

    path = Path(path)
    with nd2.ND2File(path) as f:
        for loop in f.experiment or []:
            points = getattr(getattr(loop, "parameters", None), "points", None)
            if points:
                return [p.name for p in points]
    return None


def _move_axis_to_front(arr: np.ndarray, axes: list[str], label: str) -> tuple[np.ndarray, list[str]]:
    if label not in axes or axes[0] == label:
        return arr, axes
    i = axes.index(label)
    arr = np.moveaxis(arr, i, 0)
    axes = [label] + [a for a in axes if a != label]
    return arr, axes


def iter_points(arr: np.ndarray, axes: list[str], point_names: Optional[list[str]] = None):
    """Splits a P (and/or T) axis into separate per-point samples.

    Yields (point_index, point_label, sub_arr, sub_axes) where sub_arr has
    axes reordered to lead with 'C' (i.e. (C, [Z], Y, X)) — ready for
    extract_channels(). point_label is the ND2 position name if available
    (e.g. 'B02_0000'), else a zero-padded index string.

    If neither P nor T is present (single field of view, no loop), yields
    exactly one item: (0, '0', arr_with_C_first, axes_with_C_first).
    """
    loop_axes = [a for a in LOOP_AXES if a in axes and arr.shape[axes.index(a)] > 1]

    if not loop_axes:
        sub_arr, sub_axes = _move_axis_to_front(arr, axes, "C")
        yield 0, "0", sub_arr, sub_axes
        return

    if len(loop_axes) > 1:
        raise NotImplementedError(
            f"ND2 file has more than one multi-value loop axis {loop_axes} "
            "(e.g. both multi-point AND multi-timepoint) — not supported yet."
        )

    loop_axis = loop_axes[0]
    loop_dim = axes.index(loop_axis)
    n = arr.shape[loop_dim]

    if point_names is not None and len(point_names) != n:
        point_names = None  # metadata didn't line up; fall back to indices

    remaining_axes = [a for a in axes if a != loop_axis]
    for i in range(n):
        sub_arr = np.take(arr, i, axis=loop_dim)
        sub_arr, sub_axes = _move_axis_to_front(sub_arr, remaining_axes, "C")
        label = point_names[i] if point_names else str(i).zfill(len(str(n - 1)))
        yield i, label, sub_arr, sub_axes


def load_nd2_point(path: str | Path, point_index: int):
    """Convenience wrapper: loads the ND2 file and returns just one point's
    (sub_arr, channel_meta) as (C, [Z], Y, X) — for re-opening a specific
    point later (e.g. in a downstream measurement notebook) without having
    to re-derive axis handling by hand.
    """
    arr, channel_meta, axes = load_nd2(path)
    point_names = get_point_names(path)
    for i, _label, sub_arr, _sub_axes in iter_points(arr, axes, point_names):
        if i == point_index:
            return sub_arr, channel_meta
    raise IndexError(f"point_index {point_index} out of range for {path}")


def resolve_channel_index(
    channel_meta: list[dict],
    channel_cfg: dict,
    tolerance_nm: float = 20,
) -> int:
    """Pick a channel index from ND2 metadata, in priority order:
    1. explicit `index`
    2. exact (case-insensitive) match of `name` against the ND2 channel name
       (e.g. "UV - 405", "Green - 488", "Red - 561")
    3. nearest-wavelength match to `wavelength_nm`
    """
    if channel_cfg.get("index") is not None:
        return int(channel_cfg["index"])

    target_name = channel_cfg.get("name")
    if target_name:
        for ch in channel_meta:
            if str(ch.get("name", "")).strip().lower() == str(target_name).strip().lower():
                return ch["index"]
        if channel_cfg.get("wavelength_nm") is None:
            available = [c.get("name") for c in channel_meta]
            raise ValueError(
                f"No channel named '{target_name}' found in ND2 metadata. "
                f"Available channel names: {available}. "
                "Update 'name' in the panel/config to match exactly, or set an explicit 'index'."
            )

    target_wl = channel_cfg.get("wavelength_nm")
    if target_wl is None:
        raise ValueError(
            f"Channel config {channel_cfg} needs 'index', a matching 'name', or 'wavelength_nm'"
        )

    best_idx: Optional[int] = None
    best_diff = float("inf")
    for ch in channel_meta:
        wl = ch.get("excitation_nm") or ch.get("emission_nm")
        if wl is None:
            continue
        diff = abs(wl - target_wl)
        if diff < best_diff:
            best_diff = diff
            best_idx = ch["index"]

    if best_idx is None or best_diff > tolerance_nm:
        available = [(c["index"], c["name"], c.get("excitation_nm"), c.get("emission_nm")) for c in channel_meta]
        raise ValueError(
            f"Could not confidently match a channel to {target_wl} nm "
            f"(closest diff={best_diff}). Set an explicit 'index' or exact 'name'. "
            f"Available channels: {available}"
        )
    return best_idx


def max_project_if_stack(channel_img: np.ndarray, enabled: bool = True) -> np.ndarray:
    """channel_img may be (Z, Y, X) or (Y, X). Max-projects a genuine Z focal
    stack if present and enabled. Never call this on a P/T axis — split
    those out first with iter_points()."""
    if channel_img.ndim == 3:
        if enabled:
            return channel_img.max(axis=0)
        mid = channel_img.shape[0] // 2
        return channel_img[mid]
    return channel_img


def extract_channels(
    arr: np.ndarray,
    channel_meta: list[dict],
    channel_specs: dict[str, dict],
    zstack_cfg: dict | None = None,
    tolerance_nm: float = 20,
) -> dict[str, np.ndarray]:
    """Generic multi-channel extractor for a SINGLE point/field of view, i.e.
    arr must already be (C, [Z], Y, X) — run it through iter_points() first
    if the source ND2 has a P (or T) axis. channel_specs is
    {role_name: channel_cfg}, e.g. {'nuclear': {...}, 'cytosolic': {...}}
    for segmentation, or {'target': {...}} for a project-specific readout
    channel. Returns {role_name: img2d}, Z-projected per zstack_cfg.
    """
    zstack_cfg = zstack_cfg or {}
    project = zstack_cfg.get("max_project", True)

    out = {}
    for role, spec in channel_specs.items():
        idx = resolve_channel_index(channel_meta, spec, tolerance_nm)
        img = arr[idx]
        out[role] = max_project_if_stack(img, enabled=project)
    return out
