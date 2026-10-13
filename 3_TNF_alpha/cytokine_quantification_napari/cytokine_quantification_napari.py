# ============================================================================
#  INTRACELLULAR CYTOKINE QUANTIFICATION — Ilastik pipeline in napari
#  (fast-scrub version)
#
#  Paste this WHOLE block into the napari console (">_" button, bottom-left),
#  or run it as:  python cytokine_quantification_napari.py
#
#  PERFORMANCE REWORK (this version):
#    The original script re-read all 3 channel TIFFs from disk (and searched
#    for / generated the Ilastik probability map) EVERY time you moved the
#    FOV scrubber, synchronously on the GUI thread -> napari froze while
#    dragging.
#
#    This version borrows the pattern from the frame-scrubbing (ERK-KTR)
#    pipeline: images are loaded into memory ONCE, up front, when a
#    condition is added (like that pipeline's single `imread(path)` load of
#    the whole stack). Probability maps are also generated in one Ilastik
#    batch call per condition instead of one-at-a-time. Scrubbing between
#    FOVs afterwards is a pure in-memory lookup + segmentation, same as how
#    the KTR pipeline's frame slider just indexes an already-loaded array.
#
#  WORKFLOW:
#    1. Setup — point to Ilastik executable + .ilp project
#    2. Add conditions — select TIFFs (auto-grouped into FOVs); this
#       PRELOADS every FOV's images + probability map into memory
#    3. Scrub through FOVs with the combined Position module (slider +
#       spinbox + count + remove button) — no disk I/O per scrub anymore
#    4. Run batch → per-cell CSV, summary CSV, scatter + distribution plots
#
#  DEPENDENCIES:
#    pip install napari[all] magicgui tifffile scikit-image scipy pandas
#              matplotlib h5py
# ============================================================================

import os
import re
import subprocess
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, clear_border

import napari
from magicgui import magicgui
from magicgui.widgets import Label
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider,
                             QSpinBox, QLabel, QPushButton, QProgressBar,
                             QComboBox)
from qtpy.QtCore import Qt
import matplotlib.pyplot as plt

# ---- Channel mapping (edit if your channel order differs) ------------------
CH_DAPI = "c001"
CH_CYTOKINE = "c002"
CH_MEMBRANE = "c003"
CH_LABELS = {CH_DAPI: "405 DAPI", CH_CYTOKINE: "488 cytokine", CH_MEMBRANE: "561 membrane"}


# ============================================================================
#  Viewer + global state
# ============================================================================
_created = napari.current_viewer() is None
viewer = napari.current_viewer() or napari.Viewer(
    title="Cytokine Quantification (Ilastik pipeline)"
)

_PROB_FALLBACK_DIR = Path(tempfile.mkdtemp(prefix="cytokine_prob_"))

S = {
    # discovered FOVs: {fov_prefix: {ch_key: Path, ...}}
    "fovs": {},
    "fov_names": [],
    # conditions: {cond_name: [fov_prefix, ...]}
    "conditions": {},
    "cond_order": [],
    # IN-MEMORY CACHE -- this is what makes scrubbing fast.
    # {fov_prefix: {"img_405": arr, "img_488": arr, "img_561": arr,
    #               "prob_map": arr or None}}
    "fov_data": {},
    # current preview
    "current_fov": None,
    "nuc_labels": None,
    "cell_labels": None,
}


# ============================================================================
#  TIFF auto-grouping
# ============================================================================
_CH_PATTERN = re.compile(r"^(.+)_(c\d+)\.tiff?$", re.IGNORECASE)


def _group_tiffs(file_list):
    """Group a list of TIFF paths by FOV prefix (everything before _cNNN).

    Returns {prefix: {ch_key: Path}} for FOVs that have all three channels.
    Also returns a list of skipped prefixes (incomplete channel sets).
    """
    raw = {}
    for p in file_list:
        p = Path(p)
        m = _CH_PATTERN.match(p.name)
        if m:
            prefix, ch = m.group(1), m.group(2).lower()
            raw.setdefault(prefix, {})[ch] = p

    complete, skipped = {}, []
    for prefix, chs in raw.items():
        if CH_DAPI in chs and CH_CYTOKINE in chs and CH_MEMBRANE in chs:
            complete[prefix] = chs
        else:
            skipped.append((prefix, list(chs.keys())))

    if skipped:
        print(f"[group] {len(skipped)} incomplete FOV(s) skipped (missing channels).")
    return complete, skipped


# ============================================================================
#  Segmentation functions  (matches cell_tracker exactly)
# ============================================================================
import inspect as _inspect
_RSO_KW = ("max_size" if "max_size" in _inspect.signature(remove_small_objects).parameters
            else "min_size")


def _remove_small(label_img, min_px):
    return remove_small_objects(label_img, **{_RSO_KW: min_px})


def segment_nuclei(nuclear_img, smoothing_sigma=0.5, min_size_px=20,
                    min_signal_range=300.0, threshold_multiplier=1.0):
    img = nuclear_img.astype(np.float64)
    if smoothing_sigma > 0:
        img = gaussian_filter(img, sigma=smoothing_sigma)
    if img.max() <= img.min():
        return np.zeros(img.shape, dtype=np.int32)
    if min_signal_range > 0:
        gap = np.percentile(img, 99.5) - np.percentile(img, 50)
        if gap < min_signal_range:
            return np.zeros(img.shape, dtype=np.int32)
    thresh = threshold_otsu(img) * threshold_multiplier
    mask = img > thresh
    mask = _remove_small(mask, min_size_px)
    return label(mask).astype(np.int32)


def smooth_probabilities(prob, sigma):
    if sigma <= 0:
        return prob
    return gaussian_filter(prob, sigma=(sigma, sigma, 0))


def load_probabilities(path):
    """Load Ilastik Probabilities export (.h5 or .tif) → (Y,X,n_classes)."""
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = "exported_data" if "exported_data" in f else list(f.keys())[0]
            arr = np.asarray(f[key][()])
    else:
        arr = tifffile.imread(str(path))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D probability array, got shape {arr.shape}")
    if arr.shape[-1] > min(arr.shape):
        arr = np.moveaxis(arr, int(np.argmin(arr.shape)), -1)
    return arr


def _remove_large(label_img, max_px):
    """Drop any labeled object bigger than max_px (0/None = no-op).

    Mirror image of remove_small_objects/_remove_small -- this is what
    knocks out a plate-edge artifact that watershed picks up as one huge
    "cell": it's a normal object by shape, just far bigger in area than any
    real cell in the frame, so a per-object area ceiling catches it cleanly
    without touching the real cells around it.
    """
    if not max_px:
        return label_img
    props = regionprops_table(label_img, properties=("label", "area"))
    too_big = props["label"][props["area"] > max_px]
    if len(too_big) == 0:
        return label_img
    out = label_img.copy()
    out[np.isin(out, too_big)] = 0
    return out


def cells_from_nucleus_and_membrane(membrane_prob, nucleus_labels,
                                     membrane_class_idx=0,
                                     probability_threshold=0.5,
                                     min_cell_size_px=20,
                                     max_cell_size_px=0,
                                     smoothing_sigma=0.5):
    prob = smooth_probabilities(membrane_prob, smoothing_sigma)
    cell_prob = prob[..., membrane_class_idx]
    fg = (cell_prob > probability_threshold) | (nucleus_labels > 0)
    inst = watershed(-cell_prob, markers=nucleus_labels, mask=fg)
    inst = _remove_small(inst, min_cell_size_px)
    inst = _remove_large(inst, max_cell_size_px)
    return inst.astype(np.int32)


# ============================================================================
#  Ilastik headless integration
# ============================================================================
def run_ilastik_batch(ilastik_exe, ilp_path, input_files, output_dir):
    """Run Ilastik headless on a LIST of membrane TIFFs at once -> .h5s.

    Called once per condition-add on every FOV that still needs a prob map,
    instead of once per FOV per scrub tick.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ilastik_exe), "--headless",
        f"--project={ilp_path}",
        "--readonly",  # skip Ilastik rewriting the .ilp file after each run
        "--export_source=Probabilities",
        "--output_format=hdf5",
        f"--output_filename_format={output_dir / '{nickname}.h5'}",
        *[str(f) for f in input_files],
    ]

    # Ilastik's headless mode is conservative by default -- it will NOT
    # use all your cores/RAM unless told to via these env vars. If Task
    # Manager shows CPU/RAM headroom while this runs, this is why.
    env = os.environ.copy()
    n_threads = max(1, (os.cpu_count() or 4) - 1)  # leave one core free
    env.setdefault("LAZYFLOW_THREADS", str(n_threads))
    try:
        import psutil
        avail_mb = int(psutil.virtual_memory().available / 1e6 * 0.7)
    except Exception:
        avail_mb = 8000  # conservative fallback if psutil isn't installed
    env.setdefault("LAZYFLOW_TOTAL_RAM_MB", str(avail_mb))

    print(f"[ilastik] Running headless batch on {len(input_files)} file(s) "
          f"(threads={env['LAZYFLOW_THREADS']}, ram_mb={env['LAZYFLOW_TOTAL_RAM_MB']}) ...")
    print("[ilastik] (streaming Ilastik's own output below -- a gap of a "
          "minute or more between lines on a big/first-time image is "
          "normal, not stuck)")
    t0 = time.perf_counter()

    # Stream stdout live instead of subprocess.run's capture_output, which
    # blocks silently and only shows anything once the WHOLE batch is done
    # -- that silence is what reads as "stuck" even when Ilastik is working.
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"[ilastik]   {line}")
            lines.append(line)
    proc.wait()
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Ilastik headless failed (exit {proc.returncode}) after "
            f"{elapsed:.1f}s.\noutput tail:\n" + "\n".join(lines[-40:])
        )
    print(f"[ilastik] Batch done in {elapsed:.1f}s "
          f"({elapsed / max(len(input_files), 1):.1f}s/file avg).")
    return proc


def _prob_cache_dir(membrane_path):
    d = Path(membrane_path).parent / "probability_maps"
    try:
        d.mkdir(exist_ok=True)
    except OSError:
        return _PROB_FALLBACK_DIR
    return d


def _find_prob_map(membrane_path):
    """Search for an existing Ilastik probability map (.h5) on disk.

    Checks the two expected locations first (direct exists() checks, cheap
    even on a slow/synced drive). Only falls back to scanning the parent
    directory for a "prob*"-named sibling folder -- an iterdir() call that
    can be very slow on a OneDrive/synced folder with many files -- if
    those direct checks come up empty.
    """
    membrane_path = Path(membrane_path)
    membrane_stem = membrane_path.stem
    tiff_parent = membrane_path.parent

    fast_dirs = [tiff_parent / "probability_maps", _PROB_FALLBACK_DIR, tiff_parent]
    for search_dir in fast_dirs:
        if not search_dir.is_dir():
            continue
        for ext in (".h5", ".hdf5"):
            p = search_dir / f"{membrane_stem}{ext}"
            if p.exists():
                return p

    # Slow path: only reached if the fast checks above found nothing.
    try:
        if tiff_parent.is_dir():
            for d in tiff_parent.iterdir():
                if d.is_dir() and "prob" in d.name.lower() and d not in fast_dirs:
                    for ext in (".h5", ".hdf5"):
                        p = d / f"{membrane_stem}{ext}"
                        if p.exists():
                            return p
    except OSError:
        pass
    return None


def _ilastik_exe():
    v = setup_widget.ilastik_executable.value
    return str(v) if v and Path(v).is_file() else None


def _ilp_path():
    v = setup_widget.ilastik_project.value
    return str(v) if v and Path(v).is_file() else None


# ============================================================================
#  PRELOAD -- the core fix. Loads images + prob maps for a batch of FOVs
#  into S["fov_data"] ONCE, so scrubbing afterwards never touches disk.
#  Mirrors the KTR pipeline's `data = imread(path)` single up-front load.
# ============================================================================
def _to_2d(img):
    if img.ndim > 2:
        img = np.squeeze(img)
    if img.ndim > 2:
        img = img.max(axis=0)
    return img


def _preload_fovs(prefixes, progress_cb=None):
    """Load the 3 channel TIFFs for each new FOV into memory (once)."""
    total = len(prefixes)
    t_start = time.perf_counter()
    n_loaded = 0
    for i, prefix in enumerate(prefixes):
        if prefix in S["fov_data"]:
            if progress_cb:
                progress_cb(i + 1, total, prefix, "cached")
            continue
        chs = S["fovs"][prefix]
        t0 = time.perf_counter()
        entry = {
            "img_405": _to_2d(tifffile.imread(str(chs[CH_DAPI])).astype(np.float32)),
            "img_488": _to_2d(tifffile.imread(str(chs[CH_CYTOKINE])).astype(np.float32)),
            "img_561": _to_2d(tifffile.imread(str(chs[CH_MEMBRANE])).astype(np.float32)),
            "prob_map": None,
        }
        S["fov_data"][prefix] = entry
        n_loaded += 1
        print(f"[tiff] {prefix}: {time.perf_counter() - t0:.2f}s "
              f"(shape={entry['img_405'].shape})")
        if progress_cb:
            progress_cb(i + 1, total, prefix, "loaded")
    if n_loaded:
        elapsed = time.perf_counter() - t_start
        print(f"[tiff] Loaded {n_loaded} FOV(s) from disk in {elapsed:.1f}s "
              f"({elapsed / n_loaded:.2f}s/FOV avg).")


def _preload_prob_maps(prefixes, progress_cb=None):
    """Ensure every FOV in `prefixes` has a probability map, using ONE
    batch Ilastik call for whatever isn't already cached on disk, then
    loads every map into memory.
    """
    exe, ilp = _ilastik_exe(), _ilp_path()

    to_generate = []          # membrane TIFF paths needing a fresh run
    to_generate_fov = []      # matching fov prefix, same order
    for prefix in prefixes:
        entry = S["fov_data"].get(prefix)
        if entry is None:
            continue
        if entry["prob_map"] is not None:
            continue
        membrane_path = S["fovs"][prefix][CH_MEMBRANE]
        cached = _find_prob_map(membrane_path)
        if cached is None:
            if exe and ilp:
                to_generate.append(membrane_path)
                to_generate_fov.append(prefix)
            # else: no exe/ilp configured -- leave prob_map as None, cell
            # instances just won't be available until Setup is fixed.

    if to_generate:
        out_dir = _prob_cache_dir(to_generate[0])
        try:
            run_ilastik_batch(exe, ilp, to_generate, out_dir)
        except Exception as e:
            print(f"[prob] Batch Ilastik run failed: {e}")

    total = len(prefixes)
    t_start = time.perf_counter()
    n_loaded = 0
    for i, prefix in enumerate(prefixes):
        entry = S["fov_data"].get(prefix)
        if entry is None:
            continue
        if entry["prob_map"] is None:
            membrane_path = S["fovs"][prefix][CH_MEMBRANE]
            t_find = time.perf_counter()
            cached = _find_prob_map(membrane_path)
            find_elapsed = time.perf_counter() - t_find
            if find_elapsed > 0.05:
                print(f"[prob] {prefix}: _find_prob_map took {find_elapsed:.2f}s")
            if cached is not None:
                t0 = time.perf_counter()
                try:
                    entry["prob_map"] = load_probabilities(cached)
                    n_loaded += 1
                    print(f"[prob] {prefix}: loaded {cached.name} in "
                          f"{time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[prob] {prefix}: failed to load {cached}: {e}")
        if progress_cb:
            status = "ok" if entry["prob_map"] is not None else "MISSING (check Setup paths)"
            progress_cb(i + 1, total, prefix, status)
    if n_loaded:
        elapsed = time.perf_counter() - t_start
        print(f"[prob] Loaded {n_loaded} cached probability map(s) from disk "
              f"in {elapsed:.1f}s ({elapsed / n_loaded:.2f}s/file avg).")


# ============================================================================
#  Measurement  (matches cell_tracker.measure)
# ============================================================================
def measure_intensity_per_cell(label_img, intensity_img, channel_name="488",
                                exclude_border=False, **extra_cols):
    if exclude_border:
        label_img = clear_border(label_img)
    props = regionprops_table(
        label_img, intensity_image=intensity_img,
        properties=("label", "area", "mean_intensity", "max_intensity",
                     "centroid"),
    )
    df = pd.DataFrame(props).rename(columns={
        "label": "cell_id", "area": "area_px",
        "mean_intensity": f"mean_intensity_{channel_name}",
        "max_intensity": f"max_intensity_{channel_name}",
        "centroid-0": "centroid_row", "centroid-1": "centroid_col",
    })
    df[f"integrated_intensity_{channel_name}"] = (
        df["area_px"] * df[f"mean_intensity_{channel_name}"]
    )
    for i, (k, v) in enumerate(extra_cols.items()):
        df.insert(i, k, v)
    return df


# ============================================================================
#  Display helpers
# ============================================================================
def _set_layer(name, data, kind, **kw):
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        adder = viewer.add_labels if kind == "labels" else viewer.add_image
        adder(data, name=name, **kw)


def _display_fov(fov_prefix):
    """Pull already-loaded arrays from S['fov_data'] and show them.

    No disk I/O here -- this is the whole point. Equivalent to how the KTR
    script's frame scrubber just re-indexes its in-memory `data` array.
    """
    entry = S["fov_data"].get(fov_prefix)
    if entry is None:
        # Shouldn't normally happen -- preload runs at "Add condition" time.
        print(f"[display] {fov_prefix} not preloaded yet -- loading now "
              f"(this one FOV will be slow; everything after is cached).")
        _preload_fovs([fov_prefix])
        _preload_prob_maps([fov_prefix])
        entry = S["fov_data"][fov_prefix]

    S["current_fov"] = fov_prefix
    S["nuc_labels"] = None
    S["cell_labels"] = None

    _set_layer("405 DAPI", entry["img_405"], "image",
               colormap="blue", blending="additive")
    _set_layer("488 cytokine", entry["img_488"], "image",
               colormap="green", blending="additive")
    _set_layer("561 membrane", entry["img_561"], "image",
               colormap="red", blending="additive")
    shape = entry["img_405"].shape
    _set_layer("nuclei", np.zeros(shape, np.int32), "labels")
    _set_layer("cell instances", np.zeros(shape, np.int32), "labels")


def _load_fov_and_segment(fov_name):
    """Switch to a FOV (in-memory lookup) and re-run segmentation on it."""
    if fov_name not in S["fovs"]:
        return
    _display_fov(fov_name)
    scrub_widget.update_position_label()

    prob_map = S["fov_data"][fov_name]["prob_map"]
    if prob_map is None:
        print(f"[seg] {fov_name}: no probability map available -- "
              f"nuclei will show, cell instances will not.")

    seg_widget(
        nuclear_smoothing=seg_widget.nuclear_smoothing.value,
        nuclear_threshold_mult=seg_widget.nuclear_threshold_mult.value,
        min_nucleus_size_px=seg_widget.min_nucleus_size_px.value,
        min_signal_range=seg_widget.min_signal_range.value,
        membrane_smoothing=seg_widget.membrane_smoothing.value,
        probability_threshold=seg_widget.probability_threshold.value,
        min_cell_size_px=seg_widget.min_cell_size_px.value,
        max_cell_size_px=seg_widget.max_cell_size_px.value,
        max_nuclei_per_frame=seg_widget.max_nuclei_per_frame.value,
        max_membrane_area_frac=seg_widget.max_membrane_area_frac.value,
    )


def _refresh_cond_status():
    if not S["cond_order"]:
        cond_status_label.value = "No conditions added yet."
        return
    parts = [f"{c} (n={len(S['conditions'][c])})" for c in S["cond_order"]]
    cond_status_label.value = (
        f"{len(S['cond_order'])} conditions:  " + " | ".join(parts)
    )


def _clipboard(text):
    try:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(text)
            return True
    except Exception:
        pass
    return False


# ============================================================================
#  Status label
# ============================================================================
cond_status_label = Label(value="No conditions added yet.")


# ============================================================================
#  WIDGET 1 — Setup (Ilastik paths)
# ============================================================================
@magicgui(
    call_button=False,
    ilastik_executable={
        "widget_type": "FileEdit", "mode": "r",
        "label": "Ilastik executable",
        "tooltip": ("Path to ilastik.exe (Windows) or run_ilastik.sh "
                     "(Mac/Linux)."),
    },
    ilastik_project={
        "widget_type": "FileEdit", "mode": "r",
        "filter": "Ilastik projects (*.ilp)",
        "label": "Ilastik project (.ilp)",
    },
)
def setup_widget(
    ilastik_executable=Path(r"C:\Program Files\ilastik-1.4.2\ilastik.exe"),
    ilastik_project=Path(r"C:\Users\ECHOe\OneDrive\Desktop\TNFa\cell tracker\red\membrane_segmentation.ilp"),
):
    pass


# ============================================================================
#  PRELOAD PROGRESS BAR (plain Qt, shown during "Add condition")
# ============================================================================
preload_progress = QProgressBar()
preload_progress.setValue(0)
preload_progress.setTextVisible(True)
preload_progress.setFormat("Preload idle")


def _preload_progress_cb(i, total, name, status):
    preload_progress.setMaximum(total)
    preload_progress.setValue(i)
    preload_progress.setFormat(f"Preloading {i}/{total}: {name} [{status}]")
    # Let Qt repaint the progress bar / stay responsive during the batch
    # load. This is a one-time cost per condition-add, not per scrub.
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()


# ============================================================================
#  WIDGET 2 — Add condition (directly select TIFFs) -- now preloads
# ============================================================================
@magicgui(
    call_button="Add condition",
    condition_name={"widget_type": "LineEdit", "label": "Condition name"},
    images={
        "widget_type": "FileEdit", "mode": "rm",
        "filter": "TIFFs (*.tif *.tiff)",
        "label": "Select TIFFs",
        "tooltip": ("Multi-select all channel TIFFs for this condition. "
                     "Files are auto-grouped into FOVs by the _cNNN suffix. "
                     "All FOVs are preloaded into memory now, so scrubbing "
                     "afterwards is instant."),
    },
)
def add_cond_widget(condition_name="cond_1", images=()):
    if isinstance(images, (list, tuple)):
        files = [Path(f) for f in images if f and Path(f).is_file()]
    else:
        p = Path(images)
        files = [p] if p.is_file() else []

    if not files:
        print("[condition] Select one or more TIFF files first.")
        return

    cname = (condition_name or f"cond_{len(S['cond_order'])+1}").strip()

    grouped, skipped = _group_tiffs(files)
    if not grouped:
        print("[condition] No complete FOVs found in selected files. "
              "Each FOV needs _c001, _c002, and _c003 TIFFs.")
        return

    for prefix, chs in grouped.items():
        S["fovs"][prefix] = chs

    if cname not in S["conditions"]:
        S["conditions"][cname] = []
    if cname not in S["cond_order"]:
        S["cond_order"].append(cname)

    added = []
    for prefix in sorted(grouped.keys()):
        if prefix not in S["conditions"][cname]:
            S["conditions"][cname].append(prefix)
            added.append(prefix)

    # ---- PRELOAD: images + probability maps, once, right now -------------
    new_prefixes = sorted(grouped.keys())
    print(f"[preload] Loading {len(new_prefixes)} FOV(s) into memory ...")
    _preload_fovs(new_prefixes, progress_cb=_preload_progress_cb)
    print(f"[preload] Generating/loading probability maps ...")
    _preload_prob_maps(new_prefixes, progress_cb=_preload_progress_cb)
    preload_progress.setFormat(f"Preload done ({len(new_prefixes)} FOVs)")
    print(f"[preload] Done. {cname} is ready to scrub with no further disk I/O.")

    S["fov_names"] = sorted(S["fovs"].keys())
    _refresh_cond_status()
    scrub_widget.set_condition_choices(list(S["cond_order"]))
    scrub_widget.refresh_position_range()
    _refresh_unload_choices()
    add_cond_widget.condition_name.value = f"cond_{len(S['cond_order'])+1}"

    msg = f"Added {len(added)} FOV(s) to '{cname}' (total: {len(S['conditions'][cname])})"
    print(f"[condition] {msg}")
    if skipped:
        print(f"[condition] {len(skipped)} incomplete FOV(s) skipped.")
    try:
        from napari.utils.notifications import show_info
        show_info(msg)
    except Exception:
        pass

    if S["current_fov"] is None and scrub_widget.active_fov_count() > 0:
        scrub_widget.set_index(0)


# ============================================================================
#  WIDGET 3 — Combined FOV Position module
#  (slider + spinbox "clicker" with count + remove-position button, all in
#  one dock widget, per request)
# ============================================================================
class FovScrubberWidget(QWidget):
    """Single module: drag/click through FOVs, see position + count, and
    remove the currently-shown FOV from its condition -- all in one place.
    Because everything is preloaded (see _preload_fovs/_preload_prob_maps),
    moving this slider does pure in-memory lookups, no disk I/O.
    """

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # -- condition row: pick which condition to scrub through, or
        # delete it -- same choices as the "Unload from RAM" dropdown ------
        cond_row = QHBoxLayout()
        cond_row.addWidget(QLabel("Condition:"))
        self._cond_combo = QComboBox()
        self._cond_combo.addItem("All conditions")
        self._cond_combo.currentTextChanged.connect(self._on_condition_change)
        cond_row.addWidget(self._cond_combo, stretch=1)
        outer.addLayout(cond_row)

        self._delete_cond_btn = QPushButton("Delete selected condition")
        self._delete_cond_btn.clicked.connect(self._delete_selected_condition)
        outer.addWidget(self._delete_cond_btn)

        self._active_condition = None  # None = scrub through all conditions

        # -- top row: slider + spinbox --------------------------------
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Position:"))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        top_row.addWidget(self._slider, stretch=1)

        self._spin = QSpinBox()
        self._spin.setMinimum(0)
        self._spin.setMaximum(0)
        self._spin.setFixedWidth(70)
        top_row.addWidget(self._spin)
        outer.addLayout(top_row)

        # -- status row: "3 / 12 -- fov_name" ---------------------------
        self._status = QLabel("No positions loaded")
        outer.addWidget(self._status)

        # -- remove-position row ----------------------------------------
        self._remove_btn = QPushButton("Remove current position")
        self._remove_btn.clicked.connect(self._remove_current)
        outer.addWidget(self._remove_btn)

        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)
        self._changing = False

    # -- linking slider/spinbox --------------------------------------------
    def _from_slider(self, val):
        if self._changing:
            return
        self._changing = True
        self._spin.setValue(val)
        self._changing = False
        self._navigate(val)

    def _from_spin(self, val):
        if self._changing:
            return
        self._changing = True
        self._slider.setValue(val)
        self._changing = False
        self._navigate(val)

    def _active_fov_names(self):
        """The FOV list currently being scrubbed -- either every loaded
        FOV, or just the ones in the condition picked in the dropdown."""
        if self._active_condition and self._active_condition in S["conditions"]:
            return sorted(S["conditions"][self._active_condition])
        return S["fov_names"]

    def _navigate(self, idx):
        names = self._active_fov_names()
        if not names:
            return
        idx = min(idx, len(names) - 1)
        fov = names[idx]
        if fov != S["current_fov"]:
            _load_fov_and_segment(fov)

    # -- condition dropdown: filter scrub range, or delete a condition -----
    def _on_condition_change(self, text):
        self._active_condition = None if text in ("All conditions", "") else text
        names = self._active_fov_names()
        self.set_fov_count(len(names))
        if names:
            self.set_index(0)
        else:
            S["current_fov"] = None
            self._status.setText("No positions loaded")

    def _delete_selected_condition(self):
        cname = self._cond_combo.currentText()
        if not cname or cname == "All conditions" or cname not in S["conditions"]:
            print("[delete] Select a specific condition first.")
            return
        fov_list = list(S["conditions"][cname])
        del S["conditions"][cname]
        S["cond_order"].remove(cname)
        for fov in fov_list:
            S["fovs"].pop(fov, None)
            S["fov_data"].pop(fov, None)
        S["fov_names"] = sorted(S["fovs"].keys())
        print(f"[delete] Removed condition '{cname}' ({len(fov_list)} FOV(s)).")

        _refresh_cond_status()
        _refresh_unload_choices()
        self.set_condition_choices(list(S["cond_order"]))  # resets combo to "All conditions"

        names = self._active_fov_names()
        self.set_fov_count(len(names))
        if names:
            self.set_index(0)
        else:
            S["current_fov"] = None
            self._status.setText("No positions loaded")

    # -- remove current position --------------------------------------------
    def _remove_current(self):
        fov = S["current_fov"]
        if fov is None:
            print("[remove] No position currently shown.")
            return
        owning_cond = None
        for cname, fov_list in S["conditions"].items():
            if fov in fov_list:
                owning_cond = cname
                break
        if owning_cond is None:
            print(f"[remove] '{fov}' isn't part of any condition.")
            return

        S["conditions"][owning_cond].remove(fov)
        print(f"[remove] Removed '{fov}' from '{owning_cond}'.")
        cond_emptied = not S["conditions"][owning_cond]
        if cond_emptied:
            del S["conditions"][owning_cond]
            S["cond_order"].remove(owning_cond)

        # Drop it from the global FOV set + cache too so it can't be
        # scrubbed back to (still fully re-addable via "Add condition").
        S["fovs"].pop(fov, None)
        S["fov_data"].pop(fov, None)
        S["fov_names"] = sorted(S["fovs"].keys())
        _refresh_cond_status()
        _refresh_unload_choices()
        if cond_emptied:
            self.set_condition_choices(list(S["cond_order"]))

        names = self._active_fov_names()
        self.set_fov_count(len(names))
        if names:
            self.set_index(min(self._spin.value(), len(names) - 1))
        else:
            S["current_fov"] = None
            self._status.setText("No positions loaded")

    # -- public helpers used by add_cond_widget / batch flows --------------
    def refresh_position_range(self):
        """Recompute slider/spinbox range from the current active list
        (all conditions, or just the one picked in the dropdown)."""
        self.set_fov_count(len(self._active_fov_names()))

    def active_fov_count(self):
        return len(self._active_fov_names())

    def set_fov_count(self, count):
        new_max = max(count - 1, 0)
        self._slider.setMaximum(new_max)
        self._spin.setMaximum(new_max)
        self.update_position_label()

    def set_index(self, idx):
        self._changing = True
        self._slider.setValue(idx)
        self._spin.setValue(idx)
        self._changing = False
        self._navigate(idx)

    def set_condition_choices(self, choices):
        """Repopulate the condition dropdown -- same choice list as the
        Unload from RAM dropdown, plus "All conditions" at the top."""
        current = self._cond_combo.currentText()
        self._cond_combo.blockSignals(True)
        self._cond_combo.clear()
        self._cond_combo.addItem("All conditions")
        self._cond_combo.addItems(choices)
        idx = self._cond_combo.findText(current)
        self._cond_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._cond_combo.blockSignals(False)
        selected = self._cond_combo.currentText()
        self._active_condition = None if selected == "All conditions" else selected

    def update_position_label(self):
        names = self._active_fov_names()
        if not names or S["current_fov"] is None:
            self._status.setText("No positions loaded")
            return
        try:
            idx = names.index(S["current_fov"])
            name = S["current_fov"]
            if len(name) > 40:
                name = "..." + name[-37:]
            self._status.setText(f"{idx + 1} / {len(names)}  —  {name}")
        except ValueError:
            self._status.setText(f"0 / {len(names)}")


scrub_widget = FovScrubberWidget()


# ============================================================================
#  WIDGET 4 — Unload one condition's images from RAM
#  (registration/position stays -- only the in-memory arrays are freed, so
#  you can work through replicates one at a time without all of them
#  sitting in memory. Scrubbing back to an unloaded FOV re-loads just that
#  one FOV on demand via the fallback in _display_fov.)
# ============================================================================
@magicgui(
    call_button="Unload from RAM",
    condition={"widget_type": "ComboBox", "choices": ["(none)"],
               "label": "Condition"},
)
def unload_widget(condition="(none)"):
    if not condition or condition == "(none)" or condition not in S["conditions"]:
        print("[unload] Select a condition first.")
        return
    fov_list = S["conditions"][condition]
    freed = 0
    for fov in fov_list:
        if fov in S["fov_data"]:
            del S["fov_data"][fov]
            freed += 1
    print(f"[unload] Freed {freed} FOV(s) from '{condition}' "
          f"({len(S['fov_data'])} FOV(s) still resident in RAM).")
    if S["current_fov"] in fov_list:
        print("[unload] Note: currently-displayed FOV was unloaded -- "
              "scrubbing away and back will reload it on demand.")


def _refresh_unload_choices():
    unload_widget.condition.choices = list(S["cond_order"]) or ["(none)"]


# ============================================================================
#  WIDGET 4b — Clear all conditions
# ============================================================================
@magicgui(call_button="Clear all conditions")
def clear_cond_widget():
    S["conditions"].clear()
    S["cond_order"].clear()
    S["fovs"].clear()
    S["fov_names"].clear()
    S["fov_data"].clear()
    S["current_fov"] = None
    scrub_widget.set_condition_choices([])
    scrub_widget.set_fov_count(0)
    scrub_widget.update_position_label()
    _refresh_cond_status()
    _refresh_unload_choices()
    preload_progress.setValue(0)
    preload_progress.setFormat("Preload idle")
    print("[conditions] Cleared all.")


# ============================================================================
#  WIDGET 5 — Segmentation parameters (live preview, operates on the
#  already-in-memory current FOV -- fast, matches the KTR script's
#  update_segmentation() which also only touches one in-memory slice)
# ============================================================================
@magicgui(
    auto_call=True,
    nuclear_smoothing={
        "widget_type": "FloatSlider", "min": 0.0, "max": 5.0,
        "step": 0.1, "value": 0.5, "label": "Nuclear smoothing sigma",
    },
    nuclear_threshold_mult={
        "widget_type": "FloatSlider", "min": 0.1, "max": 2.0,
        "step": 0.05, "value": 1.0,
        "label": "Nuclear threshold (×Otsu)",
    },
    min_nucleus_size_px={
        "widget_type": "SpinBox", "min": 1, "max": 10000,
        "step": 5, "value": 20, "label": "Min nucleus size (px)",
    },
    min_signal_range={
        "widget_type": "SpinBox", "min": 0, "max": 5000,
        "step": 10, "value": 0, "label": "Min signal range (nuclear)",
    },
    membrane_smoothing={
        "widget_type": "FloatSlider", "min": 0.0, "max": 5.0,
        "step": 0.1, "value": 0.5, "label": "Membrane smoothing sigma",
    },
    probability_threshold={
        "widget_type": "FloatSlider", "min": 0.0, "max": 1.0,
        "step": 0.01, "value": 0.5, "label": "Membrane prob threshold",
    },
    min_cell_size_px={
        "widget_type": "SpinBox", "min": 1, "max": 100000,
        "step": 5, "value": 20, "label": "Min cell size (px)",
    },
    max_cell_size_px={
        "widget_type": "SpinBox", "min": 0, "max": 500000,
        "step": 500, "value": 20000, "label": "Max cell size (px, 0 = off)",
    },
    max_nuclei_per_frame={
        "widget_type": "SpinBox", "min": 0, "max": 1000,
        "step": 5, "value": 150, "label": "Max nuclei / frame (0 = off)",
    },
    max_membrane_area_frac={
        "widget_type": "FloatSlider", "min": 0.0, "max": 1.0,
        "step": 0.01, "value": 0.30,
        "label": "Max membrane area frac (0 = off)",
    },
)
def seg_widget(nuclear_smoothing=0.5, nuclear_threshold_mult=1.0,
                min_nucleus_size_px=20, min_signal_range=0,
                membrane_smoothing=0.5, probability_threshold=0.5,
                min_cell_size_px=20, max_cell_size_px=20000,
                max_nuclei_per_frame=150, max_membrane_area_frac=0.30):
    if S["current_fov"] is None:
        return
    entry = S["fov_data"][S["current_fov"]]
    img_405 = entry["img_405"]
    prob_map = entry["prob_map"]

    nuc = segment_nuclei(
        img_405,
        smoothing_sigma=nuclear_smoothing,
        min_size_px=min_nucleus_size_px,
        min_signal_range=min_signal_range,
        threshold_multiplier=nuclear_threshold_mult,
    )
    S["nuc_labels"] = nuc
    viewer.layers["nuclei"].data = nuc
    n_nuc = int(len(np.unique(nuc)) - 1)

    if prob_map is not None:
        cells = cells_from_nucleus_and_membrane(
            prob_map, nuc,
            membrane_class_idx=0,
            probability_threshold=probability_threshold,
            min_cell_size_px=min_cell_size_px,
            max_cell_size_px=max_cell_size_px,
            smoothing_sigma=membrane_smoothing,
        )
        S["cell_labels"] = cells
        viewer.layers["cell instances"].data = cells
        n_cells = int(len(np.unique(cells)) - 1)
    else:
        S["cell_labels"] = None
        n_cells = "— (no prob map)"

    flags = []
    if max_nuclei_per_frame > 0 and isinstance(n_nuc, int) and n_nuc > max_nuclei_per_frame:
        flags.append(f"nuclei={n_nuc}>{max_nuclei_per_frame}")
    if prob_map is not None and max_membrane_area_frac > 0:
        ps = smooth_probabilities(prob_map, membrane_smoothing)
        mf = float((ps[..., 0] > probability_threshold).mean())
        if mf > max_membrane_area_frac:
            flags.append(f"memb_area={mf:.3f}>{max_membrane_area_frac}")

    status = f"Nuclei: {n_nuc}  |  Cells: {n_cells}"
    if flags:
        status += f"  | EXCLUDED: {'; '.join(flags)}"
    print(f"[seg] {status}")


# ============================================================================
#  WIDGET 6 — Run batch comparison across all conditions
#  (uses the in-memory cache when available, falls back to disk for any
#  FOV that somehow wasn't preloaded)
# ============================================================================
@magicgui(
    call_button="Run batch & plot",
    exclude_border={
        "widget_type": "CheckBox", "value": False,
        "label": "Exclude border cells",
    },
    error_bars={
        "widget_type": "ComboBox", "choices": ["SEM", "SD", "none"],
        "label": "Error bars",
    },
)
def batch_widget(exclude_border=False, error_bars="SEM"):
    if not S["cond_order"]:
        print("[batch] Add at least one condition first.")
        return

    sp = dict(
        nuclear_smoothing=seg_widget.nuclear_smoothing.value,
        nuclear_threshold_mult=seg_widget.nuclear_threshold_mult.value,
        min_nucleus_size_px=seg_widget.min_nucleus_size_px.value,
        membrane_smoothing=seg_widget.membrane_smoothing.value,
        probability_threshold=seg_widget.probability_threshold.value,
        min_cell_size_px=seg_widget.min_cell_size_px.value,
        max_cell_size_px=seg_widget.max_cell_size_px.value,
        min_signal_range=seg_widget.min_signal_range.value,
        max_nuclei=seg_widget.max_nuclei_per_frame.value,
        max_memb_frac=seg_widget.max_membrane_area_frac.value,
    )

    print("\n[batch] Segmentation settings used for this run:")
    print(f"  Nuclear smoothing sigma:        {sp['nuclear_smoothing']}")
    print(f"  Nuclear threshold (x Otsu):     {sp['nuclear_threshold_mult']}")
    print(f"  Min nucleus size (px):          {sp['min_nucleus_size_px']}")
    print(f"  Min signal range (nuclear):     {sp['min_signal_range']}")
    print(f"  Membrane smoothing sigma:       {sp['membrane_smoothing']}")
    print(f"  Membrane prob threshold:        {sp['probability_threshold']}")
    print(f"  Min cell size (px):             {sp['min_cell_size_px']}")
    print(f"  Max cell size (px, 0=off):      {sp['max_cell_size_px']}")
    print(f"  Max nuclei/frame (0=off):       {sp['max_nuclei']}")
    print(f"  Max membrane area frac (0=off): {sp['max_memb_frac']}")
    print(f"  Exclude border cells:           {exclude_border}")
    print(f"  Error bars:                     {error_bars}\n")

    all_dfs = []
    per_cond = {}

    for cname in S["cond_order"]:
        fov_list = S["conditions"][cname]
        print(f"\n[batch] Condition '{cname}': {len(fov_list)} FOVs")
        cond_dfs = []

        # make sure every FOV in this condition is preloaded (no-op if
        # it already is, which it normally will be)
        _preload_fovs(fov_list, progress_cb=_preload_progress_cb)
        _preload_prob_maps(fov_list, progress_cb=_preload_progress_cb)

        for fov in fov_list:
            entry = S["fov_data"].get(fov)
            if entry is None:
                print(f"  [skip] {fov}: FOV data not found")
                continue
            try:
                img_405 = entry["img_405"]
                img_488 = entry["img_488"]
                prob = entry["prob_map"]
                if prob is None:
                    print(f"  [skip] {fov}: no probability map available")
                    continue

                nuc = segment_nuclei(
                    img_405,
                    smoothing_sigma=sp["nuclear_smoothing"],
                    min_size_px=sp["min_nucleus_size_px"],
                    min_signal_range=sp["min_signal_range"],
                    threshold_multiplier=sp["nuclear_threshold_mult"],
                )

                n_nuc = int(len(np.unique(nuc)) - 1)
                ps = smooth_probabilities(prob, sp["membrane_smoothing"])
                mf = float((ps[..., 0] > sp["probability_threshold"]).mean())
                excluded = False
                if sp["max_nuclei"] > 0 and n_nuc > sp["max_nuclei"]:
                    excluded = True
                if sp["max_memb_frac"] > 0 and mf > sp["max_memb_frac"]:
                    excluded = True
                if excluded:
                    print(f"  [excluded] {fov}: nuclei={n_nuc}, memb_area={mf:.3f}")
                    continue

                cells = cells_from_nucleus_and_membrane(
                    prob, nuc,
                    membrane_class_idx=0,
                    probability_threshold=sp["probability_threshold"],
                    min_cell_size_px=sp["min_cell_size_px"],
                    max_cell_size_px=sp["max_cell_size_px"],
                    smoothing_sigma=sp["membrane_smoothing"],
                )

                df = measure_intensity_per_cell(
                    cells, img_488, channel_name="488",
                    exclude_border=exclude_border,
                    condition=cname, fov=fov,
                )
                cond_dfs.append(df)
                print(f"  {fov}: {len(df)} cells")

            except Exception as e:
                print(f"  [error] {fov}: {e}")

        per_cond[cname] = (pd.concat(cond_dfs, ignore_index=True)
                           if cond_dfs else pd.DataFrame())
        all_dfs.extend(cond_dfs)

    if not all_dfs:
        print("[batch] No cells measured. Check Ilastik paths and parameters.")
        return

    master = pd.concat(all_dfs, ignore_index=True)

    summ_rows = []
    for cname in S["cond_order"]:
        df = per_cond.get(cname, pd.DataFrame())
        if df.empty:
            continue
        col = "mean_intensity_488"
        summ_rows.append({
            "condition": cname,
            "n_cells": len(df),
            "n_fovs": df["fov"].nunique() if "fov" in df.columns else 0,
            "mean_MFI_488": df[col].mean(),
            "median_MFI_488": df[col].median(),
            "std_MFI_488": df[col].std(),
        })
    summary = pd.DataFrame(summ_rows)

    tsv = master.to_csv(sep="\t", index=False)
    if _clipboard(tsv):
        print(f"\n[clipboard] Per-cell data copied ({len(master)} cells) "
              f"— paste into Excel / Prism")
    else:
        print("\n[clipboard] Could not access clipboard — printing summary:")
        print(summary.to_string(index=False))

    conds = [c for c in S["cond_order"]
             if c in per_cond and not per_cond[c].empty]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conds), 1)))

    def _err(series, mode):
        v = series.dropna().to_numpy()
        m = float(np.mean(v)) if len(v) else 0.0
        if mode == "none" or len(v) < 2:
            return m, 0.0
        sd = float(np.std(v, ddof=1))
        return m, (sd if mode == "SD" else sd / np.sqrt(len(v)))

    fig1, ax1 = plt.subplots(figsize=(max(4, 1.5 * len(conds)), 5))
    rng = np.random.default_rng(42)
    for i, c in enumerate(conds):
        vals = per_cond[c]["mean_intensity_488"].dropna().to_numpy()
        jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals,
                    alpha=0.35, s=6, color=colors[i])
        m, e = _err(per_cond[c]["mean_intensity_488"], error_bars)
        ax1.errorbar(i, m, yerr=e, fmt="_", markersize=18, color="black",
                     capsize=6, linewidth=2)
    ax1.set_xticks(range(len(conds)))
    ax1.set_xticklabels(conds, rotation=45, ha="right")
    ax1.set_ylabel("Per-cell mean 488 intensity")
    ax1.set_title("Per-cell MFI by condition")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(conds):
        vals = per_cond[c]["mean_intensity_488"].dropna().to_numpy()
        ax2.hist(vals, bins=40, alpha=0.45, density=True,
                 label=f"{c} (n={len(vals)})", color=colors[i])
    ax2.set_xlabel("Per-cell mean 488 intensity")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of per-cell MFI")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    if len(conds) == 2:
        try:
            from scipy import stats as _st
            a, b = conds
            va = per_cond[a]["mean_intensity_488"].dropna()
            vb = per_cond[b]["mean_intensity_488"].dropna()
            if len(va) > 1 and len(vb) > 1:
                t, pv = _st.ttest_ind(va, vb, equal_var=False)
                print(f"\n[stats] Welch t-test ({a} vs {b}): "
                      f"t = {t:.3f}, p = {pv:.4g}")
        except Exception:
            pass

    print(f"\n[batch] Done — {len(master)} cells across "
          f"{len(conds)} conditions")
    print(f"[batch] Summary:\n{summary.to_string(index=False)}")
    plt.show(block=False)


# ============================================================================
#  Dock all widgets
# ============================================================================
viewer.window.add_dock_widget(setup_widget,
    name="1 · Setup (Ilastik paths)", area="right")
viewer.window.add_dock_widget(add_cond_widget,
    name="2 · Add condition (preloads FOVs)", area="right")
viewer.window.add_dock_widget(preload_progress,
    name="Preload progress", area="right")
viewer.window.add_dock_widget(cond_status_label,
    name="Conditions loaded", area="right")
viewer.window.add_dock_widget(unload_widget,
    name="Unload a condition from RAM", area="right")
viewer.window.add_dock_widget(clear_cond_widget,
    name="Clear conditions", area="right")
viewer.window.add_dock_widget(scrub_widget,
    name="3 · Position (scrub / count / remove)", area="right")
viewer.window.add_dock_widget(seg_widget,
    name="4 · Segmentation (live)", area="right")
viewer.window.add_dock_widget(batch_widget,
    name="5 · Run batch & plot", area="right")

print("=" * 70)
print("  Cytokine Quantification GUI ready (fast-scrub version).")
print()
print("  1. Confirm Ilastik paths in Setup (pre-filled)")
print("  2. Add conditions — TIFFs are grouped into FOVs and PRELOADED")
print("     into memory right away (progress bar shows this happening).")
print("  3. Scrub FOVs with the Position module — pure in-memory lookup,")
print("     no disk I/O, no freezing.")
print("  4. Run batch → plots displayed, per-cell data → clipboard")
print("=" * 70)

if _created:
    napari.run()
