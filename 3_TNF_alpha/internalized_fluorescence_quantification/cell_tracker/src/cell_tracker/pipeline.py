"""High-level orchestration tying together: channel extraction -> ilastik
membrane pixel classification -> instance segmentation. This is the "cell
segmentation pipeline" that project-specific pipelines call; everything
after this (measuring a target channel, aggregating by condition, plotting,
stats) is left to the calling project, using cell_tracker.measure on the
manifest this returns.

Design (see cell_tracker/README.md for the full picture):
  - ilastik is trained on ONE marker channel only (a panel's "membrane"
    channel, e.g. Red - 561) and classifies each pixel as membrane vs
    background. It never sees a nuclear channel.
  - If the project has a nuclear channel (nuclear_channel is given), nuclei
    are found directly in Python via
    cell_tracker.nuclear_segmentation.segment_nuclei (Otsu threshold — no
    ilastik involved), and each detected nucleus seeds a watershed that
    grows through the (smoothed, thresholded) membrane probability map
    until it hits a neighboring cell or the background
    (instance_segmentation.cells_from_nucleus_and_membrane).
  - If the project has NO nuclear channel (nuclear_channel=None, the
    default), instances are split directly from the membrane mask's shape
    instead, via distance-transform watershed
    (instance_segmentation.cells_from_membrane_mask) — no seed channel
    needed at all. Used for projects imaging only a cell-body marker (e.g.
    Erika small beads' Green - 488 macrophage marker).
  - A panel can optionally add a THIRD ilastik pixel class marking where two
    cells visibly touch (e.g. "cell_contact", set via panel.yaml's
    ilastik.contact_class). When present, cells_from_membrane_mask carves
    those pixels out of the foreground mask before splitting instances, so
    ilastik's own learned touching-cell boundary drives the split instead of
    relying only on the geometric min_seed_distance_px heuristic. Panels
    without a contact_class behave exactly as before — this is fully
    opt-in.
  - QC images are written for only the FIRST point processed in a run by
    default (qc_first_only=True), to keep QC generation fast, or for an
    explicit list of points (qc_point_indices=[...]) when you want to check
    specific images (e.g. ones you already know look wrong). qc_selectors
    additionally allows exact (replicate, condition, point_index) targeting
    when point_index alone is ambiguous across files — e.g. always showing
    QC for each replicate's background-subtraction reference point.
  - Whole frames can be flagged as unusable via a panel's
    max_nuclei_per_frame / max_membrane_area_fraction (see panel.yaml) —
    e.g. a blank field of view whose noise still trips up nuclear detection,
    or an oversaturated/artifact frame. max_nuclei_per_frame only applies
    when there's a nuclear channel. Flagged points are NOT dropped from the
    manifest (their segmentation still runs and gets QC'd like any other
    point) — they're marked via an `excluded` column so downstream
    measurement notebooks can filter them out before aggregating results.

One ND2 FILE can contain multiple POINTS (multi-position acquisitions). This
module expands each discovered file into one manifest row per point — every
point is segmented and measured independently, never blended together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes as ndi_fill_holes
from tqdm.auto import tqdm

from .discovery import make_sample_id
from .ilastik_utils import check_ilastik_setup, export_training_stack, run_ilastik_headless
from .instance_segmentation import (
    cells_from_membrane_mask,
    cells_from_nucleus_and_membrane,
    load_probabilities,
    smooth_probabilities,
)
from .io_utils import extract_channels, get_point_names, load_nd2, iter_points
from .nuclear_segmentation import segment_nuclei
from .qc import make_membrane_only_qc, make_pipeline_qc


def run_segmentation_pipeline(
    files_df: pd.DataFrame,
    panel: dict,
    ilastik_executable: str | Path,
    work_dir: str | Path,
    nuclear_channel: dict | None = None,
    zstack_cfg: dict | None = None,
    force_export: bool = False,
    force_segment: bool = False,
    qc_dir: str | Path | None = None,
    qc_first_only: bool = True,
    qc_point_indices: list[int] | None = None,
    qc_selectors: list[tuple] | None = None,
):
    """Runs the full segmentation pipeline over every point in every ND2 file
    in `files_df` (from discovery.discover_project_files, for a full run, or
    discovery.discover_flat_files, for a quick test run against a small
    always-available test dataset — the caller picks which one to pass in,
    that's the test/full toggle):

      1. for each file, split out every point (field of view); extract the
         panel's membrane channel (exported as a single-channel TIFF for
         ilastik) AND, if nuclear_channel is given, the project's nuclear
         channel too (kept as a plain TIFF, read directly in Python — never
         shown to ilastik), skipping points already exported unless
         force_export=True
      2. batch-run ilastik headless with the panel's trained Pixel
         Classification project to export the membrane probability map
      3. instance-segment each point — nucleus-seeded watershed if
         nuclear_channel was given, distance-transform watershed on the
         membrane mask alone if not
      4. write a QC panel (input image next to what was identified), if
         qc_dir is given — for the first point only by default, or for
         whichever points are listed in qc_point_indices if given

    nuclear_channel: a channel_cfg dict (name/wavelength_nm/index), e.g.
    cfg['nuclear_channel'] from a project's config.yaml — same shape as any
    entry in panel['channels']. Leave as None (the default) for projects
    with no nuclear stain at all.

    qc_point_indices: an explicit list of `point_index` values (the same
    zero-based index in each manifest row, e.g. [5, 16, 24]) to always write
    QC for, regardless of qc_first_only — use this to check specific images
    you already suspect look wrong, rather than whatever happens to be
    first. When given, this REPLACES the qc_first_only selection (it does
    not also include the first point unless 0 is in the list). NOTE:
    point_index alone isn't unique across files — a value here matches that
    point_index in EVERY file/replicate/condition, not just one.

    qc_selectors: an explicit list of (replicate, condition, point_index)
    tuples to always write QC for, exact-matching all three fields — use
    this when point_index alone would be ambiguous across files (e.g. to
    target one specific replicate's background-subtraction reference
    point, not that same point_index in every other file too). Unlike
    qc_point_indices, this ADDS to whatever qc_first_only/qc_point_indices
    already selects rather than replacing it. Values are compared as
    strings, so int/str mismatches between config and manifest don't matter.

    Returns (manifest_df, manifest_csv_path). manifest_df has one row per
    POINT with columns: nd2_path, replicate, condition, filename, file_id,
    point_index, point_label, sample_id, membrane_stack_path,
    nuclear_image_path (None if no nuclear_channel), probability_path,
    nucleus_label_path (None if no nuclear_channel), segmentation_path,
    n_nuclei (None if no nuclear_channel), membrane_area_fraction, excluded,
    exclude_reason — everything a downstream measurement notebook needs.
    """
    has_nuclear = nuclear_channel is not None

    work_dir = Path(work_dir)
    membrane_dir = work_dir / "ilastik_stacks"
    prob_dir = work_dir / "probabilities"
    nucleus_dir = work_dir / "nuclei"
    seg_dir = work_dir / "segmentation"
    for d in (membrane_dir, prob_dir, nucleus_dir, seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    check_ilastik_setup(ilastik_executable, panel["ilastik_project_path"])

    zstack_cfg = zstack_cfg or {"max_project": True}
    seg_cfg = panel.get("segmentation", {})
    smoothing_sigma = seg_cfg.get("smoothing_sigma", 0.0)
    min_cell_size_px = seg_cfg.get("min_cell_size_px", 20)
    probability_threshold = seg_cfg.get("probability_threshold", 0.5)
    min_signal_range = seg_cfg.get("min_signal_range", 0.0)
    min_seed_distance_px = seg_cfg.get("min_seed_distance_px", 10)
    fill_holes = seg_cfg.get("fill_holes", True)
    contact_probability_threshold = seg_cfg.get("contact_probability_threshold", None)
    max_nuclei_per_frame = seg_cfg.get("max_nuclei_per_frame", None)
    max_membrane_area_fraction = seg_cfg.get("max_membrane_area_fraction", None)

    qc_selector_set = (
        {(str(r), str(c), int(p)) for r, c, p in qc_selectors} if qc_selectors else None
    )

    membrane_role = panel["ilastik"].get("membrane_class", next(iter(panel["channels"])))
    # Optional third ilastik class marking where two cells touch (see
    # cells_from_membrane_mask) -- panel.yaml sets ilastik.contact_class to
    # the pixel_classes key to use, e.g. "cell_contact". None if the panel
    # doesn't have one (the default -- no code changes needed for panels
    # that stay 2-class).
    contact_role = panel["ilastik"].get("contact_class")
    contact_class_idx = panel["ilastik"]["pixel_classes"][contact_role] if contact_role else None
    combined_channels = {**panel["channels"], "nuclear": nuclear_channel} if has_nuclear else panel["channels"]

    # --- 1. expand files into points, extract membrane (+ nuclear) images --
    point_rows = []
    for file_row in tqdm(files_df.itertuples(), total=len(files_df), desc="Extracting channels"):
        point_names = get_point_names(file_row.nd2_path)
        arr, channel_meta, axes = load_nd2(file_row.nd2_path)

        for point_index, point_label, sub_arr, _sub_axes in iter_points(arr, axes, point_names):
            sample_id = make_sample_id(file_row.file_id, point_label)
            membrane_path = membrane_dir / f"{sample_id}.tif"
            nuclear_image_path = nucleus_dir / f"{sample_id}_nuclear.tif" if has_nuclear else None

            needs_export = force_export or not membrane_path.exists()
            if has_nuclear:
                needs_export = needs_export or not nuclear_image_path.exists()

            if needs_export:
                imgs = extract_channels(sub_arr, channel_meta, combined_channels, zstack_cfg)
                export_training_stack(
                    {membrane_role: imgs[membrane_role]}, panel["channel_stack_order"], membrane_path
                )
                if has_nuclear:
                    tifffile.imwrite(nuclear_image_path, imgs["nuclear"].astype(np.uint16))

            point_rows.append(
                {
                    "nd2_path": file_row.nd2_path,
                    "replicate": file_row.replicate,
                    "condition": file_row.condition,
                    "filename": file_row.filename,
                    "file_id": file_row.file_id,
                    "point_index": point_index,
                    "point_label": point_label,
                    "sample_id": sample_id,
                    "membrane_stack_path": membrane_path,
                    "nuclear_image_path": nuclear_image_path,
                }
            )

    manifest = pd.DataFrame(point_rows)
    membrane_paths = list(manifest["membrane_stack_path"])

    # --- 2. batch headless ilastik: membrane probability map ---------------
    ilastik_cfg = panel["ilastik"]
    export_source = ilastik_cfg.get("export_source", "Probabilities")
    prob_paths = [prob_dir / f"{sid}.h5" for sid in manifest["sample_id"]]
    missing_prob = [p for p in prob_paths if not p.exists()]
    if force_segment or missing_prob:
        run_ilastik_headless(
            ilastik_executable=ilastik_executable,
            ilastik_project_path=panel["ilastik_project_path"],
            input_files=membrane_paths,
            output_dir=prob_dir,
            export_source=export_source,
            output_format="hdf5",
        )
    manifest["probability_path"] = prob_paths

    # --- 3. instance segmentation -------------------------------------------
    membrane_class_idx = ilastik_cfg["pixel_classes"][membrane_role]

    nucleus_label_paths = []
    seg_paths = []
    n_nuclei_list = []
    membrane_area_fraction_list = []
    excluded_list = []
    exclude_reason_list = []
    for i, row in enumerate(
        tqdm(manifest.itertuples(), total=len(manifest), desc="Segmenting cells")
    ):
        seg_path = seg_dir / f"{row.sample_id}_segmentation.tif"
        nucleus_label_path = nucleus_dir / f"{row.sample_id}_nuclei_labels.tif" if has_nuclear else None

        nuclear_img = tifffile.imread(row.nuclear_image_path) if has_nuclear else None
        prob = load_probabilities(row.probability_path)

        needs_segment = force_segment or not seg_path.exists()
        if has_nuclear:
            needs_segment = needs_segment or not nucleus_label_path.exists()

        if needs_segment:
            if has_nuclear:
                nucleus_labels = segment_nuclei(
                    nuclear_img,
                    smoothing_sigma=smoothing_sigma,
                    min_size_px=min_cell_size_px,
                    min_signal_range=min_signal_range,
                )
                tifffile.imwrite(nucleus_label_path, nucleus_labels)

                instances = cells_from_nucleus_and_membrane(
                    prob,
                    nucleus_labels,
                    membrane_class_idx=membrane_class_idx,
                    probability_threshold=probability_threshold,
                    min_cell_size_px=min_cell_size_px,
                    smoothing_sigma=smoothing_sigma,
                    fill_holes=fill_holes,
                )
            else:
                nucleus_labels = None
                instances = cells_from_membrane_mask(
                    prob,
                    membrane_class_idx=membrane_class_idx,
                    probability_threshold=probability_threshold,
                    min_cell_size_px=min_cell_size_px,
                    smoothing_sigma=smoothing_sigma,
                    min_seed_distance_px=min_seed_distance_px,
                    fill_holes=fill_holes,
                    contact_class_idx=contact_class_idx,
                    contact_probability_threshold=contact_probability_threshold,
                )
            tifffile.imwrite(seg_path, instances.astype(np.int32))
        else:
            nucleus_labels = tifffile.imread(nucleus_label_path) if has_nuclear else None
            instances = tifffile.imread(seg_path)

        nucleus_label_paths.append(nucleus_label_path)
        seg_paths.append(seg_path)

        # --- 4. per-frame quality metrics (computed for EVERY point, not ---
        # just QC'd ones, since exclusion has to apply to the whole dataset)
        smoothed_prob = smooth_probabilities(prob, smoothing_sigma)
        membrane_mask = smoothed_prob[..., membrane_class_idx] > probability_threshold
        if fill_holes:
            membrane_mask = ndi_fill_holes(membrane_mask)
        n_nuclei = int(len(np.unique(nucleus_labels)) - 1) if has_nuclear else None
        membrane_area_fraction = float(membrane_mask.mean())

        reasons = []
        if has_nuclear and max_nuclei_per_frame is not None and n_nuclei > max_nuclei_per_frame:
            reasons.append(f"n_nuclei={n_nuclei} > max_nuclei_per_frame={max_nuclei_per_frame}")
        if (
            max_membrane_area_fraction is not None
            and membrane_area_fraction > max_membrane_area_fraction
        ):
            reasons.append(
                f"membrane_area_fraction={membrane_area_fraction:.3f} > "
                f"max_membrane_area_fraction={max_membrane_area_fraction}"
            )
        n_nuclei_list.append(n_nuclei)
        membrane_area_fraction_list.append(membrane_area_fraction)
        excluded_list.append(bool(reasons))
        exclude_reason_list.append("; ".join(reasons) if reasons else None)

        # --- 5. QC: input image next to what was identified ----------------
        # qc_point_indices, if given, takes over selection entirely (an
        # explicit list of points to check). Otherwise fall back to
        # qc_first_only: just the first point, or every point if False.
        # QC is written regardless of whether a point ends up excluded, so
        # you can see WHY it got flagged.
        if qc_point_indices is not None:
            write_qc = row.point_index in qc_point_indices
        else:
            write_qc = (i == 0) or (not qc_first_only)

        if qc_selector_set is not None:
            write_qc = write_qc or (
                (str(row.replicate), str(row.condition), int(row.point_index)) in qc_selector_set
            )

        if qc_dir is not None and write_qc:
            membrane_img = tifffile.imread(row.membrane_stack_path)
            membrane_img = np.squeeze(membrane_img)

            if has_nuclear:
                make_pipeline_qc(
                    nuclear_img,
                    nucleus_labels,
                    membrane_img,
                    membrane_mask,
                    instances,
                    Path(qc_dir) / f"{row.sample_id}_qc.png",
                    title=row.sample_id,
                )
            else:
                make_membrane_only_qc(
                    membrane_img,
                    membrane_mask,
                    instances,
                    Path(qc_dir) / f"{row.sample_id}_qc.png",
                    title=row.sample_id,
                )

    manifest["nucleus_label_path"] = nucleus_label_paths
    manifest["segmentation_path"] = seg_paths
    manifest["n_nuclei"] = n_nuclei_list
    manifest["membrane_area_fraction"] = membrane_area_fraction_list
    manifest["excluded"] = excluded_list
    manifest["exclude_reason"] = exclude_reason_list

    manifest_path = work_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest, manifest_path
