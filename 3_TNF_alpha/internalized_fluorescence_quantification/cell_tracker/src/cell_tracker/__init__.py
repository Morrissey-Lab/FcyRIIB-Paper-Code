"""cell_tracker — reusable ND2 -> ilastik -> per-cell instance segmentation pipeline.

Design:
- ilastik does PIXEL classification on ONE marker channel only (a panel's
  "membrane" channel, e.g. a cytoplasm/membrane marker), trained once per
  panel and reused across every project that images that marker — no
  retraining per project. ilastik never sees the nuclear channel.
- If the project has a nuclear channel, nuclei are identified separately,
  directly in Python (nuclear_segmentation.segment_nuclei — Otsu threshold,
  no ilastik involved), and per-cell INSTANCE labels are produced via
  watershed seeded from those nuclei and constrained to the (smoothed,
  thresholded) ilastik membrane probability map
  (instance_segmentation.cells_from_nucleus_and_membrane).
- If the project has NO nuclear channel, instances are split directly from
  the membrane mask's shape instead, via distance-transform watershed
  (instance_segmentation.cells_from_membrane_mask) — no seed channel
  needed. Pass nuclear_channel=None (the default) to
  run_segmentation_pipeline for this case.
- Either way, this is what turns one semantic membrane/background
  probability map into individually countable/measurable cells.
- Everything downstream of the instance segmentation (measuring any target
  channel per cell, aggregating by replicate/condition, plotting, stats) is
  intentionally left to project-specific notebooks, which import the
  `measure` module and build on top of the segmentation manifest.
"""

from .config import load_config
from .discovery import discover_project_files, discover_flat_files, make_file_id, make_sample_id
from .io_utils import (
    load_nd2,
    load_nd2_point,
    get_point_names,
    iter_points,
    resolve_channel_index,
    extract_channels,
    max_project_if_stack,
)
from .panels import load_panel, list_panels, scaffold_panel
from .ilastik_utils import (
    export_training_stack,
    check_ilastik_setup,
    run_ilastik_headless,
)
from .nuclear_segmentation import segment_nuclei
from .instance_segmentation import (
    load_probabilities,
    cells_from_nucleus_and_membrane,
    cells_from_membrane_mask,
    smooth_probabilities,
)
from .measure import (
    load_segmentation,
    background_subtract,
    compute_segmentation_background,
    measure_intensity_per_cell,
    aggregate_results,
)
from .qc import (
    make_qc_overlay,
    make_class_probability_overlay,
    make_pipeline_qc,
    make_membrane_only_qc,
)
from .pipeline import run_segmentation_pipeline

__all__ = [
    "load_config",
    "discover_project_files",
    "discover_flat_files",
    "make_file_id",
    "make_sample_id",
    "load_nd2",
    "load_nd2_point",
    "get_point_names",
    "iter_points",
    "resolve_channel_index",
    "extract_channels",
    "max_project_if_stack",
    "load_panel",
    "list_panels",
    "scaffold_panel",
    "export_training_stack",
    "check_ilastik_setup",
    "run_ilastik_headless",
    "segment_nuclei",
    "load_probabilities",
    "cells_from_nucleus_and_membrane",
    "cells_from_membrane_mask",
    "smooth_probabilities",
    "load_segmentation",
    "background_subtract",
    "compute_segmentation_background",
    "measure_intensity_per_cell",
    "aggregate_results",
    "make_qc_overlay",
    "make_class_probability_overlay",
    "make_pipeline_qc",
    "make_membrane_only_qc",
    "run_segmentation_pipeline",
]

__version__ = "0.2.0"
