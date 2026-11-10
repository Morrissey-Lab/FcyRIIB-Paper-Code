"""Discover ND2 files in a project's <replicate>/<condition>/ folder tree.

Note: one ND2 FILE can contain multiple POINTS (multi-position acquisitions —
several distinct fields of view in one file). discover_project_files() only
walks the filesystem (cheap, no file I/O) and returns one row per file; it's
run_segmentation_pipeline() (in pipeline.py), which actually opens each file,
that expands a multi-point file into one manifest row per point using
make_sample_id() below.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def make_file_id(replicate: str, condition: str, filename: str) -> str:
    """A unique, filesystem-safe identifier for one ND2 FILE (which may
    contain multiple points/fields of view)."""
    stem = Path(filename).stem
    return f"{replicate}__{condition}__{stem}"


def make_sample_id(file_id: str, point_label: str) -> str:
    """A unique, filesystem-safe identifier for one POINT (field of view)
    within a file, used as the base filename for every downstream artifact
    (training stack, probability map, segmentation mask, QC image) so they
    can all live in flat directories without collisions."""
    return f"{file_id}__pt{point_label}"


def discover_project_files(project_dir: str | Path) -> pd.DataFrame:
    """Walks project_dir expecting exactly:

        project_dir/<replicate>/<condition>/<file>.nd2

    Returns a DataFrame with columns: nd2_path, replicate, condition,
    filename, file_id — one row per ND2 FILE (not per point; a file may
    contain multiple points, expanded later by run_segmentation_pipeline).

    Any .nd2 file that isn't exactly two folders deep is SKIPPED with a
    warning (not a hard error) — e.g. a stray duplicate left sitting
    directly in project_dir won't block discovery of correctly-nested files
    elsewhere in the tree. Raises only if project_dir doesn't exist, or if
    no correctly-nested .nd2 files are found at all.
    """
    project_dir = Path(project_dir)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    nd2_paths = sorted(project_dir.rglob("*.nd2"))
    if not nd2_paths:
        raise FileNotFoundError(f"No .nd2 files found under {project_dir}")

    rows = []
    skipped = []
    for nd2_path in nd2_paths:
        rel_parts = nd2_path.relative_to(project_dir).parts
        if len(rel_parts) != 3:
            skipped.append((nd2_path, rel_parts))
            continue
        replicate, condition, filename = rel_parts
        rows.append(
            {
                "nd2_path": nd2_path,
                "replicate": replicate,
                "condition": condition,
                "filename": filename,
                "file_id": make_file_id(replicate, condition, filename),
            }
        )

    if skipped:
        examples = "\n".join(f"  - {p}  (path parts: {parts})" for p, parts in skipped[:10])
        warnings.warn(
            "Skipped .nd2 file(s) that aren't exactly two folders deep as "
            "<project_dir>/<replicate>/<condition>/<file>.nd2 "
            f"({len(skipped)} file(s)):\n{examples}\n"
            "Move/delete them (or nest them under a replicate/condition "
            "folder) so they're picked up too.",
            stacklevel=2,
        )

    if not rows:
        raise ValueError(
            f"No correctly-nested .nd2 files found under {project_dir} "
            "(everything found was skipped — see warning above). Expected "
            "<project_dir>/<replicate>/<condition>/<file>.nd2."
        )

    df = pd.DataFrame(rows)
    dupes = df[df.duplicated("file_id", keep=False)]
    if not dupes.empty:
        raise ValueError(
            "Duplicate (replicate, condition, filename) combinations found — "
            f"file IDs must be unique:\n{dupes[['replicate', 'condition', 'filename']]}"
        )
    return df


def discover_flat_files(
    test_dir: str | Path,
    replicate: str = "test",
    condition: str = "test",
) -> pd.DataFrame:
    """Like discover_project_files, but for a flat folder of .nd2 files with
    no <replicate>/<condition> nesting — used for a project's small,
    always-available test dataset (e.g. data/test/). Every file found is
    tagged with a constant replicate/condition label so it flows through
    exactly the same downstream code (manifest columns, file_id/sample_id
    scheme) as the full <project_dir>/<replicate>/<condition>/ tree.

    This is the toggle mechanism: a project's notebook picks either this
    function (test run) or discover_project_files (full run) based on a
    `use_test_data` flag in its config.yaml, then passes the resulting
    DataFrame into cell_tracker.run_segmentation_pipeline the same way
    either way.
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    nd2_paths = sorted(test_dir.glob("*.nd2"))
    if not nd2_paths:
        raise FileNotFoundError(f"No .nd2 files found directly under {test_dir}")

    rows = [
        {
            "nd2_path": p,
            "replicate": replicate,
            "condition": condition,
            "filename": p.name,
            "file_id": make_file_id(replicate, condition, p.name),
        }
        for p in nd2_paths
    ]
    return pd.DataFrame(rows)
