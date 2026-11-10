"""Export ilastik input stacks and run ilastik headless (Pixel Classification)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import tifffile


def export_training_stack(
    channel_imgs: dict[str, np.ndarray],
    channel_order: list[str],
    out_path: str | Path,
) -> Path:
    """Stacks the given channels (in channel_order) into a multi-channel
    (C, Y, X) uint16 TIFF for ilastik. Works for any number of channels, so
    the same function serves both panel training-stack export and any other
    multi-channel composite a panel might need.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack([channel_imgs[c] for c in channel_order], axis=0).astype(np.uint16)
    tifffile.imwrite(out_path, stack, imagej=True, metadata={"axes": "CYX"})
    return out_path


def check_ilastik_setup(ilastik_executable: str | Path, ilastik_project_path: str | Path) -> None:
    ilastik_bin = Path(ilastik_executable)
    if not ilastik_bin.exists():
        raise FileNotFoundError(
            f"ilastik executable not found at {ilastik_bin}. "
            "Update the ilastik executable path in your project config.yaml."
        )
    project_path = Path(ilastik_project_path)
    if not project_path.exists():
        raise FileNotFoundError(
            f"ilastik project not found at {project_path}. Train this panel first "
            "— see cell tracker/notebooks/00_train_and_test_panel.ipynb."
        )

    if not _has_trained_labels(project_path):
        raise RuntimeError(
            f"{project_path} exists but has no trained labels yet — running "
            "headless against it will fail with a cryptic ilastik crash "
            "('no classifier' / 'ExportPath ... not ready'). Open it in the "
            "ilastik GUI, add a few single-channel membrane images, draw "
            "label brushstrokes for membrane/background, and save (see this "
            "panel's README.md / notebooks/00_train_and_test_panel.ipynb) "
            "before running this again."
        )


def _has_trained_labels(project_path: Path) -> bool:
    """Best-effort check for whether a Pixel Classification .ilp has any
    labels drawn (a freshly scaffolded/untrained project has an empty,
    uninitialized LabelNames dataset). Fails open (returns True) if the
    project structure doesn't match what we expect, so an unrelated file
    format quirk never blocks a legitimately trained project."""
    try:
        import h5py

        with h5py.File(project_path, "r") as f:
            label_names = f["PixelClassification/LabelNames"][()]
            return len(label_names) > 0 and all(
                isinstance(n, (bytes, str)) for n in label_names
            )
    except Exception:
        return True


def run_ilastik_headless(
    ilastik_executable: str | Path,
    ilastik_project_path: str | Path,
    input_files: list[Path],
    output_dir: str | Path,
    export_source: str = "Probabilities",
    output_format: str = "hdf5",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Runs ilastik headless on a batch of input TIFFs, exporting
    `export_source` (default: 'Probabilities', a per-pixel-class probability
    stack) to output_dir. Output files are named <input_stem>.<ext> via
    ilastik's {nickname} template, so keep input_files uniquely named
    (cell_tracker's pipeline.py uses each sample's sample_id for this).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = extra_args or []

    ext = "h5" if output_format == "hdf5" else output_format
    output_filename_format = str(output_dir / f"{{nickname}}.{ext}")

    cmd = [
        str(ilastik_executable),
        "--headless",
        f"--project={ilastik_project_path}",
        f"--export_source={export_source}",
        f"--output_format={output_format}",
        f"--output_filename_format={output_filename_format}",
        *extra_args,
        *[str(f) for f in input_files],
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ilastik headless run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result
