# Internalized Fluorescence Quantification (ilastik + cell_tracker)

Measures **internalized / engulfment fluorescence** (Far Red - 640 readout)
per macrophage. Unlike the `cytokine_quantification_napari` pipeline next to
it, this dataset has **no nuclear stain** — cells are imaged in only two
channels (Green - 488 membrane marker and Far Red - 640 target). Cells are
segmented from the Green-488 membrane with an ilastik classifier, then split
into single-cell masks by **distance-transform watershed** (no nucleus to
seed from).

This is a notebook pipeline (not a napari script). It was authored by
**Andrew Manion** and is built on his reusable `cell_tracker` package, which
is bundled here (`cell_tracker/`).

## What's in this folder
```
internalized_fluorescence_quantification/
├── config.yaml          # all settings for a run (paths, panel, background, etc.)
├── requirements.txt      # installs the bundled cell_tracker package + Jupyter
├── notebooks/
│   ├── 01_run_cell_segmentation.ipynb     # segment cells (calls cell_tracker)
│   └── 02_measure_target_intensity.ipynb  # measure Far Red-640 per cell
└── cell_tracker/         # the bundled analysis package (green panel + its trained .ilp)
```
The trained ilastik classifier (`cell_tracker/green/membrane_segmentation.ilp`)
is included, so you don't need to retrain it.

## Data you must add
Raw microscopy is **not** included (same as the other folders leave out
TIFFs). Put your `.nd2` files in, following this structure:
```
data/raw/<replicate>/<condition>/<file>.nd2      # full dataset
data/test/<file>.nd2                              # a small test file
```
`config.yaml`'s `use_test_data` toggles which one runs (`true` = test folder,
`false` = full `data/raw` tree).

## One-time setup
1. Install the packages (ideally in a fresh environment):
   ```
   pip install -r requirements.txt
   ```
   This installs the bundled `cell_tracker` package (editable) plus Jupyter.
2. Install **ilastik** (the program) from https://www.ilastik.org/download —
   it is not a pip package.
3. In `config.yaml`, set `segmentation.ilastik_executable` to the path of
   ilastik's headless launcher **on your machine** (`ilastik.exe` on Windows,
   `run_ilastik.sh` on Mac/Linux). The value currently in the file is the
   original author's Mac path — change it to yours.
4. In `config.yaml`, set `background_subtraction.reference_points` (one
   `{condition, point_index}` per replicate) once you know which point best
   represents background in each replicate.

## How to run
1. Open `notebooks/01_run_cell_segmentation.ipynb`, **Run All** → writes
   per-cell instance masks + `manifest.csv` into `data/segmentation_work/`,
   and QC panels into `data/qc/`.
2. Open `notebooks/02_measure_target_intensity.ipynb`, **Run All** → reads
   the manifest, measures per-cell Far Red-640 intensity, and writes
   `data/results/mfi_summary.csv`.

(The notebooks read `../config.yaml`, so keep them inside `notebooks/`.)

## Notes
- `cell_tracker` here is a bundled copy of Andrew's package (version pinned to
  what produced these results). The full package (including the `red` panel
  used by the nuclear TNFα work) lives with him; only the `green` panel needed
  by this pipeline is included here.
- See `cell_tracker/green/README.md` for how the green classifier was trained
  and tuned, and `cell_tracker/README.md` for the package design.
