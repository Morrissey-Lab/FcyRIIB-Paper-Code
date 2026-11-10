# cell tracker

Reusable ND2 → ilastik → per-cell instance segmentation pipeline. This is
**not** a project pipeline itself — it's a shared library that project
pipelines (e.g. `python pipelines/Erika TNFa`) install and call, so you
never have to retrain ilastik per project.

This package lives inside `cell segmentation/ilastik/` because it's one of
potentially several tools trained on ilastik panels — future panels for
markers unrelated to this package would nest as siblings of `cell tracker/`
directly under `cell segmentation/ilastik/`, keeping every ilastik-trained
tool discoverable under one umbrella folder.

## How segmentation works here

1. ilastik does **Pixel Classification on ONE marker channel only** — a
   panel's "membrane" channel (e.g. Red - 561, a cytoplasm/membrane marker),
   classified into `membrane` vs `background`. This is the part that's
   specific to a marker and needs training. ilastik never sees the nuclear
   channel at all.
2. Nuclei are identified separately, entirely in Python
   (`nuclear_segmentation.segment_nuclei` — Gaussian smoothing + automatic
   Otsu threshold + small-object removal + connected-component labeling).
   No ilastik training needed for nuclei.
3. Individual cells are split out in Python via **seeded watershed**
   (`instance_segmentation.cells_from_nucleus_and_membrane`): each detected
   nucleus is a seed; watershed grows it through the (smoothed, thresholded)
   membrane probability map until it meets a neighboring cell or the
   background.

This split (ilastik for the membrane pixel class, Python for nuclei and for
instances) is why retraining isn't needed per project: the pixel classifier
only has to learn what your membrane marker looks like once, and nuclear
detection never needs ilastik at all.

## Layout

```
cell tracker/
├── pyproject.toml              # installable as the `cell_tracker` package
├── src/cell_tracker/
│   ├── discovery.py              # walk <project>/<replicate>/<condition>/*.nd2, or a flat test folder
│   ├── io_utils.py                # ND2 loading, channel-by-name/wavelength resolution
│   ├── panels.py                  # load/list/scaffold single-channel panels (the .ilp registry)
│   ├── ilastik_utils.py           # export stacks, run ilastik headless
│   ├── nuclear_segmentation.py    # pure-Python nuclear detection (Otsu) -- no ilastik
│   ├── instance_segmentation.py   # nucleus-seeded watershed: membrane probability -> per-cell labels
│   ├── measure.py                 # regionprops per-cell intensity measurement
│   ├── qc.py                      # QC panels (input image next to what was identified)
│   └── pipeline.py                # run_segmentation_pipeline() — the main entry point
├── red/                          # <- a trained panel (Red - 561 membrane marker)
│   ├── panel.yaml                   # channel definition, pixel-class mapping, tuning
│   ├── membrane_segmentation.ilp    # the trained ilastik project (train via notebook 00)
│   └── README.md
└── notebooks/
    └── 00_train_and_test_panel.ipynb   # train/tune a panel, test end-to-end
```

Every panel is a subfolder directly inside this package (like `red/` above)
— `panels_root` for this package's own panels is just the package folder
itself, and `panel_name` is that subfolder's name.

## Using it from a project pipeline

```bash
# from inside the project folder, e.g. python pipelines/Erika TNFa
pip install -e "../cell segmentation/ilastik/cell tracker"
```

```python
from cell_tracker import load_panel, discover_flat_files, run_segmentation_pipeline

panel = load_panel("../cell segmentation/ilastik/cell tracker", "red")

files_df = discover_flat_files("data/test")  # or discover_project_files("data/raw")

manifest, manifest_path = run_segmentation_pipeline(
    files_df=files_df,
    panel=panel,
    nuclear_channel={"name": "UV - 405", "wavelength_nm": 405, "index": None},
    ilastik_executable="/Applications/ilastik-1.4.1.post1-arm64-OSX.app/Contents/ilastik-release/run_ilastik.sh",
    work_dir="data/segmentation_work",
    qc_dir="data/qc",
)
```

`manifest` is a DataFrame — one row per **point** (field of view; one ND2
file can contain many) — with `replicate`, `condition`, `nd2_path`, and
`segmentation_path` (the per-cell instance label mask). From there, use
`cell_tracker.measure` to measure whatever channel(s) the project cares
about per cell; see Erika TNFa's `02_measure_target_intensity.ipynb` for a
worked example. This is the "modular" seam — new analyses (extra channels,
morphology, ratios, whatever comes next) plug in after the manifest without
touching segmentation at all.

## Adding a new panel

Different marker? Don't touch an existing panel — add a new one, right
alongside `red/`:

```python
from cell_tracker import scaffold_panel

scaffold_panel(
    panels_root=".",  # this package folder -- panel_dir will be "./<panel_name>"
    panel_name="green",
    channel_name="Green - 488",
    wavelength_nm=488,
)
```

Then train it via `notebooks/00_train_and_test_panel.ipynb` (point
`PANEL_NAME` at the new name).

**Panels for a different tool entirely** (not this `cell_tracker` package)
should nest as a sibling of `cell tracker/` directly under
`cell segmentation/ilastik/` — e.g. `cell segmentation/ilastik/some-other-tool/`
— so every ilastik-trained panel across every tool stays discoverable under
one shared `ilastik/` folder.

## Requirements

`pip install -r requirements.txt` installs this package (editable) plus
Jupyter. ilastik itself is a separate desktop install:
https://www.ilastik.org/download
