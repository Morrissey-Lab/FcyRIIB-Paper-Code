# TNF-alpha Figure

Makes the TNF-alpha violin-plot figures from single-cell intensity data.

## What's in this folder
- `TNF_alpha_fig.ipynb` — the notebook that makes the figure
- `tnfa_data_iso_subtracted.csv` — the input data (included)
- `requirements.txt` — the Python packages needed
## Upstream image-analysis pipelines
Two pipelines produce the per-cell intensity data (each self-contained, with its own README):

- `cytokine_quantification_napari/` — TNFα fluorescence per cell, **using the nuclear signal** (napari + ilastik).
- `internalized_fluorescence_quantification/` — internalized/engulfment fluorescence, **no nuclear marker** (single-cell mask from the Green-488 membrane; notebooks + the bundled `cell_tracker` package).

## How to run
1. Install the packages (one time):
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```
   jupyter notebook TNF_alpha_fig.ipynb
   ```
3. Run all cells (menu: **Run → Run All Cells**).

Figures are saved into this folder as PDF.
