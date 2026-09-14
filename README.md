# FcγRIIb Paper — Code & Figures

Code to reproduce the figures in the FcγRIIb paper. Each subfolder is fully
self-contained: it holds the script(s), all the data files they need, and its
own README with run instructions. Every folder is ready to run as-is.

## Folders
| Folder | What it makes |
|--------|---------------|
| `1_PIP3_model/` | PIP3 ODE model and its figures (no data files needed) |
| `2_fit_stiff_directions/` | Single-cell fits + parameter sensitivity, IgG vs. IgG+DNA |
| `3_TNF_alpha/` | TNF-alpha violin-plot figures + `cytokine_quantification_napari/` (upstream napari tool) |
| `4_ERK_figures/` | ERK heatmaps + trace analysis notebooks, plus `erk_ktr_pipeline.py` (upstream napari tool) |

Most folders are Jupyter notebooks. Two folders also include a **napari**
image-analysis script (`cytokine_quantification_napari/` and
`4_ERK_figures/erk_ktr_pipeline.py`) — these are the upstream steps that turn
raw microscopy images into the data the notebooks plot. Each has its own
instructions in its README.

## One-time setup (if you've never run Python notebooks)
1. Install Python 3 from https://www.python.org/downloads/ (check "Add to PATH" on Windows).
2. Install Jupyter:
   ```
   pip install jupyter
   ```
Then follow the README inside whichever folder you want to run.

## How to run any folder
1. Open the folder and read its `README.md`.
2. `pip install -r requirements.txt`
3. Open the notebook in Jupyter and choose **Run → Run All Cells**.
