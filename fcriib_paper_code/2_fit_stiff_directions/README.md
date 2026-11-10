# Fit Stiff Directions

Fits single-cell traces and analyzes parameter sensitivity ("stiff directions") for the IgG vs. IgG+DNA conditions.

## What's in this folder
- `fit_stiff_directions_v2.ipynb` — the notebook
- `IgG.csv` — data for the IgG condition (included)
- `IgG+DNA.csv` — data for the IgG+DNA condition (included)
- `unified_fits_20000cap_zerofloor.npz` — saved fit results (included; see below)
- `requirements.txt` — the Python packages needed

> Note: the data file was called `IgGDNA.csv` elsewhere; it has been renamed to
> `IgG+DNA.csv` here because that is the exact name the notebook looks for. Keep
> the name as-is.

## How to run
1. Install the packages (one time):
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```
   jupyter notebook fit_stiff_directions_v2.ipynb
   ```
3. Run all cells (menu: **Run → Run All Cells**).

The included `.npz` file holds pre-computed fits, so the notebook loads them
instantly instead of re-fitting (which is slow). If you delete that file, the
notebook will re-fit from scratch on the next run and save a new one. Figures
are saved into this folder as PNG/PDF.
