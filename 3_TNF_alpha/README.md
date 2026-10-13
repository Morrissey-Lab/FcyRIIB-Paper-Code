# TNF-alpha Figure

Makes the TNF-alpha violin-plot figures from single-cell intensity data.

## What's in this folder
- `TNF_alpha_fig.ipynb` — the notebook that makes the figure
- `tnfa_data_iso_subtracted.csv` — the input data (included)
- `requirements.txt` — the Python packages needed
- `cytokine_quantification_napari/` — the upstream napari image-analysis tool
  that produces this kind of data (self-contained; see its own README)

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
