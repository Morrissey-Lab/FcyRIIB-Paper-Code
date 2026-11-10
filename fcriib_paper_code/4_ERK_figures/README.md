# ERK Figures

The two ERK notebooks that go together: the heatmap figures and the trace/peak analysis.

## What's in this folder
- `fig_6_erk.ipynb` — ERK heatmaps and summary figures
- `Erk_KTR_Trace_Analysis.ipynb` — per-cell ERK trace analysis (peaks, stats)
- `requirements.txt` — the Python packages needed
- data files (all included; see table below)

## Data files (all included)
These are already in this folder. The notebooks read them from the same folder:

| File           | Used by                                   |
|----------------|-------------------------------------------|
| `1K.csv`       | `fig_6_erk.ipynb`                         |
| `ALL CONC.csv` | `fig_6_erk.ipynb`, `Erk_KTR_Trace_Analysis.ipynb` |
| `LPS.csv`      | `fig_6_erk.ipynb`                         |
| `2A.csv`       | `fig_6_erk.ipynb`                         |
| `Blank.csv`    | `fig_6_erk.ipynb`                         |

## How to run
1. Install the packages (one time):
   ```
   pip install -r requirements.txt
   ```
2. Open a notebook, e.g.:
   ```
   jupyter notebook fig_6_erk.ipynb
   ```
3. Run all cells (menu: **Run → Run All Cells**).

`Erk_KTR_Trace_Analysis.ipynb` also writes result tables (`erk_results.csv`,
`erk_summary.csv`) into this folder. Figures are saved here as PNG/PDF.

---

## Bonus: `erk_ktr_pipeline.py` (napari image-analysis pipeline)

This is the **upstream** step that turns raw microscopy movies into ERK KTR
traces. It runs inside napari (not as a notebook) and is fully self-contained —
it needs no accessory files, only a TIFF you select when you run it.

**What it needs:** a multi-frame TIFF (you pick it in the file dialog).
**What it makes:** `CN_ratio_<well>.csv`, `CN_ratio_ALL_WELLS.csv`, and plots,
saved to your **Documents** folder.

How to run:
1. Install the packages (one time), ideally in a fresh environment:
   ```
   pip install "napari[all]" trackpy tifffile scikit-image scipy pandas matplotlib
   ```
2. Open napari (`napari`), open the console (`>_`, bottom-left), and paste in
   the whole contents of `erk_ktr_pipeline.py` — or run `python erk_ktr_pipeline.py`.
3. Use the button it adds to select your TIFF; results are written to Documents.

