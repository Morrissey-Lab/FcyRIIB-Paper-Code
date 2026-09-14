# Cytokine Quantification (napari + Ilastik)

Interactive napari tool that measures intracellular cytokine signal per cell.
It segments cells using an Ilastik membrane probability map, lets you scrub
through fields of view, and exports per-cell and summary CSVs.

This is the **upstream image-analysis** step for the TNF-alpha figure. It runs
inside napari, not as a Jupyter notebook.

## What's in this folder
- `cytokine_quantification_napari.py` — the script
- `requirements.txt` — the Python packages needed
- `membrane_segmentation.ilp` — **you must add this** (see below)

## Before you run — two things to set up
1. **Add the Ilastik project file.** Put `membrane_segmentation.ilp` in this
   folder. (No `.yaml`/`.json` is needed — just this one `.ilp`.)
2. **Install Ilastik** (the program) from https://www.ilastik.org/download.html.
   This is separate software; the script calls it to make the probability maps.

## How to run
1. Install the Python packages (one time), ideally in a fresh environment:
   ```
   pip install -r requirements.txt
   ```
2. Open napari:
   ```
   napari
   ```
3. Open the napari console (the `>_` button, bottom-left), then paste in the
   whole contents of `cytokine_quantification_napari.py`.
   (Or run it directly: `python cytokine_quantification_napari.py`.)
4. In the **Setup** panel, point it to:
   - your Ilastik executable (`ilastik.exe` on Windows, `run_ilastik.sh` on Mac/Linux)
   - the `membrane_segmentation.ilp` in this folder
5. **Add conditions** → select your channel TIFFs (they auto-group into FOVs).
6. Scrub through FOVs, then **Run batch** to export the per-cell CSV, summary
   CSV, and plots.

## Notes
- Your image TIFFs are chosen at run time — they are **not** stored in the repo.
- Channel mapping is set near the top of the script (`CH_DAPI`, `CH_CYTOKINE`,
  `CH_MEMBRANE`); edit those if your channel order differs.
