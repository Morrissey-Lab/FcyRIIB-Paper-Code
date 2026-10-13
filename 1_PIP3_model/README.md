# PIP3 Model & Figures

Generates the PIP3 model figures using an ODE simulation. **No input data files are needed** — the model computes everything from equations.

## What's in this folder
- `V4_PIP3_Model_and_Figures.ipynb` — the notebook that runs the model and makes the figures
- `requirements.txt` — the Python packages needed

## How to run
1. Install the packages (one time):
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```
   jupyter notebook V4_PIP3_Model_and_Figures.ipynb
   ```
3. Run all cells (menu: **Run → Run All Cells**).

Figures are saved as PDF/PNG files into this folder when you run the cells.

## Important: run order for the minimal-model cell
The **minimal-model cell** (the one titled `MINIMAL MODEL`) reuses the variable
names `y0`, `t`, and `params` for its own small 5-variable model. This
overwrites the full-model `y0`/`t`/`params` defined in **Cell 1 (Setup)**.

Because of this, if you run that minimal-model cell and then go back to run the
**PIP3 peak-time contour** cell or the **IFFL delay** cell, they will error
(they expect Cell 1's full-model values).

**Fix:** just re-run **Cell 1 (Setup)** before running those cells again. A
clean top-to-bottom run also works as long as you re-run Cell 1 after the
minimal-model cell. (Everything runs fine; this is only about variable
reuse, not a problem with the results.)
