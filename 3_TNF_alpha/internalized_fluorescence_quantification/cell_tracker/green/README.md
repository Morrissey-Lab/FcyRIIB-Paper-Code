# Panel: green ("membrane segmentation" for Green - 488)

Part of the `cell tracker` package (`cell segmentation/ilastik/cell tracker/`)
— this panel is the `green/` subfolder inside it.

Cytoplasm/membrane marker (Green - 488) ONLY, classified by ilastik Pixel
Classification into three classes: `membrane`, `background`, and
`cell_contact` (where two touching cells' walls press together). Single-
channel panel — ilastik never sees a nuclear channel (Erika small beads,
the project using this panel, has no nuclear stain at all).

**History:** this panel originally reused `red`'s Red-561-trained classifier
applied to the Green-488 channel, to avoid training a second model. On real
data that didn't generalize well enough (probability values were
near-saturated, over-classifying almost the entire frame as membrane), so
it's now trained independently instead — same idea as `red`, just its own
`.ilp` and its own brush strokes.

## No nuclear channel, no nucleus-seeded watershed

Because there's no nuclear stain, instances are **not** split via
nucleus-seeded watershed the way `red`'s consuming project (Erika TNFa)
does it. There's no nucleus to seed a watershed from, so
`cell_tracker.instance_segmentation.cells_from_membrane_mask` is used
instead — cells are split directly from the membrane mask's shape via
distance-transform watershed (the classic technique for separating touching
blobs with no separate marker channel). Call
`run_segmentation_pipeline(..., nuclear_channel=None)` (or just omit
`nuclear_channel` — it defaults to `None`) to use this pathway.

## Why a third "cell_contact" class

The base 2-class (membrane/background) version of this pathway splits
touching cells purely geometrically — it finds the "widest" point of each
blob in the mask (a distance-transform peak) and assumes two peaks far
enough apart are separate cells (`min_seed_distance_px`). That's a decent
fallback but it's guessing; it has no real information about where one
cell's edge actually ends and the next begins when they're pressed
together.

`cell_contact` fixes that at the source: ilastik is trained to directly
recognize the thin boundary line where two touching cells' membranes press
against each other, as its own class distinct from both "inside a cell"
(membrane) and "empty space" (background). At segmentation time, any pixel
classified as contact gets carved OUT of the foreground mask before the
distance transform runs — turning ilastik's learned boundary into a real
physical gap between the two cells, so they get their own genuine
distance-transform peaks instead of one merged blob that
`min_seed_distance_px` has to awkwardly split apart. See
`ilastik.contact_class` in `panel.yaml` and
`cell_tracker.instance_segmentation.cells_from_membrane_mask`'s docstring
for the implementation.

This is why `fill_holes` is off for this panel (see Tuning below) — a
contact gap is, by construction, a hole fully enclosed by foreground on
both sides, and `fill_holes` would fill it straight back in.

## Status

`membrane_segmentation.ilp` here does not exist yet — create it fresh in
the ilastik GUI (see below) and save it into this folder. After that, every
project using this panel (currently just Erika small beads) reuses it with
no retraining.

## Training checklist

Because this panel is single-channel, training is simple — no "Stack
Across" dialog needed at all.

1. `tiffs/` already has 20 single-channel Green-488 TIFFs (one
   `..._488.tif` per point) exported from Erika small beads' test file —
   ready to load straight into ilastik. (To regenerate them, or export more
   from other data, run the export cell in
   `cell tracker/notebooks/00_train_and_test_panel_green.ipynb`.)
2. Open ilastik -> **New Project -> Pixel Classification**. Save the new
   project as `membrane_segmentation.ilp` directly into this folder
   (`cell segmentation/ilastik/cell tracker/green/`).
3. In Input Data, **"Add New" -> "Add separate image(s)..."** and select
   as many `tiffs/..._488.tif` files as you want to train on — each becomes
   its own lane. No stacking/volume dialog needed since there's only one
   channel per file.
4. Create exactly 3 labels, **in this order** (ilastik assigns pixel-class
   indices 0, 1, 2... in creation order, and this must match `panel.yaml`'s
   `ilastik.pixel_classes`): `membrane`, `background`, `cell_contact`. If
   you want a different order, edit `panel.yaml` to match.
5. Brush-label a handful of clear examples of each class across a few
   images, train, and visually check the live prediction overlay:
   - `membrane`: inside a cell body, same as before.
   - `background`: clearly empty space, not just "not obviously a cell".
     Given the old reused classifier over-called almost everything as
     membrane, this class especially needs clean, unambiguous examples so
     the model has real contrast to learn from.
   - `cell_contact`: the thin line/zone right where two touching cells'
     edges press against each other — pick spots with a real, judgeable
     boundary between two macrophages, not just any spot inside a cluster.
     This is the class that makes touching-cell separation reliable instead
     of geometric guesswork, so it's worth a few extra minutes finding
     good, unambiguous examples across several images.
6. Set the Prediction Export source to **Probabilities**, save the project.
7. Re-run `notebooks/00_train_and_test_panel_green.ipynb`'s test cell to
   confirm `run_segmentation_pipeline` produces sane per-cell instance
   masks (distance-transform watershed constrained by this panel's
   membrane probability map) on the test file.

## Tuning

Once trained, check the QC panel (two rows: membrane threshold mask, final
cell instances) and adjust `panel.yaml`'s `segmentation:` block, then
re-run with `force_segment=True` (no need to re-run ilastik):

- `smoothing_sigma` (default `0`, off) — Gaussian-smooths the membrane
  probability map before thresholding. Off by default for this panel —
  smoothing blurs exactly the kind of thin, precise boundary the
  `cell_contact` class is meant to draw, so leave it at 0 unless the
  post-contact-carving mask looks genuinely speckly/noisy. If you do raise
  it, expect it to soften contact gaps too, working against them.
- `probability_threshold` (default 0.5) — cutoff on the membrane
  probability used to build the binary mask that gets split into cells.
  **Must be strictly less than 1** — probabilities cap at 1.0, so a
  threshold of exactly 1 (or higher) can never be exceeded and silently
  produces a permanently empty mask (0 cells, 0% membrane area) regardless
  of the image.
- `min_cell_size_px` (default 20) — drops final cell regions smaller than
  this.
- `min_seed_distance_px` (default 10) — minimum pixel distance required
  between two distance-map peaks to count as separate cells. Lower it if
  touching/tightly-packed cells are merging into one; raise it if single
  cells are getting split into pieces. A purely geometric fallback — with
  `contact_class` trained and working, this should matter a lot less than
  it did before.
- `contact_class` (`cell_contact`) — which `pixel_classes` entry marks the
  touching-cell boundary (see "Why a third cell_contact class" above).
  Pixels classified as contact are carved out of the foreground mask before
  splitting, giving touching cells real, independent distance-transform
  peaks. Set to `null` to fall back to plain geometric splitting.
- `contact_probability_threshold` (default `null`, falls back to
  `probability_threshold`) — separate confidence cutoff for the contact
  class, if it needs a different bar than membrane does.
- `fill_holes` (default `false`, **off**) — fills any background-classified
  hole in the mask that's fully enclosed by foreground (scipy's
  `binary_fill_holes`). Off for this panel because it's incompatible with
  `contact_class`: a contact gap IS a hole fully enclosed by foreground on
  both sides, so `fill_holes` would fill it straight back in and silently
  undo the split. If you ever turn `contact_class` off, `fill_holes` is
  safe to turn back on — it can help with interior misclassification the
  same way it does for `red` (see that panel's README), it just can't
  coexist with contact-based splitting.
- `max_membrane_area_fraction` — frame-level exclusion, flags (doesn't
  drop) any point whose membrane-positive area exceeds this. Currently
  `null` (disabled) for Erika small beads, the only project using this
  panel — macrophage coverage legitimately varies a lot by design, so a
  fixed area cutoff isn't a meaningful quality signal here. Set a value
  (0-1 scale) if that changes for this or a future project. There's no
  nucleus-count equivalent here (no nuclear channel).
