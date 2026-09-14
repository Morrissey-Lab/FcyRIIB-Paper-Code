import numpy as np
import pandas as pd
import trackpy as tp
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import expand_labels, watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from tifffile import imread
from qtpy.QtWidgets import (QPushButton, QFileDialog, QLabel, QVBoxLayout,
                             QWidget, QSlider, QHBoxLayout, QSpinBox,
                             QCheckBox, QGroupBox, QDoubleSpinBox,
                             QProgressBar)
from qtpy.QtCore import Qt

# =============================================================================
# SETTINGS
# =============================================================================
H2B_CHANNEL      = 1
KTR_CHANNEL      = 0
SEARCH_RANGE     = 50
MEMORY           = 3
MIN_TRACK_LENGTH = 10
SAVE_DIR         = os.path.expanduser('~/Documents')

state = {'data': None, 'well_name': None}
all_results = []

# =============================================================================
# CORE SEGMENTATION
# =============================================================================
def segment_frame(frame_h2b, threshold, min_size, sigma, use_watershed,
                  ws_min_dist, ring_px):
    smoothed = gaussian(frame_h2b.astype(float), sigma=sigma)
    binary   = smoothed > threshold
    binary   = remove_small_objects(binary, min_size=min_size)

    if use_watershed:
        distance = distance_transform_edt(binary)
        coords   = peak_local_max(
            distance,
            min_distance=int(ws_min_dist),
            labels=binary
        )
        mask                  = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers               = label(mask)
        nuclei                = watershed(-distance, markers, mask=binary)
    else:
        nuclei = label(binary)
        nuclei = remove_small_objects(nuclei, min_size=min_size)

    expanded  = expand_labels(nuclei, distance=ring_px)
    cyto_ring = expanded.copy()
    cyto_ring[nuclei > 0] = 0

    return nuclei, cyto_ring, binary

# =============================================================================
# ADAPTIVE THRESHOLD
# =============================================================================
def segment_frame_adaptive(frame_h2b, threshold, min_size, sigma,
                            use_watershed, ws_min_dist, ring_px,
                            step_pct=0.05):
    step  = max(1, int(threshold * step_pct))
    floor = max(1, int(threshold * 0.10))
    t     = threshold

    while t >= floor:
        nuclei, cyto_ring, binary = segment_frame(
            frame_h2b, t, min_size, sigma, use_watershed, ws_min_dist, ring_px
        )
        if nuclei.max() >= 1:
            return nuclei, cyto_ring, binary, t
        t -= step

    return nuclei, cyto_ring, binary, t

# =============================================================================
# COLLECT CURRENT PARAMETERS FROM WIDGETS
# =============================================================================
def get_params():
    return dict(
        threshold = thresh_slider.value(),
        min_size  = minsize_spin.value(),
        sigma     = sigma_spin.value(),
        use_ws    = ws_checkbox.isChecked(),
        ws_dist   = wsdist_spin.value(),
        ring_px   = ring_spin.value(),
        adaptive  = adaptive_checkbox.isChecked(),
    )

# =============================================================================
# SEGMENTATION PREVIEW
# =============================================================================
def update_segmentation(*args):
    data = state['data']
    if data is None:
        return

    p         = get_params()
    frame_idx = preview_slider.value()
    frame_h2b = data[frame_idx, H2B_CHANNEL]

    nuclei, cyto_ring, binary = segment_frame(
        frame_h2b,
        p['threshold'], p['min_size'], p['sigma'],
        p['use_ws'],    p['ws_dist'],  p['ring_px']
    )

    count_label.setText(f'Nuclei detected - frame {frame_idx}: {nuclei.max()}')

    def _upsert(name, arr, add_fn):
        layer_names = [lay.name for lay in viewer.layers]
        if name in layer_names:
            viewer.layers[name].data = arr
        else:
            add_fn(arr)

    _upsert('binary_mask',
            binary.astype(np.uint8),
            lambda a: viewer.add_labels(a, name='binary_mask', opacity=0.25))
    _upsert('nuclei_labels',
            nuclei,
            lambda a: viewer.add_labels(a, name='nuclei_labels'))
    _upsert('cyto_ring',
            cyto_ring,
            lambda a: viewer.add_labels(a, name='cyto_ring', opacity=0.45))

# =============================================================================
# AUTO-THRESHOLD
# =============================================================================
def auto_threshold():
    data = state['data']
    if data is None:
        return
    frame_idx = preview_slider.value()
    smoothed  = gaussian(
        data[frame_idx, H2B_CHANNEL].astype(float),
        sigma=sigma_spin.value()
    )
    otsu_val = int(threshold_otsu(smoothed))
    thresh_slider.setValue(otsu_val)
    thresh_label.setText(f'Threshold: {otsu_val}  (Otsu, frame {frame_idx})')

# =============================================================================
# LOAD TIF
# =============================================================================
def load_tif():
    path, _ = QFileDialog.getOpenFileName(
        None, 'Select TIF file', '', 'TIFF files (*.tif *.tiff)'
    )
    if not path:
        return

    state['well_name'] = os.path.splitext(os.path.basename(path))[0]
    data               = imread(path)
    print(f'Loaded: {path}')
    print(f'  shape={data.shape}  dtype={data.dtype}')
    print(f'  intensity  min={data.min()}  max={data.max()}  mean={data.mean():.1f}')

    if data.ndim == 4 and data.shape[1] > data.shape[-1]:
        data = np.moveaxis(data, -1, 1)

    state['data'] = data
    n_frames      = data.shape[0]

    preview_slider.setRange(0, n_frames - 1)
    preview_slider.setValue(0)
    preview_spin.setRange(0, n_frames - 1)
    preview_spin.setValue(0)
    preview_frame_label.setText(f'Preview frame: 0 / {n_frames - 1}')

    viewer.layers.clear()
    viewer.add_image(data[:, H2B_CHANNEL], name='H2B',
                     colormap='magenta', blending='additive')
    viewer.add_image(data[:, KTR_CHANNEL], name='KTR',
                     colormap='green',   blending='additive')

    img_max  = int(data[:, H2B_CHANNEL].max())
    thresh_slider.setRange(0, img_max)

    smoothed = gaussian(data[0, H2B_CHANNEL].astype(float),
                        sigma=sigma_spin.value())
    otsu_val = int(threshold_otsu(smoothed))
    thresh_slider.setValue(otsu_val)
    thresh_label.setText(f'Threshold: {otsu_val}  (Otsu auto)')

    print(f'H2B range: {data[:, H2B_CHANNEL].min()} - {data[:, H2B_CHANNEL].max()}')
    print(f'{n_frames} frames loaded. Tune controls and scrub through '
          f'frames, then click Confirm & Measure.')

    update_segmentation()

# =============================================================================
# PLOT CN RATIO TRACES  -  Figure 1
# One figure per well. Each tracked cell is a separate line, coloured by
# particle ID. A bold black line shows the population median per frame.
# Saved as a PNG next to the CSV.
# =============================================================================
def plot_cn_traces(tracked, well_name, save_dir):
    particles = tracked['particle'].unique()
    n         = len(particles)
    frames    = np.arange(tracked['frame'].max() + 1)

    colours = cm.tab20(np.linspace(0, 1, min(n, 20)))

    fig, ax = plt.subplots(figsize=(12, 5), num=f'Fig1 - {well_name}')

    for i, pid in enumerate(sorted(particles)):
        cell = tracked[tracked['particle'] == pid].sort_values('frame')
        ax.plot(
            cell['frame'], cell['CN_ratio'],
            color=colours[i % 20],
            linewidth=0.9,
            alpha=0.6,
            label=f'cell {pid}' if n <= 20 else None
        )

    median_trace = (
        tracked.groupby('frame')['CN_ratio']
               .median()
               .reindex(frames)
    )
    ax.plot(
        median_trace.index, median_trace.values,
        color='black', linewidth=2.2, label='median', zorder=5
    )

    ax.axhline(1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('C/N ratio  (KTR cytoplasm / nucleus)', fontsize=12)
    ax.set_title(
        f'{well_name}  -  ERK-KTR C/N ratio\n'
        f'{n} tracked cells  |  '
        f'median final frame = {median_trace.dropna().iloc[-1]:.3f}',
        fontsize=13
    )

    if n <= 20:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.5)

    fig.tight_layout()

    plot_path = os.path.join(save_dir, f'CN_ratio_{well_name}.png')
    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f'Plot saved: {plot_path}')


# =============================================================================
# CONFIRM AND MEASURE  (all frames)
# =============================================================================
def confirm_and_measure():
    data      = state['data']
    well_name = state['well_name']
    if data is None:
        print('No TIF loaded.')
        return

    p        = get_params()
    n_frames = data.shape[0]

    print(f'\nMeasuring {well_name}  ({n_frames} frames)')
    print(f'  sigma={p["sigma"]}  threshold={p["threshold"]}  '
          f'min_size={p["min_size"]}')
    print(f'  watershed={p["use_ws"]}  ws_min_dist={p["ws_dist"]}  '
          f'ring={p["ring_px"]}px')
    print(f'  adaptive threshold={p["adaptive"]}')

    progress_bar.setRange(0, n_frames)
    progress_bar.setValue(0)

    well_results   = []
    adapted_frames = []

    for t in range(n_frames):
        frame_h2b = data[t, H2B_CHANNEL]
        frame_ktr = data[t, KTR_CHANNEL]

        if p['adaptive']:
            nuclei, cyto_ring, _, t_used = segment_frame_adaptive(
                frame_h2b,
                p['threshold'], p['min_size'], p['sigma'],
                p['use_ws'],    p['ws_dist'],  p['ring_px']
            )
            if t_used != p['threshold']:
                adapted_frames.append((t, t_used))
        else:
            nuclei, cyto_ring, _ = segment_frame(
                frame_h2b,
                p['threshold'], p['min_size'], p['sigma'],
                p['use_ws'],    p['ws_dist'],  p['ring_px']
            )

        if nuclei.max() == 0:
            print(f'  WARNING frame {t}: 0 nuclei detected - frame skipped.')
            progress_bar.setValue(t + 1)
            continue

        nuc_ktr  = regionprops_table(
            nuclei, intensity_image=frame_ktr,
            properties=['label', 'intensity_mean', 'centroid', 'area']
        )
        cyto_ktr = regionprops_table(
            cyto_ring, intensity_image=frame_ktr,
            properties=['label', 'intensity_mean']
        )
        nuc_h2b  = regionprops_table(
            nuclei, intensity_image=frame_h2b,
            properties=['label', 'intensity_mean']
        )

        ndf = pd.DataFrame(nuc_ktr).rename(columns={
            'intensity_mean': 'nuclear_ktr_mean',
            'centroid-0':     'y',
            'centroid-1':     'x'
        })
        cdf = pd.DataFrame(cyto_ktr).rename(columns={
            'intensity_mean': 'cyto_ktr_mean'
        })
        hdf = pd.DataFrame(nuc_h2b).rename(columns={
            'intensity_mean': 'h2b_mean'
        })

        df             = ndf.merge(cdf, on='label').merge(hdf, on='label')
        df['CN_ratio'] = df['cyto_ktr_mean'] / df['nuclear_ktr_mean']
        df['frame']    = t
        df['well']     = well_name

        well_results.append(df)
        progress_bar.setValue(t + 1)

        if t % 10 == 0:
            print(f'  frame {t:>4}/{n_frames}  |  '
                  f'nuclei={nuclei.max():<4}  '
                  f'median CN={df["CN_ratio"].median():.3f}')

    if adapted_frames:
        print(f'\n  Adaptive threshold fired on {len(adapted_frames)} frames:')
        for fr, tv in adapted_frames:
            print(f'    frame {fr:>4}  threshold lowered to {tv}')

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------
    well_df = pd.concat(well_results, ignore_index=True)
    tracked = tp.link(
        well_df,
        search_range=SEARCH_RANGE,
        memory=MEMORY,
        pos_columns=['x', 'y'],
        t_column='frame'
    )
    tracked = tp.filter_stubs(tracked, threshold=MIN_TRACK_LENGTH)
    tracked = tracked.reset_index(drop=True)

    n_cells = tracked.particle.nunique()
    print(f'\nTracking complete - {n_cells} cells passed filter '
          f'(>={MIN_TRACK_LENGTH} frames)')

    # -------------------------------------------------------------------------
    # CN ratio overlay
    # -------------------------------------------------------------------------
    _add_cn_overlay(data, tracked, p)

    # -------------------------------------------------------------------------
    # Save CSV + copy pivot table to clipboard
    # -------------------------------------------------------------------------
    save_path = os.path.join(SAVE_DIR, f'CN_ratio_{well_name}.csv')
    tracked.to_csv(save_path, index=False)
    print(f'CSV saved: {save_path}')

    # Build wide-format table: one column group per cell, side by side
    # Columns: frame | cell_0_cyto | cell_0_nuclear | cell_0_CN | cell_1_cyto | ...
    particles_sorted = sorted(tracked['particle'].unique())
    frames_sorted    = sorted(tracked['frame'].unique())

    wide = pd.DataFrame({'frame': frames_sorted})
    for pid in particles_sorted:
        cell = (
            tracked[tracked['particle'] == pid]
            .set_index('frame')[['cyto_ktr_mean', 'nuclear_ktr_mean', 'CN_ratio']]
            .round(4)
            .rename(columns={
                'cyto_ktr_mean':    f'cell_{int(pid)}_cyto',
                'nuclear_ktr_mean': f'cell_{int(pid)}_nuclear',
                'CN_ratio':         f'cell_{int(pid)}_CN',
            })
        )
        wide = wide.merge(cell, on='frame', how='left')

    try:
        wide.to_clipboard(index=False)
        print('Copied to clipboard (wide format, one column group per cell) - paste into Excel or Prism.')
    except Exception as e:
        print(f'Clipboard copy failed ({e}). Data is saved to CSV at {save_path}')

    plot_cn_traces(tracked, well_name, SAVE_DIR)

    all_results.append({'name': well_name, 'data': tracked})
    progress_bar.setValue(0)

# =============================================================================
# SAVE ALL WELLS - combined CSV + combined plot
# =============================================================================
def save_all():
    if not all_results:
        print('No wells measured yet.')
        return

    combined  = pd.concat([r['data'] for r in all_results], ignore_index=True)
    save_path = os.path.join(SAVE_DIR, 'CN_ratio_ALL_WELLS.csv')
    combined.to_csv(save_path, index=False)
    print(f'Combined CSV saved: {save_path}')

    try:
        combined.to_clipboard(index=False)
        print('Combined C/N data copied to clipboard - paste directly into Excel or Prism.')
    except Exception as e:
        print(f'Clipboard copy failed ({e}). Data is saved to CSV at {save_path}')


    # One subplot per well, stacked vertically
    n_wells = len(all_results)
    fig, axes = plt.subplots(n_wells, 1,
                             figsize=(12, 4 * n_wells),
                             sharex=False, squeeze=False)

    for idx, result in enumerate(all_results):
        ax        = axes[idx][0]
        tracked   = result['data']
        wname     = result['name']
        particles = tracked['particle'].unique()
        n         = len(particles)
        colours   = cm.tab20(np.linspace(0, 1, min(n, 20)))

        for i, pid in enumerate(sorted(particles)):
            cell = tracked[tracked['particle'] == pid].sort_values('frame')
            ax.plot(cell['frame'], cell['CN_ratio'],
                    color=colours[i % 20], linewidth=0.9, alpha=0.6)

        median_trace = tracked.groupby('frame')['CN_ratio'].median()
        ax.plot(median_trace.index, median_trace.values,
                color='black', linewidth=2.2, label='median', zorder=5)

        ax.axhline(1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylabel('C/N ratio', fontsize=11)
        ax.set_title(f'{wname}  ({n} cells)', fontsize=12)

    axes[-1][0].set_xlabel('Frame', fontsize=11)
    fig.suptitle('ERK-KTR C/N ratio - all wells', fontsize=14, y=1.01)
    fig.tight_layout()

    plot_path = os.path.join(SAVE_DIR, 'CN_ratio_ALL_WELLS.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Combined plot saved: {plot_path}')

# =============================================================================
# CN RATIO OVERLAY LAYER
# =============================================================================
def _add_cn_overlay(data, tracked, p):
    n_frames  = data.shape[0]
    h, w      = data.shape[2], data.shape[3]
    cn_volume = np.zeros((n_frames, h, w), dtype=np.float32)

    for t in range(n_frames):
        frame_h2b = data[t, H2B_CHANNEL]
        if p['adaptive']:
            nuclei, _, _, _ = segment_frame_adaptive(
                frame_h2b,
                p['threshold'], p['min_size'], p['sigma'],
                p['use_ws'],    p['ws_dist'],  p['ring_px']
            )
        else:
            nuclei, _, _ = segment_frame(
                frame_h2b,
                p['threshold'], p['min_size'], p['sigma'],
                p['use_ws'],    p['ws_dist'],  p['ring_px']
            )
        frame_data = tracked[tracked['frame'] == t]
        for _, row in frame_data.iterrows():
            lbl = int(row['label'])
            cn_volume[t][nuclei == lbl] = row['CN_ratio']

    layer_names = [lay.name for lay in viewer.layers]
    if 'CN_ratio_overlay' in layer_names:
        viewer.layers.remove('CN_ratio_overlay')

    viewer.add_image(
        cn_volume,
        name='CN_ratio_overlay',
        colormap='RdYlGn',
        blending='additive',
        contrast_limits=[0, 3],
        opacity=0.7
    )
    print('CN_ratio_overlay layer added - scrub through frames to QC tracking.')

# =============================================================================
# BUILD WIDGET
# =============================================================================
widget = QWidget()
layout = QVBoxLayout()

btn_load = QPushButton('1. Load TIF')
btn_load.clicked.connect(load_tif)

# Nucleus segmentation
nuc_group  = QGroupBox('Nucleus segmentation (H2B channel)')
nuc_layout = QVBoxLayout()

sigma_row = QHBoxLayout()
sigma_row.addWidget(QLabel('Blur sigma:'))
sigma_spin = QDoubleSpinBox()
sigma_spin.setRange(0.0, 20.0)
sigma_spin.setSingleStep(0.5)
sigma_spin.setValue(2.0)
sigma_spin.setToolTip('Gaussian pre-smoothing. Increase (2-5) for noisy images.')
sigma_spin.valueChanged.connect(update_segmentation)
sigma_row.addWidget(sigma_spin)

thresh_label  = QLabel('Threshold: ---')
thresh_slider = QSlider(Qt.Horizontal)
thresh_slider.setRange(0, 65535)
thresh_slider.setValue(500)
def on_thresh(val):
    thresh_label.setText(f'Threshold: {val}')
    update_segmentation()
thresh_slider.valueChanged.connect(on_thresh)

btn_otsu = QPushButton('Auto-threshold (Otsu) on current frame')
btn_otsu.clicked.connect(auto_threshold)

minsize_row = QHBoxLayout()
minsize_row.addWidget(QLabel('Min nucleus area (px2):'))
minsize_spin = QSpinBox()
minsize_spin.setRange(1, 50000)
minsize_spin.setValue(200)
minsize_spin.setToolTip('Remove objects smaller than this. Raise to cut debris.')
minsize_spin.valueChanged.connect(update_segmentation)
minsize_row.addWidget(minsize_spin)

nuc_layout.addLayout(sigma_row)
nuc_layout.addWidget(thresh_label)
nuc_layout.addWidget(thresh_slider)
nuc_layout.addWidget(btn_otsu)
nuc_layout.addLayout(minsize_row)
nuc_group.setLayout(nuc_layout)

# Watershed
ws_group  = QGroupBox('Watershed  (split touching nuclei)')
ws_layout = QVBoxLayout()

ws_checkbox = QCheckBox('Enable watershed')
ws_checkbox.setChecked(True)
ws_checkbox.stateChanged.connect(update_segmentation)

wsdist_row = QHBoxLayout()
wsdist_row.addWidget(QLabel('Min peak distance (px):'))
wsdist_spin = QSpinBox()
wsdist_spin.setRange(1, 100)
wsdist_spin.setValue(10)
wsdist_spin.setToolTip(
    'Approximate nucleus radius in pixels.\n'
    'Too low: one nucleus split into fragments.\n'
    'Too high: touching nuclei stay merged.'
)
wsdist_spin.valueChanged.connect(update_segmentation)
wsdist_row.addWidget(wsdist_spin)

ws_layout.addWidget(ws_checkbox)
ws_layout.addLayout(wsdist_row)
ws_group.setLayout(ws_layout)

# Cytoplasm ring
cyto_group  = QGroupBox('Cytoplasmic ring (KTR channel)')
cyto_layout = QVBoxLayout()

ring_row = QHBoxLayout()
ring_row.addWidget(QLabel('Ring width (px):'))
ring_spin = QSpinBox()
ring_spin.setRange(1, 100)
ring_spin.setValue(5)
ring_spin.setToolTip('Keep small (3-7 px) in dense monolayers.')
ring_spin.valueChanged.connect(update_segmentation)
ring_row.addWidget(ring_spin)

cyto_layout.addLayout(ring_row)
cyto_group.setLayout(cyto_layout)

# Adaptive threshold
adapt_group  = QGroupBox('Adaptive threshold (dimming correction)')
adapt_layout = QVBoxLayout()

adaptive_checkbox = QCheckBox('Enable adaptive threshold')
adaptive_checkbox.setChecked(True)
adaptive_checkbox.setToolTip(
    'If a frame returns 0 nuclei, steps threshold down in 5% increments\n'
    'to a floor of 10% of the starting value.\n'
    'Adjusted frames are logged to the console after measurement.'
)

adapt_note = QLabel(
    'Steps down 5% per iteration, floor at 10% of starting threshold.\n'
    'Adjusted frames are logged to console after measurement.'
)
adapt_note.setStyleSheet('color: #aaaaaa; font-size: 10px;')

adapt_layout.addWidget(adaptive_checkbox)
adapt_layout.addWidget(adapt_note)
adapt_group.setLayout(adapt_layout)

# Frame scrubber
scrub_group  = QGroupBox('Segmentation preview - scrub through frames')
scrub_layout = QVBoxLayout()

preview_frame_label = QLabel('Preview frame: 0 / ---')

preview_row = QHBoxLayout()

preview_slider = QSlider(Qt.Horizontal)
preview_slider.setRange(0, 0)
preview_slider.setValue(0)
preview_slider.setToolTip('Drag to scrub through frames.')

preview_spin = QSpinBox()
preview_spin.setRange(0, 0)
preview_spin.setValue(0)
preview_spin.setFixedWidth(65)
preview_spin.setToolTip('Type or click arrows to jump to a specific frame.')

# Keep slider and spinbox in sync without double-firing update_segmentation
_scrub_updating = [False]

def on_preview_frame(val):
    if _scrub_updating[0]:
        return
    _scrub_updating[0] = True
    preview_spin.setValue(val)
    _scrub_updating[0] = False
    n = state['data'].shape[0] - 1 if state['data'] is not None else 0
    preview_frame_label.setText(f'Preview frame: {val} / {n}')
    if state['data'] is not None:
        viewer.dims.set_point(0, val)
    update_segmentation()

def on_preview_spin(val):
    if _scrub_updating[0]:
        return
    _scrub_updating[0] = True
    preview_slider.setValue(val)
    _scrub_updating[0] = False
    n = state['data'].shape[0] - 1 if state['data'] is not None else 0
    preview_frame_label.setText(f'Preview frame: {val} / {n}')
    if state['data'] is not None:
        viewer.dims.set_point(0, val)
    update_segmentation()

preview_slider.valueChanged.connect(on_preview_frame)
preview_spin.valueChanged.connect(on_preview_spin)

preview_row.addWidget(preview_slider)
preview_row.addWidget(preview_spin)

scrub_layout.addWidget(preview_frame_label)
scrub_layout.addLayout(preview_row)
scrub_group.setLayout(scrub_layout)

count_label = QLabel('Nuclei detected: ---')
count_label.setStyleSheet('color: #88cc88; font-weight: bold;')

progress_bar = QProgressBar()
progress_bar.setValue(0)
progress_bar.setTextVisible(True)
progress_bar.setFormat('Measuring frame %v / %m')

btn_confirm = QPushButton('2. Confirm & Measure (all frames)')
btn_confirm.clicked.connect(confirm_and_measure)

btn_save_all = QPushButton('3. Save All Wells (combined CSV + plot)')
btn_save_all.clicked.connect(save_all)

layout.addWidget(btn_load)
layout.addWidget(nuc_group)
layout.addWidget(ws_group)
layout.addWidget(cyto_group)
layout.addWidget(adapt_group)
layout.addWidget(scrub_group)
layout.addWidget(count_label)
layout.addWidget(progress_bar)
layout.addWidget(btn_confirm)
layout.addWidget(btn_save_all)
widget.setLayout(layout)

viewer.window.add_dock_widget(widget, name='ERK-KTR Pipeline', area='right')
print('Pipeline ready - click Load TIF to begin.')
