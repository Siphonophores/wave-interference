#!/usr/bin/env python3
"""
Interactive interference-03 UI with exact same math as original.
Generates dot-pattern wave interference with real-time parameter controls.
Based on interference-03.py with UI pattern from ripple-1.py.

Run:
    python interference-03-ui.py

Needs:
    pip install numpy matplotlib svgwrite
"""

import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import svgwrite

# ───── CONFIGURABLE CONSTANTS (same as interference-03.py) ──────────────────
WAVELENGTH    = 2.75                   # λ in coordinate units
SOURCES       = [(-1, 0.0), (1, 0.0)]
GRID_NX, GRID_NY = 200, 200           # grid resolution
NUM_LEVELS    = 100                    # quantization levels
EXTENT        = 5.0                   # half-width of the data window
DOT_SIZE      = 4                     # size of dots in matplotlib preview
SVG_WIDTH     = 800                   # output SVG width in px
SVG_HEIGHT    = 800                   # output SVG height in px
OUTPUT_PREFIX = "interference_dots"   # base name for SVG files
DOT_RADIUS    = 1                     # radius of each dot in px

# SVG output
SAVE_SVG      = True                  # SVG will ONLY be written when you click Export
# ───────────────────────────────────────────────────────────────────────────

def get_next_filename(prefix: str, directory: str = "interference-03-svg") -> str:
    """
    Find existing files matching prefix*.svg in the specified directory and return the next filename
    without overwriting: prefix.svg, prefix_1.svg, prefix_2.svg, ...
    """
    pattern = os.path.join(directory, f"{prefix}*.svg")
    existing = glob.glob(pattern)
    nums = []
    for f in existing:
        name = os.path.basename(f)
        m = re.match(rf"^{prefix}_(\d+)\.svg$", name)
        if m:
            nums.append(int(m.group(1)))
        elif name == f"{prefix}.svg":
            nums.append(0)
    next_n = max(nums) + 1 if nums else 0
    if next_n == 0:
        return f"{prefix}.svg"
    return f"{prefix}_{next_n}.svg"


def compute_field(x, y, sources, wavelength):
    """
    Superpose cosine waves from each point source.
    (Exact same function as interference-03.py)
    """
    k = 2 * np.pi / wavelength
    Z = np.zeros_like(x)
    for sx, sy in sources:
        r = np.hypot(x - sx, y - sy)
        Z += np.cos(k * r)
    return Z


def generate_interference_pattern(wavelength, sources, grid_nx, grid_ny, extent, num_levels):
    """
    Generate interference pattern exactly like interference-03.py
    Returns visible_points (dots to show) for preview and complete grid data
    """
    # 1) create a uniform grid in data-space (same as original)
    xs = np.linspace(-extent, extent, grid_nx)
    ys = np.linspace(-extent, extent, grid_ny)
    X, Y = np.meshgrid(xs, ys)
    
    # 2) compute and normalize the interference field (same as original)
    Z = compute_field(X, Y, sources, wavelength)
    Znorm = (Z - Z.min()) / (Z.max() - Z.min())  # scale to [0,1]
    
    # 3) quantize to integer levels 0..NUM_LEVELS-1 (same as original)
    Q = np.floor(Znorm * num_levels).astype(int)
    
    # 4) extract visible points (even levels only) - EXACTLY like original
    visible_points = []
    
    for i in range(grid_nx):
        for j in range(grid_ny):
            level = Q[j, i]
            if level % 2 == 0:
                # Convert grid indices to data coordinates
                x = xs[i]
                y = ys[j]
                visible_points.append((x, y))
    
    return visible_points, Q, xs, ys


def main():
    """Interactive UI: tweak interference parameters live and export SVG on demand."""
    global WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS
    
    # Initial computation
    visible_points, Q, xs, ys = generate_interference_pattern(
        WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS
    )
    
    # Extract coordinates for plotting
    if visible_points:
        x_vis = [p[0] for p in visible_points]
        y_vis = [p[1] for p in visible_points]
    else:
        x_vis, y_vis = [], []
    
    # ── Figure & layout (sliders left, figure right) ──────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)
    
    # Left column for controls
    left_ax = fig.add_subplot(gs[0, 0])
    left_ax.set_axis_off()
    
    # Right column for plot
    ax_plot = fig.add_subplot(gs[0, 1])
    ax_plot.set_aspect('equal')
    ax_plot.set_xlim(-EXTENT*1.1, EXTENT*1.1)
    ax_plot.set_ylim(-EXTENT*1.1, EXTENT*1.1)
    ax_plot.set_title('Interference Pattern Preview', fontsize=14)
    ax_plot.grid(True, alpha=0.3)
    
    # Plot the interference pattern
    scatter = ax_plot.scatter(x_vis, y_vis, s=DOT_SIZE, c='black', alpha=0.8)
    
    # Add source positions (will be updated dynamically)
    source_scatter = ax_plot.scatter([s[0] for s in SOURCES], [s[1] for s in SOURCES], 
                                   s=80, c='red', marker='x', linewidth=3, label='Sources')
    ax_plot.legend()
    
    # ── Wave Parameters Box ──────────────────────────────────────────────
    ax_wave = plt.axes([0.05, 0.85, 0.35, 0.025])
    ax_extent = plt.axes([0.05, 0.80, 0.35, 0.025])
    
    wave_slider = Slider(ax_wave, "Wavelength", 0.5, 8.0, valinit=WAVELENGTH, valstep=0.05)
    extent_slider = Slider(ax_extent, "Extent", 2.0, 10.0, valinit=EXTENT, valstep=0.1)
    
    # Wave parameters title
    wave_title = plt.text(0.22, 0.88, 'Wave Parameters', transform=fig.transFigure, 
                         fontsize=12, weight='bold', ha='center')
    
    # ── Grid Parameters Box ──────────────────────────────────────────────
    ax_grid_res = plt.axes([0.05, 0.70, 0.35, 0.025])
    ax_levels = plt.axes([0.05, 0.65, 0.35, 0.025])
    
    grid_res_slider = Slider(ax_grid_res, "Grid Resolution", 50, 400, valinit=GRID_NX, valstep=10)
    levels_slider = Slider(ax_levels, "Quantization Levels", 20, 200, valinit=NUM_LEVELS, valstep=5)
    
    # Grid parameters title
    grid_title = plt.text(0.22, 0.73, 'Grid Parameters', transform=fig.transFigure, 
                         fontsize=12, weight='bold', ha='center')
    
    # ── Source Positions Box ─────────────────────────────────────────────
    ax_src1x = plt.axes([0.05, 0.55, 0.35, 0.025])
    ax_src1y = plt.axes([0.05, 0.50, 0.35, 0.025])
    ax_src2x = plt.axes([0.05, 0.45, 0.35, 0.025])
    ax_src2y = plt.axes([0.05, 0.40, 0.35, 0.025])
    
    src1x_slider = Slider(ax_src1x, "Source 1 X", -5.0, 5.0, valinit=SOURCES[0][0], valstep=0.1)
    src1y_slider = Slider(ax_src1y, "Source 1 Y", -5.0, 5.0, valinit=SOURCES[0][1], valstep=0.1)
    src2x_slider = Slider(ax_src2x, "Source 2 X", -5.0, 5.0, valinit=SOURCES[1][0], valstep=0.1)
    src2y_slider = Slider(ax_src2y, "Source 2 Y", -5.0, 5.0, valinit=SOURCES[1][1], valstep=0.1)
    
    # Source positions title
    source_title = plt.text(0.22, 0.58, 'Source Positions', transform=fig.transFigure, 
                           fontsize=12, weight='bold', ha='center')
    
    # ── Stats display ────────────────────────────────────────────────────
    stats_text = plt.text(0.05, 0.30, '', transform=fig.transFigure, 
                         fontsize=10, verticalalignment='top')
    
    # ── Export button ────────────────────────────────────────────────────
    bax = plt.axes([0.15, 0.05, 0.15, 0.08])
    export_btn = Button(bax, 'Export SVG')
    
    def update_stats():
        """Update the statistics display."""
        total_points = GRID_NX * GRID_NY
        visibility_pct = (len(visible_points)/total_points*100) if total_points else 0
        stats_text.set_text(
            f"Grid Points: {total_points}\n"
            f"Visible Points: {len(visible_points)}\n"
            f"Visibility: {visibility_pct:.1f}%\n"
            f"─────────────────\n"
            f"Wavelength: {WAVELENGTH:.2f}\n"
            f"Extent: {EXTENT:.1f}\n"
            f"Grid: {GRID_NX}×{GRID_NY}\n"
            f"Levels: {NUM_LEVELS}\n"
            f"─────────────────\n"
            f"Source 1: ({SOURCES[0][0]:.1f}, {SOURCES[0][1]:.1f})\n"
            f"Source 2: ({SOURCES[1][0]:.1f}, {SOURCES[1][1]:.1f})\n"
            f"Distance: {np.hypot(SOURCES[1][0]-SOURCES[0][0], SOURCES[1][1]-SOURCES[0][1]):.1f}\n"
            f"─────────────────\n"
            f"Process:\n"
            f"1. Uniform Grid\n"
            f"2. Wave Superposition\n"
            f"3. Quantization\n"
            f"4. Even Levels Only"
        )
    
    update_stats()
    
    # ── Callbacks ────────────────────────────────────────────────────────
    def redraw():
        """Recompute pattern and refresh the plot."""
        global visible_points, xs, ys
        
        visible_points, Q, xs, ys = generate_interference_pattern(
            WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS
        )
        
        if visible_points:
            x_vis = [p[0] for p in visible_points]
            y_vis = [p[1] for p in visible_points]
        else:
            x_vis, y_vis = [], []
        
        # Update scatter plot
        scatter.set_offsets(np.column_stack([x_vis, y_vis]) if x_vis else np.empty((0, 2)))
        
        # Update source positions
        source_scatter.set_offsets(np.array([[s[0], s[1]] for s in SOURCES]))
        
        # Update plot limits
        ax_plot.set_xlim(-EXTENT*1.1, EXTENT*1.1)
        ax_plot.set_ylim(-EXTENT*1.1, EXTENT*1.1)
        
        update_stats()
        fig.canvas.draw_idle()
    
    def on_slider_change(val):
        global WAVELENGTH, EXTENT, GRID_NX, GRID_NY, NUM_LEVELS, SOURCES
        WAVELENGTH = wave_slider.val
        EXTENT = extent_slider.val
        GRID_NX = GRID_NY = int(grid_res_slider.val)  # Keep square grid
        NUM_LEVELS = int(levels_slider.val)
        SOURCES = [(src1x_slider.val, src1y_slider.val), (src2x_slider.val, src2y_slider.val)]
        redraw()
    
    wave_slider.on_changed(on_slider_change)
    extent_slider.on_changed(on_slider_change)
    grid_res_slider.on_changed(on_slider_change)
    levels_slider.on_changed(on_slider_change)
    src1x_slider.on_changed(on_slider_change)
    src1y_slider.on_changed(on_slider_change)
    src2x_slider.on_changed(on_slider_change)
    src2y_slider.on_changed(on_slider_change)
    
    def on_export(event):
        """Save SVG with current parameters using EXACT same data as preview."""
        if not SAVE_SVG:
            return
        
        # Update parameters from sliders
        global WAVELENGTH, EXTENT, GRID_NX, GRID_NY, NUM_LEVELS, SOURCES
        WAVELENGTH = wave_slider.val
        EXTENT = extent_slider.val
        GRID_NX = GRID_NY = int(grid_res_slider.val)
        NUM_LEVELS = int(levels_slider.val)
        SOURCES = [(src1x_slider.val, src1y_slider.val), (src2x_slider.val, src2y_slider.val)]
        
        # Get output filename in the interference-03-svg directory
        output_file = os.path.join("interference-03-svg", get_next_filename(OUTPUT_PREFIX))
        
        # Use the EXACT same computation as the preview
        export_visible_points, export_Q, export_xs, export_ys = generate_interference_pattern(
            WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS
        )
        
        # Prepare SVG document
        dwg = svgwrite.Drawing(output_file, size=(SVG_WIDTH, SVG_HEIGHT))
        grp = dwg.g(id="interference_dots", fill="black", stroke="none")
        
        # Convert each visible point from data coordinates to SVG coordinates
        # This ensures exact correspondence with the preview
        for x_data, y_data in export_visible_points:
            # Convert from data coordinates (-EXTENT to +EXTENT) to SVG coordinates (0 to SVG_WIDTH/HEIGHT)
            px = ((x_data + EXTENT) / (2 * EXTENT)) * SVG_WIDTH
            py = SVG_HEIGHT - ((y_data + EXTENT) / (2 * EXTENT)) * SVG_HEIGHT
            grp.add(dwg.circle(center=(px, py), r=DOT_RADIUS))
        
        # Save SVG
        dwg.add(grp)
        dwg.save()
        print(f"✓ Wrote dot pattern to {output_file} ({len(export_visible_points)} dots, grid {GRID_NX}×{GRID_NY}, levels={NUM_LEVELS})")
    
    export_btn.on_clicked(on_export)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()