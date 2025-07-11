#!/usr/bin/env python3
"""
Interactive interference-04 UI with circular cropping.
Generates dot-pattern wave interference with circular crop applied to both preview and export.
Based on interference-03-ui.py with added circular masking functionality.

Run:
    python interference-04-ui.py

Needs:
    pip install numpy matplotlib svgwrite
"""

import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import svgwrite

# ───── CONFIGURABLE CONSTANTS ──────────────────────────────────────────────
WAVELENGTH    = 2.75                   # λ in coordinate units
SOURCES       = [(-1, 0.0), (1, 0.0)]
GRID_NX, GRID_NY = 200, 200           # grid resolution
NUM_LEVELS    = 100                    # quantization levels
EXTENT        = 5.0                   # half-width of the data window
DOT_SIZE      = 4                     # size of dots in matplotlib preview
SVG_WIDTH     = 800                   # output SVG width in px
SVG_HEIGHT    = 800                   # output SVG height in px
OUTPUT_PREFIX = "interference_dots_circular"   # base name for SVG files
DOT_RADIUS    = 1                     # radius of each dot in px
CIRCLE_RADIUS = 4.5                   # radius of the circular crop in data units
CULLING_INTENSITY = 0.5               # intensity of radial culling (0=none, 1=maximum)

# SVG output
SAVE_SVG      = True                  # SVG will ONLY be written when you click Export
# ───────────────────────────────────────────────────────────────────────────

def get_next_filename(prefix: str, directory: str = "interference-04-svg") -> str:
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


def is_inside_circle(x, y, center_x=0, center_y=0, radius=CIRCLE_RADIUS):
    """
    Check if point (x, y) is inside the circular crop area.
    """
    return np.hypot(x - center_x, y - center_y) <= radius


def radial_cull_probability(x, y, center_x=0, center_y=0, max_radius=CIRCLE_RADIUS, intensity=CULLING_INTENSITY):
    """
    Calculate the probability of culling (removing) a dot based on its distance from center.
    Returns a value between 0 (never cull) and 1 (always cull).
    Points at center have low cull probability, points at edge have high cull probability.
    """
    if intensity == 0:
        return 0.0  # No culling
    
    # Calculate normalized distance from center (0 to 1)
    distance = np.hypot(x - center_x, y - center_y)
    normalized_distance = min(distance / max_radius, 1.0)
    
    # Apply culling intensity with a smooth curve
    # Using quadratic function for smooth gradient
    cull_probability = intensity * (normalized_distance ** 2)
    
    return min(cull_probability, 1.0)


def generate_interference_pattern_circular(wavelength, sources, grid_nx, grid_ny, extent, num_levels, circle_radius, culling_intensity=CULLING_INTENSITY):
    """
    Generate interference pattern with circular cropping and radial density gradient applied.
    Returns visible_points (dots to show) for preview and complete grid data.
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
    
    # 4) extract visible points (even levels only) with circular cropping and radial culling
    visible_points = []
    
    # Set random seed for reproducible results during UI interaction
    np.random.seed(42)
    
    for i in range(grid_nx):
        for j in range(grid_ny):
            level = Q[j, i]
            if level % 2 == 0:
                # Convert grid indices to data coordinates
                x = xs[i]
                y = ys[j]
                
                # Apply circular crop
                if is_inside_circle(x, y, 0, 0, circle_radius):
                    # Apply radial culling
                    cull_prob = radial_cull_probability(x, y, 0, 0, circle_radius, culling_intensity)
                    random_value = np.random.random()
                    
                    # Keep the dot if random value is greater than cull probability
                    if random_value > cull_prob:
                        visible_points.append((x, y))
    
    return visible_points, Q, xs, ys


def main():
    """Interactive UI: tweak interference parameters live and export circular SVG on demand."""
    global WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS, CIRCLE_RADIUS, CULLING_INTENSITY
    
    # Initial computation
    visible_points, Q, xs, ys = generate_interference_pattern_circular(
        WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS, CIRCLE_RADIUS, CULLING_INTENSITY
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
    ax_plot.set_title('Circular Interference Pattern Preview', fontsize=14)
    ax_plot.grid(True, alpha=0.3)
    
    # Plot the interference pattern
    scatter = ax_plot.scatter(x_vis, y_vis, s=DOT_SIZE, c='black', alpha=0.8)
    
    # Add circular boundary visualization
    circle_boundary = Circle((0, 0), CIRCLE_RADIUS, fill=False, color='blue', 
                           linewidth=2, linestyle='--', alpha=0.7, label='Circular Crop')
    ax_plot.add_patch(circle_boundary)
    
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
    
    # ── Circular Crop Box ─────────────────────────────────────────────────
    ax_circle_radius = plt.axes([0.05, 0.35, 0.35, 0.025])
    ax_culling_intensity = plt.axes([0.05, 0.30, 0.35, 0.025])
    
    circle_radius_slider = Slider(ax_circle_radius, "Circle Radius", 1.0, 8.0, valinit=CIRCLE_RADIUS, valstep=0.1)
    culling_intensity_slider = Slider(ax_culling_intensity, "Density Gradient", 0.0, 1.0, valinit=CULLING_INTENSITY, valstep=0.05)
    
    # Circular crop title
    circle_title = plt.text(0.22, 0.38, 'Circular Crop & Density', transform=fig.transFigure, 
                           fontsize=12, weight='bold', ha='center')
    
    # ── Stats display ────────────────────────────────────────────────────
    stats_text = plt.text(0.05, 0.20, '', transform=fig.transFigure, 
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
            f"Circle Radius: {CIRCLE_RADIUS:.1f}\n"
            f"Density Gradient: {CULLING_INTENSITY:.2f}\n"
            f"─────────────────\n"
            f"Source 1: ({SOURCES[0][0]:.1f}, {SOURCES[0][1]:.1f})\n"
            f"Source 2: ({SOURCES[1][0]:.1f}, {SOURCES[1][1]:.1f})\n"
            f"Distance: {np.hypot(SOURCES[1][0]-SOURCES[0][0], SOURCES[1][1]-SOURCES[0][1]):.1f}\n"
            f"─────────────────\n"
            f"Process:\n"
            f"1. Uniform Grid\n"
            f"2. Wave Superposition\n"
            f"3. Quantization\n"
            f"4. Circular Crop\n"
            f"5. Radial Culling\n"
            f"6. Even Levels Only"
        )
    
    update_stats()
    
    # ── Callbacks ────────────────────────────────────────────────────────
    def redraw():
        """Recompute pattern and refresh the plot."""
        global visible_points, xs, ys
        
        visible_points, Q, xs, ys = generate_interference_pattern_circular(
            WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS, CIRCLE_RADIUS, CULLING_INTENSITY
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
        
        # Update circular boundary
        circle_boundary.set_radius(CIRCLE_RADIUS)
        
        # Update plot limits
        ax_plot.set_xlim(-EXTENT*1.1, EXTENT*1.1)
        ax_plot.set_ylim(-EXTENT*1.1, EXTENT*1.1)
        
        update_stats()
        fig.canvas.draw_idle()
    
    def on_slider_change(val):
        global WAVELENGTH, EXTENT, GRID_NX, GRID_NY, NUM_LEVELS, SOURCES, CIRCLE_RADIUS, CULLING_INTENSITY
        WAVELENGTH = wave_slider.val
        EXTENT = extent_slider.val
        GRID_NX = GRID_NY = int(grid_res_slider.val)  # Keep square grid
        NUM_LEVELS = int(levels_slider.val)
        SOURCES = [(src1x_slider.val, src1y_slider.val), (src2x_slider.val, src2y_slider.val)]
        CIRCLE_RADIUS = circle_radius_slider.val
        CULLING_INTENSITY = culling_intensity_slider.val
        redraw()
    
    wave_slider.on_changed(on_slider_change)
    extent_slider.on_changed(on_slider_change)
    grid_res_slider.on_changed(on_slider_change)
    levels_slider.on_changed(on_slider_change)
    src1x_slider.on_changed(on_slider_change)
    src1y_slider.on_changed(on_slider_change)
    src2x_slider.on_changed(on_slider_change)
    src2y_slider.on_changed(on_slider_change)
    circle_radius_slider.on_changed(on_slider_change)
    culling_intensity_slider.on_changed(on_slider_change)
    
    def on_export(event):
        """Save circular SVG with current parameters using EXACT same data as preview."""
        if not SAVE_SVG:
            return
        
        # Update parameters from sliders
        global WAVELENGTH, EXTENT, GRID_NX, GRID_NY, NUM_LEVELS, SOURCES, CIRCLE_RADIUS, CULLING_INTENSITY
        WAVELENGTH = wave_slider.val
        EXTENT = extent_slider.val
        GRID_NX = GRID_NY = int(grid_res_slider.val)
        NUM_LEVELS = int(levels_slider.val)
        SOURCES = [(src1x_slider.val, src1y_slider.val), (src2x_slider.val, src2y_slider.val)]
        CIRCLE_RADIUS = circle_radius_slider.val
        CULLING_INTENSITY = culling_intensity_slider.val
        
        # Get output filename in the interference-04-svg directory
        output_file = os.path.join("interference-04-svg", get_next_filename(OUTPUT_PREFIX))
        
        # Use the EXACT same computation as the preview
        export_visible_points, export_Q, export_xs, export_ys = generate_interference_pattern_circular(
            WAVELENGTH, SOURCES, GRID_NX, GRID_NY, EXTENT, NUM_LEVELS, CIRCLE_RADIUS, CULLING_INTENSITY
        )
        
        # Prepare SVG document
        dwg = svgwrite.Drawing(output_file, size=(SVG_WIDTH, SVG_HEIGHT))
        
        # Add circular clipping path
        clip_path = dwg.defs.add(dwg.clipPath(id="circularClip"))
        svg_center = SVG_WIDTH / 2  # Assuming square SVG
        svg_radius = (CIRCLE_RADIUS / EXTENT) * (SVG_WIDTH / 2)
        clip_path.add(dwg.circle(center=(svg_center, svg_center), r=svg_radius))
        
        # Create group with clipping applied
        grp = dwg.g(id="interference_dots_circular", fill="black", stroke="none", 
                   clip_path="url(#circularClip)")
        
        # Convert each visible point from data coordinates to SVG coordinates
        # This ensures exact correspondence with the preview
        for x_data, y_data in export_visible_points:
            # Convert from data coordinates (-EXTENT to +EXTENT) to SVG coordinates (0 to SVG_WIDTH/HEIGHT)
            px = ((x_data + EXTENT) / (2 * EXTENT)) * SVG_WIDTH
            py = SVG_HEIGHT - ((y_data + EXTENT) / (2 * EXTENT)) * SVG_HEIGHT
            grp.add(dwg.circle(center=(px, py), r=DOT_RADIUS))
        
        # Save SVG (no boundary circle in export)
        dwg.add(grp)
        dwg.save()
        print(f"✓ Wrote circular dot pattern to {output_file} ({len(export_visible_points)} dots, grid {GRID_NX}×{GRID_NY}, levels={NUM_LEVELS}, radius={CIRCLE_RADIUS:.1f}, culling={CULLING_INTENSITY:.2f})")
    
    export_btn.on_clicked(on_export)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()