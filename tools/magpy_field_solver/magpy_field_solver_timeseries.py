"""
magpy_field_solver_timeseries.py
---------------------------------
Biot-Savart magnetic field solver for a tokamak coil set driven by a
BATCH of time-series current setpoints.

Input A — coil geometry CSV (from parametric_coil_generator)
Input B — current time-series CSV (from coil_demo_sim.py).
          Contains MULTIPLE timesteps covering a full demo sequence.

NOTE: This script processes all timesteps in the CSV in one batch run.
      A separate single-timestep variant (for real-time / streaming use,
      e.g. driven by MQTT) will be developed subsequently.

Voltage -> current conversion
------------------------------
The synthetic data CSV stores PSU setpoint voltages (V), not currents (A).
  TF banks (ch0, ch1) - Lianshu PSU:  gain supplied via --tf-gain  [A/V]
  PF banks (ch2, ch3) - ZJIVnV PSU:   gain supplied via --pf-gain  [A/V]

Outputs
-------
  --output-gif      : animated GIF of the R-Z (poloidal) field map over time
  --output-png      : static PNG of the R-Z peak-field frame
  --output-gif-mid  : animated GIF of the X-Y (midplane, z=0) field map over time
  --output-png-mid  : static PNG of the midplane peak-field frame
"""

import argparse
import csv
import numpy as np
import magpylib as magpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Coil geometry loader
# ---------------------------------------------------------------------------

def load_coils(csv_path):
    """
    Read the coil geometry CSV produced by parametric_coil_generator.
    Returns dict: {coil_id: {"type": "TF"|"PF", "vertices": [(x,y,z), ...]}}
    """
    coils = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["coil_id"]
            if cid not in coils:
                coils[cid] = {"type": row["coil_type"], "vertices": []}
            coils[cid]["vertices"].append(
                (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            )
    return coils


# ---------------------------------------------------------------------------
# Time-series current loader
# ---------------------------------------------------------------------------

def load_timeseries(csv_path, tf_gain, pf_gain):
    """
    Read the MiniMak synthetic current CSV (batch of timesteps).
    Converts PSU voltages -> coil currents using the supplied gain factors.

    NOTE: This function ingests ALL timesteps from the file at once (batch mode).
          For single-timestep streaming use, see magpy_field_solver_realtime.py.
    """
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "t_s":   float(row["t_s"]),
                "I_tf1": float(row["V_ch0"]) * tf_gain,
                "I_tf2": float(row["V_ch1"]) * tf_gain,
                "I_pf1": float(row["V_ch2"]) * pf_gain,
                "I_pf2": float(row["V_ch3"]) * pf_gain,
            })
    print(f"Loaded {len(records)} timesteps  "
          f"(t = {records[0]['t_s']:.3f} s to {records[-1]['t_s']:.3f} s)")
    return records


# ---------------------------------------------------------------------------
# Magpylib source builder
# ---------------------------------------------------------------------------

def build_sources(coils):
    sources = {}
    for cid, data in coils.items():
        sources[cid] = magpy.current.Polyline(
            current=0.0,
            vertices=data["vertices"],
        )
    return sources


def set_currents(sources, coils, record):
    for cid, src in sources.items():
        if coils[cid]["type"] == "TF":
            src.current = record["I_tf1"]
        elif cid == "PF_upper":
            src.current = record["I_pf1"]
        elif cid == "PF_lower":
            src.current = record["I_pf2"]


# ---------------------------------------------------------------------------
# Field grids
# ---------------------------------------------------------------------------

def build_rz_grid(r_range, z_range, grid_n):
    """R-Z poloidal cross-section grid at Y=0."""
    r = np.linspace(r_range[0], r_range[1], grid_n)
    z = np.linspace(z_range[0], z_range[1], grid_n)
    R, Z = np.meshgrid(r, z)
    points = np.column_stack([R.ravel(), np.zeros(R.size), Z.ravel()])
    return R, Z, points


def build_midplane_grid(r_max, grid_n):
    """X-Y midplane grid at Z=0 (top-down view)."""
    v = np.linspace(-r_max, r_max, grid_n)
    X, Y = np.meshgrid(v, v)
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return X, Y, points


def compute_rz_field(sources, points, grid_shape):
    collection = magpy.Collection(*sources.values(), override_parent=True)
    B = magpy.getB(collection, points)
    Br   = B[:, 0].reshape(grid_shape)
    Bz   = B[:, 2].reshape(grid_shape)
    Bmag = np.sqrt(Br**2 + Bz**2)
    return Br, Bz, Bmag


def compute_midplane_field(sources, points, grid_shape):
    """Return Bx, By, |B_total| on the Z=0 midplane."""
    collection = magpy.Collection(*sources.values(), override_parent=True)
    B = magpy.getB(collection, points)
    Bx   = B[:, 0].reshape(grid_shape)
    By   = B[:, 1].reshape(grid_shape)
    Bmag = np.linalg.norm(B, axis=1).reshape(grid_shape)
    return Bx, By, Bmag


# ---------------------------------------------------------------------------
# Coil cross-section markers for R-Z plot
# ---------------------------------------------------------------------------

def coil_markers(coils):
    markers = []
    for cid, data in coils.items():
        verts = np.array(data["vertices"])
        tol = np.max(np.abs(verts[:, 1])) * 0.05 + 1e-6
        mask = np.abs(verts[:, 1]) < tol
        color = "steelblue" if data["type"] == "TF" else "tomato"
        markers.append((verts[mask, 0], verts[mask, 2], color))
    return markers


# ---------------------------------------------------------------------------
# Frame renderers
# ---------------------------------------------------------------------------

def render_rz(ax, R, Z, Bmag_i, Br_i, Bz_i, vmin, vmax, markers, title):
    ax.cla()
    ax.contourf(R, Z, np.log10(Bmag_i + 1e-12),
                levels=np.linspace(vmin, vmax, 50), cmap="inferno", extend="both")
    ax.streamplot(R[0, :], Z[:, 0], Br_i, Bz_i,
                  color="white", linewidth=0.6, density=1.5, arrowsize=0.8)
    for xs, zs, color in markers:
        ax.scatter(xs, zs, color=color, s=8, zorder=5)
    ax.legend(handles=[Patch(color="steelblue", label="TF"),
                       Patch(color="tomato",    label="PF")],
              loc="upper right", fontsize=7)
    ax.set_xlabel("R (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=8)


def render_midplane(ax, X, Y, Bmag_i, Bx_i, By_i, vmin, vmax, title):
    ax.cla()
    ax.contourf(X, Y, np.log10(Bmag_i + 1e-12),
                levels=np.linspace(vmin, vmax, 50), cmap="viridis", extend="both")
    ax.streamplot(X[0, :], Y[:, 0], Bx_i, By_i,
                  color="white", linewidth=0.6, density=1.2, arrowsize=0.8)
    # Reference circle at approximate magnetic axis radius
    theta = np.linspace(0, 2 * np.pi, 200)
    r_axis = np.sqrt((X**2 + Y**2).mean())
    ax.plot(r_axis * np.cos(theta), r_axis * np.sin(theta),
            "--", color="white", lw=0.5, alpha=0.4)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=8)


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def run_batch(coils, sources, timeseries, R, Z, rz_points, X, Y, mid_points, args):
    """
    Compute B field for ALL timesteps (batch mode), then write:
      - R-Z animated GIF + peak PNG
      - Midplane (X-Y, Z=0) animated GIF + peak PNG

    NOTE: all timesteps are held in memory. For single-step streaming see
    magpy_field_solver_realtime.py.
    """
    n_steps    = len(timeseries)
    rz_shape   = R.shape
    mid_shape  = X.shape

    print(f"Computing B field for {n_steps} timesteps on "
          f"{args.grid_n}x{args.grid_n} grid - please wait...")

    Br_all    = np.zeros((n_steps, *rz_shape))
    Bz_all    = np.zeros((n_steps, *rz_shape))
    Bmag_rz   = np.zeros((n_steps, *rz_shape))
    Bx_all    = np.zeros((n_steps, *mid_shape))
    By_all    = np.zeros((n_steps, *mid_shape))
    Bmag_mid  = np.zeros((n_steps, *mid_shape))

    for i, record in enumerate(timeseries):
        set_currents(sources, coils, record)
        Br_all[i], Bz_all[i], Bmag_rz[i]  = compute_rz_field(sources, rz_points, rz_shape)
        Bx_all[i], By_all[i], Bmag_mid[i] = compute_midplane_field(sources, mid_points, mid_shape)
        if (i + 1) % 10 == 0 or i == n_steps - 1:
            print(f"  Step {i+1}/{n_steps}  t={record['t_s']:.3f} s")

    # Shared colour scale across both views
    all_bmag = np.concatenate([Bmag_rz.ravel(), Bmag_mid.ravel()])
    vmin = np.log10(all_bmag.max() * 1e-4 + 1e-12)
    vmax = np.log10(all_bmag.max() + 1e-12)
    markers = coil_markers(coils)

    def frame_title(rec):
        return (f"t = {rec['t_s']:.3f} s  |  "
                f"I_TF = {rec['I_tf1']:.1f} A  "
                f"I_PF+ = {rec['I_pf1']:.1f} A  "
                f"I_PF- = {rec['I_pf2']:.1f} A")

    # --- R-Z animation ---
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.colorbar(plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="inferno"),
        ax=ax, label="log10 |B| (T)")

    def update_rz(i):
        render_rz(ax, R, Z, Bmag_rz[i], Br_all[i], Bz_all[i],
                  vmin, vmax, markers,
                  f"R-Z field  |  {frame_title(timeseries[i])}")
        return []

    FuncAnimation(fig, update_rz, frames=n_steps, interval=100, blit=False
                  ).save(args.output_gif, writer=PillowWriter(fps=10))
    plt.close()
    print(f"Written: {args.output_gif}")

    # --- Midplane animation ---
    fig_m, ax_m = plt.subplots(figsize=(7, 7))
    plt.colorbar(plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis"),
        ax=ax_m, label="log10 |B| (T)")

    def update_mid(i):
        render_midplane(ax_m, X, Y, Bmag_mid[i], Bx_all[i], By_all[i],
                        vmin, vmax,
                        f"Midplane (Z=0)  |  {frame_title(timeseries[i])}")
        return []

    FuncAnimation(fig_m, update_mid, frames=n_steps, interval=100, blit=False
                  ).save(args.output_gif_mid, writer=PillowWriter(fps=10))
    plt.close()
    print(f"Written: {args.output_gif_mid}")

    # --- R-Z peak PNG ---
    peak = int(np.argmax(Bmag_rz.max(axis=(1, 2))))
    fig2, ax2 = plt.subplots(figsize=(7, 8))
    plt.colorbar(plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="inferno"),
        ax=ax2, label="log10 |B| (T)")
    render_rz(ax2, R, Z, Bmag_rz[peak], Br_all[peak], Bz_all[peak],
              vmin, vmax, markers,
              f"R-Z peak frame  |  {frame_title(timeseries[peak])}")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150, format="png")
    plt.close()
    print(f"Written: {args.output_png}  (peak at t={timeseries[peak]['t_s']:.3f} s)")

    # --- Midplane peak PNG ---
    peak_m = int(np.argmax(Bmag_mid.max(axis=(1, 2))))
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    plt.colorbar(plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis"),
        ax=ax3, label="log10 |B| (T)")
    render_midplane(ax3, X, Y, Bmag_mid[peak_m], Bx_all[peak_m], By_all[peak_m],
                    vmin, vmax,
                    f"Midplane peak  |  {frame_title(timeseries[peak_m])}")
    plt.tight_layout()
    plt.savefig(args.output_png_mid, dpi=150, format="png")
    plt.close()
    print(f"Written: {args.output_png_mid}  (peak at t={timeseries[peak_m]['t_s']:.3f} s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Biot-Savart field solver - BATCH time-series mode.\n"
            "Accepts a full MiniMak current CSV (multiple timesteps) and the\n"
            "coil geometry CSV, and outputs R-Z and midplane animations + PNGs.\n"
            "For single-timestep real-time use see magpy_field_solver_realtime.py."
        )
    )
    parser.add_argument("--input-coils",     required=True,
                        help="Coil geometry CSV from parametric_coil_generator")
    parser.add_argument("--input-currents",  required=True,
                        help="Current time-series CSV from coil_demo_sim.py (all timesteps)")
    parser.add_argument("--tf-gain",         type=float, default=2000.0,
                        help="TF PSU gain [A/V]")
    parser.add_argument("--pf-gain",         type=float, default=1000.0,
                        help="PF PSU gain [A/V]")
    parser.add_argument("--r-min",           type=float, default=0.0,   help="R grid min [m]")
    parser.add_argument("--r-max",           type=float, default=0.5,   help="R grid max [m]")
    parser.add_argument("--z-min",           type=float, default=-0.4,  help="Z grid min [m]")
    parser.add_argument("--z-max",           type=float, default=0.4,   help="Z grid max [m]")
    parser.add_argument("--grid-n",          type=int,   default=40,
                        help="Grid resolution NxN")
    parser.add_argument("--output-gif",      required=True,
                        help="R-Z animated GIF output path")
    parser.add_argument("--output-png",      required=True,
                        help="R-Z peak-frame PNG output path")
    parser.add_argument("--output-gif-mid",  required=True,
                        help="Midplane animated GIF output path")
    parser.add_argument("--output-png-mid",  required=True,
                        help="Midplane peak-frame PNG output path")
    args = parser.parse_args()

    coils      = load_coils(args.input_coils)
    timeseries = load_timeseries(args.input_currents, args.tf_gain, args.pf_gain)
    sources    = build_sources(coils)

    R, Z, rz_points  = build_rz_grid(
        r_range=(args.r_min, args.r_max),
        z_range=(args.z_min, args.z_max),
        grid_n=args.grid_n,
    )
    X, Y, mid_points = build_midplane_grid(args.r_max, args.grid_n)

    run_batch(coils, sources, timeseries, R, Z, rz_points, X, Y, mid_points, args)


if __name__ == "__main__":
    main()
