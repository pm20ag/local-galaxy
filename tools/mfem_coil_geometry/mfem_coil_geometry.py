#!/usr/bin/env python3
"""
mfem_coil_geometry.py
------------------------
Standalone MiniMak coil geometry tool.

Defines the electrical and physical parameters for each MiniMak coil bank
and outputs a geometry CSV compatible with the MFEM Time Series Demo tool.

Extracted from mfem_coil_demo.py so that geometry definition is decoupled
from voltage sequence generation — the geometry CSV can now be produced once
and reused across multiple MFEM timeseries runs.

Outputs
-------
  --output-csv  : geometry CSV  (coil_id, coil_name, radius_m, turns, resistance_ohm)
  --output-png  : 3-D coil geometry plot
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


COIL_NAMES = {
    "tf1": "TF bank 1",
    "tf2": "TF bank 2",
    "pf1": "PF bank 1",
    "pf2": "PF bank 2",
}


# ---------------------------------------------------------------------------
# Geometry builder
# ---------------------------------------------------------------------------

def build_geometry(args):
    return {
        "tf1": {"radius": args.tf1_radius, "turns": args.tf_turns, "resistance": args.tf_resistance},
        "tf2": {"radius": args.tf2_radius, "turns": args.tf_turns, "resistance": args.tf_resistance},
        "pf1": {"radius": args.pf1_radius, "turns": args.pf_turns, "resistance": args.pf_resistance},
        "pf2": {"radius": args.pf2_radius, "turns": args.pf_turns, "resistance": args.pf_resistance},
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_geometry_csv(path, geometry):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["coil_id", "coil_name", "radius_m", "turns", "resistance_ohm"])
        for coil_id, coil in geometry.items():
            writer.writerow([
                coil_id,
                COIL_NAMES[coil_id],
                coil["radius"],
                coil["turns"],
                coil["resistance"],
            ])
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Geometry plot
# ---------------------------------------------------------------------------

def write_geometry_plot(path, geometry):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    theta = [i * 2 * math.pi / 100 for i in range(101)]

    # TF coils — 8 coils arranged toroidally, split into two banks of 4
    major_radius = 0.23
    tf_loop_radius = geometry["tf1"]["radius"]
    n_tf = 8
    for i in range(n_tf):
        phi = 2 * math.pi * i / n_tf
        x_c = major_radius * math.cos(phi)
        y_c = major_radius * math.sin(phi)
        e_r = (math.cos(phi), math.sin(phi), 0.0)
        e_z = (0.0, 0.0, 1.0)

        x_vals = [x_c + tf_loop_radius * math.cos(t) * e_r[0] + tf_loop_radius * math.sin(t) * e_z[0] for t in theta]
        y_vals = [y_c + tf_loop_radius * math.cos(t) * e_r[1] + tf_loop_radius * math.sin(t) * e_z[1] for t in theta]
        z_vals = [       tf_loop_radius * math.cos(t) * e_r[2] + tf_loop_radius * math.sin(t) * e_z[2] for t in theta]

        bank_label = "TF bank 1" if i < n_tf // 2 else "TF bank 2"
        color = "steelblue" if i < n_tf // 2 else "cornflowerblue"
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=1.0,
                label=bank_label if i in (0, n_tf // 2) else None)

    # PF coils — two horizontal rings
    pf_data = [("pf1", geometry["pf1"]["radius"], 0.15), ("pf2", geometry["pf2"]["radius"], -0.15)]
    for coil_id, r, z0 in pf_data:
        ax.plot([r * math.cos(t) for t in theta],
                [r * math.sin(t) for t in theta],
                [z0] * len(theta),
                color="tomato", linewidth=2.5,
                label=COIL_NAMES[coil_id])

    # Machine axis
    ax.plot([0, 0], [0, 0], [-0.2, 0.2], linestyle="--", linewidth=0.8,
            color="black", label="Machine axis")

    axis_limit = max(geometry["tf1"]["radius"], geometry["pf1"]["radius"], 0.3) + 0.1
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-0.25, 0.25)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"MiniMak coil geometry\n"
        f"TF r={geometry['tf1']['radius']} m | "
        f"PF1 r={geometry['pf1']['radius']} m | "
        f"PF2 r={geometry['pf2']['radius']} m"
    )
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, format="png")
    plt.close()
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MiniMak coil geometry definition tool")

    parser.add_argument("--tf1-radius",    type=float, required=True, help="TF bank 1 coil radius [m]")
    parser.add_argument("--tf2-radius",    type=float, required=True, help="TF bank 2 coil radius [m]")
    parser.add_argument("--pf1-radius",    type=float, required=True, help="PF bank 1 coil radius [m]")
    parser.add_argument("--pf2-radius",    type=float, required=True, help="PF bank 2 coil radius [m]")
    parser.add_argument("--tf-turns",      type=int,   required=True, help="Number of turns per TF coil")
    parser.add_argument("--pf-turns",      type=int,   required=True, help="Number of turns per PF coil")
    parser.add_argument("--tf-resistance", type=float, required=True, help="TF coil resistance [ohm]")
    parser.add_argument("--pf-resistance", type=float, required=True, help="PF coil resistance [ohm]")
    parser.add_argument("--output-csv",    required=True, help="Output geometry CSV path")
    parser.add_argument("--output-png",    required=True, help="Output geometry plot PNG path")

    args = parser.parse_args()

    geometry = build_geometry(args)
    write_geometry_csv(args.output_csv, geometry)
    write_geometry_plot(args.output_png, geometry)


if __name__ == "__main__":
    main()
