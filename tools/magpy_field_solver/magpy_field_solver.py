import argparse
import csv
import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_coils(tsv_path):
    """Read coils.tsv and return a dict of {coil_id: {type, vertices}}."""
    coils = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = row["coil_id"]
            if cid not in coils:
                coils[cid] = {"type": row["coil_type"], "vertices": []}
            coils[cid]["vertices"].append(
                (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            )
    return coils


def build_sources(coils, tf_current, pf_current):
    """Create a magpylib Collection of Polyline sources from coil data."""
    sources = []
    for cid, data in coils.items():
        current = tf_current if data["type"] == "TF" else pf_current
        src = magpy.current.Polyline(
            current=current,
            vertices=data["vertices"],
        )
        src._coil_id = cid
        sources.append(src)
    return sources


def compute_rz_field(sources, r_range, z_range, grid_n):
    """
    Compute B field on a 2D R-Z grid (Y=0 plane, X=R).
    Returns R, Z meshgrids and Br, Bz component arrays.
    """
    r = np.linspace(r_range[0], r_range[1], grid_n)
    z = np.linspace(z_range[0], z_range[1], grid_n)
    R, Z = np.meshgrid(r, z)

    # Observation points in 3D: (R, 0, Z)
    points = np.column_stack([R.ravel(), np.zeros(R.size), Z.ravel()])
    collection = magpy.Collection(*sources)
    B = magpy.getB(collection, points)

    Br = B[:, 0].reshape(R.shape)   # X component = radial in Y=0 plane
    Bz = B[:, 2].reshape(R.shape)   # Z component
    Bmag = np.sqrt(Br**2 + Bz**2)

    return R, Z, Br, Bz, Bmag


def plot_field(R, Z, Br, Bz, Bmag, coils, output_path):
    fig, ax = plt.subplots(figsize=(8, 9))

    # Field magnitude as filled contour background
    cf = ax.contourf(R, Z, np.log10(Bmag + 1e-12), levels=50, cmap="inferno")
    plt.colorbar(cf, ax=ax, label="log₁₀ |B| (T)")

    # Field line streamplot
    r = R[0, :]
    z = Z[:, 0]
    ax.streamplot(r, z, Br, Bz, color="white", linewidth=0.6,
                  density=1.5, arrowsize=0.8)

    # Mark coil cross-sections (where each coil crosses the Y=0 plane)
    for cid, data in coils.items():
        verts = np.array(data["vertices"])
        # Points near the Y=0 plane
        mask = np.abs(verts[:, 1]) < (np.max(np.abs(verts[:, 1])) * 0.05 + 1e-6)
        xs = verts[mask, 0]
        zs = verts[mask, 2]
        color = "steelblue" if data["type"] == "TF" else "tomato"
        ax.scatter(xs, zs, color=color, s=8, zorder=5)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue", label="TF coil cross-section"),
        Patch(color="tomato",    label="PF coil cross-section"),
    ], loc="upper right", fontsize=8)

    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Magnetic field — R-Z plane (Y = 0)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format="png")
    plt.close()
    print(f"Written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Magpylib Biot-Savart field solver for tokamak coils")
    parser.add_argument("--input-csv",    required=True,       help="Coil coordinate TSV from parametric_coil_generator")
    parser.add_argument("--tf-current",   type=float, default=10000.0, help="Current in each TF coil [A]")
    parser.add_argument("--pf-current",   type=float, default=5000.0,  help="Current in each PF coil [A]")
    parser.add_argument("--r-min",        type=float, default=0.0,     help="R grid minimum [m]")
    parser.add_argument("--r-max",        type=float, default=0.5,     help="R grid maximum [m]")
    parser.add_argument("--z-min",        type=float, default=-0.4,    help="Z grid minimum [m]")
    parser.add_argument("--z-max",        type=float, default=0.4,     help="Z grid maximum [m]")
    parser.add_argument("--grid-n",       type=int,   default=60,      help="Grid resolution (NxN)")
    parser.add_argument("--output-png",   required=True,               help="Output field map PNG path")
    args = parser.parse_args()

    coils = load_coils(args.input_csv)
    print(f"Loaded {len(coils)} coils: {list(coils.keys())}")

    sources = build_sources(coils, args.tf_current, args.pf_current)
    print(f"Computing B field on {args.grid_n}x{args.grid_n} R-Z grid...")

    R, Z, Br, Bz, Bmag = compute_rz_field(
        sources,
        r_range=(args.r_min, args.r_max),
        z_range=(args.z_min, args.z_max),
        grid_n=args.grid_n,
    )

    plot_field(R, Z, Br, Bz, Bmag, coils, args.output_png)


if __name__ == "__main__":
    main()
