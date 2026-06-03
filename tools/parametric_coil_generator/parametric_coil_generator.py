import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def make_tf_coil(phi_deg, major_radius, coil_radius, n_points):
    phi = np.radians(phi_deg)
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])
    centre = major_radius * r_hat
    pts = centre + coil_radius * (np.outer(np.cos(theta), r_hat) + np.outer(np.sin(theta), z_hat))
    return pts[:, 0], pts[:, 1], pts[:, 2]


def make_pf_coil(z_offset, pf_coil_radius, n_points):
    phi = np.linspace(0, 2 * np.pi, n_points + 1)
    x = pf_coil_radius * np.cos(phi)
    y = pf_coil_radius * np.sin(phi)
    z = np.full_like(phi, z_offset)
    return x, y, z


def generate_coils(major_radius, tf_coil_radius, n_tf_coils, pf_coil_radius, pf_coil_z, n_points):
    coils = []

    tf_angles = np.linspace(0, 360, n_tf_coils, endpoint=False)
    for i, phi in enumerate(tf_angles):
        x, y, z = make_tf_coil(phi, major_radius, tf_coil_radius, n_points)
        bank = 1 if i < n_tf_coils // 2 else 2
        coils.append({"type": "TF", "id": f"TF_{i:02d}", "bank": bank, "phi_deg": phi,
                       "x": x, "y": y, "z": z})

    for sign, label in [(+1, "upper"), (-1, "lower")]:
        x, y, z = make_pf_coil(sign * pf_coil_z, pf_coil_radius, n_points)
        coils.append({"type": "PF", "id": f"PF_{label}", "z_offset": sign * pf_coil_z,
                       "x": x, "y": y, "z": z})

    return coils


def write_csv(coils, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coil_id", "coil_type", "bank", "point_index", "x_m", "y_m", "z_m"])
        for coil in coils:
            bank = coil.get("bank", "")
            for i, (x, y, z) in enumerate(zip(coil["x"], coil["y"], coil["z"])):
                writer.writerow([coil["id"], coil["type"], bank, i,
                                  f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


def write_png(coils, params, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    tf_plotted = False
    pf_plotted = False
    for coil in coils:
        x, y, z = coil["x"], coil["y"], coil["z"]
        if coil["type"] == "TF":
            ax.plot(x, y, z, color="steelblue", linewidth=1.0,
                    label="TF coil" if not tf_plotted else "")
            tf_plotted = True
        else:
            ax.plot(x, y, z, color="tomato", linewidth=2.5,
                    label="PF coil" if not pf_plotted else "")
            pf_plotted = True

    z_range = params["tf_coil_radius_m"] * 1.3
    ax.plot([0, 0], [0, 0], [-z_range, z_range],
            color="black", linewidth=0.8, linestyle="--", label="Machine axis")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Tokamak coil geometry\n"
        f"R₀={params['major_radius_m']} m  |  "
        f"{params['n_tf_coils']} TF coils (r={params['tf_coil_radius_m']} m)  |  "
        f"2 PF coils (z=±{params['pf_coil_z_m']} m)"
    )
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format='png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parametric tokamak coil generator")
    parser.add_argument("--major-radius",    type=float, default=0.23,
                        help="Major radius R0 [m]")
    parser.add_argument("--tf-coil-radius",  type=float, default=0.075,
                        help="TF coil loop radius [m]")
    parser.add_argument("--n-tf-coils",      type=int,   default=8,
                        help="Number of TF coils")
    parser.add_argument("--pf-coil-radius",  type=float, default=0.12,
                        help="PF coil ring radius from machine axis [m]")
    parser.add_argument("--pf-coil-z",       type=float, default=0.15,
                        help="PF coil distance from midplane [m]")
    parser.add_argument("--n-points",        type=int,   default=200,
                        help="Points per coil loop")
    parser.add_argument("--output-png",      required=True,
                        help="Output PNG plot file path")
    parser.add_argument("--output-csv",      required=True,
                        help="Output CSV (tab-separated) coordinate file path")
    args = parser.parse_args()

    params = {
        "major_radius_m":   args.major_radius,
        "tf_coil_radius_m": args.tf_coil_radius,
        "n_tf_coils":       args.n_tf_coils,
        "pf_coil_radius_m": args.pf_coil_radius,
        "pf_coil_z_m":      args.pf_coil_z,
        "n_points":         args.n_points,
    }

    coils = generate_coils(
        major_radius=args.major_radius,
        tf_coil_radius=args.tf_coil_radius,
        n_tf_coils=args.n_tf_coils,
        pf_coil_radius=args.pf_coil_radius,
        pf_coil_z=args.pf_coil_z,
        n_points=args.n_points,
    )

    write_csv(coils, args.output_csv)
    write_png(coils, params, args.output_png)

    print(f"Written: {args.output_csv}")
    print(f"Written: {args.output_png}")


if __name__ == "__main__":
    main()
