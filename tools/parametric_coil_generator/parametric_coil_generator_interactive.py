import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
        coils.append({"type": "TF", "id": f"TF_{i:02d}", "phi_deg": phi,
                      "x": x, "y": y, "z": z})

    for sign, label in [(+1, "upper"), (-1, "lower")]:
        x, y, z = make_pf_coil(sign * pf_coil_z, pf_coil_radius, n_points)
        coils.append({"type": "PF", "id": f"PF_{label}", "z_offset": sign * pf_coil_z,
                      "x": x, "y": y, "z": z})

    return coils


def plot_coils(coils, major_radius, tf_coil_radius, n_tf_coils, pf_coil_radius, pf_coil_z):
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

    z_range = tf_coil_radius * 1.3
    ax.plot([0, 0], [0, 0], [-z_range, z_range],
            color="black", linewidth=0.8, linestyle="--", label="Machine axis")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Tokamak coil geometry\n"
        f"R₀={major_radius} m  |  {n_tf_coils} TF coils (r={tf_coil_radius} m)  |  "
        f"2 PF coils (r={pf_coil_radius} m, z=±{pf_coil_z} m)"
    )
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Parametric tokamak coil generator — interactive plot")
    parser.add_argument("--major-radius",   type=float, default=0.23,  help="Major radius R0 [m]")
    parser.add_argument("--tf-coil-radius", type=float, default=0.075, help="TF coil loop radius [m]")
    parser.add_argument("--n-tf-coils",     type=int,   default=8,     help="Number of TF coils")
    parser.add_argument("--pf-coil-radius", type=float, default=0.12,  help="PF coil ring radius from machine axis [m]")
    parser.add_argument("--pf-coil-z",      type=float, default=0.15,  help="PF coil distance from midplane [m]")
    parser.add_argument("--n-points",       type=int,   default=200,   help="Points per coil loop")
    args = parser.parse_args()

    coils = generate_coils(
        major_radius=args.major_radius,
        tf_coil_radius=args.tf_coil_radius,
        n_tf_coils=args.n_tf_coils,
        pf_coil_radius=args.pf_coil_radius,
        pf_coil_z=args.pf_coil_z,
        n_points=args.n_points,
    )

    plot_coils(coils, args.major_radius, args.tf_coil_radius,
               args.n_tf_coils, args.pf_coil_radius, args.pf_coil_z)


if __name__ == "__main__":
    main()
