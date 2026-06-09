#!/usr/bin/env python3
"""
vtk_field_plot.py
-----------------
Renders isosurface contours of a 3D scalar field stored in a VTK XML
StructuredGrid (.vts) file and saves the result as a PNG.

Intended for quick in-Galaxy visual validation of volumetric field data
(e.g. Bz_total from the MFEM Time Series Demo or Magpy Field Solver)
before feeding the same file into the Omniverse pipeline.

Inputs
------
  --input-vtk    : VTK XML StructuredGrid file (.vts)
  --field-name   : Scalar field to visualise (default: Bz_total)
  --n-contours   : Number of isosurface levels (default: 7)
  --colormap     : Matplotlib colormap name (default: plasma)
  --output-png   : Output PNG path

Usage
-----
  python vtk_field_plot.py --input-vtk field.vts --output-png plot.png
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")

import pyvista as pv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render VTK StructuredGrid scalar field as isosurface PNG"
    )
    parser.add_argument("--input-vtk",   required=True, help="VTK XML StructuredGrid (.vts) file")
    parser.add_argument("--field-name",  default="Bz_total", help="Scalar field name to visualise")
    parser.add_argument("--n-contours",  type=int,   default=7,      help="Number of isosurface contour levels")
    parser.add_argument("--colormap",    default="plasma",            help="Matplotlib colormap name")
    parser.add_argument("--output-png",  required=True, help="Output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the VTK file — pyvista detects StructuredGrid from the XML header
    print(f"Reading: {args.input_vtk}")
    mesh = pv.read(args.input_vtk)
    print(f"  Type      : {type(mesh).__name__}")
    print(f"  Dimensions: {mesh.dimensions}")
    print(f"  Fields    : {list(mesh.point_data.keys())}")

    if args.field_name not in mesh.point_data:
        available = list(mesh.point_data.keys())
        print(
            f"ERROR: field '{args.field_name}' not found in VTK file.\n"
            f"Available fields: {available}"
        )
        sys.exit(1)

    field_data = mesh.point_data[args.field_name]
    f_min = float(field_data.min())
    f_max = float(field_data.max())
    print(f"  {args.field_name}: min={f_min:.4e}  max={f_max:.4e}")

    # Generate isosurfaces
    iso = mesh.contour(isosurfaces=args.n_contours, scalars=args.field_name)
    print(f"  Isosurface points: {iso.n_points}")

    if iso.n_points == 0:
        print("WARNING: No isosurface points generated — field may be uniform or n_contours too low.")
        print("Falling back to slice plot through midplane.")
        iso = mesh.slice(normal="z")

    # Off-screen render — three views: front, top, isometric
    pv.start_xvfb()  # needed in headless Docker environments
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 600))

    titles = ["Front (XZ)", "Top (XY)", "Isometric"]
    cameras = [
        (0, -1, 0),   # looking along -Y
        (0, 0,  1),   # looking down -Z (top)
        (1, -1, 1),   # isometric
    ]

    for col, (title, cam_dir) in enumerate(zip(titles, cameras)):
        pl.subplot(0, col)
        pl.add_mesh(
            iso,
            scalars=args.field_name,
            cmap=args.colormap,
            show_scalar_bar=(col == 2),
            scalar_bar_args={"title": f"{args.field_name} (T)", "fmt": "%.3e"},
        )
        pl.add_axes()
        pl.view_vector(cam_dir)
        pl.add_title(title, font_size=10)

    pl.screenshot(args.output_png)
    pl.close()
    print(f"Written: {args.output_png}")


if __name__ == "__main__":
    main()
