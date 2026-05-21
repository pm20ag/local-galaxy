import argparse
import os
import numpy as np
import meshio

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--heating_input',  type=str, required=True)
parser.add_argument('--dpa_input',      type=str, required=True)
parser.add_argument('--heating_exodus', type=str, required=True)
parser.add_argument('--dpa_exodus',     type=str, required=True)
args = parser.parse_args()

# ── Resolve paths ─────────────────────────────────────────────────────────────
args.heating_input  = os.path.abspath(args.heating_input)
args.dpa_input      = os.path.abspath(args.dpa_input)
args.heating_exodus = os.path.abspath(args.heating_exodus)
args.dpa_exodus     = os.path.abspath(args.dpa_exodus)

# ── Load CSVs from OpenMC ─────────────────────────────────────────────────────
def load_csv(filepath):
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    x   = data['x']
    y   = data['y']
    z   = data['z']
    val = data[data.dtype.names[3]]
    return x, y, z, val

print("Loading heating field...")
hx, hy, hz, h_val = load_csv(args.heating_input)

print("Loading DPA field...")
dx, dy, dz, d_val = load_csv(args.dpa_input)

# ── Verify grids match ────────────────────────────────────────────────────────
if not (np.allclose(hx, dx) and np.allclose(hy, dy) and np.allclose(hz, dz)):
    raise ValueError("Heating and DPA meshes do not share the same coordinates.")

x, y, z = hx, hy, hz

# ── Build mesh ────────────────────────────────────────────────────────────────
print("Building mesh...")

ux = np.unique(x)
uy = np.unique(y)
uz = np.unique(z)

nx, ny, nz = len(ux), len(uy), len(uz)

points = np.column_stack([x, y, z])

idx_map = {}
for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
    ix = np.searchsorted(ux, xi)
    iy = np.searchsorted(uy, yi)
    iz = np.searchsorted(uz, zi)
    idx_map[(ix, iy, iz)] = i

cells = []
for ix in range(nx - 1):
    for iy in range(ny - 1):
        for iz in range(nz - 1):
            try:
                n000 = idx_map[(ix,   iy,   iz  )]
                n100 = idx_map[(ix+1, iy,   iz  )]
                n110 = idx_map[(ix+1, iy+1, iz  )]
                n010 = idx_map[(ix,   iy+1, iz  )]
                n001 = idx_map[(ix,   iy,   iz+1)]
                n101 = idx_map[(ix+1, iy,   iz+1)]
                n111 = idx_map[(ix+1, iy+1, iz+1)]
                n011 = idx_map[(ix,   iy+1, iz+1)]
                cells.append([n000, n100, n110, n010,
                               n001, n101, n111, n011])
            except KeyError:
                continue

cells_array = np.array(cells)

# ── Write EXODUS files ────────────────────────────────────────────────────────
print("Writing EXODUS files...")

mesh_heating = meshio.Mesh(
    points=points,
    cells=[('hexahedron', cells_array)],
    point_data={'heating_W_per_cm3': h_val}
)
meshio.write(args.heating_exodus, mesh_heating)

mesh_dpa = meshio.Mesh(
    points=points,
    cells=[('hexahedron', cells_array)],
    point_data={'damage_energy_eV_per_cm3': d_val}
)
meshio.write(args.dpa_exodus, mesh_dpa)

print(f'Done.')
print(f'  Heating EXODUS: {args.heating_exodus}')
print(f'  DPA EXODUS:     {args.dpa_exodus}')
print(f'  Points:         {len(points)}')
print(f'  Hex cells:      {len(cells_array)}')