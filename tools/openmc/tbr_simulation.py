import argparse
import os
import numpy as np
import openmc

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--materials',       type=str, nargs='+', required=True)
parser.add_argument('--thicknesses',     type=float, nargs='+', required=True)
parser.add_argument('--enrichment',      type=float, required=True)
parser.add_argument('--batches',         type=int, required=True)
parser.add_argument('--particles',       type=int, required=True)
parser.add_argument('--mesh_resolution', type=int, required=True)
parser.add_argument('--output',          type=str, required=True)
parser.add_argument('--tbr_mean_output',  type=str, required=True)
parser.add_argument('--tbr_stdev_output', type=str, required=True)
parser.add_argument('--heating_output',  type=str, required=True)
parser.add_argument('--dpa_output',      type=str, required=True)
args = parser.parse_args()

# ── Validate inputs ───────────────────────────────────────────────────────────
if len(args.materials) != len(args.thicknesses):
    raise ValueError("Number of materials must match number of thicknesses.")

layers = [{'material': m, 'thickness': t}
          for m, t in zip(args.materials, args.thicknesses)]

# ── Resolve output paths before any chdir ────────────────────────────────────
args.output         = os.path.abspath(args.output)
args.tbr_mean_output  = os.path.abspath(args.tbr_mean_output)
args.tbr_stdev_output = os.path.abspath(args.tbr_stdev_output)
args.heating_output = os.path.abspath(args.heating_output)
args.dpa_output     = os.path.abspath(args.dpa_output)

# ── Working directory ─────────────────────────────────────────────────────────
run_dir = 'run'
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

# ── Material library ──────────────────────────────────────────────────────────
def make_material(name, enrichment=0.9):
    if name == 'lithium':
        mat = openmc.Material(name='lithium')
        mat.add_nuclide('Li6', enrichment)
        li7 = 1.0 - enrichment
        if li7 > 0:
            mat.add_nuclide('Li7', li7)
        mat.set_density('g/cm3', 0.534)

    elif name == 'tungsten':
        mat = openmc.Material(name='tungsten')
        mat.add_element('W', 1.0)
        mat.set_density('g/cm3', 19.3)

    elif name == 'steel':
        mat = openmc.Material(name='steel')
        mat.add_element('Fe', 0.65)
        mat.add_element('Cr', 0.17)
        mat.add_element('Ni', 0.12)
        mat.add_element('Mo', 0.025)
        mat.add_element('Mn', 0.02)
        mat.add_element('Si', 0.015)
        mat.set_density('g/cm3', 7.99)

    elif name == 'beryllium':
        mat = openmc.Material(name='beryllium')
        mat.add_element('Be', 1.0)
        mat.set_density('g/cm3', 1.85)

    else:
        raise ValueError(f"Unknown material: {name}. "
                         f"Supported: lithium, tungsten, steel, beryllium")
    return mat

# ── Build materials + geometry ────────────────────────────────────────────────
materials_list = []
cells          = []
inner_radius   = 0.0

for i, layer in enumerate(layers):
    mat_name     = layer['material']
    thickness    = layer['thickness']
    outer_radius = inner_radius + thickness

    mat = make_material(mat_name, enrichment=args.enrichment)
    materials_list.append(mat)

    inner_surf = openmc.Sphere(r=inner_radius) if inner_radius > 0 else None
    outer_surf = openmc.Sphere(r=outer_radius)

    if i == len(layers) - 1:
        outer_surf.boundary_type = 'vacuum'

    region = -outer_surf if inner_surf is None else (-outer_surf & +inner_surf)
    cell   = openmc.Cell(fill=mat, region=region, name=f'layer_{i}_{mat_name}')
    cells.append(cell)
    inner_radius = outer_radius

total_radius = inner_radius

materials = openmc.Materials(materials_list)
materials.export_to_xml()

universe = openmc.Universe(cells=cells)
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

# ── Settings ──────────────────────────────────────────────────────────────────
settings           = openmc.Settings()
settings.batches   = args.batches
settings.particles = args.particles
settings.run_mode  = 'fixed source'

source        = openmc.IndependentSource()
source.space  = openmc.stats.Point((0, 0, 0))
source.energy = openmc.stats.Discrete([14.1e6], [1.0])
settings.source = source
settings.export_to_xml()

# ── Tallies ───────────────────────────────────────────────────────────────────
tallies = openmc.Tallies()

# TBR — lithium cells only
lithium_mats = [m for m in materials_list if m.name == 'lithium']
if not lithium_mats:
    raise ValueError("No lithium layer found — cannot compute TBR.")

tbr_tally          = openmc.Tally(name='TBR')
tbr_tally.filters  = [openmc.MaterialFilter(lithium_mats)]
tbr_tally.nuclides = ['Li6']
tbr_tally.scores   = ['(n,Xt)']
tallies.append(tbr_tally)

# Mesh tallies
mesh             = openmc.RegularMesh()
mesh.dimension   = [args.mesh_resolution, args.mesh_resolution, args.mesh_resolution]
mesh.lower_left  = [-total_radius, -total_radius, -total_radius]
mesh.upper_right = [ total_radius,  total_radius,  total_radius]
mesh_filter      = openmc.MeshFilter(mesh)

heating_tally         = openmc.Tally(name='heating')
heating_tally.filters = [mesh_filter]
heating_tally.scores  = ['heating']
tallies.append(heating_tally)

dpa_tally         = openmc.Tally(name='dpa')
dpa_tally.filters = [mesh_filter]
dpa_tally.scores  = ['damage-energy']
tallies.append(dpa_tally)

tallies.export_to_xml()

# ── Run ───────────────────────────────────────────────────────────────────────
openmc.run()

# ── Extract results ───────────────────────────────────────────────────────────
sp = openmc.StatePoint(f'statepoint.{args.batches}.h5')

tbr_tally = sp.get_tally(name='TBR')
tbr_mean  = tbr_tally.mean.sum()
tbr_std   = tbr_tally.std_dev.sum()

heating_tally = sp.get_tally(name='heating')
heating_mean  = heating_tally.get_values(scores=['heating']).reshape(
    args.mesh_resolution, args.mesh_resolution, args.mesh_resolution)

dpa_tally = sp.get_tally(name='dpa')
dpa_mean  = dpa_tally.get_values(scores=['damage-energy']).reshape(
    args.mesh_resolution, args.mesh_resolution, args.mesh_resolution)

# ── Write outputs ─────────────────────────────────────────────────────────────
with open(args.output, 'w') as f:
    f.write('=== OpenMC Multiphysics Simulation ===\n\n')
    f.write('Layers:\n')
    for layer in layers:
        f.write(f"  {layer['material']:12s}  {layer['thickness']} cm\n")
    f.write(f'\nLi-6 enrichment:  {args.enrichment*100:.1f}%\n')
    f.write(f'Batches:          {args.batches}\n')
    f.write(f'Particles/batch:  {args.particles}\n')
    f.write(f'Mesh resolution:  {args.mesh_resolution}^3\n\n')
    f.write(f'TBR mean:         {tbr_mean:.4f}\n')
    f.write(f'TBR std dev:      {tbr_std:.4f}\n')
    f.write(f'\nTotal geometry radius: {total_radius:.2f} cm\n')

with open(args.tbr_mean_output, 'w') as f:
    f.write(f'{tbr_mean:.6f}\n')

with open(args.tbr_stdev_output, 'w') as f:
    f.write(f'{tbr_std:.6f}\n')

mesh_coords = np.linspace(-total_radius, total_radius, args.mesh_resolution)
with open(args.heating_output, 'w') as f:
    f.write('x,y,z,heating_eV_per_source\n')
    for ix in range(args.mesh_resolution):
        for iy in range(args.mesh_resolution):
            for iz in range(args.mesh_resolution):
                val = heating_mean[ix, iy, iz]
                if val > 0:
                    f.write(f'{mesh_coords[ix]:.4f},'
                            f'{mesh_coords[iy]:.4f},'
                            f'{mesh_coords[iz]:.4f},'
                            f'{val:.6e}\n')

with open(args.dpa_output, 'w') as f:
    f.write('x,y,z,damage_energy_eV_per_source\n')
    for ix in range(args.mesh_resolution):
        for iy in range(args.mesh_resolution):
            for iz in range(args.mesh_resolution):
                val = dpa_mean[ix, iy, iz]
                if val > 0:
                    f.write(f'{mesh_coords[ix]:.4f},'
                            f'{mesh_coords[iy]:.4f},'
                            f'{mesh_coords[iz]:.4f},'
                            f'{val:.6e}\n')

print(f'TBR mean: {tbr_mean:.4f} +/- {tbr_std:.4f}')
print(f'Heating and DPA mesh tallies written.')