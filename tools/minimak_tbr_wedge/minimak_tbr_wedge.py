"""
MiniMak TBR Wedge Simulation
----------------------------
CSG wedge model (10° toroidal slice, reflective BCs) of the MiniMak
tritium breeder blanket. Uses a volumetric 14.1 MeV D-T neutron source
inside the plasma torus.

Blanket stack (inboard to outboard):
  plasma | first wall (W or SS316) | beryllium multiplier | FLiBe blanket | void

Geometry defaults mirror the parametric_coil_generator MiniMak defaults.
All major parameters are set at the top of this file.
"""

import argparse
import math
import json
import openmc

# ---------------------------------------------------------------------------
# PARAMETERS  (edit here or override via CLI flags)
# ---------------------------------------------------------------------------

MAJOR_RADIUS_M     = 0.23    # plasma magnetic axis [m]
MINOR_RADIUS_M     = 0.045   # plasma minor radius  [m]
FIRST_WALL_GAP_M   = 0.025   # standoff from plasma edge to first wall inner face [m]
FW_THICK_M         = 0.003   # first wall thickness [m]  (3 mm tungsten)
BE_THICK_M         = 0.0     # beryllium multiplier thickness [m]  (0 = off)
BLANKET_THICK_M    = 0.05    # FLiBe blanket thickness [m]  <-- primary sweep parameter
WEDGE_ANGLE_DEG    = 10.0    # toroidal wedge angle [degrees]
N_PARTICLES        = 50_000  # particles per batch
N_BATCHES          = 50
N_INACTIVE         = 10

FLIBE_ENRICHMENT   = 0.075   # Li-6 atom fraction in lithium (natural = 0.075)
FW_MATERIAL        = "tungsten"  # "tungsten" or "ss316"
BLANKET_MATERIAL   = "flibe"     # "flibe" | "pbli" | "li"

# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="MiniMak TBR wedge simulation")
parser.add_argument("--blanket-thick",  type=float, default=BLANKET_THICK_M)
parser.add_argument("--major-radius",   type=float, default=MAJOR_RADIUS_M)
parser.add_argument("--minor-radius",   type=float, default=MINOR_RADIUS_M)
parser.add_argument("--fw-gap",         type=float, default=FIRST_WALL_GAP_M)
parser.add_argument("--fw-thick",       type=float, default=FW_THICK_M,
                    help="First wall thickness [m]")
parser.add_argument("--fw-material",    type=str,   default=FW_MATERIAL,
                    choices=["tungsten", "ss316"],
                    help="First wall material")
parser.add_argument("--be-thick",       type=float, default=BE_THICK_M,
                    help="Beryllium multiplier thickness [m]  (0 = no Be layer)")
parser.add_argument("--particles",      type=int,   default=N_PARTICLES)
parser.add_argument("--batches",        type=int,   default=N_BATCHES)
parser.add_argument("--li6-enrichment",   type=float, default=FLIBE_ENRICHMENT)
parser.add_argument("--blanket-material", type=str,   default=BLANKET_MATERIAL,
                    choices=["flibe", "pbli", "li"],
                    help="Blanket material: flibe=Li2BeF4, pbli=Pb-16Li eutectic, li=pure lithium")
parser.add_argument("--output",           type=str,   default="tbr_result.json")
parser.add_argument("threads",          type=int,   nargs="?", default=1)
args = parser.parse_args()

R0          = args.major_radius
a           = args.minor_radius
fw_gap      = args.fw_gap
fw_thick    = args.fw_thick
be_thick    = args.be_thick
t_b         = args.blanket_thick
enrich          = args.li6_enrichment
fw_mat_name     = args.fw_material
blanket_mat_name = args.blanket_material

# Derived radii in cm
R0_cm       = R0       * 100
a_cm        = a        * 100
fw_gap_cm   = fw_gap   * 100
fw_thick_cm = fw_thick * 100
be_thick_cm = be_thick * 100
t_b_cm      = t_b      * 100

plasma_edge_cm  = a_cm
fw_inner_cm     = plasma_edge_cm + fw_gap_cm
fw_outer_cm     = fw_inner_cm    + fw_thick_cm
be_outer_cm     = fw_outer_cm    + be_thick_cm          # == fw_outer_cm if be_thick=0
blanket_inner_cm = be_outer_cm
blanket_outer_cm = blanket_inner_cm + t_b_cm
vessel_outer_cm  = blanket_outer_cm + 5.0

half_wedge_rad = math.radians(WEDGE_ANGLE_DEG / 2.0)

print(f"Geometry stack (minor radii from plasma axis):")
print(f"  plasma edge:    {plasma_edge_cm:.2f} cm")
print(f"  FW inner/outer: {fw_inner_cm:.2f} / {fw_outer_cm:.2f} cm  ({fw_mat_name}, {fw_thick_cm:.1f} mm)")
if be_thick_cm > 0:
    print(f"  Be outer:       {be_outer_cm:.2f} cm  ({be_thick_cm:.1f} mm)")
print(f"  blanket outer:  {blanket_outer_cm:.2f} cm  ({t_b_cm:.1f} mm {blanket_mat_name})")

# ---------------------------------------------------------------------------
# MATERIALS
# ---------------------------------------------------------------------------

vacuum = openmc.Material(name="vacuum")
vacuum.add_nuclide("He4", 1.0)
vacuum.set_density("g/cm3", 1e-10)

# First wall
if fw_mat_name == "tungsten":
    first_wall_mat = openmc.Material(name="tungsten")
    first_wall_mat.add_nuclide("W182", 0.265, "ao")
    first_wall_mat.add_nuclide("W183", 0.143, "ao")
    first_wall_mat.add_nuclide("W184", 0.306, "ao")
    first_wall_mat.add_nuclide("W186", 0.286, "ao")
    first_wall_mat.set_density("g/cm3", 19.3)
else:  # ss316
    first_wall_mat = openmc.Material(name="ss316")
    first_wall_mat.add_nuclide("Fe54",  0.037,  "wo")
    first_wall_mat.add_nuclide("Fe56",  0.582,  "wo")
    first_wall_mat.add_nuclide("Fe57",  0.013,  "wo")
    first_wall_mat.add_nuclide("Fe58",  0.002,  "wo")
    first_wall_mat.add_nuclide("Cr50",  0.007,  "wo")
    first_wall_mat.add_nuclide("Cr52",  0.152,  "wo")
    first_wall_mat.add_nuclide("Cr53",  0.017,  "wo")
    first_wall_mat.add_nuclide("Cr54",  0.004,  "wo")
    first_wall_mat.add_nuclide("Ni58",  0.080,  "wo")
    first_wall_mat.add_nuclide("Ni60",  0.031,  "wo")
    first_wall_mat.add_nuclide("Ni61",  0.001,  "wo")
    first_wall_mat.add_nuclide("Ni62",  0.004,  "wo")
    first_wall_mat.add_nuclide("Mo92",  0.003,  "wo")
    first_wall_mat.add_nuclide("Mo94",  0.002,  "wo")
    first_wall_mat.add_nuclide("Mo95",  0.003,  "wo")
    first_wall_mat.add_nuclide("Mo96",  0.003,  "wo")
    first_wall_mat.add_nuclide("Mo97",  0.002,  "wo")
    first_wall_mat.add_nuclide("Mo98",  0.005,  "wo")
    first_wall_mat.add_nuclide("Mo100", 0.002,  "wo")
    first_wall_mat.set_density("g/cm3", 7.99)

# Beryllium multiplier
beryllium = openmc.Material(name="beryllium")
beryllium.add_nuclide("Be9", 1.0, "ao")
beryllium.set_density("g/cm3", 1.85)

# Blanket material — selected via --blanket-material
if blanket_mat_name == "flibe":
    # FLiBe  Li2BeF4  ~1.94 g/cm3 at operating temperature
    blanket_mat = openmc.Material(name="FLiBe")
    blanket_mat.set_density("g/cm3", 1.94)
    blanket_mat.add_nuclide("Li6",  enrich       * 2, "ao")
    blanket_mat.add_nuclide("Li7", (1 - enrich)  * 2, "ao")
    blanket_mat.add_nuclide("Be9",  1.0,              "ao")
    blanket_mat.add_nuclide("F19",  4.0,              "ao")

elif blanket_mat_name == "pbli":
    # Pb-16Li eutectic  (84.2 at% Pb, 15.8 at% Li)  ~9.32 g/cm3 at 330°C
    # Standard DEMO/ITER test blanket module composition
    blanket_mat = openmc.Material(name="PbLi")
    blanket_mat.set_density("g/cm3", 9.32)
    blanket_mat.add_nuclide("Pb204", 0.014 * 0.842, "ao")
    blanket_mat.add_nuclide("Pb206", 0.241 * 0.842, "ao")
    blanket_mat.add_nuclide("Pb207", 0.221 * 0.842, "ao")
    blanket_mat.add_nuclide("Pb208", 0.524 * 0.842, "ao")
    blanket_mat.add_nuclide("Li6",   enrich        * 0.158, "ao")
    blanket_mat.add_nuclide("Li7",  (1 - enrich)   * 0.158, "ao")

elif blanket_mat_name == "li":
    # Pure liquid lithium  ~0.512 g/cm3 at operating temperature
    blanket_mat = openmc.Material(name="PureLi")
    blanket_mat.set_density("g/cm3", 0.512)
    blanket_mat.add_nuclide("Li6",  enrich,       "ao")
    blanket_mat.add_nuclide("Li7", (1 - enrich),  "ao")

mat_list = [vacuum, first_wall_mat, blanket_mat]
if be_thick_cm > 0:
    mat_list.insert(2, beryllium)
materials = openmc.Materials(mat_list)
materials.export_to_xml()

# ---------------------------------------------------------------------------
# GEOMETRY  — CSG torus wedge
# ---------------------------------------------------------------------------

plasma_torus   = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=plasma_edge_cm,   c=plasma_edge_cm)
fw_inner_torus = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=fw_inner_cm,      c=fw_inner_cm)
fw_outer_torus = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=fw_outer_cm,      c=fw_outer_cm)
blanket_torus  = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=blanket_outer_cm, c=blanket_outer_cm)
vessel_torus   = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=vessel_outer_cm,  c=vessel_outer_cm)

plane_pos = openmc.Plane(a= math.sin(half_wedge_rad), b=-math.cos(half_wedge_rad),
                          c=0, d=0, boundary_type="reflective")
plane_neg = openmc.Plane(a=-math.sin(half_wedge_rad), b=-math.cos(half_wedge_rad),
                          c=0, d=0, boundary_type="reflective")
outer_sphere = openmc.Sphere(r=vessel_outer_cm + R0_cm + 10, boundary_type="vacuum")

wedge_interior = +plane_pos & -plane_neg & -outer_sphere

plasma_cell   = openmc.Cell(name="plasma",      fill=vacuum,         region=-plasma_torus  & wedge_interior)
gap_cell      = openmc.Cell(name="gap",         fill=vacuum,         region=+plasma_torus  & -fw_inner_torus & wedge_interior)
fw_cell       = openmc.Cell(name="first_wall",  fill=first_wall_mat, region=+fw_inner_torus & -fw_outer_torus & wedge_interior)

if be_thick_cm > 0:
    be_outer_torus = openmc.ZTorus(x0=0, y0=0, z0=0, a=R0_cm, b=be_outer_cm, c=be_outer_cm)
    be_cell      = openmc.Cell(name="beryllium",   fill=beryllium,    region=+fw_outer_torus & -be_outer_torus & wedge_interior)
    blanket_cell = openmc.Cell(name="blanket",     fill=blanket_mat,  region=+be_outer_torus & -blanket_torus  & wedge_interior)
    cells = [plasma_cell, gap_cell, fw_cell, be_cell, blanket_cell]
else:
    blanket_cell = openmc.Cell(name="blanket",     fill=blanket_mat,  region=+fw_outer_torus & -blanket_torus  & wedge_interior)
    cells = [plasma_cell, gap_cell, fw_cell, blanket_cell]

vessel_cell = openmc.Cell(name="vessel_shell", fill=None, region=+blanket_torus & -vessel_torus & wedge_interior)
outer_void  = openmc.Cell(name="outer_void",   fill=None, region=+vessel_torus  & wedge_interior)
cells += [vessel_cell, outer_void]

universe = openmc.Universe(cells=cells)
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

# ---------------------------------------------------------------------------
# GEOMETRY PLOT  — R-Z cross section for visual QA
# ---------------------------------------------------------------------------

plot_colours = {vacuum: (100, 180, 255), first_wall_mat: (180, 180, 180), blanket_mat: (255, 160, 80)}
if be_thick_cm > 0:
    plot_colours[beryllium] = (150, 220, 150)

plot = openmc.Plot()
plot.basis    = 'xz'
plot.origin   = (R0_cm, 0, 0)
plot.width    = (vessel_outer_cm * 2 + 10, vessel_outer_cm * 2 + 10)
plot.pixels   = (600, 600)
plot.color_by = 'material'
plot.colors   = plot_colours
plots = openmc.Plots([plot])
plots.export_to_xml()
openmc.plot_geometry()

import shutil as _shutil
_shutil.move("plot_1.png", "geometry_plot.png")

# ---------------------------------------------------------------------------
# SOURCE  — 14.1 MeV isotropic, rejection-sampled toroidal plasma volume
#
# OpenMC's source.domains restricts sampling to a specific cell.
# Internally OpenMC draws from the bounding box and rejects any point that
# falls outside the named cell — this is rejection sampling handled by the
# transport engine itself, with no extra code needed on our side.
# The result is a source that is uniform within the true torus cross-section
# rather than the rectangular bounding box.
#
# TODO: upgrade to a peaked plasma density profile (e.g. parabolic or
#       Gaussian in minor radius) to represent the fusion reaction rate
#       distribution more physically. This matters most for first-wall
#       heat load calculations; the effect on global TBR is secondary.
# ---------------------------------------------------------------------------

source = openmc.IndependentSource()
source.space = openmc.stats.Box(
    lower_left=(R0_cm - plasma_edge_cm, -plasma_edge_cm, -plasma_edge_cm),
    upper_right=(R0_cm + plasma_edge_cm,  plasma_edge_cm,  plasma_edge_cm),
)
source.domains = [plasma_cell]   # reject any sample outside the plasma torus cell
source.angle   = openmc.stats.Isotropic()
source.energy  = openmc.stats.Discrete([14.1e6], [1.0])
print("Source: Box + cell rejection sampling (uniform within plasma torus)")

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

settings = openmc.Settings()
settings.source    = source
settings.batches   = args.batches
settings.inactive  = N_INACTIVE
settings.particles = args.particles
settings.run_mode  = "fixed source"
settings.export_to_xml()

# ---------------------------------------------------------------------------
# TALLIES  — (n,Xt) in blanket only
# ---------------------------------------------------------------------------

trit_tally = openmc.Tally(name="TBR")
trit_tally.filters = [openmc.CellFilter(blanket_cell)]
trit_tally.scores  = ["(n,Xt)"]

tallies = openmc.Tallies([trit_tally])
tallies.export_to_xml()

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

openmc.run(threads=args.threads)

# ---------------------------------------------------------------------------
# EXTRACT RESULT
# ---------------------------------------------------------------------------

import glob as _glob

_sp_files = sorted(_glob.glob("statepoint.*.h5"))
if not _sp_files:
    raise FileNotFoundError("No statepoint file found after run.")
sp    = openmc.StatePoint(_sp_files[-1])
tally = sp.get_tally(name="TBR")
tbr_mean = float(tally.mean.flatten()[0])
tbr_std  = float(tally.std_dev.flatten()[0])

print(f"\nTBR = {tbr_mean:.4f} ± {tbr_std:.4f}")
print(f"R0={R0}m  a={a}m  FW={fw_mat_name} {fw_thick*1000:.0f}mm  "
      f"Be={be_thick*1000:.0f}mm  blanket={t_b}m ({blanket_mat_name})  Li-6={enrich*100:.1f}%")

result = {
    "TBR":              tbr_mean,
    "TBR_std":          tbr_std,
    "major_radius_m":   R0,
    "minor_radius_m":   a,
    "fw_material":      fw_mat_name,
    "fw_thick_m":       fw_thick,
    "be_thick_m":       be_thick,
    "blanket_material": blanket_mat_name,
    "blanket_thick_m":  t_b,
    "li6_enrichment":   enrich,
    "n_particles":      args.particles,
    "n_batches":        args.batches,
}

with open(args.output, "w") as f:
    json.dump(result, f, indent=2)

print(f"Result written to {args.output}")
