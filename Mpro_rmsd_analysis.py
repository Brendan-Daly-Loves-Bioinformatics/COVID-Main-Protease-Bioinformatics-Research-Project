"""COVID-19 Main Protease Bioinformatics Research Project
Download Protein Data Base files and download the structures, read the structures,
remove H2O molecules, and save the new structures"""

from Bio.PDB import PDBList, PDBParser, PDBIO, Select, Superimposer, NeighborSearch
import numpy as np

# Download PDB files
pdbl = PDBList()
pdb_ids = ['6LU7', '7TLL']

for pdb in pdb_ids:
    pdbl.retrieve_pdb_file(pdb, pdir='pdb_files', file_format='pdb')

# Parse the structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure("Mpro_WT", "pdb_files/pdb6lu7.ent")
print(structure)

# Class to remove water molecules
class NoWater(Select):
    def accept_residue(self, residue):
        return residue.get_id()[0] != 'W'

# Save structure without water molecules
io = PDBIO()
io.set_structure(structure)
io.save("pdb_files/6lu7_no_water.ent", NoWater())

"""Calculation of RMSDs, which shows how different the wild type structure is from the Omicron structure"""

# Paths to the downloaded files
wt_path = "pdb_files/pdb6lu7.ent"
mut_path = "pdb_files/pdb7tll.ent"

# load structures (keep them in memory)
wt_struct = parser.get_structure("WT", wt_path)
mut_struct = parser.get_structure("Mut", mut_path)

# Get first model = first chain for both
wt_chain = wt_struct[0]["A"]
mut_chain = mut_struct[0]["A"]

"""extract CA atoms (alpha-carbons)"""

def get_ca_atoms(chain):
    return [res["CA"] for res in chain if "CA" in res]

# Global Cα alignment
wt_ca = get_ca_atoms(wt_chain)
mut_ca = get_ca_atoms(mut_chain)

# Ensure equal length (needed for alignment)
min_len = min(len(wt_ca), len(mut_ca))
wt_ca = wt_ca[:min_len]
mut_ca = mut_ca[:min_len]

super_imposer = Superimposer()
super_imposer.set_atoms(wt_ca, mut_ca)
# apply the transformation to the mutant structure coordinates so subsequent analyses use aligned coords
super_imposer.apply(mut_struct.get_atoms())
# This code and similar codes are highly important, as using data analytics allowed me to find
# that the original code gave unexpected Å values magnitudes larger than they really are.

print("GLOBAL RMSD (Cα atoms)")
print(f"RMSD: {super_imposer.rms:.3f} Å\n")
# A low global RMSD of 0.510 Å indicates the wild type and mutant
# overall structures are quite similar.

"""Active Site RMSD (His41, Cys145)"""

active_ids = [41, 145]  # catalytic dyad

def get_specific_residues(chain, ids):
    atoms = []
    for residue in chain:
        if residue.id[1] in ids and "CA" in residue:
            atoms.append(residue["CA"])
    return atoms

# Note: mut_chain references mut_struct atoms; because we applied the super_imposer to mut_struct above,
# mut_chain now contains aligned coordinates.
wt_active = get_specific_residues(wt_chain, active_ids)
mut_active = get_specific_residues(mut_chain, active_ids)

super2 = Superimposer()
super2.set_atoms(wt_active, mut_active)

print("ACTIVE SITE RMSD (His41 + Cys145)")
print(f"RMSD: {super2.rms:.3f} Å\n")
# A very low active site RMSD of 0.089 Å indicates the wild type and mutant
# active site structures are nearly identical

"""Local RMSD near mutation site (residue 132)"""

def get_residues_near(chain, center_res, window=3):
    atoms = []
    for residue in chain:
        resnum = residue.id[1]
        if "CA" in residue and abs(resnum - center_res) <= window:
            atoms.append(residue["CA"].coord)
    return atoms

wt_mut_region = get_residues_near(wt_chain, 132)
mut_mut_region = get_residues_near(mut_chain, 132)

# compute local RMSD on aligned coordinates
wt_region = np.array(wt_mut_region)
mut_region = np.array(mut_mut_region)

# guard against empty regions or unequal lengths
if wt_region.size == 0 or mut_region.size == 0 or wt_region.shape != mut_region.shape:
    local_rmsd = float('nan')
else:
    local_rmsd = np.sqrt(((wt_region - mut_region)**2).mean())

print("LOCAL RMSD around mutation (residue 132 ±3)")
print(f"RMSD: {local_rmsd:.3f} Å\n")
# A very low RMSD of 0.194 Å around the mutation (residue 132 ±3) indicates the
# mutation doesn't change Omicron's structure very much.

"""In conclusion, we can see from the RMSD calculations that the structural differences between
the wild type and Omicron SARS-CoV-2 strains aren't very big. However, the RMSD alone doesn't give
us the information we need to know the functional differences, which could still be very large."""

"""Comparing WT (6LU7) vs Mutant (7TLL) SARS-CoV-2 Mpro"""

# 1. Compare catalytic dyad geometry (His41 - Cys145)

def ca_distance(chain, res1, res2):
    # Distance between CA atoms of 2 residues.
    a1 = chain[res1]["CA"].coord
    a2 = chain[res2]["CA"].coord
    return np.linalg.norm(a1 - a2)

wt_dyad = ca_distance(wt_chain, 41, 145)
mut_dyad = ca_distance(mut_chain, 41, 145)

print("Catalytic Dyad Geometry")
print(f"WT   His41–Cys145 distance:  {wt_dyad:.3f} Å")
print(f"Mut  His41–Cys145 distance:  {mut_dyad:.3f} Å\n")
# A WT His41-Cys145 distance of 9.387 Å versus the Omicron distance of 9.210 Å
# indicates that the catalytic dyad distance only changes by 0.177 Å in Omicron

# 2. Compare local structure around mutation (residue 132)

def region_coords(chain, center=132, window=3):
    coords = []
    for res in chain:
        if "CA" in res and abs(res.id[1] - center) <= window:
            coords.append(res["CA"].coord)
    return np.array(coords)

wt_region = region_coords(wt_chain)
mut_region = region_coords(mut_chain)

# ensure shapes match before calculation
if wt_region.size == 0 or mut_region.size == 0 or wt_region.shape != mut_region.shape:
    local_rmsd_region = float('nan')
else:
    local_rmsd_region = np.sqrt(((wt_region - mut_region)**2).mean())

print("Local Structural Change (Residue 132 Region)")
print(f"Local RMSD around 132 ±3: {local_rmsd_region:.3f} Å\n")
# The local structural change of the mutation at Residue 132 ±3 has an RMSD value of 0.194 Å,
# which indicates that Omicron isn't that different in structure from the WT at this location,
# despite the large functional difference the Proline to Histidine mutation likely causes.

# 3. Detect hydrogen bonds formed ONLY by His132 in mutant

def find_hbonds(chain, residue_number, cutoff=3.5):
    # Find atoms near residue for possible H-bonds
    residue = chain[residue_number]
    atom_list = [atom for atom in chain.get_atoms()]
    ns = NeighborSearch(atom_list)

    close_atoms = []
    for atom in residue:
        neighbors = ns.search(atom.coord, cutoff)
        for nbr in neighbors:
            if nbr.get_parent().id[1] != residue_number:  # avoid self
                close_atoms.append((atom, nbr))
    return close_atoms

print("Hydrogen Bond Environment at Position 132")

# WT = Pro132 (no polar atom = cannot H-bond)
wt_contacts = find_hbonds(wt_chain, 132)
mut_contacts = find_hbonds(mut_chain, 132)

print(f"WT contacts near Pro132: {len(wt_contacts)} (Proline cannot H-bond)")
print(f"Mut contacts near His132: {len(mut_contacts)} possible interactions\n")
# The WT has proline, which has no H atoms on its backbone N atom and doesn't accept H-bonds,
# so there are likely only Van der Waals forces present. Omicron has a histidine instead, which
# can donate and accept H-bonds and is also able to form an alpha-helix unlike the rigid proline.

# 4. Per-residue deviation (per-residue RMSD)

def per_residue_rmsd(wt_chain, mut_chain):
    diffs = {}
    for res in wt_chain:
        idx = res.id[1]
        if "CA" in res and idx in mut_chain and "CA" in mut_chain[idx]:
            d = np.linalg.norm(res["CA"].coord - mut_chain[idx]["CA"].coord)
            diffs[idx] = d
    return diffs

differences = per_residue_rmsd(wt_chain, mut_chain)

print("Per-Residue Cα Deviations (Å)")
for res_num, diff in differences.items():
    if diff > 1.0:  # highlight meaningful shifts
        print(f"Residue {res_num}: {diff:.2f} Å")
# Overall, there are only a few residues with meaningful alpha carbon deviations.
# However, the large 4.61 Å deviation at residue 302 indicates I should explore further.
