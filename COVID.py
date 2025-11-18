"""COVID Main Protease Bioinformatics Research Project
Download Protein Data Base files and download the structures, read the structures,
remove ligands and H2O molecules, and save the new structures"""

from Bio.PDB import PDBList, PDBParser, PDBIO, Select, Superimposer, NeighborSearch, PPBuilder
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
import os

# Make a tuple for the two Mpro files and a list of the protein IDs
pdbl = PDBList()
pdb_ids = ['6LU7', '7TLL']

# For each protein in pdb_ids, we will retrieve the pdb file to the tuple
for pdb in pdb_ids:
    pdbl.retrieve_pdb_file(pdb, pdir='pdb_files', file_format='pdb')

# Parse each protein structure (QUIET=True to avoid warnings)
parser = PDBParser(QUIET=True)
WT_structure = parser.get_structure("WT", "pdb_files/pdb6lu7.ent")
Mut_structure = parser.get_structure("Mut", "pdb_files/pdb7tll.ent")
print(WT_structure)
print(Mut_structure)

# Print out any molecules to be removed for wild type
WT_ligands = set()

for model in WT_structure:
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag.strip():    # ligand or water
                WT_ligands.add(residue.get_resname())

print("WT ligands to be removed:", WT_ligands)

# Print out any molecules to be removed for mutant
Mut_ligands = set()

for model in Mut_structure:
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            if hetflag.strip():    # ligand or water
                Mut_ligands.add(residue.get_resname())

print("Mut ligands to be removed:", Mut_ligands)

# I used class to allow PDBIO to use the select containing the functions
class RemoveLigandSet(Select):
    def __init__(self, remove_resnames):
        # Convert to set for fast lookup
        self.remove = set(remove_resnames)

    def accept_residue(self, residue):
        resname = residue.get_resname().strip()

        # Remove if residue name is in the deletion set
        if resname in self.remove:
            return False

        else: return True

# Uses the class containing functions to remove mentioned ligands and saves the
# cleaned WT structure (Must use class as io takes select not functions)
WT_remove = {'HOH', '010', 'PJE', '02J'}

io = PDBIO()
io.set_structure(WT_structure)
io.save("WT_clean.pdb", select=RemoveLigandSet(WT_remove))

# Uses the class to remove mentioned ligands and saves the cleaned Mut structure
Mut_remove = {'HOH', '4WI'}

io = PDBIO()
io.set_structure(Mut_structure)
io.save("Mut_clean.pdb", select=RemoveLigandSet(Mut_remove))

"""Calculation of RMSDs, which helps show how different the wild type structure 
is from the Omicron structure. This is to help us know about why some ligands have
higher affinity and some don't on AutoDock Vina"""

# Path to cleaned files (Mut_clean is from the modeller software!)
wt_path = "WT_clean.pdb"
mut_path = "Mut_clean.B99990001.pdb"

# Load the structures
clean_WT_structure = parser.get_structure("WT", wt_path)
clean_Mut_structure =parser.get_structure("Mut", mut_path)

# Function to remove any extra polypeptide inhibitors, as AVL was found
def remove_nonA_chains(structure):
    model = structure[0]
    chains_to_delete = [chain.id for chain in model if chain.id != "A"]
    for chain_id in chains_to_delete:
        model.detach_child(chain_id)
    return structure

# Even cleaner structures with extra chains removed
clean_WT_structure = remove_nonA_chains(clean_WT_structure)
clean_Mut_structure = remove_nonA_chains(clean_Mut_structure)

# Confirms I am using chain 1 for both
WT_chain = clean_WT_structure[0]["A"]
# Attempt to get chain A first, then fallback to blank chain (keep getting errors)
if "A" in clean_Mut_structure[0]:
    Mut_chain = clean_Mut_structure[0]["A"]
elif " " in clean_Mut_structure[0]:
    Mut_chain = clean_Mut_structure[0][" "]
else:
    raise ValueError("No chain A or blank chain found in mutant structure")

# This function was necessary because using only alpha carbons for RMSD gave data with less
# Value to the project. In order to take the RMSD using heavy atoms however, the atoms
# must match in both the wild type and the mutant, which was a problem without this
# function for the mutation site proline vs histidine
def get_matching_heavy_atoms(chain1, chain2, res_ids=None):
    atoms1, atoms2 = [], []

    # Select residues
    if res_ids is None:
        res_ids1 = [res.id[1] for res in chain1]
        res_ids2 = [res.id[1] for res in chain2]
        res_ids = sorted(set(res_ids1) & set(res_ids2))  # only shared residues

    # Match atoms residue by residue
    res1_map = {res.id[1]: res for res in chain1}
    res2_map = {res.id[1]: res for res in chain2}

    for res_id in res_ids:
        if res_id in res1_map and res_id in res2_map:
            atoms_res1 = {atom.name: atom for atom in res1_map[res_id] if atom.element != 'H'}
            atoms_res2 = {atom.name: atom for atom in res2_map[res_id] if atom.element != 'H'}
            shared_atoms = set(atoms_res1.keys()) & set(atoms_res2.keys())
            for atom_name in sorted(shared_atoms):
                atoms1.append(atoms_res1[atom_name])
                atoms2.append(atoms_res2[atom_name])

    return atoms1, atoms2

# This function calculates RMSD for given chains or specific residues
def calc_rmsd(chain1, chain2, res_ids=None):
    atoms1, atoms2 = get_matching_heavy_atoms(chain1, chain2, res_ids)
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    return sup.rms

"""Amino Acid Sequences"""

# Print amino acid sequence
ppb = PPBuilder()

print("WT amino acid sequence:")
for pp in ppb.build_peptides(clean_WT_structure):
    print(pp.get_sequence())

print("Mutant amino acid sequence:")
for pp in ppb.build_peptides(clean_Mut_structure):
    print(pp.get_sequence())

"""RMSD (Root Mean Square Deviation) Calculations"""

"""Global RMSD"""

global_rmsd = calc_rmsd(WT_chain, Mut_chain)
print("Global RMSD:", global_rmsd)
# A global RMSD of 0.758 confirms the overall structures are almost identical

"""Active site RMSD (His41, Cys145)"""

active_rmsd = calc_rmsd(WT_chain, Mut_chain, res_ids=[41, 145])
print("Active Site RMSD:", active_rmsd)
# An active site RMSD of 0.303 confirms the active site structures are almost identical

"""Mutation site RMSD (Residue 132 ± 3 Residues)"""

mutation_rmsd = calc_rmsd(WT_chain, Mut_chain, res_ids=[129,130,131,132,133,134,135])
print("Mutation Site RMSD:", mutation_rmsd)
# A mutation site RMSD of 0.538 shows that the proline to histidine mutation
# has a very small impact structurally. However, this alone doesn't tell us
# about the functionalities it might change

"""RMSF (Root Mean Square Fluctuation) Calculations"""

parser = PDBParser(QUIET=True)

# Import all the new Modeller files in separately to avoid errors
m1 = parser.get_structure("m1", "Mut_clean.B99990001.pdb")
m2 = parser.get_structure("m2", "Mut_clean.B99990002.pdb")
m3 = parser.get_structure("m3", "Mut_clean.B99990003.pdb")
m4 = parser.get_structure("m4", "Mut_clean.B99990004.pdb")
m5 = parser.get_structure("m5", "Mut_clean.B99990005.pdb")
m6 = parser.get_structure("m6", "Mut_clean.B99990006.pdb")
m7 = parser.get_structure("m7", "Mut_clean.B99990007.pdb")
m8 = parser.get_structure("m8", "Mut_clean.B99990008.pdb")
m9 = parser.get_structure("m9", "Mut_clean.B99990009.pdb")
m10 = parser.get_structure("m10", "Mut_clean.B99990010.pdb")

models = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]

# extracts alpha carbons by residue, as this is better for RMSF using Modeller
def get_ca_by_residue(model):
    chain = model[0]["A"]  # assumes chain A
    coords = {}
    for res in chain:
        if "CA" in res:
            coords[res.id[1]] = res["CA"].coord
    return coords

# Collect all CA coordinates
all_coords = []

for model in models:
    ca_dict = get_ca_by_residue(model)
    all_coords.append(ca_dict)

# Residues present in all models
resids = sorted(set.intersection(*(set(d.keys()) for d in all_coords)))

# Convert to array shape: (n models, n residues, 3)
coords_array = np.array([[model_dict[r] for r in resids] for model_dict in all_coords])

# Solve for RMSF equation
mean_coords = coords_array.mean(axis=0)
fluctuations = coords_array - mean_coords
rmsf = np.sqrt((fluctuations ** 2).sum(axis=2).mean(axis=0))

# Print RMSF
print("Pseudo-RMSF per residue:")
for resid, value in zip(resids, rmsf):
    print(f"Residue {resid} RMSF: {value} Å")

