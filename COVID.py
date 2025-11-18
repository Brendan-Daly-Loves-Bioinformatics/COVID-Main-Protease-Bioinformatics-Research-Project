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

# Path to cleaned files
wt_path = "WT_clean.pdb"
mut_path = "Mut_clean.pdb"

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
Mut_chain = clean_Mut_structure[0]["A"]

# Get alpha carbons (CA is alpha carbon in Bio.PDB)
def get_CA_atoms(chain):
    return [res["CA"] for res in chain if "CA" in res]

# Extracts alpha carbons
WT_CA = get_CA_atoms(WT_chain)
Mut_CA = get_CA_atoms(Mut_chain)

# Makes the chains the same length (as the smaller chain) to avoid errors
def match_ca_by_resnum(WT_chain, Mut_chain):
    wt_ca = {res.id[1]: res['CA'] for res in WT_chain if 'CA' in res}
    mut_ca = {res.id[1]: res['CA'] for res in Mut_chain if 'CA' in res}
    shared_residues = sorted(set(wt_ca.keys()) & set(mut_ca.keys()))

    wt_list = [wt_ca[i] for i in shared_residues]
    mut_list = [mut_ca[i] for i in shared_residues]

    return wt_list, mut_list

WT_CA, Mut_CA = match_ca_by_resnum(WT_chain, Mut_chain)

# Aligns the chains as accurately as possible to avoid overestimated RMSD values (important!)
super_imposer = Superimposer()
super_imposer.set_atoms(WT_CA, Mut_CA)
# Only Apply to the mutant structure to align with the wild type structure
super_imposer.apply(clean_Mut_structure.get_atoms())

# Print amino acid sequence
ppb = PPBuilder()

print("WT amino acid sequence:")
for pp in ppb.build_peptides(clean_WT_structure):
    print(pp.get_sequence())

print("Mutant amino acid sequence:")
for pp in ppb.build_peptides(clean_Mut_structure):
    print(pp.get_sequence())

"""Global RMSD (The entire chain)"""

print("Global RMSD:", super_imposer.rms, "Å")

"""Active site RMSD (His41-Cys145)"""
