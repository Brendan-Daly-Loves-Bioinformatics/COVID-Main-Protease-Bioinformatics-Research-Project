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

active_rmsd = calc_rmsd(WT_chain, Mut_chain, res_ids=[41,145])
print("Active Site RMSD:", active_rmsd)
# An active site RMSD of 0.303 confirms the active site structures are almost identical

"""Mutation site RMSD (Residue 132 ± 3 Residues)"""

mutation_rmsd = calc_rmsd(WT_chain, Mut_chain, res_ids=[129,130,131,132,133,134,135])
print("Mutation Site RMSD:", mutation_rmsd)
# A mutation site RMSD of 0.538 shows that the proline to histidine mutation
# has a very small impact structurally. However, this alone doesn't tell us
# about the functionalities it might change

"""RMSF (Root Mean Square Fluctuation) Calculations"""
"""Disclaimer: This is psuedo RMSF done by Modeller, which creates predicted models.
This data will be standardized to be as accurate as possible without having multiple 
official models on the PDB."""

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

"""Standardizing RMSF Values using Z-scores"""

# List of RMSF values (in Å)
rmsf = [4.125117778778076, 3.941920757293701, 3.628969669342041, 3.511352300643921,
3.317049980163574, 3.227311611175537, 3.118635416030884, 3.0229620933532715,
3.195312261581421, 3.18601393699646, 3.317148447036743, 3.120443105697632,
3.0938572883605957, 3.3726556301116943, 3.413538932800293, 3.322784900665283,
3.532853126525879, 3.853985548019409, 4.191998481750488, 4.358037948608398,
4.814262390136719, 5.0808563232421875, 5.551892280578613, 5.580079078674316,
5.108872890472412, 4.770974159240723, 4.371620178222656, 3.94586181640625,
3.7294609546661377, 3.3965160846710205, 3.436171770095825, 3.237279176712036,
3.366741895675659, 3.6271188259124756, 3.6160194873809814, 3.663118362426758,
3.584904432296753, 3.814030647277832, 4.003724098205566, 4.345101356506348,
4.647446632385254, 4.69962739944458, 5.035475254058838, 5.304577350616455,
5.685229778289795, 5.851285934448242, 6.195043563842773, 5.817751407623291,
5.468283176422119, 5.744658946990967, 5.6659650802612305, 5.533994674682617,
5.514123439788818, 5.117008686065674, 5.22272253036499, 5.692392826080322,
5.5405144691467285, 5.205822467803955, 5.528003215789795, 5.787921905517578,
5.4802727699279785, 5.30540657043457, 4.9968156814575195, 5.276282787322998,
5.293361663818359, 4.861313819885254, 4.7991042137146, 4.396470546722412,
4.38861083984375, 4.0992512702941895, 4.29058313369751, 4.61832332611084,
4.467094898223877, 4.575855255126953, 4.351718902587891, 4.585549831390381,
4.518230438232422, 4.539163589477539, 4.516717910766602, 4.43719482421875,
4.3418145179748535, 4.297795295715332, 3.9702649116516113, 4.054344177246094,
4.033186435699463, 3.8627679347991943, 4.051924705505371, 3.8801424503326416,
4.065907955169678, 4.037608623504639, 4.079970836639404, 4.179750919342041,
3.949312686920166, 3.6547048091888428, 3.518054962158203, 3.5141396522521973,
3.372722864151001, 3.173780918121338, 3.027665376663208, 2.954332113265991,
2.8244504928588867, 2.795207977294922, 2.840653419494629, 2.8023271560668945,
2.8686625957489014, 2.8013718128204346, 2.9087648391723633, 3.0177218914031982,
2.950547218322754, 2.7840373516082764, 2.790470600128174, 2.7973501682281494,
2.936072587966919, 3.1356730461120605, 3.3107783794403076, 3.629667043685913,
3.830594301223755, 4.243416786193848, 4.3948283195495605, 4.039864540100098,
4.037213325500488, 3.802610397338867, 4.002189636230469, 3.7594752311706543,
3.420173168182373, 3.3240206241607666, 3.10557222366333, 3.072626829147339,
2.98911190032959, 3.03546404838562, 3.2371304035186768, 3.321044445037842,
3.599424123764038, 3.4099228382110596, 3.519227981567383, 3.397117853164673,
3.5923850536346436, 3.5976014137268066, 3.9299607276916504, 4.080965518951416,
4.500877380371094, 4.785480499267578, 4.4794840812683105, 4.107175827026367,
4.014388084411621, 3.707160234451294, 3.4849791526794434, 3.1658527851104736,
2.9516162872314453, 2.7793116569519043, 2.716578960418701, 2.7879269123077393,
2.8657000064849854, 3.0915966033935547, 3.041856288909912, 2.8691794872283936,
2.7582459449768066, 2.6988000869750977, 2.7715492248535156, 2.8040595054626465,
3.0274851322174072, 3.3079020977020264, 3.602489471435547, 3.9472150802612305,
4.210803985595703, 4.299952507019043, 4.462052822113037, 4.857496738433838,
4.596476078033447, 4.4217095375061035, 4.007078170776367, 3.845586061477661,
3.724903106689453, 3.5667834281921387, 3.4082765579223633, 3.1636569499969482,
3.135972499847412, 3.3300435543060303, 3.5881357192993164, 3.574352741241455,
3.6380953788757324, 3.4356627464294434, 3.702704429626465, 4.114251136779785,
4.191122055053711, 4.43215274810791, 4.500709056854248, 4.972390174865723,
5.219740390777588, 5.355783939361572, 5.282332420349121, 4.83148717880249,
4.61305570602417, 4.173761367797852, 4.128598690032959, 4.12824010848999,
3.7970917224884033, 3.630594730377197, 3.449028730392456, 3.217376470565796,
3.3284213542938232, 3.203796863555908, 3.0861809253692627, 3.2966532707214355,
3.388625383377075, 3.2714474201202393, 3.324647903442383, 3.558227062225342,
3.615730047225952, 3.565424919128418, 3.798487424850464, 4.004673480987549,
3.9887828826904297, 4.0707244873046875, 4.354973316192627, 4.234198570251465,
4.396008014678955, 4.502581596374512, 4.246830940246582, 4.330277442932129,
4.561215877532959, 4.832911491394043, 4.647979259490967, 4.497035980224609,
4.395991325378418, 4.402826309204102, 4.144333362579346, 4.298749923706055,
4.342700958251953, 4.0409722328186035, 3.9704082012176514, 4.238893985748291,
4.131556510925293, 3.862522602081299, 4.029278755187988, 4.235933303833008,
4.019806385040283, 3.8915648460388184, 3.632007360458374, 3.509538173675537,
3.66394305229187, 3.591641664505005, 3.7006430625915527, 3.893902063369751,
3.652387857437134, 3.423811912536621, 3.663745641708374, 3.725431442260742,
3.4206812381744385, 3.501494884490967, 3.8188629150390625, 3.794398307800293,
3.692443370819092, 3.963555097579956, 4.255002498626709, 4.196661949157715,
4.189355850219727, 4.483919620513916, 4.266959190368652, 4.228021144866943,
3.909878730773926, 4.087145805358887, 4.202282905578613, 3.9049975872039795,
3.8486640453338623, 4.125258445739746, 4.052846908569336, 3.8289835453033447,
4.001033782958984, 4.227889537811279, 4.07925271987915, 4.063037872314453,
4.343613147735596, 4.526361465454102, 4.334885120391846, 4.369444847106934,
4.668821811676025, 4.901081562042236, 4.5958170890808105, 4.3098297119140625,
4.002807140350342, 3.906907558441162, 4.143468856811523, 4.007009506225586,
4.200405120849609, 3.907005786895752, 3.631688356399536, 3.4211316108703613,
3.1675870418548584, 3.08850359916687, 3.0232467651367188, 2.9262006282806396,
3.0715012550354004, 2.9851205348968506, 2.9803922176361084, 3.1987411975860596,
3.339165210723877, 3.247377395629883, 3.425527572631836, 3.6363649368286133,
3.831080675125122, 3.950465202331543, 4.210178375244141, 4.48406982421875,
4.61696720123291, 4.7703938484191895]

# Converts the list into a Numpy array for math
rmsf = np.array(rmsf)

# Calculate the mean and standard deviation of the RMSF values
mean = np.mean(rmsf)
std = np.std(rmsf)

print("Mean RMSF:", mean)
print("Standard Deviation:", std)

# Calculate the Z-scores, where i is the residue number and z is the Z-score,
# which starts at 1 instead of 0 like Python
z_scores = (rmsf-mean)/std

for i, z in enumerate(z_scores, start=1):
    print(f"Residue {i}: Z-score = {z}")
# For z scores, a value of 0 is balanced, a value of < -1 is quite rigid, and a value
# of > 1 is quite flexible. To summarize, this data tells me about the secondary structure.
# Beta-sheets are very rigid, alpha-helices are quite rigid, and loops, turns, and surface
# regions are very flexible. Areas with high Z-scores/RMSF mean that the RMSD values at these
# locations have less value, as there is more probability involved. Areas with low
# Z-scores/RMSF indicate that the RMSD values have more value, as less probability is involved

