import os
import pickle
from Bio.PDB import PDBParser, Polypeptide
from model.ESM import *
parser = PDBParser(QUIET=True)

# Function to extract amino acid sequence from a PDB file
def pdb_to_sequence(pdb_path):
    try:
        structure = parser.get_structure('protein', pdb_path)
    except ValueError:
        return None
    sequence = ""

    for residue in structure.get_residues():
        if 'CA' in residue:
            aa_code = Polypeptide.three_to_one(residue.get_resname())
            sequence += aa_code
    return sequence

pdb_directory = "/swissprot/"
pdb_files = [f for f in os.listdir(pdb_directory) if os.path.splitext(f)[1] == ".pdb"]
print("The Number of files:", len(pdb_files))

tokenized_sequences = []

# Loop through PDB files, convert to sequences, and tokenize
for i, pdb_file in enumerate(pdb_files):
    pdb_path = os.path.join(pdb_directory, pdb_file)
    sequence = pdb_to_sequence(pdb_path)
    tokenized_sequence = esm_tokenizer(sequence, return_tensors="pt", padding=True)["input_ids"]
    tokenized_sequences.append(tokenized_sequence)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} files processed")

print("Done")

# Save tokenized sequences to a pickle file
with open('/swissprot/sequences.pkl', 'wb') as f:
    pickle.dump(tokenized_sequences, f)
