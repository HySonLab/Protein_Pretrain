import numpy as np
import torch
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from Bio.PDB import PDBParser, PPBuilder, Polypeptide

# Initialize a PDB parser and PPBuilder for protein structure parsing
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

# Define the standard amino acids and create a LabelEncoder for encoding them
amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
label_encoder = LabelEncoder()
label_encoder.fit(list(amino_acids))
num_amino_acids = len(amino_acids)

# Function to one-hot encode amino acid sequences
def one_hot_encode_amino_acid(sequence):
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot

# Function to convert a PDB file to a PyTorch Geometric graph data object
def pdb_to_graph(pdb_path, k_neighbors=5):
    try:
      structure = parser.get_structure('protein', pdb_path)
    except ValueError:
      return None

    node_features = []
    coordinates = []
    sequence = ""

    for residue in structure.get_residues():
      if 'CA' in residue:
        aa_code = Polypeptide.three_to_one(residue.get_resname())
        sequence += aa_code
        coordinates.append(residue['CA'].get_coord())

    coordinates = np.array(coordinates, dtype=np.float32)
    try:
      node_features = one_hot_encode_amino_acid(sequence)
    except IndexError:
      return None
    x = torch.tensor(node_features, dtype=torch.float32)

    # Calculate K-nearest neighbors (KNN) edge indices
    edge_index = kneighbors_graph(coordinates, k_neighbors, mode='connectivity', include_self=False)
    edge_index = edge_index.nonzero()
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    # Generate negative edge samples
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=edge_index.size(1) // 2
    )

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    return data

# Directory containing PDB files
pdb_directory = 'pretrain/data/swissprot/'
pdb_files = [f for f in os.listdir(pdb_directory) if os.path.splitext(f)[1] == ".pdb"]
print("The Number of files:", len(pdb_files))

# Process PDB files to create graph data objects
graphs = []
for i, pdb_file in enumerate(pdb_files):
    pdb_path = os.path.join(pdb_directory, pdb_file)
    data = pdb_to_graph(pdb_path)
    if data:
        graphs.append(data)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} files processed")

# Save the dataset of graph data objects to a file
with open('pretrain/data/swissprot/graphs.pkl', 'wb') as f:
    pickle.dump(graphs, f)
print("Done")
