import pickle
import numpy as np
from Bio import PDB
import os
import torch
from tqdm import tqdm

# Function to convert a PDB file to a point cloud
def pdb_to_point_cloud(pdb_path, desired_num_points=2048):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except ValueError:
        return None
    coordinates = []
    for atom in structure.get_atoms():
        coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    num_points = coordinates.shape[0]

    # Pad or truncate the point cloud to the desired number of points
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]

    # Center and normalize the point cloud
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    coordinates = torch.FloatTensor(coordinates).permute(1, 0)
    return coordinates

# Directory containing PDB files
pdb_directory = "./pretrain/data/swissprot"
pdb_files = [f for f in os.listdir(pdb_directory) if os.path.splitext(f)[1] == ".pdb"]
print("The Number of files:", len(pdb_files))
dataset = []

# Process PDB files to create point clouds
for i, pdb_file in tqdm(enumerate(pdb_files)):
    pdb_path = os.path.join(pdb_directory, pdb_file)
    data = pdb_to_point_cloud(pdb_path)
    dataset.append(data)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1} files processed")
print("Done")

# Save the dataset of point clouds to a file
with open(f'{pdb_directory}/pointclouds.pkl', 'wb') as f:
    pickle.dump(dataset, f)
