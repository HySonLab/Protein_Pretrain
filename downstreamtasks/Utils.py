import atom3d.datasets as da
from atom3d.filters import filters
import atom3d.protein.sequence as seq
import atom3d.util.formats as fo
import argparse
import ast
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
import pickle
from math import sqrt
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import torch
from tqdm import tqdm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import transformers
import pandas as pd
import os
import glob
import gpytorch
import h5py
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from xgboost import XGBClassifier
from Bio.PDB import PDBParser, PPBuilder, Polypeptide
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

import sys
script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = f"{script_directory}/../"
sys.path.append(model_path)

from model.Auto_Fusion import *
from model.ESM import *
from model.VGAE import *
from model.PAE import *

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU.")

model_token = "facebook/esm2_t30_150M_UR50D"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_token)

amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
label_encoder = LabelEncoder()
label_encoder.fit(list(amino_acids))
num_amino_acids = len(amino_acids)

def one_hot_encode_amino_acid(sequence = None, amino_acid_indices=None):
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot

def z_score_standardization(tensor):
  mean = tensor.mean()
  std = tensor.std()
  if std != 0:
      standardized_tensor = (tensor - mean) / std
  else:
      standardized_tensor = tensor  # Handle the case when std is 0
  return standardized_tensor

def process_encoded_graph(encoded_graph, edge_index, fixed_size = 640, feature_dim = 10):
    num_nodes = encoded_graph.size(0)

    if num_nodes > fixed_size:
        ratio = fixed_size / num_nodes
        with torch.no_grad():
            pooling_layer = TopKPooling(in_channels=feature_dim, ratio=ratio)
            pooled_x, edge_index, edge_attr, batch, perm, score = pooling_layer(encoded_graph, edge_index)
        processed_encoded_graph = pooled_x
    else:
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)

    return processed_encoded_graph[:fixed_size]

def read_pdb(pdb_path):
  structure = parser.get_structure('protein', pdb_path)

  # Graph
  coordinates = []
  sequence = ""
  k_neighbors = 5
  for residue in structure.get_residues():
    if 'CA' in residue:
        try:  
            aa_code = Polypeptide.three_to_one(residue.get_resname())
        except KeyError:
            aa_code = "X"
        sequence += aa_code
        coordinates.append(residue['CA'].get_coord())        
  coordinates = np.array(coordinates, dtype=np.float32)
  node_features = one_hot_encode_amino_acid(sequence)
  x = torch.tensor(node_features, dtype=torch.float32)
  edge_index = kneighbors_graph(coordinates, k_neighbors, mode='connectivity', include_self=False)
  edge_index = edge_index.nonzero()
  edge_index = np.array(edge_index)
  edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
  neg_edge_index = negative_sampling(
      edge_index= edge_index,
      num_nodes= x.size(0),
      num_neg_samples= edge_index.size(1)//2
  )
  graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

  # Point Cloud
  coordinates = []
  desired_num_points = 2048
  for atom in structure.get_atoms():
      coordinates.append(atom.get_coord())
  coordinates = np.array(coordinates, dtype=np.float32)
  num_points = coordinates.shape[0]
  if num_points < desired_num_points:
      padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
      coordinates = np.concatenate((coordinates, padding), axis=0)
  elif num_points > desired_num_points:
      coordinates = coordinates[:desired_num_points, :]
  coordinates = torch.tensor(coordinates, dtype=torch.float32)
  coordinates -= coordinates.mean(0)
  d = np.sqrt((coordinates ** 2).sum(1))
  coordinates /= d.max()
  point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

  # Sequence
  sequence = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"]

  return sequence, graph, point_cloud

def read_atom3d(atoms_df):


    atoms_df = filters.standard_residue_filter(atoms_df)
    # Graph
    C_alpha_df = atoms_df[atoms_df['name'] == "CA"]
    residue_coords = fo.get_coordinates_from_df(C_alpha_df)
    residue_coords = np.array(residue_coords, dtype=np.float32)
    sequence = ""
    for chain_sequence in seq.get_chain_sequences(atoms_df): sequence += chain_sequence[1]
    node_features = one_hot_encode_amino_acid(sequence)
    x = torch.tensor(node_features, dtype=torch.float32)
    k_neighbors = min(5, len(sequence) - 1)
    edge_index = kneighbors_graph(residue_coords, k_neighbors, mode='connectivity', include_self=False)
    edge_index = edge_index.nonzero()
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
    neg_edge_index = negative_sampling(
        edge_index= edge_index,
        num_nodes= x.size(0),
        num_neg_samples= edge_index.size(1)//2
    )
    graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)
   
    # Point Cloud
    atom_coords = fo.get_coordinates_from_df(atoms_df)
    desired_num_points = 2048
    atom_coords = np.array(atom_coords, dtype=np.float32)
    num_points = atom_coords.shape[0]
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        atom_coords = np.concatenate((atom_coords, padding), axis=0)
    elif num_points > desired_num_points:
        atom_coords = atom_coords[:desired_num_points, :]
    atom_coords = torch.tensor(atom_coords, dtype=torch.float32)
    atom_coords -= atom_coords.mean(0)
    d = np.sqrt((atom_coords ** 2).sum(1))
    atom_coords /= d.max()
    point_cloud = torch.FloatTensor(atom_coords).permute(1, 0)


    # Sequence
    sequence = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=500)["input_ids"]
    return sequence, graph, point_cloud

def get_multimodal_representation(protein_path, ESM, VGAE, PAE, Fusion):

    # Check file_extension
    file_name, file_extension = os.path.splitext(protein_path)
    if file_extension == "pdb":
        sequence, graph, point_cloud = read_pdb(protein_path)
    elif file_extension == "hdf5":
        sequence, graph, point_cloud = read_hdf5(protein_path)
    
    # Using GPU
    graph = graph.to(device)
    point_cloud = point_cloud.to(device)
    sequence = sequence.to(device)

    # Pass the sequence data through ESM for encoding
    with torch.no_grad():
        encoded_sequence = ESM(sequence, output_hidden_states=True)['hidden_states'][-1][0,-1].to("cpu")
        encoded_sequence = z_score_standardization(encoded_sequence)

    # Pass the graph data through VGAE for encoding
    with torch.no_grad():
        encoded_graph = VGAE.encode(graph.x, graph.edge_index).to("cpu")
        encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index.to("cpu"))
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Pass the point cloud data through PAE for encoding
    with torch.no_grad():
        encoded_point_cloud = PAE.encode(point_cloud[None, :]).squeeze().to("cpu")
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    concatenated_data = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0).unsqueeze(0).to(device)
    multimodal_representation = Fusion.encode(concatenated_data).squeeze().to("cpu")
    return  (multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud)

def get_ligand_representation(ligand_smiles):
    mol = Chem.MolFromSmiles(ligand_smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Change radius as needed
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)
    return fingerprint_tensor

def read_hdf5(hdf5_path):
    hdf5_file = h5py.File(hdf5_path, "r")
    # Sequence Processing
    amino_acid_indices = hdf5_file['amino_types'][:]
    amino_acid_indices[amino_acid_indices > 20] = 20
    amino_acid_indices[amino_acid_indices == - 1] = 20
    sequence = ''.join(label_encoder.inverse_transform(amino_acid_indices))
    sequence_token = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"]

    # Graph Processing
    amino_pos =  hdf5_file['amino_pos'][:]
    k_neighbors = 5
    coordinates = np.array(amino_pos, dtype=np.float32).squeeze()
    node_features = one_hot_encode_amino_acid(sequence)
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = kneighbors_graph(coordinates, k_neighbors, mode='connectivity', include_self=False)
    edge_index = edge_index.nonzero()
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
    neg_edge_index = negative_sampling(
        edge_index= edge_index,
        num_nodes= x.size(0),
        num_neg_samples= edge_index.size(1)//2
    )
    graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    # Point Cloud Processing
    atom_pos = hdf5_file['atom_pos'][:]
    desired_num_points = 2048
    coordinates = np.array(atom_pos, dtype=np.float32).squeeze()
    num_points = coordinates.shape[0]
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

    return sequence_token, graph, point_cloud

def suffle(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2

def get_cindex(Y, P):
    return concordance_index(Y, P)

# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)

# Prepare for rm2
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)

# Prepare for rm2
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)

def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))

def get_rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def get_mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def get_pearson(y,f):
    rp = pearsonr(y,f)[0]
    return rp

def get_spearman(y,f):
    sp = spearmanr(y,f)[0]

    return sp