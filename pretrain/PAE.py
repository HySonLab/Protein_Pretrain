import warnings
# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")

import pickle
import math

from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset

from tensorboardX import SummaryWriter

import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
model_path = f"{script_directory}/../"
sys.path.append(model_path)

from model.PAE import *

# Create a summary writer for logging
writer = SummaryWriter(log_dir="/log/PAE")

# Define a custom dataset class for handling point cloud data
class PointClouds(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]

# Load the point cloud data
with open('/data/swissprot/pointcloud.pkl', 'rb') as f:
    print("Loading data ...")
    dataset = pickle.load(f)
print("Data loaded successfully.")

# Create instances of the custom dataset class for train, validation, and test sets
dataset = PointClouds(dataset)

# Split the dataset into train, validation, and test sets
train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2

train_size = int(len(dataset) * train_ratio)
valid_size = int(len(dataset) * valid_ratio)
test_size = len(dataset) - train_size - valid_size

print("Train dataset size:", train_size)
print("Validation dataset size:", valid_size)
print("Test dataset size:", test_size)

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size]
)

batch_size = 256

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a custom Chamfer Distance loss function
class ChamferDistance(nn.Module):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # Compute pairwise L2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)

# Create an optimizer for the PAE (Point Cloud Auto-Encoder) model
optimizer = optim.Adam(pae_model.parameters(), lr=0.001)

# Function to apply random rotations to input data
def random_rotation_matrix(batch_size):
    # Generate random angles for rotation
    angles = torch.rand((batch_size, 3)) * 2 * math.pi

    # Compute cosines and sines of the angles
    cosines = torch.cos(angles)
    sines = torch.sin(angles)

    # Create the rotation matrices for each axis
    Rx = torch.stack([torch.tensor([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]], dtype=torch.float32) for c, s in zip(cosines[:, 0], sines[:, 0])])

    Ry = torch.stack([torch.tensor([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]], dtype=torch.float32) for c, s in zip(cosines[:, 1], sines[:, 1])])

    Rz = torch.stack([torch.tensor([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]], dtype=torch.float32) for c, s in zip(cosines[:, 2], sines[:, 2])])

    # Multiply the rotation matrices to get the final rotation matrices
    rotation_matrices = torch.matmul(Rz, torch.matmul(Ry, Rx))

    return rotation_matrices

# Initialize the Chamfer Distance loss
chamfer_distance = ChamferDistance()

# Training function
def train():
    pae_model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        rotation_matrix = random_rotation_matrix(data.size(0)).to(device)
        data = torch.matmul(rotation_matrix, data)

        optimizer.zero_grad()
        encoding = pae_model.encode(data)

        restoration = pae_model.decode(encoding)
        loss = chamfer_distance(data, restoration)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss

# Validation function
def validation():
    pae_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            encoding = pae_model.encode(data)
            restoration = pae_model.decode(encoding)

            # Calculate the Chamfer distance loss on the validation set
            val_loss = chamfer_distance(data, restoration)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(valid_loader)
    return average_val_loss

num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    train_loss = train()
    val_loss = validation()
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

# Define the file path for saving the model
script_directory = os.path.dirname(os.path.abspath(__file__))
PATH = f"{script_directory}/../model/PAE.pt"

# Save the PAE model
torch.save(pae_model, PATH)
print("Model saved")

# Testing function
def test(pae_model):
    pae_model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            encoding = pae_model.encode(data)
            restoration = pae_model.decode(encoding)

            # Calculate the Chamfer distance loss on the test set
            test_loss = chamfer_distance(data, restoration)
            total_test_loss += test_loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    return average_test_loss

pae_model = torch.load(PATH)
# Evaluate the model on the test dataset
test_loss = test(pae_model)
print(f"Average Chamfer Distance on Test Set: {test_loss:.4f}")
