import warnings
# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")
import pickle
from model.Auto_Fusion import *
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset

writer = SummaryWriter(log_dir="pretrain/log/Fusion")

class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Load the preprocessed data
with open('pretrain/data/swissprot/fusion.pkl', 'rb') as f:
    print("Loading data ...")
    dataset = pickle.load(f)
print("Data loaded successfully.")

# Create a custom dataset from the loaded data
dataset = MultimodalDataset(dataset)

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

# Split the dataset into train, validation, and test sets using random_split
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size]
)

batch_size = 256

# Create data loaders for training, validation, and test
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function (Mean Squared Error) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

# Training function
def train():
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        encoding = model.encode(batch)
        restoration = model.decode(encoding)
        loss = criterion(batch, restoration)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss

# Validation function
def validation():
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            encoding = model.encode(batch)
            restoration = model.decode(encoding)
            val_loss = criterion(batch, restoration)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(valid_loader)
    return average_val_loss

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train()
    val_loss = validation()
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

# Save the fusion model to a file
PATH = "/model/Fusion.pt"
torch.save(fusion_model, PATH)

# Load the saved model
model = torch.load(PATH)

# Test function
def test():
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            encoding = model.encode(data)
            restoration = model.decode(encoding)
            test_loss = criterion(data, restoration)
            total_test_loss += test_loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    return average_test_loss

# Evaluate the model on the test dataset
test_loss = test()
print(f"Average MSE on Test Set: {test_loss:.4f}")