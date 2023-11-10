import warnings
# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")

import random
random.seed(0)
import pickle
from model.VGAE import *
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="pretrain/log/VGAE")

with open('pretrain/data/swissprot/graph.pkl', 'rb') as f:
    print("Loading data ...")
    graphs = pickle.load(f)
print("Data loaded successfully.")

def split_multi_graphs(graphs, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    # Shuffle the list of graphs to ensure randomness
    random.shuffle(graphs)

    # Calculate the number of graphs for each split
    total_graphs = len(graphs)
    num_train = int(total_graphs * train_ratio)
    num_test = int(total_graphs * test_ratio)
    num_valid = int(total_graphs * valid_ratio)

    # Split the dataset
    train_graphs = graphs[:num_train]
    test_graphs = graphs[num_train:num_train + num_test]
    valid_graphs = graphs[num_train + num_test:]

    return train_graphs, test_graphs, valid_graphs

train_graphs, test_graphs, valid_graphs = split_multi_graphs(graphs)

batch_size = 256
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.001)

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index, data.neg_edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)

def validation(model, valid_loader):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data in valid_loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index, data.neg_edge_index)
            val_loss += loss.item()
        val_loss /= len(valid_loader)
        return val_loss
    
# Training loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train_loss = train(vgae_model, train_loader, optimizer)
    val_loss = validation(vgae_model, valid_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

PATH = "/model/VGAE.pt"
torch.save(vgae_model, PATH)
print("Model saved")

def test_model(model, test_loader):
    model.eval()
    AUC = []
    AP = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            z = model.encode(data.x,data.edge_index) # encode train
            auc, ap = model.test(z, data.edge_index, data.neg_edge_index)
            AUC.append(auc)
            AP.append(ap)
    return sum(AUC)/len(AUC), sum(AP)/len(AP)

model = torch.load(PATH)
AUC, AP = test_model(model,test_loader)
print(f"AUC: {AUC}, AP: {AP}")