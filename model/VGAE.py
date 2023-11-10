from torch_geometric.nn import GCNConv, VGAE
import torch

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()

        # Define GCN layers for the encoder
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        # Forward pass through the GCN layers with ReLU activation
        x = self.conv1(x, edge_index).relu()

        # Calculate mean (mu) and log standard deviation (logstd)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)

        return mu, logstd

# Define the output dimensions for the model
out_channels = 10
num_features = 21

# Create an instance of VGAE using the VariationalGCNEncoder
vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))

# Check if GPU is available, and move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgae_model = vgae_model.to(device)
