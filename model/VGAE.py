from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
import torch

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

out_channels = 10
num_features = 21
vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgae_model = vgae_model.to(device)