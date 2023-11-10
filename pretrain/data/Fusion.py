import pickle
from model.ESM import *
from model.VGAE import *
from model.PAE import *
from model.Auto_Fusion import *
from torch_geometric.nn import TopKPooling

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU.")

# Load pre-trained models
vgae_model = torch.load("/model/VGAE.pt", map_location=device)
pae_model = torch.load("/model/PAE.pt", map_location=device)
esm_model = transformers.AutoModelForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D")
esm_model = esm_model.to(device)

data_folder = '/pretrain/data/swissprot/'

with open(f'{data_folder}graphs.pkl', 'rb') as f:
    print("Loading graph data ...")
    graph_data = pickle.load(f)
print("Graph data loaded successfully.")

with open(f'{data_folder}pointcloud.pkl', 'rb') as f:
    print("Loading point cloud data ...")
    point_cloud_data = pickle.load(f)
print("Point Cloud data loaded successfully.")

with open(f'{data_folder}sequences.pkl', 'rb') as f:
    print("Loading sequence data ...")
    sequence_data = pickle.load(f)
print("Sequence data loaded successfully.")

# Function for Z-score standardization
def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor

# Function to process and adjust encoded graph data to a fixed size
def process_encoded_graph(encoded_graph, edge_index, fixed_size=640, feature_dim=10):
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
    return processed_encoded_graph

processed_data_list = []

for i, (graph, point_cloud, sequence) in enumerate(zip(graph_data, point_cloud_data, sequence_data)):
    # Encode sequence data using ESM
    with torch.no_grad():
        encoded_sequence = esm_model(sequence, output_hidden_states=True)['hidden_states'][-1][0, -1]
        encoded_sequence = z_score_standardization(encoded_sequence)

    # Encode graph data using VGAE
    with torch.no_grad():
        encoded_graph = vgae_model.encode(graph.x, graph.edge_index)
        encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index)
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Encode point cloud data using PAE
    with torch.no_grad():
        encoded_point_cloud = pae_model.encode(point_cloud[None, :]).squeeze()
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    concatenated_data = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0)
    processed_data_list.append(concatenated_data)
print("Done")

# Print the shapes of the encoded data
print("Encoded Sequence Shape:", encoded_sequence.shape)
print("Encoded Graph Shape:", encoded_graph.shape)
print("Encoded Point Cloud Shape:", encoded_point_cloud.shape)
print(concatenated_data.shape)

with open(f'{data_folder}fusion.pkl', 'wb') as f:
    pickle.dump(processed_data_list, f)
