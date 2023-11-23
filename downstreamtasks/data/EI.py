from Utils import *
data_folder = '/ProtDD/'
# Load pre-trained models
script_directory = os.path.dirname(os.path.abspath(__file__))
vgae_model = torch.load(f"{script_directory}/../../model/VGAE.pt", map_location=device)
pae_model = torch.load(f"{script_directory}/../../model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load(f"{script_directory}/../../model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

# Read the label CSV file
df = pd.read_csv(f'/{data_folder}data.csv')
print("Number of samples:", len(df))

# Initialize empty lists to store multimodal representations
mulmodal = []
sequence = []
graph = []
point_cloud = []

# Iterate through the HDF5 files and extract multimodal representations
for i, hdf5_file in tqdm(enumerate(df['id']), total=len(df['id'])):
    hdf5_path = f'/{data_folder}/hdf5/{hdf5_file}.hdf5'
    
    # Get multimodal representations from the HDF5 file using pre-trained models
    multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation(hdf5_path, esm_model, vgae_model, pae_model, fusion_model)
    
    # Append the representations to their respective lists
    mulmodal.append(multimodal_representation.detach().numpy())
    sequence.append(encoded_sequence.detach().numpy())
    graph.append(encoded_graph.detach().numpy())
    point_cloud.append(encoded_point_cloud.detach().numpy())

# Save the multimodal representations to pickle files
with open(f'/{data_folder}multimodal.pkl', 'wb') as f:
    pickle.dump(mulmodal, f)

with open(f'/{data_folder}sequence.pkl', 'wb') as f:
    pickle.dump(sequence, f)

with open(f'/{data_folder}graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open(f'/{data_folder}point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)