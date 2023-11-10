from Utils import *
data_folder = '/downstreamttasks/ProtDD/'

df = pd.read_csv(f'/{data_folder}data.csv')
# Load your pre-trained models
vgae_model = torch.load("/model/VGAE.pt", map_location=device)
pae_model = torch.load("/model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load("/model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

mulmodal = []
sequence = []
graph = []
point_cloud = []

for i, hdf5_file in tqdm(enumerate(df['id']), total = len(df['id'])):
    hdf5_path = f'/{data_folder}/{hdf5_file}.hdf5'
    multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation_from_hdf5(hdf5_path, esm_model, vgae_model, pae_model, fusion_model)
    mulmodal.append(multimodal_representation.detach().numpy())
    sequence.append(encoded_sequence.detach().numpy())
    graph.append(encoded_graph.detach().numpy())
    point_cloud.append(encoded_point_cloud.detach().numpy())

with open(f'/{data_folder}multimodal.pkl', 'wb') as f:
    pickle.dump(mulmodal, f)

with open(f'/{data_folder}sequence.pkl', 'wb') as f:
    pickle.dump(sequence, f)

with open(f'/{data_folder}graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open(f'/{data_folder}point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)