from Utils import *
data_folder = '/ProtDD'
# Load pre-trained models
script_directory = os.path.dirname(os.path.abspath(__file__))
vgae_model = torch.load(f"{script_directory}/../../model/VGAE.pt", map_location=device)
pae_model = torch.load(f"{script_directory}/../../model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load(f"{script_directory}/../../model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

# Attempt to read label.csv
try:
    df = pd.read_csv(f'{data_folder}/label.csv')
# Create label.csv
except:
    with open(f'{data_folder}/amino_enzymes.txt', 'r') as f:
        enzyme_ids = [line.strip() for line in f]

    with open(f'{data_folder}/amino_no_enzymes.txt', 'r') as f:
        non_enzyme_ids = [line.strip() for line in f]

    fold_dataframes = []
    for fold_id in range(10):
        with open(f'{data_folder}/amino_fold_{fold_id}.txt', 'r') as f:
            protein_ids = [line.strip() for line in f]

        labels = [1 if id in enzyme_ids else 0 for id in protein_ids]
        fold_df = pd.DataFrame({'id': protein_ids, 'label': labels, 'fold_id': fold_id})
        fold_dataframes.append(fold_df)

    df = pd.concat(fold_dataframes, ignore_index=True)
    df.to_csv(f'{data_folder}/label.csv', index=False)

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
with open(f'{data_folder}/multimodal.pkl', 'wb') as f:
    pickle.dump(mulmodal, f)

with open(f'{data_folder}/sequence.pkl', 'wb') as f:
    pickle.dump(sequence, f)

with open(f'{data_folder}/graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open(f'{data_folder}/point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)