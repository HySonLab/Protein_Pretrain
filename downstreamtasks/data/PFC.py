from Utils import *

# Load pre-trained models
script_directory = os.path.dirname(os.path.abspath(__file__))
vgae_model = torch.load(f"{script_directory}/../../model/VGAE.pt", map_location=device)
pae_model = torch.load(f"{script_directory}/../../model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load(f"{script_directory}/../../model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

print("Pre-trained models loaded successfully.")

data_folder = '/SCOPe1.75'  # Define the data folder
hdf5_folder_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

# Iterate through different folders containing text files
for folder in hdf5_folder_name:
    with open(f'{data_folder}/{folder}.txt', 'r') as file:
        lines = file.readlines()
        hdf5_file_names = [line.split()[0] for line in lines]

    # Initialize lists to store multimodal representations
    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []

    # Iterate through HDF5 files in the current folder
    for i, hdf5_file_name in tqdm(enumerate(hdf5_file_names), total=len(hdf5_file_names)):
        hdf5_path = f'{data_folder}/{folder}/{hdf5_file_name}.hdf5'

        # Get multimodal representations from the HDF5 file using pre-trained models
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation(
            hdf5_path, esm_model, vgae_model, pae_model, fusion_model)

        # Append the representations to their respective lists
        mulmodal.append(multimodal_representation.detach().numpy())
        sequence.append(encoded_sequence.detach().numpy())
        graph.append(encoded_graph.detach().numpy())
        point_cloud.append(encoded_point_cloud.detach().numpy())

    # Save the multimodal representations to pickle files in the target folder
    with open(f'{data_folder}/{folder}/multimodal.pkl', 'wb') as f:
        pickle.dump(mulmodal, f)

    with open(f'{data_folder}/{folder}/sequence.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    with open(f'{data_folder}/{folder}/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)

    with open(f'{data_folder}/{folder}/point_cloud.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)

    print(f"Saved {folder} folder")
