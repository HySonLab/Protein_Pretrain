from Utils import *
# Load your pre-trained models
vgae_model = torch.load("/model/VGAE.pt", map_location=device)
pae_model = torch.load("/model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load("/model/Fusion.pt", map_location=device)

print("Pre-trained models loaded successfully.")
data_folder = '/downstreamtasks/data/SCOPe1.75/'
hdf5_folder_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

for folder in hdf5_folder_name:
    with open(f'{data_folder}/{folder}.txt', 'r') as file:
        lines = file.readlines()
        hdf5_file_names = [line.split()[0] for line in lines]
    
    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []
    target_path = data_folder + folder
    for i, hdf5_file_name in tqdm(enumerate(hdf5_file_names), total = len(hdf5_file_names)):
        hdf5_path = f'{data_folder}{folder}/{hdf5_file_name}.hdf5'
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation_from_hdf5(hdf5_path, esm_model, vgae_model, pae_model, fusion_model)
        
        mulmodal.append(multimodal_representation.detach().numpy())
        sequence.append(encoded_sequence.detach().numpy())
        graph.append(encoded_graph.detach().numpy())
        point_cloud.append(encoded_point_cloud.detach().numpy())
   
    with open(f'{target_path}/multimodal.pkl', 'wb') as f:
        pickle.dump(mulmodal, f)

    with open(f'{target_path}/sequence.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    with open(f'{target_path}/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)

    with open(f'{target_path}/point_cloud.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)
    
    print(f"Saved {folder} folder")