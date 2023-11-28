from Utils import *

data_folder = '/Atom3D_MSP/split-by-sequence-identity-30/data'
folders = ['train', 'test', 'val']

# Load pre-trained models
script_directory = os.path.dirname(os.path.abspath(__file__))
vgae_model = torch.load(f"{script_directory}/../../model/VGAE.pt", map_location=device)
pae_model = torch.load(f"{script_directory}/../../model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load(f"{script_directory}/../../model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

for folder in folders:
    dataset = da.load_dataset(f'{data_folder}/{folder}', 'lmdb')

    label = []
    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []

    for i, struct in tqdm(enumerate(dataset), total = len(dataset)):
        original_atoms = struct['original_atoms']
        mutated_atoms = struct['mutated_atoms']
        multimodal_representation1, encoded_sequence1, encoded_graph1, encoded_point_cloud1 = get_multimodal_representation(original_atoms, esm_model, vgae_model, pae_model, fusion_model)
        multimodal_representation2, encoded_sequence2, encoded_graph2, encoded_point_cloud2 = get_multimodal_representation(original_atoms, esm_model, vgae_model, pae_model, fusion_model)
        multimodal_representation = torch.cat((multimodal_representation1, multimodal_representation2), dim=0)
        encoded_sequence = torch.cat((encoded_sequence1, encoded_sequence2), dim=0)
        encoded_graph = torch.cat((encoded_graph1, encoded_graph2), dim=0)
        encoded_point_cloud = torch.cat((encoded_point_cloud1, encoded_point_cloud2), dim=0)
        mulmodal.append(multimodal_representation.detach().numpy())
        sequence.append(encoded_sequence.detach().numpy())
        graph.append(encoded_graph.detach().numpy())
        point_cloud.append(encoded_point_cloud.detach().numpy())
        label.append(int(struct['label']))
       
    with open(f'{data_folder}/{folder}/multimodal.pkl', 'wb') as f:
        pickle.dump(mulmodal, f)

    with open(f'{data_folder}/{folder}/sequence.pkl', 'wb') as f:
        pickle.dump(sequence, f)

    with open(f'{data_folder}/{folder}/graph.pkl', 'wb') as f:
        pickle.dump(graph, f)

    with open(f'{data_folder}/{folder}/point_cloud.pkl', 'wb') as f:
        pickle.dump(point_cloud, f)
   
    with open(f'{data_folder}/{folder}/label.pkl', 'wb') as f:
        pickle.dump(label, f)