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

# Specify the dataset you are working with
dataset = ['KIBA', 'DAVIS', "PDBbind"]
dataset_id = 0
data_folder = f'/{dataset[dataset_id]}/'

# Read the label CSV file
df = pd.read_csv(f'{data_folder}label.csv')
print("Number of samples:", len(df))

mulmodal = []
sequence = []
graph = []
point_cloud = []

# Iterate through the dataset to process each sample
for i, (ligand_smiles, protein_name) in tqdm(enumerate(zip(df["ligand"], df["protein"]), total=len(df))):
    pdb_path = f"/downstreamtask/data/{dataset[dataset_id]}/pdb/{protein_name}.pdb"
    multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation(pdb_path, esm_model, vgae_model, pae_model, fusion_model)
    ligand_representation = get_ligand_representation(ligand_smiles)

    # Concatenate multimodal representation with ligand representation
    mulmodal_feature = torch.cat((multimodal_representation, ligand_representation), dim=0).detach().numpy()
    sequence_feature = torch.cat((encoded_sequence, ligand_representation), dim=0).detach().numpy()
    graph_feature = torch.cat((encoded_graph, ligand_representation), dim=0).detach().numpy()
    point_cloud_feature = torch.cat((encoded_point_cloud, ligand_representation), dim=0).detach().numpy()

    # Append the features to their respective lists
    mulmodal.append(mulmodal_feature)
    sequence.append(sequence_feature)
    graph.append(graph_feature)
    point_cloud.append(point_cloud_feature)

# Save the features to pickle files
with open(f'{data_folder}multimodal.pkl', 'wb') as f:
    pickle.dump(mulmodal, f)

with open(f'{data_folder}sequence.pkl', 'wb') as f:
    pickle.dump(sequence, f)

with open(f'{data_folder}graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open(f'{data_folder}point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)
