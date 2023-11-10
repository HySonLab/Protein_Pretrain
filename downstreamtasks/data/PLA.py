from Utils import *
# Load your pre-trained models
vgae_model = torch.load("/model/VGAE.pt", map_location=device)
pae_model = torch.load("/model/PAE.pt", map_location=device)
model_token = "facebook/esm2_t30_150M_UR50D"
esm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_token)
esm_model = esm_model.to(device)
fusion_model = torch.load("/model/Fusion.pt", map_location=device)
print("Pre-trained models loaded successfully.")

dataset = ['KIBA', 'DAVIS', "PDBbind"]
dataset_id = 0
data_folder = f'/downstreamtasks/data/{dataset_id}/'

df = pd.read_csv(f'{data_folder}label.csv')
print("Number of samples:", df.size)
mulmodal = []
sequence = []
graph = []
point_cloud = []
for i, (ligand_smiles, protein_name) in tqdm(enumerate(zip(df["ligand"], df["protein"])),total=len(df)):
  pdb_path = f"/downstreamtask/data/KIBA/pdb/{protein_name}.pdb"
  multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_multimodal_representation(pdb_path, esm_model, vgae_model, pae_model, fusion_model)
  ligand_representation = get_ligand_representation(ligand_smiles)

  mulmodal_feature = torch.cat((multimodal_representation, ligand_representation), dim=0).detach().numpy()
  sequence_feature =  torch.cat((encoded_sequence, ligand_representation), dim=0).detach().numpy()
  graph_feature =  torch.cat((encoded_graph, ligand_representation), dim=0).detach().numpy()
  point_cloud_feature =  torch.cat((encoded_point_cloud, ligand_representation), dim=0).detach().numpy()
  
  mulmodal.append(mulmodal_feature)
  sequence.append(sequence_feature)
  graph.append(graph_feature)
  point_cloud.append(point_cloud_feature)

with open(f'{data_folder}multimodal.pkl', 'wb') as f:
    pickle.dump(mulmodal, f)

with open(f'{data_folder}sequence.pkl', 'wb') as f:
    pickle.dump(sequence, f)

with open(f'{data_folder}graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

with open(f'{data_folder}point_cloud.pkl', 'wb') as f:
    pickle.dump(point_cloud, f)