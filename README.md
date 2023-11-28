# Multimodal Pretraining for Unsupervised Protein Representation Learning

![framework](./figures/framework.png)

### Paper

### Contributors
* Viet Thanh Duy Nguyen
* Truong Son Hy (Correspondent / PI)

### Data Downloading
* For pretraining:
    * Create ```/pretrain/data/swissprot``` directory.
    * Download the data at [Swiss Prot](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar).
    * Move the downloaded files into the ```/pretrain/data/swissprot``` directory and extract it.
* For downstream tasks:
    * Create ```downstreamtasks/data/{dataset_name}``` directory, where dataset_name is in {Atom3D_MSP, DAVIS, KIBA, PDBbind, SCOPe1.75, D&D}.
    * Download the data at: [Atom3D_MSP](https://zenodo.org/records/4962515/files/MSP-split-by-sequence-identity-30.tar.gz), [DAVIS](https://drive.google.com/file/d/1kobzvO9aZcCAOWqXodEZi9xbr4PuGHYY/view?usp=drive_link),
    [KIBA](https://drive.google.com/file/d/1X8LQZYjShhKo0YOkTZ-zfzftN1V-Wbyx/view?usp=drive_link), [PDBbind](), [SCOPe1.75](https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar), [D&D](https://drive.google.com/uc?export=download&id=1KTs5cUYhG60C6WagFp4Pg8xeMgvbLfhB).
    * Move the downloaded files into the ```downstreamtasks/data/{dataset_name}``` directory and extract it.

### Data Preprocessing
* For pretraining, run following commands:
```
cd /pretrain/data/
python *.py
```
* For downstream tasks, run following commands:
```
cd /downstreamtasks/data/
python *.py
```
### Pretraining
* For pretraining all models, run following commands:
```
cd /pretraining/
python *.py
```
* For pretraining a specific model, run the following commands, where model_name is either VGAE, PAE, or Fusion:
```
cd /pretraining/
python {model_name}.py
```
### Downstream Tasks
* For conducting experiment for all the downstream tasks, run following commands:
```
cd /downstreamtasks/
python *.py
```
* For conducting experiment for a specific task, run the following commands, where task_name is in {EI, MSP, PFC, PLA}:
```
cd /downstreamtasks/
python {task_name}.py
```
### Citation