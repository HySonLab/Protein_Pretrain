# Multimodal Pretraining for Unsupervised Protein Representation Learning

![framework](./figures/framework.png)

### Paper

### Contributors
* Viet Thanh Duy Nguyen
* Truong Son Hy (Correspondent / PI)

### Data Downloading
* For pretraining:
    1. Create ```/pretrain/data/swissprot``` directory.
    2. Download the data at [Swiss Prot](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar).
    3. Move the downloaded files into the ```/pretrain/data/swissprot``` directory and extract it.
* For downstream tasks:
    1. Create ```downstreamtasks/data/{dataset_name}``` directory.
    2. Download the data at: [Atom3D_MSP](https://zenodo.org/records/4962515/files/MSP-split-by-sequence-identity-30.tar.gz), [DAVIS](),
    [KIBA](), [PDBbind](), [SCOPe1.75](https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar), [D&D](https://drive.google.com/uc?export=download&id=1KTs5cUYhG60C6WagFp4Pg8xeMgvbLfhB).
    3. Move the downloaded files into the ```downstreamtasks/data/{dataset_name}``` directory and extract it.

### Data Preprocessing
* For pretraining, run these commands:
    ```
    cd /pretrain/data/
    python *.py
    ```
* For downstream tasks, run these commands:
    ```
    cd /downstreamtasks/data/
    python *.py
    ```
### Pretraining

### Downstream Tasks

### Citation