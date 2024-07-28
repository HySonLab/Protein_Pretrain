# Multimodal Pretraining for Unsupervised Protein Representation Learning

Published at journal *Biology Methods & Protocols* (Oxford University Press): 

https://doi.org/10.1093/biomethods/bpae043

Preprint:

https://doi.org/10.1101/2023.11.29.569288

![framework](./figures/framework.png)

### Contributors
* Viet Thanh Duy Nguyen
* Truong Son Hy (Correspondent / PI)

### Environment Setup

```
git clone https://github.com/HySonLab/Protein_Pretrain.git
cd Protein_Pretrain
conda env create -f environment.yml
conda activate MPRL
```

### Data Downloading
* For pretraining:
    * Create ```/pretrain/data/swissprot``` directory.
    * Download the data at [Swiss Prot](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar).
    * Move the downloaded files into the ```/pretrain/data/swissprot``` directory and extract it.
* For downstream tasks:
    * Create ```downstreamtasks/data/{dataset_name}``` directory, where dataset_name is in {Atom3D_MSP, DAVIS, KIBA, PDBbind, SCOPe1.75, D&D}.
    * Download the data at: [Atom3D_MSP](https://zenodo.org/records/4962515/files/MSP-split-by-sequence-identity-30.tar.gz), [DAVIS](https://drive.google.com/file/d/1kobzvO9aZcCAOWqXodEZi9xbr4PuGHYY/view?usp=drive_link),
    [KIBA](https://drive.google.com/file/d/1X8LQZYjShhKo0YOkTZ-zfzftN1V-Wbyx/view?usp=drive_link), [PDBbind](http://pdbbind.org.cn/download.php), [SCOPe1.75](https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar), [D&D](https://drive.google.com/uc?export=download&id=1KTs5cUYhG60C6WagFp4Pg8xeMgvbLfhB).
    * Move the downloaded files into the ```downstreamtasks/data/{dataset_name}``` directory and extract it.

### Data Preprocessing
1. For pretraining, run following commands:
```
python ./pretrain/data/{model_name}.py
```
Replace {model_name} with the specific model identifier:
* VGAE: Variational Graph Autoencoder
* PAE: PointNet Autoencoder
* Auto-Fusion

2. For downstream tasks, run following commands:
```
python ./downstreamtasks/data/{task_name}.py
```
Replace {task_name} with the specific task identifier:
* PLA: Protein-ligand Binding Affinity
* PFC: Protein Fold Classification
* EI: Enzyme Identification
* MSP: Mutation Stability Prediction

### Pretraining
You have two options for using our models for pretraining: training from scratch or using our pre-trained model checkpoints.
#### Training from Scratch
If you prefer to train the model yourself, run the following commands:
```
python ./pretrain/{model_name}.py --mode your_mode
```
* Command-line Arguments
    * `--mode`: Select the mode (`train` or `test`).
#### Using Pre-trained Model Checkpoints
For those who want to bypass the training phase, we provide pretrained model checkpoints that you can use directly in your projects. Download and integrate the pretrained model checkpoints from our shared drive: [checkpoints](https://drive.google.com/drive/folders/1_BBRPe2aC7rIg3HB6b5KkucvN1WOexwY?usp=sharing)

### Downstream Tasks
* Run following commands:
```
python ./downstreamtasks/{task_name}.py --mode your_mode --modal your_modal 
```
* Command-line Arguments
    * `--modal`: Select the modality (`sequence`, `graph`, `point_cloud` or `multimodal`).
    * `--mode`: Select the mode (`train` or `test`).
    * `--test_dataset` (Only available for PFC task): Select the test dataset for testing (`test_family`, `test_fold`, or `test_superfamily`).
    * `--dataset` (Only available for PLA task): Select the dataset (`DAVIS`, `KIBA` or `PDBBind`)
### Please cite our paper!

```bibtex
@article{10.1093/biomethods/bpae043,
    author = {Duy Nguyen, Viet Thanh and Son Hy, Truong},
    title = "{Multimodal pretraining for unsupervised protein representation learning}",
    journal = {Biology Methods and Protocols},
    pages = {bpae043},
    year = {2024},
    month = {06},
    abstract = "{Proteins are complex biomolecules essential for numerous biological processes, making them crucial targets for advancements in molecular biology, medical research, and drug design. Understanding their intricate, hierarchical structures and functions is vital for progress in these fields. To capture this complexity, we introduce MPRL—Multimodal Protein Representation Learning, a novel framework for symmetry-preserving multimodal pretraining that learns unified, unsupervised protein representations by integrating primary and tertiary structures. MPRL employs Evolutionary Scale Modeling (ESM-2) for sequence analysis, Variational Graph Auto-Encoders (VGAE) for residue-level graphs, and PointNet Autoencoder (PAE) for 3D point clouds of atoms, each designed to capture the spatial and evolutionary intricacies of proteins while preserving critical symmetries. By leveraging Auto-Fusion to synthesize joint representations from these pretrained models, MPRL ensures robust and comprehensive protein representations. Our extensive evaluation demonstrates that MPRL significantly enhances performance in various tasks such as protein-ligand binding affinity prediction, protein fold classification, enzyme activity identification, and mutation stability prediction. This framework advances the understanding of protein dynamics and facilitates future research in the field. Our source code is publicly available at https://github.com/HySonLab/Protein\_Pretrain.}",
    issn = {2396-8923},
    doi = {10.1093/biomethods/bpae043},
    url = {https://doi.org/10.1093/biomethods/bpae043},
    eprint = {https://academic.oup.com/biomethods/advance-article-pdf/doi/10.1093/biomethods/bpae043/58272243/bpae043.pdf},
}
```

```bibtex
@article{Nguyen2023.11.29.569288,
	author = {Viet Thanh Duy Nguyen and Truong Son Hy},
	title = {Multimodal Pretraining for Unsupervised Protein Representation Learning},
	elocation-id = {2023.11.29.569288},
	year = {2023},
	doi = {10.1101/2023.11.29.569288},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {In this paper, we introduce a framework of symmetry-preserving multimodal pretraining to learn a unified representation on proteins in an unsupervised manner that can take into account primary and tertiary structures. For each structure, we propose the corresponding pretraining method on sequence, graph and 3D point clouds based on large language models and generative models. We present a novel way to combining representations from multiple sources of information into a single global representation for proteins. We carefully analyze the performance of our framework in the pretraining tasks. For the fine-tuning tasks, our experiments have shown that our new multimodal representation can achieve competitive results in protein-ligand binding affinity prediction, protein fold classification, enzyme identification and mutation stability prediction. We expect that this work will accelerate future research in proteins. Our source code in PyTorch deep learning framework is publicly available at https://github.com/HySonLab/Protein_PretrainCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/12/02/2023.11.29.569288},
	eprint = {https://www.biorxiv.org/content/early/2023/12/02/2023.11.29.569288.full.pdf},
	journal = {bioRxiv}
}
```
