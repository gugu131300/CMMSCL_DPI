# CMMSCL_DPI
--------

# CMMSCL-DPI : Cross-Modal Multi-Structural Contrastive Learning for Predicting Drug-Protein Interactions
--------

# Abstract
--------
Predicting drug-protein interactions (DPI) is essential for effective and safe drug discovery. Although deep learning methods have been extensively applied to DPI prediction, effectively leveraging the multi-structural and multi-modal data of drugs and proteins to enhance prediction accuracy remains a significant challenge. This study proposed CMMSCL-DPI, a cross-modal multi-structural contrastive learning model. CMMSCL-DPI applies contrastive learning to the multi-dimensional structural features of proteins and drugs separately and integrates interaction features from a DPI heterogeneous graph network to facilitate cross-modal contrastive learning. This approach effectively captures the key differences and similarities between proteins and drugs, significantly enhancing the model's generalization capabilities for novel drug-target pairs. Experimental results across three benchmark datasets demonstrate that CMMSCL-DPI outperforms five state-of-the-art baseline models in overall performance. Additionally, the model successfully identified an unreported drug-protein interaction, which was subsequently validated through all-atom molecular dynamics simulations. This case study not only confirms the predictive accuracy of CMMSCL-DPI but also underscores its potential in discovering novel protein-ligand interactions. In summary, CMMSCL-DPI exhibits high efficiency and broad applicability in advancing the drug discovery process.

# Requirement
-------
- python 3.10
- cudatoolkit 11.3.1
- pytorch 2.1.0+cu121
- rdkit 2023.9.6
- deepchem 2.8.0
- mdanalysis 2.7.0
- scipy 1.13.0
- dgl 2.4.0+cu118
- fair-esm 2.0.0

# Datasets
-------
D84、D92M、Davis

# How to run
-------

# 1、Training and testing on existing datasets（Davis、D84、D92M）
-------
## Preprocessing protein data
-------
### Obtain protein PDB data
python /data processing/download_all_pdb.py

### Preprocess to obtain protein graph
python protein_graph.py

### Preprocessing compound data
python compound_graph.py

## Train and Test the model
-------
python train_cl2RWR_class.py



# 2、Training and testing on other datasets
-------

## Preprocessing protein data
-------
### Obtain protein PDB data
This project aims to extract the Uniprot ID of proteins from the Davis dataset and query the corresponding PDB structure file in the PDB database.
If there is no structure file for a protein in the PDB database, the PDB structure predicted by AlphaFold is used as an alternative.
In addition, if a protein corresponds to multiple PDB structure files, the structure file with the best resolution and coverage is selected as the final structure.

Input:
Contains the Uniprot ID of the protein for querying the PDB database.   For example：XXX/XXX.csv
Provides a mapping from Uniprot ID to PDB ID for downloading PDB structure files.     For example：XXX/XXX.tsv

python /data processing/download_all_pdb.py

### Preprocess to obtain protein graph
python protein_graph.py

### Preprocessing compound data
python compound_graph.py

## Train and Test the model
-------
python train_cl2RWR_class.py




