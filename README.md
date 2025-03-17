# CMMSCL-DPI
**Cross-Modal Multi-Structural Contrastive Learning for Predicting Drug-Protein Interactions**
--------

# 📌 Abstract 
Predicting drug-protein interactions (DPI) is essential for effective and safe drug discovery. Although deep learning methods have been extensively applied to DPI prediction, effectively leveraging the multi-structural and multi-modal data of drugs and proteins to enhance prediction accuracy remains a significant challenge. This study proposed CMMSCL-DPI, a cross-modal multi-structural contrastive learning model. CMMSCL-DPI applies contrastive learning to the multi-dimensional structural features of proteins and drugs separately and integrates interaction features from a DPI heterogeneous graph network to facilitate cross-modal contrastive learning. This approach effectively captures the key differences and similarities between proteins and drugs, significantly enhancing the model's generalization capabilities for novel drug-target pairs. Experimental results across three benchmark datasets demonstrate that CMMSCL-DPI outperforms five state-of-the-art baseline models in overall performance. Additionally, the model successfully identified an unreported drug-protein interaction, which was subsequently validated through all-atom molecular dynamics simulations. This case study not only confirms the predictive accuracy of CMMSCL-DPI but also underscores its potential in discovering novel protein-ligand interactions. In summary, CMMSCL-DPI exhibits high efficiency and broad applicability in advancing the drug discovery process.

CMMSCL-DPI is a **cross-modal multi-structural contrastive learning model** that:
- Extracts **multi-dimensional structural features** of proteins & drugs.
- Integrates **interaction features from a DPI heterogeneous graph network**.
- Enhances generalization to **novel drug-target pairs** via **contrastive learning**.

# 🔬 Key Results: 
- **Outperforms** 5 state-of-the-art models on three benchmark datasets.
- Successfully **identified a novel drug-protein interaction**, later validated via **molecular dynamics simulations**.
- Demonstrates **high efficiency** and **broad applicability** in drug discovery.
--------

# 🔧  Requirements: 
Ensure the following dependencies are installed:
```
- python 3.10
- cudatoolkit 11.3.1
- pytorch 2.1.0+cu121
- rdkit 2023.9.6
- deepchem 2.8.0
- mdanalysis 2.7.0
- scipy 1.13.0
- dgl 2.4.0+cu118
- fair-esm 2.0.0
```
-------


# 🗂  Datasets
CMMSCL-DPI supports the following datasets:
D84、D92M、Davis

-------


# 🚀  How to run 
## 1、Training and testing on exsiting datasets

### Step 1: Obtain protein PDB structure files
python /data_processing/download_all_pdb.py
### Step 2: Convert protein structures into graphs
python protein_graph.py
### Step 3: Convert compounds into molecular graphs
python compound_graph.py
### Step 4: Train & Test the model
python train_cl2RWR_class.py

-------


## 2、Training and testing on other datasets
If using a custom dataset, follow the same steps:

### Step 1: Obtain protein PDB structure files
python data_processing/download_all_pdb.py

- To obtain the PDB structure files for proteins, the following input files are required:
protein_id.txt – This file contains the Uniprot IDs of all proteins and serves as the primary input for mapping protein sequences to structural data.
non390_pidmapping.tsv – This file establishes a mapping between Uniprot IDs and their corresponding PDB IDs, which are necessary for retrieving the structural information from the PDB database.
The PDB IDs need to be retrieved from the UniProt ID mapping tool available at:👉 https://www.uniprot.org/id-mapping
On the left panel, select "UniProtKB/AC ID" as the input.
On the right panel, choose "PDB" as the output.
This mapping file (non390_pidmapping.tsv) will then be used to fetch the relevant PDB structures for each protein.
- Use AlphaFold predictions if no PDB file is available.  
- Select the best PDB file (based on resolution & coverage).
- Convert proteins & compounds into graphs.
- Train & test the model.
### Step 2: Convert protein structures into graphs
python protein_graph.py

The input PDB files required for protein_graph.py must be obtained from the output of download_all_pdb.py.   
Ensure that you have successfully executed download_all_pdb.py beforehand to fetch and store all necessary PDB structure files.
### Step 3: Convert compounds into molecular graphs
python compound_graph.py

Inputs：CSV File with Compound Information  
The input file should contain at least the following columns:  
COMPOUND_SMILES: The SMILES representation of each compound.  
COMPOUND_ID: A unique identifier for each compound.  

### Step 4: Train & Test the model
python train_cl2RWR_class.py

-------
