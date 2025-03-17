# **CMMSCL-DPI**
**Cross-Modal Multi-Structural Contrastive Learning for Predicting Drug-Protein Interactions**
--------

# **📌 Abstract**
Predicting drug-protein interactions (DPI) is essential for effective and safe drug discovery. Although deep learning methods have been extensively applied to DPI prediction, effectively leveraging the multi-structural and multi-modal data of drugs and proteins to enhance prediction accuracy remains a significant challenge. This study proposed CMMSCL-DPI, a cross-modal multi-structural contrastive learning model. CMMSCL-DPI applies contrastive learning to the multi-dimensional structural features of proteins and drugs separately and integrates interaction features from a DPI heterogeneous graph network to facilitate cross-modal contrastive learning. This approach effectively captures the key differences and similarities between proteins and drugs, significantly enhancing the model's generalization capabilities for novel drug-target pairs. Experimental results across three benchmark datasets demonstrate that CMMSCL-DPI outperforms five state-of-the-art baseline models in overall performance. Additionally, the model successfully identified an unreported drug-protein interaction, which was subsequently validated through all-atom molecular dynamics simulations. This case study not only confirms the predictive accuracy of CMMSCL-DPI but also underscores its potential in discovering novel protein-ligand interactions. In summary, CMMSCL-DPI exhibits high efficiency and broad applicability in advancing the drug discovery process.

CMMSCL-DPI is a **cross-modal multi-structural contrastive learning model** that:
- Extracts **multi-dimensional structural features** of proteins & drugs.
- Integrates **interaction features from a DPI heterogeneous graph network**.
- Enhances generalization to **novel drug-target pairs** via **contrastive learning**.

# **🔬 Key Results:**
- **Outperforms** 5 state-of-the-art models on three benchmark datasets.
- Successfully **identified a novel drug-protein interaction**, later validated via **molecular dynamics simulations**.
- Demonstrates **high efficiency** and **broad applicability** in drug discovery.
--------

# **🔧  Requirements**
Ensure the following dependencies are installed:
```bash
- python 3.10
- cudatoolkit 11.3.1
- pytorch 2.1.0+cu121
- rdkit 2023.9.6
- deepchem 2.8.0
- mdanalysis 2.7.0
- scipy 1.13.0
- dgl 2.4.0+cu118
- fair-esm 2.0.0
-------


# **🗂  Datasets **
CMMSCL-DPI supports the following datasets:
D84、D92M、Davis
-------

# **🚀  How to run**

## 1、Training and testing on other datasets

### Step 1: Obtain protein PDB structure files
python data_processing/download_all_pdb.py

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
1、Extract Uniprot IDs and obtain PDB structure files.(Input: Contains the Uniprot ID of the protein for querying the PDB database. For example：XXX/XXX.csv
Provides a mapping from Uniprot ID to PDB ID for downloading PDB structure files. For example：XXX/XXX.tsv)
2、Use AlphaFold predictions if no PDB file is available.  
3、Select the best PDB file (based on resolution & coverage).
4、Convert proteins & compounds into graphs.
5、Train & test the model.

### Step 2: Convert protein structures into graphs
python protein_graph.py

### Step 3: Convert compounds into molecular graphs
python compound_graph.py

### Step 4: Train & Test the model
python train_cl2RWR_class.py
-------
