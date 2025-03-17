###有一些没有pdb文件映射的protein_id,现在需要下载他们的AFold的PDB文件###
import os
import requests
from tqdm import tqdm
import pandas as pd

# 下载AlphaFold预测结果pdb
def download_af_pdb(p_id_without_pdb):
    for p_name in tqdm(p_id_without_pdb):
        download_path = "https://alphafold.ebi.ac.uk/files/AF-" + p_name + "-F1-model_v4.pdb"
        save_path = "E:/OneDrive/桌面/new_paper/dataset/Davis/missPDB_AF/" + p_name + ".pdb"
        if not os.path.exists(save_path):
            pdb_file = requests.get(download_path)
            pdb_data = pdb_file.content
            open(save_path, "wb").write(pdb_data)

uniport_id2pdb_id = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/Davis/missing.csv", sep="\t", header=0)
p_id_without_pdb = uniport_id2pdb_id['Missing_Values']
download_af_pdb(p_id_without_pdb)