import os.path
import numpy as np
import requests

"""
蛋白质pdb数据来自RCSB和AlphaFold两个网站
"""
def get_protein_pdb():
    protein_name = np.loadtxt("E:/OneDrive/桌面/new_paper/dataset/Davis/Davis2_pdbid.txt", dtype=str)
    for i in range(len(protein_name)):
        p_name = protein_name[i]
        download_path ="https://files.rcsb.org/download/" + p_name + ".pdb"
        save_path = os.path.join("E:/OneDrive/桌面/new_paper/dataset/Davis/", p_name + ".pdb")
        if os.path.exists(save_path):
            continue
        pdb_file = requests.get(download_path, verify=False)
        if not pdb_file.status_code == 404:
            open(save_path, "wb").write(pdb_file.content)
        else:
            download_path = "https://alphafold.ebi.ac.uk/files/AF-" + p_name + "-F1-model_v4.pdb"
            pdb_file = requests.get(download_path)
            open(save_path, "wb").write(pdb_file.content)

        print("download file-{}:{}".format(i,p_name))

"""
获得蛋白质序列信息
构建蛋白质序列的dict
"""
# https://rest.uniprot.org/uniprotkb/Q9UI32.fasta

def get_protein_sequence():
    protein_sequence_dict = {}
    protein_name = np.loadtxt("../data/protein.txt", dtype=str)
    no_data_Protein_id = []
    for i in range(len(protein_name)):
        p_name = protein_name[i]
        print("getting {}th protein {} sequence".format(i, p_name))
        download_path = "https://rest.uniprot.org/uniprotkb/" + p_name + ".fasta"
        save_path = os.path.join("../data/protein_file", p_name + ".fasta")
        if not os.path.exists(save_path):
            seq_file = requests.get(download_path)
            fasta_data = seq_file.content
        else:
            fasta_data = open(save_path,"rb").read()

        fasta_data = str(fasta_data)[2:-1]
        if not os.path.exists(save_path):
            open(save_path, "wb").write(fasta_data)
        temp_seq = ""
        temp_seq = temp_seq.join(fasta_data.split(r"\n")[1:])
        protein_sequence_dict[p_name] = temp_seq

        """
        如果未下载到fasta序列信息，使用“ABCDE”填充
        """
        if len(temp_seq) == 0:
            no_data_Protein_id.append(p_name)
            protein_sequence_dict[p_name] = "ABCDE"

    protein_sequence_dict_path = "../data/" + "protein_sequence_dict.npy"
    np.save(protein_sequence_dict_path, protein_sequence_dict)
    """
    无序列信息的protein id
    """
    no_data_Protein_id_path = "../data/" + "no_data_Protein_id"
    np.save(no_data_Protein_id_path, no_data_Protein_id)

"""drugBank网站数据无法爬取，有CloudFlare保护"""
# drug_name = np.loadtxt("../data/drug.txt",dtype=str)
# for d_name in drug_name:
#     download_path ="https://go.drugbank.com/structures/small_molecule_drugs/" + d_name + ".smiles"
#     save_path = os.path.join("../data/drug_file", d_name + ".smiles")
#     if os.path.exists(save_path):
#         continue
#     smiles_file = requests.get(download_path)
#     smiles_file.encoding = "utf-8"
#     soup = BeautifulSoup(smiles_file.text, "html.parser")
#     smile_item = soup.find_all("pre")
#     s = smile_item.text.strip()
#     if not smiles_file.status_code == 404:
#         open(save_path, "wb").write(s)

get_protein_pdb()
#get_protein_sequence()
a = 0

