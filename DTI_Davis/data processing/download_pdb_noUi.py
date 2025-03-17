##################download_all_pdb,no Uniprot id################
import os
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

pdb_id_txt = np.loadtxt("E:/OneDrive/桌面/new_paper/dataset/human/human_pdbid_processed.txt", dtype=str)
pdb_uniprot = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/human/alpha_uniprot.tsv", sep="\t", header=0)
pdb_uniprot_dict = dict(zip(pdb_uniprot['From'], pdb_uniprot['To']))
pdb_id = pd.DataFrame(pdb_id_txt)
pdb_id = pdb_id_txt

# 下载AlphaFold预测结果pdb
def download_af_pdb(without_pdb):
    for pdb_name in tqdm(without_pdb):
        download_path = "https://alphafold.ebi.ac.uk/files/AF-" + pdb_name + "-F1-model_v4.pdb"
        save_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/AF-" + pdb_name + "-F1-model_v4.pdb"
        if not os.path.exists(save_path):
            pdb_file = requests.get(download_path)
            pdb_data = pdb_file.content
            open(save_path, "wb").write(pdb_data)

# 2. 下载每个protein对应的所有的pdb数据
def download_pdb(pdb_id):
    for i in tqdm(pdb_id):
        download_path = "https://files.rcsb.org/download/" + i + ".pdb"
        save_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/" + i + ".pdb"
        #urllib3.disable_warnings()
        if not os.path.exists(save_path):
            pdb_file = requests.get(download_path, verify=False, timeout=10)
            pdb_data = pdb_file.content
            if pdb_file.status_code == 404:
                continue
            open(save_path, "wb").write(pdb_data)

# 获得已下载的pdb_id和未下载的pdb_id
def get_redownload_pdb_id(pdb_id):
    downloaded_2pdb_id = []
    undownloaded_2pdb_id = []
    for pdb in tqdm(pdb_id):
        save_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/" + pdb + ".pdb"
        if not os.path.exists(save_path):
            undownloaded_2pdb_id.append(pdb)
        else:
            saved_pdb_file = open(save_path, "r")
            line_0 = saved_pdb_file.readline().strip().split()
            if not line_0[0] == 'HEADER':
                undownloaded_2pdb_id.append(pdb)
                saved_pdb_file.close()
                os.remove(save_path)
            else:
                downloaded_2pdb_id.append(pdb)

    return downloaded_2pdb_id, undownloaded_2pdb_id

# 3. 读取获取resolution和position数据,获得最佳pdb
def get_best_pdb(downloaded_2pdb_id, undownloaded_2pdb_id):
    # 从每一个protein的多个pdb数据中选择最佳的
    for p_name in tqdm(downloaded_2pdb_id):
        p_name_pdb_indexs = np.where(downloaded_2pdb_id == p_name)[0]
        pdb_names = downloaded_2pdb_id[p_name_pdb_indexs]
        pdb_resolution = []
        pdb_position = []
        # 获得每一个protein的所有pdb数据，将其_resolution和position值保存在列表中
        for pdb_name in pdb_names:
            pdb_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/" + pdb_name + ".pdb"
            pdb_file = open(pdb_path, "r")
            position_bool = 0
            resolution_bool = 0
            for line in pdb_file:
                # print("line", line)
                if resolution_bool == 0 and line.startswith("REMARK"):
                    line_data = line.split()
                    if len(line_data) >= 3:
                        if line_data[1] == '2' and line_data[2] == 'RESOLUTION.' and len(line_data[3]) > 0:
                            if line_data[3] == 'NOT' or line_data[3] == 'NULL':
                                pdb_resolution.append(100)
                            else:
                                pdb_resolution.append(line_data[3])
                            resolution_bool = resolution_bool + 1
            if position_bool == 0:
                pdb_file.close()
                pdb_file = open(pdb_path, "r")
                for line in pdb_file:
                    if position_bool == 1:
                        break
                    if line.startswith("DBREF"):
                        line_data = line.split()
                        if len(line_data) >= 9:
                            if line_data[1].lower() == pdb_name.lower():
                                pdb_position.append(int(line_data[9]) - int(line_data[8]))
                                position_bool = position_bool + 1

        # 选择最长position的pdb
        if len(pdb_resolution) == 1 and len(pdb_position) == 0:
            pdb_position.append(100)
        pdb_position = np.array(pdb_position)

        max_p = max(pdb_position)
        max_indexs = np.where(pdb_position == max_p)[0]
        # 选择最佳（小）resolution的pdb
        max_indexs = max_indexs.astype(np.int)
        pdb_resolution = np.array(pdb_resolution).astype(np.float)
        pdb_resolution_temp = pdb_resolution[max_indexs]
        # 获得resolution最小值
        pdb_resolution_temp_min = np.min(pdb_resolution_temp)
        pdb_resolution_temp_min_index = np.where(pdb_resolution_temp == pdb_resolution_temp_min)
        best_pdb_index = max_indexs[pdb_resolution_temp_min_index]
        best_pdb_name = pdb_names[best_pdb_index]

        best_pdb_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/" + best_pdb_name[0] + ".pdb"
        best_pdb_save_path = "E:/OneDrive/桌面/new_paper/dataset/human/best_pdb_file/" + p_name + ".pdb"
        # pdb_data = open(best_pdb_path, 'rb',encoding='utf-8').read()
        shutil.copyfile(best_pdb_path, best_pdb_save_path)
        # open(best_pdb_save_path,'wb').write(pdb_data)

        # 使用AlphaFold预测的结果pdb作为最佳的
    for p_name in tqdm(undownloaded_2pdb_id):
        pdb_path = "E:/OneDrive/桌面/new_paper/dataset/human/pdb_file/AF-" + p_name + "-F1-model_v4.pdb"
        best_pdb_save_path = "E:/OneDrive/桌面/new_paper/dataset/human/best_pdb_file/" + p_name + ".pdb"
        pdb_data = open(pdb_path, "r").read()
        open(best_pdb_save_path, "w").write(pdb_data)

# 只用download一次
#download_pdb(pdb_id)
# 检查下载的pdb是否正确，删除不正确的，获得已下载和未下载的id
downloaded_2pdb_id, undownloaded_2pdb_id = get_redownload_pdb_id(pdb_id)

undownloaded_uniprot_id = []
for pdb in undownloaded_2pdb_id:
    undownloaded_uniprot = pdb_uniprot_dict[pdb]
    undownloaded_uniprot_id.append(undownloaded_uniprot)

#download_af_pdb(undownloaded_uniprot_id)
downloaded_2pdb_id = np.array(downloaded_2pdb_id)
undownloaded_2pdb_id = np.array(undownloaded_2pdb_id)
get_best_pdb(downloaded_2pdb_id, undownloaded_2pdb_id)
a = 0