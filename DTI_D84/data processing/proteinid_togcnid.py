###############把protein_id对应的bin文件的名字改成对用的id名
import os
import shutil
import pandas as pd

# 读取 Davis_classification.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv")

# 创建目标文件夹
output_folder = "E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/protein_id_bin/"
os.makedirs(output_folder, exist_ok=True)

# 遍历所有 bin 文件并重命名
for filename in os.listdir("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/D92M_all_best_pdb_file/"):
    if filename.endswith(".bin"):
        # 获取 COMPOUND_ID 对应的 COMPOUND_GCN 值
        protein_id = filename.split(".")[0]
        protein_gcn = df.loc[df['PROTEIN_ID'] == protein_id, 'PROTEIN_GCN'].values[0]

        # 构建新的文件名
        new_filename = f"{protein_gcn}.bin"

        # 移动文件到新的文件夹中
        shutil.copy(os.path.join("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/protein_id_bin/", filename),
                    os.path.join(output_folder, new_filename))