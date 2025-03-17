import os
import pandas as pd
################把protein_graph里面按照PROTEIN_ID换成PROTEIN_GCN#############

# 读取 D84_classification.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv")

# 创建一个字典，键为 PROTEIN_ID，值为 PROTEIN_GCN
protein_dict = pd.Series(df.PROTEIN_GCN.values, index=df.PROTEIN_ID).to_dict()

# 指定 bin 文件所在的文件夹路径
bin_folder_path = "E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/protein_graph"  # 替换为你的文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(bin_folder_path):
    # 获取文件名（不包括扩展名）
    protein_id, file_extension = os.path.splitext(filename)
    # 检查文件是否为 bin 文件
    if file_extension == ".bin":
        # 获取对应的 PROTEIN_GCN
        protein_gcn = protein_dict.get(protein_id)
        if protein_gcn:
            # 构造新的文件名
            new_filename = f"{protein_gcn}.bin"
            # 获取文件的完整路径
            old_file_path = os.path.join(bin_folder_path, filename)
            new_file_path = os.path.join(bin_folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")
        else:
            print(f"No PROTEIN_GCN found for PROTEIN_ID: {protein_id}")
