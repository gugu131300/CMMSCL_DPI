####################把GPCR里面hsa和映射的PDB id的文件处理一下,只保留PDB_id##################
import pandas as pd

# 读取tsv文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/D92M_hsa_idmapping.tsv", sep="\t")

# 提取并保存"Entry"列的内容到文件
entries = df["Entry"].tolist()

with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/D92M_unid.csv", "w") as output_file:
    for entry in entries:
        output_file.write(str(entry) + "\n")
