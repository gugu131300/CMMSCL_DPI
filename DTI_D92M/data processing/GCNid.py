##############获得GCN_id##############
import pandas as pd

# 读取 D84_classification.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv")

# 根据 COMPOUND_ID 进行序号标记
protein_gcn = {}
index = 0
for protein_id in df["PROTEIN_ID"]:
    if protein_id not in protein_gcn:
        protein_gcn[protein_id] = index
        index += 1

# 创建新的 COMPOUND_GCN 列
df["PROTEIN_GCN"] = [protein_gcn[protein_id] for protein_id in df["PROTEIN_ID"]]

# 保存更新后的 DataFrame 到 D84_classification.csv 文件
df.to_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/390_classification.csv", index=False)