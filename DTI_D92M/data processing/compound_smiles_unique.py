import pandas as pd
import numpy as np

# 读取 D84_classification.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv")

# 对 COMPOUND_SMILES 列进行去重
unique_compound_smiles = df["COMPOUND_SMILES"].drop_duplicates()
# 统计 unique_compound_smiles 的个数
unique_count = unique_compound_smiles.count()
print("unique_count", unique_count)

# 将去重后的结果保存为 NumPy 数组
np.save("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/compound_smiles_unique.npy", unique_compound_smiles.to_numpy())
