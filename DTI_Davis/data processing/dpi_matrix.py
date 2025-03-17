import pandas as pd
import numpy as np

# 读取 CSV 文件
data = pd.read_csv("E:/OneDrive/桌面/CMMSCL_DPI/dataset/Davis/PPI/Davis_classification_unique.csv")

# 创建一个空的 DataFrame 用于存储对称矩阵
dpi_matrix = pd.DataFrame(index=data['COMPOUND_GCN'].unique(), columns=data['PROTEIN_GCN'].unique())

# 填充对称矩阵
for index, row in data.iterrows():
    compound = row['COMPOUND_GCN']
    protein = row['PROTEIN_GCN']
    label = row['REG_LABEL']
    dpi_matrix.loc[compound, protein] = label
    dpi_matrix.loc[protein, compound] = label  # 对称矩阵的另一半也要填充

# 将缺失值填充为0
# dpi_matrix = dpi_matrix.fillna(0)

# 保存对称矩阵到 CSV 文件
output_file = "F:/dpi_matrix.csv"
dpi_matrix.to_csv(output_file, header=True, index=True)

print(f"对称矩阵已保存到 {output_file}")
