import pandas as pd

# 读取CSV文件
df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv')

# 获取所有唯一的 PROTEIN_GCN 和 COMPOUND_GCN
proteins = df['COMPOUND_GCN'].unique()
compounds = df['PROTEIN_GCN'].unique()

# 创建一个邻接矩阵，初始化为空值
adj_matrix = pd.DataFrame(index=proteins, columns=compounds)

# 填充邻接矩阵
for _, row in df.iterrows():
    protein = row['COMPOUND_GCN']
    compound = row['PROTEIN_GCN']
    label = row['LABEL']
    adj_matrix.at[protein, compound] = label

# 保存邻接矩阵到CSV文件
adj_matrix.to_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/Y2.csv', na_rep='')

print("邻接矩阵已保存到 Y.csv 文件中")