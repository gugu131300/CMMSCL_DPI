##############合并所有的protein sequence和PDB id#################
import pandas as pd

# 读取 Davis2.csv 和 PDB.csv
Davis2_df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/Davis2/Davis2.csv")
pdb_df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/Davis2/PDB.csv")

# 合并数据框，根据 'sequence' 列进行匹配
merged_df = pd.merge(Davis2_df, pdb_df, on='sequence', how='left')

# 将匹配到的 'pdb_id' 列的值添加到 'Davis2.csv' 中
Davis2_df['pdb_id'] = merged_df['pdb_id']

# 将结果写回 'Davis2.csv'
Davis2_df.to_csv("E:/OneDrive/桌面/new_paper/dataset/Davis2/Davis2_updated.csv", index=False)
