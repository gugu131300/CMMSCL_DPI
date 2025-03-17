###############把Davis2数据集中缺失的PDB_id的存下来###########
import pandas as pd

# 读取CSV文件
df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis2/Davis2pdb.csv')

# 找到pdb_id缺失的行并提取对应的sequence列
missing_pdb_rows = df[df['pdb_id'].isnull()]['sequence']

# 创建一个新的DataFrame保存缺失的sequence
missing_pdb_df = pd.DataFrame({'missing_sequence': missing_pdb_rows})

# 将缺失的sequence保存到misspdb.csv文件
missing_pdb_df.to_csv('E:/OneDrive/桌面/new_paper/dataset/Davis2/misspdb.csv', index=False)
