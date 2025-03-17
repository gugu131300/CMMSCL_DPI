#####################missPDB去重######################
import pandas as pd

# 读取misspdb.csv文件
df_misspdb = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis2/misspdb.csv')

# 去除重复行
df_misspdb_unique = df_misspdb.drop_duplicates()

# 将去重后的DataFrame保存到新的文件
df_misspdb_unique.to_csv('E:/OneDrive/桌面/new_paper/dataset/Davis2/misspdb_unique.csv', index=False)
