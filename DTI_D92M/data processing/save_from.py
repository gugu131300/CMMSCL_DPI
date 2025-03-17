##########只保留tsv文件中from的一列，其他的全部删除############
import pandas as pd

# 读取111.tsv文件
df = pd.read_csv('E:\OneDrive\桌面\new_paper\dataset\idmapping_2024_01_21.tsv', sep='\t')

# 只保留From一列，删除其他列
df_result = df[['From']]

# 保存结果到新文件111_processed.tsv
df_result.to_csv('E:\OneDrive\桌面\new_paper\dataset\Human111_processed.tsv', sep='\t', index=False)
