##############将classification文件中去除重复的hsa和PROTEIN_ID相互作用的那一行##############
import pandas as pd

# 读取CSV文件
input_file = 'E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv'
output_file = 'E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification_unique.csv'

# 读取CSV文件到DataFrame
df = pd.read_csv(input_file)

# 去除重复行，保留第一次出现的行
df_unique = df.drop_duplicates(subset=['COMPOUND_GCN', 'PROTEIN_GCN'])

# 将去重后的DataFrame保存到新的CSV文件
df_unique.to_csv(output_file, index=False)

print(f"去重后的数据已保存到 {output_file} 文件中")