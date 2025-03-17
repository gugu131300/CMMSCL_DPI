###在后期的protein转graph里，发现有的protein_id没有转化成功，在前期下载PDB文件的时候，发现有的protein_id
###没有映射，所以导致不是所有的protein_id都被下载了
import pandas as pd

# 读取 Davis.csv 文件和 id_mapping.csv 文件
davis_df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/Davis.csv')
mapping_df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/idmapping.csv')

# 获取 Davis.csv 中的 PROTEIN_ID 列和 id_mapping.csv 中的 From 列中的值
protein_ids = set(davis_df['PROTEIN_ID'])
mapping_values = set(mapping_df['From'])

# 找到在 From 列中出现但未出现在 PROTEIN_ID 列中的值
missing_values = []

for value in protein_ids:
    if value in protein_ids and value not in mapping_values:  # 检查值是否为空
        missing_values.append(value)

# 将结果保存到 yilou.csv 文件中
yilou_df = pd.DataFrame({'Missing_Values': missing_values})
yilou_df.to_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/missing.csv', index=False)
