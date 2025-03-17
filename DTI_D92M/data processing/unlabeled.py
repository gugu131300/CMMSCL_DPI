import pandas as pd
from itertools import product

# 读取D92M_classification.csv文件
df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/model_dataprocess/D84_classification.csv')

# 获取所有唯一的PROTEIN_GCN和COMPOUND_GCN
protein_gcn_list = df['PROTEIN_GCN'].unique()
compound_gcn_list = df['COMPOUND_GCN'].unique()

# 生成所有可能的interaction对
all_possible_interactions = set(product(protein_gcn_list, compound_gcn_list))

# 生成现有的interaction对
existing_interactions = set(zip(df['PROTEIN_GCN'], df['COMPOUND_GCN']))

# 找出不存在的interaction对
nonexistent_interactions = all_possible_interactions - existing_interactions

# 创建一个空的DataFrame来存储结果
columns = ['HSA', 'PROTEIN_ID', 'PROTEIN_GCN', 'PROTEIN_SEQUENCE',
           'COMPOUND_ID', 'COMPOUND_SMILES', 'COMPOUND_GCN']
result_df = pd.DataFrame(columns=columns)

# 查找并保存未匹配的行
for protein_gcn, compound_gcn in nonexistent_interactions:
    protein_info = df[df['PROTEIN_GCN'] == protein_gcn].iloc[0]
    compound_info = df[df['COMPOUND_GCN'] == compound_gcn].iloc[0]

    new_row = {
        'HSA': protein_info['HSA'],
        'PROTEIN_ID': protein_info['PROTEIN_ID'],
        'PROTEIN_GCN': protein_info['PROTEIN_GCN'],
        'PROTEIN_SEQUENCE': protein_info['PROTEIN_SEQUENCE'],
        'COMPOUND_ID': compound_info['COMPOUND_ID'],
        'COMPOUND_SMILES': compound_info['COMPOUND_SMILES'],
        'COMPOUND_GCN': compound_info['COMPOUND_GCN']
    }
    result_df = result_df.append(new_row, ignore_index=True)

# 保存结果到新的CSV文件中
result_df.to_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/model_dataprocess/nonexistent.csv', index=False)