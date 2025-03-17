import numpy as np
import pandas as pd

# 加载 npy 文件
compound_smiles_ids = np.load('F:/unlabel_compound_smiles_ids.npy')
protein_seq_ids = np.load('F:/unlabel_protein_seq_ids.npy')

# 加载 Davis_classification.csv 文件
davis_df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/model_dataprocess/D84_unlabel_classification.csv')

# 创建一个空的 DataFrame 来保存匹配的行
matched_rows = pd.DataFrame(columns=davis_df.columns)

# 使用集合来跟踪已匹配的组合
seen_combinations = set()
unmatched_combinations = []

# 查找匹配的行
for compound_id, protein_id in zip(compound_smiles_ids, protein_seq_ids):
    combination = (compound_id, protein_id)
    if combination not in seen_combinations:
        matched_row = davis_df[(davis_df['COMPOUND_GCN'] == compound_id) & (davis_df['PROTEIN_GCN'] == protein_id)]
        if not matched_row.empty:
            matched_rows = pd.concat([matched_rows, matched_row.head(1)], ignore_index=True)
            seen_combinations.add(combination)
        else:
            unmatched_combinations.append(combination)

# 输出未找到匹配的组合数量
print(f"Number of unmatched combinations: {len(unmatched_combinations)}")
print(f"Unmatched combinations: {unmatched_combinations}")

# 将匹配的行保存到新的 CSV 文件
matched_rows.to_csv('F:/111.csv', index=False)


