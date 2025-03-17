import pandas as pd

# 读取D84_test.csv文件
test_df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/Davis_test.csv')

# 读取D84_classification.csv文件
classification_df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/Davis_classification.csv')

# 创建一个字典用于快速查找PROTEIN_ID
protein_id_dict = dict(zip(classification_df['PROTEIN_SEQUENCE'], classification_df['PROTEIN_ID']))

# 初始化PROTEIN_ID列
test_df['PROTEIN_ID'] = test_df['PROTEIN_SEQUENCE'].map(protein_id_dict)

# 将结果保存回D84_test.csv文件
test_df.to_csv('E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/D92M_test_updated.csv', index=False)
