import pandas as pd

# 读取PubChem_compound_text_processed.csv文件
compound_text_df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/PubChem_compound_text_processed.csv")

# 读取D84_classification.csv文件
classification_df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/390_classification.csv")

# 将COMPOUND_SMILES列添加到D84_classification.csv文件中
classification_df = classification_df.merge(compound_text_df, on="COMPOUND_ID", how="left")

# 保存更新后的D84_classification.csv文件
classification_df.to_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/390_classification_updated.csv", index=False)