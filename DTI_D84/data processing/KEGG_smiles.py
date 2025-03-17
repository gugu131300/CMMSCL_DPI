import pandas as pd
###########PubChem_compound_text中去除KEGG的冗余信息###########
# 读取unique_drugs.txt文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/unique_drugs.txt", "r") as file:
    unique_drugs = [line.strip() for line in file]

# 读取PubChem_compound_text.csv文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/PubChem_compound_text.csv")

# 过滤cmpdsynonym列中的冗余信息，只保留unique_drugs中的键
df['id'] = df['cmpdsynonym'].str.extract(f'({"|".join(unique_drugs)})')

# 将处理后的数据保存到新文件
df.to_csv("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/PubChem_compound_text_processed.csv", index=False)