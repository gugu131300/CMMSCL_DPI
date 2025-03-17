import pandas as pd

####################把protein和compound进行id化###################
# 读取 compound_id.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/Davis/GCN/protein_id.csv")

# 将 COMPOUND_SMILES 列进行 id 化
df['compound_id'] = pd.factorize(df['PROTEIN_SEQUENCE'])[0]

# 将结果写入 compound_id.csv 文件
df.to_csv("E:/OneDrive/桌面/new_paper/dataset/Davis/GCN/protein_id2.csv", index=False)

