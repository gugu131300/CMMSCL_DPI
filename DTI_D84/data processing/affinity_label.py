###########把亲和力转化为label############
import pandas as pd

# 读取 Davis.csv 文件
df = pd.read_csv("E:/OneDrive/桌面/new_paper/dataset/Davis/Davis.csv")

# 对 REG_LABEL 进行处理，生成新的 LABEL 列
df['LABEL'] = df['REG_LABEL'].apply(lambda x: 0 if x <= 5 else 1)

# 统计 LABEL 列中 0 和 1 的数量
label_counts = df['LABEL'].value_counts()

print("0 的数量:", label_counts[0])
print("1 的数量:", label_counts[1])

# 将结果写入文件
df.to_csv("E:/OneDrive/桌面/new_paper/dataset/Davis/Davis_processed.csv", index=False)