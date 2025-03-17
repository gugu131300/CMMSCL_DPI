import pandas as pd
import numpy as np

# 读取Davis_classification.csv文件
df = pd.read_csv('E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/model_dataprocess/D84_unlabel_classification.csv')

# 提取LABEL列
labels = df['LABEL'].values

# 将LABEL列保存为npy文件
np.save('F:/1unlabel_test.npy', labels)
