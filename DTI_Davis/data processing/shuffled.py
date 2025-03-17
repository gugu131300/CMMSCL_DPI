import csv
import random
###################打乱文件顺序###################
# 读取原始CSV文件
rows = []
with open('E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/comparable/D92M.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

# 打乱行顺序
random.shuffle(rows)

# 将打乱后的行写入新的CSV文件
with open('E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/comparable/D92M_shuffled.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print("行顺序已打乱并写入新文件 D84_MGraphDTA_shuffled.csv")