import csv
###################将GPCR的hsa对应的氨基酸序列存入csv文件中##################
# 读取 GPCR.txt 文件中的所有内容
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/GPCR.txt", "r") as gpcr_file:
    gpcr_data = gpcr_file.read()

# 读取 D84_classification.csv 文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification.csv", "r") as csv_file:
    reader = csv.reader(csv_file)
    rows = list(reader)

# 创建一个字典，将 hsa 编号与对应的氨基酸序列关联起来
sequences = {}
current_hsa = None
current_sequence = ""
for line in gpcr_data.split("\n"):
    if line.startswith(">"):
        if current_hsa is not None:
            sequences[current_hsa] = current_sequence
        current_hsa = line.strip()[1:]
        current_sequence = ""
    else:
        current_sequence += line.strip()
sequences[current_hsa] = current_sequence

# 更新 D84_classification.csv 文件中的 sequence 列
for row in rows:
    if row[0] in sequences:
        row.append(sequences[row[0]])
    else:
        row.append("")

# 写入更新后的内容到 D84_classification.csv 文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/D92M_classification2.csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(rows)
