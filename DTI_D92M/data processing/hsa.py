import csv
###################将GPCR的反应对都存入csv文件中##################
# 打开 D84_hsa.txt 文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/non390_pairs.txt", "r") as input_file:
    lines = input_file.readlines()

# 打开 D84_classification.csv 文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/111.csv", "w", newline='') as output_file:
    writer = csv.writer(output_file)

    # 遍历每一行
    for line in lines:
        # 分割每一行，得到作用对
        interactions = line.strip().split("\t")

        # 遍历每一个作用对
        for interaction in interactions:
            print("interaction", interaction)
            # 分割作用对，得到蛋白质和药物的编号
            protein, drug = interaction.split("-")

            # 写入到 CSV 文件中
            writer.writerow([protein, drug, 0])
