# 读取 drug.txt 文件并去重
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/drug_KEGG.txt", "r") as file:
    unique_drugs = set(file.read().splitlines())

# 将去重后的药物写入新文件
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/model_dataprocessing/unique_drugs.txt", "w") as file:
    for drug in unique_drugs:
        file.write(drug + "\n")