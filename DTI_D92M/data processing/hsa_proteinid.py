import csv
# 读取 new_file.txt 文件，构建 hsa 到 proteinid 的映射关系字典, 并保存到classification文件中
hsa_proteinid_pairs = []
with open('E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/non390_hsa_idmapping.tsv', 'r') as file:
    for line in file:
        hsa, proteinid = line.strip().split('\t')
        hsa_proteinid_pairs.append((hsa, proteinid))

# 读取D84_classification.csv文件，进行匹配和写入
output_rows = []
with open('E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/non390.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        hsa_value = row['hsa']
        for hsa, proteinid in hsa_proteinid_pairs:
            if hsa == hsa_value:
                row['proteinid'] = proteinid
                break
        output_rows.append(row)

# 将更新后的数据写回D84_classification.csv文件
fieldnames = ['hsa', 'proteinid']  # 假设只有这两列需要更新
with open('E:/OneDrive/桌面/new_paper/dataset/GPCR/Check390/non390.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

