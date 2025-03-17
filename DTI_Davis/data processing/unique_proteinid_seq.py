# 读取unique_proteinid_proteins.txt文件
input_file = 'E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/protein.txt'
output_file = 'E:/OneDrive/桌面/new_paper/dataset/GPCR/D92M/model_dataprocessing/protein_unique.txt'

# 用于存储唯一条目的字典
unique_entries = {}

with open(input_file, 'r') as file:
    for line in file:
        # 分割序号和序列
        parts = line.strip().split('\t')
        if len(parts) == 2:
            protein_id, sequence = parts
            # 仅在字典中不存在该序号时才添加
            if protein_id not in unique_entries:
                unique_entries[protein_id] = sequence

# 将唯一条目写入新文件
with open(output_file, 'w') as file:
    for protein_id, sequence in unique_entries.items():
        file.write(f"{protein_id}\t{sequence}\n")

print(f"去重后的数据已保存到 {output_file} 文件中")