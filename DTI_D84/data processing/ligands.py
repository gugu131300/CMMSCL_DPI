###############将新的文件改成GraphDTA项目需要的格式文件###############
# 读取文件，转换数据格式，并写入新文件
with open('E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/comparable/protein.txt', 'r') as input_file, open('E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/comparable/protein_formatted.txt', 'w') as output_file:
    for line in input_file:
        ligand_id, ligand_sequence = line.strip().split('\t')
        formatted_sequence = '"' + ligand_id + '": "' + ligand_sequence + '",'
        output_file.write(formatted_sequence + '\n')

print("序列已转换并写入新文件 ligands_formatted.txt")
