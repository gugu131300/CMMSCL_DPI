###########################删除pdb_id的最后一个字母##########################
# 读取 Davis2_pdbid.txt 文件
with open('E:/OneDrive/桌面/new_paper/dataset/Human/human_pdbid.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行，删除最后一个字母
processed_lines = [line.strip()[:-1] for line in lines]

# 保存处理后的内容到新文件 Davis2_pdbid_processed.txt
with open('E:/OneDrive/桌面/new_paper/dataset/Human/human_pdbid_processed.txt', 'w') as file:
    file.write('\n'.join(processed_lines))
