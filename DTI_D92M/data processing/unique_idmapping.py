###################将pdb对应的proteinid去idmapping中进行去重##################
import os

# 读取 D84_hsa_idmapping.tsv 文件，构建 From 到 Entry 的映射关系字典
id_mapping = {}
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/nonD84_hsa_idmapping.tsv", "r") as id_mapping_file:
    for line in id_mapping_file:
        columns = line.strip().split("\t")
        if len(columns) == 2:
            from_id, entry_id = columns
            id_mapping[entry_id] = from_id

# 打开新的 txt 文件，用于保存结果
with open("E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/nonnew_file.txt", "w") as output_file:
    # 遍历 D84_best_pdb_file 文件夹中的每个文件
    for filename in os.listdir("E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/D84_best_pdb_file"):
        # 构建文件的完整路径
        filepath = os.path.join("E:/OneDrive/桌面/new_paper/dataset/GPCR/D84/D84_best_pdb_file", filename)
        # 检查文件是否是普通文件
        if os.path.isfile(filepath):
            # 获取文件名，作为 Entry 列的值
            entry_id = os.path.splitext(filename)[0]
            # 如果 Entry 列在映射字典中
            if entry_id in id_mapping:
                # 获取对应的 From 列的值
                from_id = id_mapping[entry_id]
                # 将 From 和 Entry 写入新的 txt 文件中
                output_file.write(f"{from_id}\t{entry_id}\n")