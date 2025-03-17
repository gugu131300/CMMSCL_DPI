##################去除文件中的空格##################
input_file_path = 'E:/OneDrive/桌面/new_paper/dataset/human/human_pdbid_processed.txt'  # 输入文件路径
output_file_path = 'E:/OneDrive/桌面/new_paper/dataset/human/human_pdbid_processed.txt'  # 输出文件路径

# 打开输入文件并读取内容
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

# 去除空行
non_empty_lines = [line.strip() for line in lines if line.strip()]

# 打开输出文件并写入处理后的内容
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(non_empty_lines))

print("空行已经去除，并保存到", output_file_path)

