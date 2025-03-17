import pandas as pd

# 读取 CSV 文件
file_path = "E:/OneDrive/桌面/CMMSCL_DPI/dataset/Davis/PPI/Davis_classification_unique.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 提取 PROTEIN_ID 和 PROTEIN_SEQUENCE 列
protein_data = data[["PROTEIN_ID", "PROTEIN_SEQUENCE"]]

# 转换为字典格式：PROTEIN_ID -> PROTEIN_SEQUENCE
protein_dict = protein_data.set_index("PROTEIN_ID")["PROTEIN_SEQUENCE"].to_dict()

# 输出为字符串形式
formatted_output = ",\n".join([f'"{key}": "{value}"' for key, value in protein_dict.items()])

# 保存到 txt 文件
output_file = "F:/protein_id_sequence.txt"  # 输出文件名
with open(output_file, "w") as f:
    f.write("{\n" + formatted_output + "\n}")

print(f"提取完成，结果已保存到 {output_file}")

