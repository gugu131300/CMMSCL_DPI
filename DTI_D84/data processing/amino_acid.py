from decimal import Decimal

# 读取文件并计算每一列的均值
with open('E:/OneDrive/桌面/CMMSCL_DPI/dataset/Davis/all_assign.txt', 'r') as file:
    lines = file.readlines()
    data = []
    for line in lines:
        # 将每一行的数据拆分为列表，并转换为 Decimal 类型
        row = [Decimal(value) for value in line.strip().split()]
        data.append(row)

# 计算每一列的均值
column_sums = [sum(col) for col in zip(*data)]
column_counts = len(data)
column_means = [sum_value / column_counts for sum_value in column_sums]

# 将均值写入新行
with open('E:/OneDrive/桌面/CMMSCL_DPI/dataset/Davis/all_assign2.txt', 'a') as file:
    file.write('\n')  # 添加一个空行
    for mean in column_means:
        file.write(f'{mean:.20f}\t')  # 写入每一列的均值，保留 20 位小数
    file.write('\n')  # 添加一个换行符


