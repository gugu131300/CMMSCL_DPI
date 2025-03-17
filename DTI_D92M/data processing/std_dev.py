##############################计算标准差##############################
import math

def calculate_std_dev(numbers):
    avg = sum(numbers) / len(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / len(numbers)
    return math.sqrt(variance), avg

# 示例使用
numbers_list = [0.82178, 0.79102, 0.78963, 0.80777, 0.78126]
std_dev, std_avg = calculate_std_dev(numbers_list)
print(std_avg)
print(std_dev)  # 输出标准差