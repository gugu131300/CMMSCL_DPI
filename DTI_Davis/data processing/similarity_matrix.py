import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# # 示例相似度矩阵
# similarity_matrix = np.array([
#     [0.2, 0.3, 0.6, 0.1],
#     [0.4, 0.5, 0.7, 0.3],
#     [0.1, 0.4, 0.9, 0.5],
#     [0.6, 0.2, 0.3, 0.4]
# ])
#
# # 创建相似度矩阵图
# plt.figure(figsize=(6, 6))
# plt.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
# plt.colorbar(label='Similarity')
#
# # 设置标签和标题
# plt.title('Similarity Matrix')
# plt.xlabel('Proteins')
# plt.ylabel('Drugs')
#
# # 显示图像
# plt.show()

# 示例相似度矩阵
similarity_matrix = np.array([
    [0.2, 0.3, 0.6, 0.1],
    [0.4, 0.5, 0.7, 0.3],
    [0.1, 0.4, 0.9, 0.5],
    [0.6, 0.2, 0.3, 0.4]
])

# 创建自定义的玫红色到白色的颜色映射
custom_cmap = LinearSegmentedColormap.from_list('MagentaWhite', ['magenta', 'white'])

# 创建相似度矩阵图
plt.figure(figsize=(6, 6))
plt.imshow(similarity_matrix, cmap=custom_cmap, interpolation='nearest')
# plt.colorbar(label='Similarity')

# # 设置标签和标题
# plt.title('Similarity Matrix')
# plt.xlabel('Proteins')
# plt.ylabel('Drugs')

# 隐藏坐标轴上的数字
plt.xticks([])
plt.yticks([])

# 显示图像
plt.show()
