import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 定义头部数和层数的数组
heads = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
layers = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# 定义AUC和AUPR的矩阵
AUC = np.array([
    [0.9378, 0.9409, 0.9409, 0.93864, 0.93743, 0.9402, 0.9438, 0.94233, 0.94061], # head 2
    [0.93971, 0.93949, 0.93878, 0.93892, 0.9414, 0.94274, 0.94071, 0.9423, 0.94156], # head 3
    [0.9383, 0.94165, 0.94156, 0.93948, 0.94193, 0.93898, 0.94144, 0.9395, 0.9408], # head 4
    [0.93942, 0.93915, 0.9405, 0.94235, 0.93911, 0.94056, 0.93951, 0.93821, 0.9431], # head 5
    [0.93895, 0.93795, 0.93958, 0.94028, 0.94364, 0.94103, 0.93822, 0.93733, 0.9411], # head 6
    [0.93957, 0.94211, 0.94205, 0.93876, 0.94021, 0.94131, 0.94049, 0.94015, 0.94043], # head 7
    [0.93948, 0.94129, 0.94214, 0.93961, 0.94137, 0.93878, 0.94281, 0.9405, 0.94351], # head 8
    [0.94088, 0.94115, 0.93932, 0.94053, 0.94138, 0.94283, 0.94397, 0.942, 0.94106], # head 9
    [0.9411, 0.94142, 0.9406, 0.94031, 0.94137, 0.94063, 0.94035, 0.94128, 0.94118], # head 10
])

AUPR = np.array([
    [0.8753, 0.88116, 0.88369, 0.8795, 0.87696, 0.88133, 0.88609, 0.8838, 0.87959], # head 2
    [0.87902, 0.88035, 0.87511, 0.87764, 0.88172, 0.88566, 0.88272, 0.88361, 0.88227], # head 3
    [0.87578, 0.88322, 0.88479, 0.87898, 0.88448, 0.88184, 0.88285, 0.88051, 0.88146], # head 4
    [0.87934, 0.87735, 0.88165, 0.88237, 0.87763, 0.88342, 0.87917, 0.87689, 0.88606], # head 5
    [0.87523, 0.87576, 0.87914, 0.87733, 0.88417, 0.88222, 0.8758, 0.87678, 0.88331], # head 6
    [0.87781, 0.88279, 0.88654, 0.87651, 0.87801, 0.88195, 0.88192, 0.87872, 0.88035], # head 7
    [0.87992, 0.8806, 0.88556, 0.87929, 0.88309, 0.87609, 0.88204, 0.88177, 0.88708], # head 8
    [0.88474, 0.87981, 0.87872, 0.88366, 0.88435, 0.88393, 0.88721, 0.88418, 0.88424], # head 9
    [0.87947, 0.87982, 0.87801, 0.88038, 0.88272, 0.8814, 0.87802, 0.88483, 0.88431] # head 10
])

# 创建头部和层数的网格
hgt, lgt = np.meshgrid(heads, layers, indexing='ij')
points = np.array([hgt.flatten(), lgt.flatten()]).T

# 创建更细的网格
fine_hgt, fine_lgt = np.mgrid[2:10:100j, 2:10:100j]

# 插值AUC和AUPR
fine_AUC = griddata(points, AUC.flatten(), (fine_hgt, fine_lgt), method='cubic')
fine_AUPR = griddata(points, AUPR.flatten(), (fine_hgt, fine_lgt), method='cubic')

# 创建3D图
fig = plt.figure(figsize=(14, 5))

# AUC曲面图
# ax1 = fig.add_subplot(121, projection='3d')
# surf1 = ax1.plot_surface(fine_hgt, fine_lgt, fine_AUC, cmap='viridis', edgecolor='none')
# ax1.set_title('AUC Surface Plot')
# ax1.set_xlabel('Heads (hgt)')
# ax1.set_ylabel('Layers (lgt)')
# ax1.set_zlabel('AUC', rotation=270, labelpad=5)
# ax1.set_zlim(0.9, 1)  # 设置Z轴范围
# fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, pad=0.11)

# AUPR曲面图
ax2 = fig.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(fine_hgt, fine_lgt, fine_AUPR, cmap='magma', edgecolor='none')
ax2.set_title('AUPR Surface Plot')
ax2.set_xlabel('Heads (hgt)')
ax2.set_ylabel('Layers (lgt)')
ax2.set_zlabel('AUPR', rotation=270, labelpad=5)
ax2.set_zlim(0.78, 0.9)  # 设置Z轴范围
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, pad=0.11)

# 保存图像
plt.savefig('3D_Plots_AUPR.svg', format='svg')
plt.show()