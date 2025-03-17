############################带误差的柱状图2############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) #添加绘图框
data = pd.read_excel("E:/OneDrive/桌面/new_paper/manuscript/pic/LR.xlsx") #读取数据

index = np.arange(5)*2.5  ###设置索引，控制不同柱之间的距离用于后面的画柱状图
print(data['Lr'])
print(data['AUC'])

################关键代码行###################
plt.bar(index, data['Lr'], width=1, yerr=data['AUC'], error_kw={'ecolor':'0.9', 'capsize':5}, alpha=0.1, color='skyblue', label='lr')
#####################################################
plt.yticks(fontsize=20)  ##设置纵坐标刻度大小
plt.xticks(index, ['1e-4', '1e-1', '1e-2', '1e-3', '1e-5'], fontsize=20)#设置横坐标刻度
plt.legend(loc='best', fontsize=20)   #设置图例
plt.ylim(0.4, 1)  #设置纵坐标轴范围
plt.xlabel("LR", fontsize=20) #设置横坐标轴名称
plt.ylabel("AUC", fontsize=20)#设置纵坐标轴名称
plt.title('(a)', fontsize=20) #设置标题名称
plt.show()