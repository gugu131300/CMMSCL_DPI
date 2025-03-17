############################带误差的柱状图############################
######D84######
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 模型名称
# models = ['MCLAS-DPI', 'MGraphDTA', 'GraphDTA', 'FusionDTA', 'TransformerCPI']
#
# # AUROC数据 (平均值和标准差)
# auroc_means = [0.865712, 0.88166, 0.703003, 0.693417, 0.686913]
# auroc_stds = [0.004647, 0.016901, 0.003266, 0.016160, 0.134698]
# auroc_significance = ['', '', '']
#
# # AUPR数据 (平均值和标准差)
# aupr_means = [0.833818, 0.761833, 0.53452, 0.42423, 0.425733]
# aupr_stds = [0.009724, 0.017879, 0.027267, 0.045306, 0.26095]
# aupr_significance = ['', '', '']
#
# # x轴位置
# x = np.arange(len(models))
#
# # 条形图宽度
# width = 0.35
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制AUROC条形图
# # bars1 = ax.bar(x - width / 2, auroc_means, width, yerr=auroc_stds, capsize=5, label='AUROC', cmap='viridis', color=(250, 233, 157)(0.1, 0.2, 0.5, 0.6),
# bars1 = ax.bar(x - width / 2, auroc_means, width, yerr=auroc_stds, capsize=5, label='AUROC', color=(0.1, 0.2, 0.5, 0.6),
#                edgecolor='black')
# # 绘制AUPR条形图
# bars2 = ax.bar(x + width / 2, aupr_means, width, yerr=aupr_stds, capsize=5, label='AUPR', color=(0.1, 0.2, 0.5, 0.1),
#                edgecolor='black')
#
# # 添加显著性标记
# for i, signif in enumerate(auroc_significance):
#     if signif:
#         ax.text(x[i] - width / 2, auroc_means[i] + auroc_stds[i] + 0.01, signif, ha='center', va='bottom', fontsize=12,
#                 color='red')
#
# for i, signif in enumerate(aupr_significance):
#     if signif:
#         ax.text(x[i] + width / 2, aupr_means[i] + aupr_stds[i] + 0.01, signif, ha='center', va='bottom', fontsize=12,
#                 color='red')
#
# # 添加标签和标题
# ax.set_xlabel('Models')
# ax.set_ylabel('Scores')
# ax.set_title('Bar chart with error lines(D84)')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()
#
# plt.tight_layout()
# plt.show(block=True)

######D92M######
import numpy as np
import matplotlib.pyplot as plt

# 模型名称
models = ['MCLAS-DPI', 'MGraphDTA', 'GraphDTA', 'FusionDTA', 'TransformerCPI']

# AUROC数据 (平均值和标准差)
auroc_means = [0.8764,0.73063, 0.57167, 0.692917, 0.57936]
auroc_stds = [0.00944, 0.001184, 0.00661, 0.008669, 0.00264]
auroc_significance = ['', '', '']

# AUPR数据 (平均值和标准差)
aupr_means = [0.798292, 0.5857, 0.195853, 0.271997, 0.18918]
aupr_stds = [0.014557, 0.01499, 0.01742, 0.012955, 0.00415]
aupr_significance = ['', '', '']

# x轴位置
x = np.arange(len(models))

# 条形图宽度
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制AUROC条形图
bars1 = ax.bar(x - width / 2, auroc_means, width, yerr=auroc_stds, capsize=5, label='AUROC', color=(0.1, 0.2, 0.5, 0.6),
               edgecolor='black')
# 绘制AUPR条形图
bars2 = ax.bar(x + width / 2, aupr_means, width, yerr=aupr_stds, capsize=5, label='AUPR', color=(0.1, 0.2, 0.5, 0.1),
               edgecolor='black')

# 添加显著性标记
for i, signif in enumerate(auroc_significance):
    if signif:
        ax.text(x[i] - width / 2, auroc_means[i] + auroc_stds[i] + 0.01, signif, ha='center', va='bottom', fontsize=12,
                color='red')

for i, signif in enumerate(aupr_significance):
    if signif:
        ax.text(x[i] + width / 2, aupr_means[i] + aupr_stds[i] + 0.01, signif, ha='center', va='bottom', fontsize=12,
                color='red')

# 添加标签和标题
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Bar chart with error lines(D92M)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show(block=True)