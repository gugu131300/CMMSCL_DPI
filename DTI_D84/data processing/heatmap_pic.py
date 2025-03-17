import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

###############################################################D92M
# data = {
#     'Model': ['CMMSCL-DPI', 'MGraphDTA', 'GraphDTA', 'FusionDTA', 'TransformerCPI'],
#     'ACC': [0.8330, 0.7302, 0.8201, 0.8021, 0.6402],
#     'Prec': [0.8030, 0.7143, 0.2110, 0.3684, 0.1861],
#     'Recall': [0.7273, 0.3125, 0.0645, 0.4516, 0.4211],
#     'AUC': [0.8795, 0.7292, 0.5646, 0.7021, 0.5821],
#     'AUPR': [0.8218, 0.5696, 0.1858, 0.2862, 0.1932]
# }
#
# # Create DataFrame
# df = pd.DataFrame(data)
# df.set_index('Model', inplace=True)
#
# # Plotting heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(df, annot=True, cmap="Oranges", fmt='.4g', cbar_kws={'label': 'Performance'})
# plt.title('Model Performance Heatmap')
#
# # Save the figure as an SVG file
# plt.savefig('F:/D92M_Heatmap.svg', format='svg',dpi=600)
#
# plt.show()

################################################################D84
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    'Model': ['CMMSCL-DPI', 'MGraphDTA', 'GraphDTA', 'FusionDTA', 'TransformerCPI'],
    'ACC': [0.8220, 0.8301, 0.7902, 0.7531, 0.7221],
    'Prec': [0.8462, 0.6190, 0.5609, 0.4688, 0.2105],
    'Recall': [0.6111, 0.6842, 0.6053, 0.3947, 0.3333],
    'AUC': [0.8646, 0.8964, 0.7030, 0.6740, 0.5929],
    'AUPR': [0.8387, 0.7688, 0.5175, 0.3927, 0.2930]
}

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# Plotting heatmap
plt.figure(figsize=(10, 7))
heatmap = sns.heatmap(df, annot=True, cmap="Oranges", fmt='.4g', cbar_kws={'label': 'Performance'})
plt.title('Model Performance Heatmap')

# Save the figure as an SVG file
plt.savefig('F:/D84_Heatmap.svg', format='svg',dpi=600)

plt.show()

################################################################Davis
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Data
# data = {
#     'Model': ['CMMSCL-DPI', 'MGraphDTA', 'GraphDTA', 'FusionDTA', 'TransformerCPI'],
#     'ACC': [0.8922, 0.8331, 0.7414, 0.8412, 0.6721],
#     'Prec': [0.8401, 0.7626, 0.7037, 0.4684, 0.5815],
#     'Recall': [0.7638, 0.5493, 0.5204, 0.3516, 0.3898],
#     'AUC': [0.9429, 0.8796, 0.7735, 0.8421, 0.6980],
#     'AUPR': [0.8820, 0.7586, 0.7206, 0.3562, 0.5830]
# }
#
# # Create DataFrame
# df = pd.DataFrame(data)
# df.set_index('Model', inplace=True)
#
# # Plotting heatmap
# plt.figure(figsize=(10, 7))
# heatmap = sns.heatmap(df, annot=True, cmap="Oranges", fmt='.4g', cbar_kws={'label': 'Performance'})
# plt.title('Model Performance Heatmap')
#
# # Save the figure as an SVG file
# plt.savefig('F:/Davis_Heatmap.svg', format='svg',dpi=600)
#
# plt.show()
