import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
# todo
# 绘制可视化的图

# Load total_labels and total_pred_scores from npy files for the first model
total_labels_1 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_labels_epoches_945.npy')
total_pred_scores_1 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_scores_epoches_945.npy')

# Load total_labels and total_pred_scores from npy files for the second model
total_labels_2 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_labels_epoches_clstruc468.npy')
total_pred_scores_2 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_scores_epoches_clstruc468.npy')

total_labels_3 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_labels_epoches_delclGCN190.npy')
total_pred_scores_3 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_scores_epoches_delclGCN190.npy')

total_labels_4 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_labels_epoches_delGCN341.npy')
total_pred_scores_4 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_scores_epoches_delGCN341.npy')

total_labels_5 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_labels_epoches_delstruc93.npy')
total_pred_scores_5 = np.load('E:/OneDrive/桌面/new_paper/manuscript/D84/XR/total_scores_epoches_delstruc93.npy')

# Calculate ROC AUC and PR AUC for the first model
roc_auc_1 = roc_auc_score(total_labels_1, total_pred_scores_1)
prc_1 = average_precision_score(total_labels_1, total_pred_scores_1)

# Calculate ROC AUC and PR AUC for the second model
roc_auc_2 = roc_auc_score(total_labels_2, total_pred_scores_2)
prc_2 = average_precision_score(total_labels_2, total_pred_scores_2)

# Calculate ROC AUC and PR AUC for the second model
roc_auc_3 = roc_auc_score(total_labels_3, total_pred_scores_3)
prc_3 = average_precision_score(total_labels_3, total_pred_scores_3)

# Calculate ROC AUC and PR AUC for the second model
roc_auc_4 = roc_auc_score(total_labels_4, total_pred_scores_4)
prc_4 = average_precision_score(total_labels_4, total_pred_scores_4)

# Calculate ROC AUC and PR AUC for the second model
roc_auc_5 = roc_auc_score(total_labels_5, total_pred_scores_5)
prc_5 = average_precision_score(total_labels_5, total_pred_scores_5)

# Calculate ROC curve and PR curve for the first model
fpr_1, tpr_1, _1 = roc_curve(total_labels_1, total_pred_scores_1)
precision_1, recall_1, _1 = precision_recall_curve(total_labels_1, total_pred_scores_1)

# Calculate ROC curve and PR curve for the second model
fpr_2, tpr_2, _2 = roc_curve(total_labels_2, total_pred_scores_2)
precision_2, recall_2, _2 = precision_recall_curve(total_labels_2, total_pred_scores_2)

# Calculate ROC curve and PR curve for the third model
fpr_3, tpr_3, _3 = roc_curve(total_labels_3, total_pred_scores_3)
precision_3, recall_3, _3 = precision_recall_curve(total_labels_3, total_pred_scores_3)

# Calculate ROC curve and PR curve for the forth model
fpr_4, tpr_4, _4 = roc_curve(total_labels_4, total_pred_scores_4)
precision_4, recall_4, _4 = precision_recall_curve(total_labels_4, total_pred_scores_4)

# Calculate ROC curve and PR curve for the forth model
fpr_5, tpr_5, _5 = roc_curve(total_labels_5, total_pred_scores_5)
precision_5, recall_5, _5 = precision_recall_curve(total_labels_5, total_pred_scores_5)

# Plot ROC curves for both models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr_1, tpr_1, color=(231/255, 34/255, 43/255), lw=1.8, label=f'Model 1 ROC (AUC = {roc_auc_1:.4f})')
plt.plot(fpr_2, tpr_2, color=(79/255, 152/255, 81/255), lw=1.5, label=f'Model 2 ROC (AUC = {roc_auc_2:.4f})')
plt.plot(fpr_3, tpr_3, color=(149/255, 104/255, 189/255), lw=1.5, label=f'Model 2 ROC (AUC = {roc_auc_3:.4f})')
plt.plot(fpr_4, tpr_4, color=(243/255, 123/255, 21/255), lw=1.5, label=f'Model 2 ROC (AUC = {roc_auc_4:.4f})')
plt.plot(fpr_5, tpr_5, color=(34/255, 120/255, 179/255), lw=1.5, label=f'Model 2 ROC (AUC = {roc_auc_5:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Plot PR curves for both models
plt.subplot(1, 2, 2)
plt.plot(recall_1, precision_1, color=(231/255, 34/255, 43/255), lw=1.8, label=f'Model 1 PR')
plt.plot(recall_2, precision_2, color=(79/255, 152/255, 81/255), lw=1.5, label=f'Model 2 PR')
plt.plot(recall_3, precision_3, color=(149/255, 104/255, 189/255), lw=1.5, label=f'Model 3 PR')
plt.plot(recall_4, precision_4, color=(243/255, 123/255, 21/255), lw=1.5, label=f'Model 4 PR')
plt.plot(recall_5, precision_5, color=(34/255, 120/255, 179/255), lw=1.5, label=f'Model 5 PR')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='lower left')
plt.grid(True)

plt.tight_layout()
plt.show()