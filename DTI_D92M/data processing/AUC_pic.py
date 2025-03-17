import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

# Load total_labels and total_pred_scores from npy files
total_labels = np.load('E:/OneDrive/桌面/CMMSCL_DPI/manuscript/DAVIS/XR/total_labels_epoches_94.npy')
total_pred_scores = np.load('E:/OneDrive/桌面/CMMSCL_DPI/manuscript/DAVIS/XR/total_labels_epoches_94.npy')

# Calculate ROC AUC and PR AUC
roc_auc = roc_auc_score(total_labels, total_pred_scores)
prc = average_precision_score(total_labels, total_pred_scores)

# Calculate ROC curve and PR curve
fpr, tpr, _ = roc_curve(total_labels, total_pred_scores)
precision, recall, _ = precision_recall_curve(total_labels, total_pred_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='lightblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_high_res.tiff', format='tiff', dpi=600)
plt.show()

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='red', lw=2, label=f'PR Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='lower left')
plt.grid(True)

plt.savefig('pr_curve_high_res.tiff', format='tiff', dpi=600)
plt.show()