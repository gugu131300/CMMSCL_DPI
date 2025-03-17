import numpy as np

# 加载 compound_smiles_unique.npy 文件
smiles = np.load("E:/OneDrive/桌面/new_paper/dataset/Davis/PPI/proteins.npy", allow_pickle=True)
# seqs = np.load("F:/unlabel_protein_seq_ids.npy", allow_pickle=True)
# pred_labels = np.load("F:/unlabel_total_pred_labels.npy", allow_pickle=True)
# pred_scores = np.load("F:/unlabel_total_pred_scores.npy", allow_pickle=True)

# labels = labels.astype(int)
# np.savetxt('F:/scores.txt', pred_scores, fmt='%.8f', newline='\n')
# print("scores", pred_scores)
# print("score.shape", pred_scores.shape)
#
# np.savetxt('F:/predicted_labels.txt', pred_labels, fmt='%d', newline='\n')
# print("predicted_labels", pred_labels)
# print("predicted_labels.shape", pred_labels.shape)

np.savetxt('F:/smiles.txt', smiles,  fmt='%d', newline='\n')
print("smiles", smiles)
print("smiles.shape", smiles.shape)

# np.savetxt('F:/seqs.txt', seqs,  fmt='%d', newline='\n')
# print("seqs", seqs)
# print("seqs.shape", seqs.shape)