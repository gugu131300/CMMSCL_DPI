import numpy as np
import torch

def seq_cat(prot):
    # x = np.zeros(max_seq_len)
    default_value = 21
    x = np.full((max_seq_len), default_value)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

seq_voc = "ACDEFGHIKLMNPQRSTVWXY"
seq_dict = {v: (i) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 400

def protein_emb(protein_seqs):
    # 转化蛋白质序列为嵌入表示
    protein_embs = [seq_cat(seq) for seq in protein_seqs]
    return protein_embs

def amino_acid():
    # 读取文件并将每行数据读取为张量
    tensor_list = []
    with open('/home/zqguxingyue/DTI/Davis/all_assign_mean.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 将每行的数据拆分为列表，并转换为张量
            row = [float(value) for value in line.strip().split()]
            # tensor_row = torch.tensor(row)
            tensor_list.append(row)
        tensor_array = np.array(tensor_list)
        amino_tensor = torch.tensor(tensor_array, dtype=torch.float32)

    return amino_tensor
