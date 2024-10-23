from math import sqrt
import numpy as np
import csv
from scipy import stats
import pandas as pd
from dgl import load_graphs
import dgl

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def rm2(y,f):
    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def positive(y_true):
    return np.sum((y_true == 1))

def negative(y_true):
    return np.sum((y_true == 0))

def true_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 1))

def true_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 0, y_pred == 0))

def false_negative(y_true, y_pred):
    return np.sum(np.bitwise_and(y_true == 1, y_pred == 0))

def accuracy(y_true, y_pred):
    sample_count = 1.
    for s in y_true.shape:
        sample_count *= s

    return np.sum((y_true == y_pred)) / sample_count

def sensitive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    p = positive(y_true) + 1e-9
    return tp / p

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    n = negative(y_true) + 1e-9
    return tn / n

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    reca = recall(y_true, y_pred)
    fs = (2 * prec * reca) / (prec + reca)
    return fs

def Kfold(train_index, test_index, args, data_shuffled, fold_index):
    label_train, label_test = [], []
    proteinseq_train, proteinseq_test = [], []
    compoundsmiles_train, compoundsmiles_test = [], []
    proteins_graph_train, proteins_graph_test = [], []
    compounds_graph_train, compounds_graph_test = [], []
    proteinseq_train_id, proteinseq_test_id = [], []
    compoundsmiles_train_id, compoundsmiles_test_id = [], []

    #############################################train、test的nfold protein_id
    for no, id in enumerate(train_index):
        proteinseq_train_id.append(data_shuffled.iloc[id]['PROTEIN_GCN'])
    np.save(args.foldpath + str(fold_index) + 'proteinseq_train_id.npy', proteinseq_train_id)

    for no, id in enumerate(test_index):
        proteinseq_test_id.append(data_shuffled.iloc[id]['PROTEIN_GCN'])
    np.save(args.foldpath + str(fold_index) + 'proteinseq_test_id.npy', proteinseq_test_id)

    #############################################train、test的nfold compound_id
    for no, id in enumerate(train_index):
        compoundsmiles_train_id.append(data_shuffled.iloc[id]['COMPOUND_GCN'])
    np.save(args.foldpath + str(fold_index) + 'compoundsmiles_train_id.npy', compoundsmiles_train_id)

    for no, id in enumerate(test_index):
        compoundsmiles_test_id.append(data_shuffled.iloc[id]['COMPOUND_GCN'])
    np.save(args.foldpath + str(fold_index) + 'compoundsmiles_test_id.npy', compoundsmiles_test_id)

    #############################################train、test的nfold protein graph
    for no, id in enumerate(train_index):
        protein_graph_train, _ = load_graphs(
            'E:/OneDrive/桌面/new_paper/dataset/' + args.dataset + '/processed/protein_graph_id/' +
            data_shuffled.iloc[id]['PROTEIN_ID'] + '.bin')
        proteins_graph_train.append(protein_graph_train[0])
    dgl.save_graphs(args.foldpath + str(fold_index) + 'trainprotein_graph.bin', proteins_graph_train)
    for no, id in enumerate(test_index):
        protein_graph_test, _ = load_graphs(
            'E:/OneDrive/桌面/new_paper/dataset/' + args.dataset + '/processed/protein_graph_id/' +
            data_shuffled.iloc[id]['PROTEIN_ID'] + '.bin')
        proteins_graph_test.append(protein_graph_test[0])
        # print("id:" + str(id))
    dgl.save_graphs(args.foldpath + str(fold_index) + 'testprotein_graph.bin', proteins_graph_test)

    ###########################################train、test的nfold compound graph
    for no, id in enumerate(train_index):
        compound_graph_train, _ = load_graphs(
            'E:/OneDrive/桌面/new_paper/dataset/' + args.dataset + '/processed' + '/compound_graph_id/' + str(
                data_shuffled.iloc[id]['COMPOUND_ID']) + '.bin')
        compounds_graph_train.append(compound_graph_train[0])
    dgl.save_graphs(args.foldpath + str(fold_index) + 'traincompound_graph.bin', compounds_graph_train)
    for no, id in enumerate(test_index):
        compound_graph_test, _ = load_graphs(
            'E:/OneDrive/桌面/new_paper/dataset/' + args.dataset + '/processed' + '/compound_graph_id/' + str(
                data_shuffled.iloc[id]['COMPOUND_ID']) + '.bin')
        compounds_graph_test.append(compound_graph_test[0])
    dgl.save_graphs(args.foldpath + str(fold_index) + 'testcompound_graph.bin', compounds_graph_test)

    #############################################train、test的nfold label
    for no, id in enumerate(train_index):
        label_train.append(data_shuffled.iloc[id]['REG_LABEL'])
    np.save(args.foldpath + str(fold_index) + 'label_train.npy', label_train)
    for no, id in enumerate(test_index):
        label_test.append(data_shuffled.iloc[id]['REG_LABEL'])
    np.save(args.foldpath + str(fold_index) + 'label_test.npy', label_test)

    ###############################################train、test的protein sequence
    for no, id in enumerate(train_index):
        proteinseq_train.append(data_shuffled.iloc[id]['PROTEIN_SEQUENCE'])
    np.save(args.foldpath + str(fold_index) + 'proteinseq_train.npy', proteinseq_train)

    for no, id in enumerate(test_index):
        proteinseq_test.append(data_shuffled.iloc[id]['PROTEIN_SEQUENCE'])
    np.save(args.foldpath + str(fold_index) + 'proteinseq_test.npy', proteinseq_test)

    ###############################train、test的compound smiles
    for no, id in enumerate(train_index):
        compoundsmiles_train.append(data_shuffled.iloc[id]['COMPOUND_SMILES'])
    output_file = args.foldpath + str(fold_index) + 'compoundsmiles_train.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['COMPOUND_SMILES'])  # 添加列名
        for smiles in compoundsmiles_train:
            csvwriter.writerow([smiles])

    for no, id in enumerate(test_index):
        compoundsmiles_test.append(data_shuffled.iloc[id]['COMPOUND_SMILES'])
    output_file = args.foldpath + str(fold_index) + 'compoundsmiles_test.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['COMPOUND_SMILES'])  # 添加列名
        for smiles in compoundsmiles_test:
            csvwriter.writerow([smiles])

def dpi_matrix(path):

    # 读取 CSV 文件
    data = pd.read_csv(path)
    # 创建一个空的 DataFrame 用于存储对称矩阵
    dpi_matrix = pd.DataFrame(index=data['COMPOUND_GCN'].unique(), columns=data['PROTEIN_GCN'].unique())

    # 填充对称矩阵
    for index, row in data.iterrows():
        compound = row['COMPOUND_GCN']
        protein = row['PROTEIN_GCN']
        label = row['LABEL']
        dpi_matrix.loc[compound, protein] = label
    
    # 将缺失值填充为0
    dpi_matrix = dpi_matrix.fillna(0)

    return dpi_matrix

def gaussian_kernel(x,y,sigma=1.0):
    distance = np.linalg.norm(x-y)
    similarity = np.exp(-distance**2 / (2 * (sigma**2)))
    return similarity

def compute_similarity_matrix(X, sigma=1.0):
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = gaussian_kernel(X[i], X[j], sigma)
    return S

def feature_nomalize(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return (x - mu) / sigma

# 用Random Walk对相似度网络进行特征提取
def diffusionRWR(A, maxiter, restartProb):
    n = A.shape[0]

    # Add self-edge to isolated nodes
    A += np.diag(np.sum(A, axis=0) == 0)
    dia_I = np.eye(n)

    # Normalize the adjacency matrix
    def renorm(M):
        return M / np.sum(M, axis=0)
    P = renorm(A)

    # Personalized PageRank
    restart = np.eye(n)
    Q = np.eye(n)
    for i in range(maxiter):
        Q_new = (1 - restartProb) * np.dot(P, Q) + restartProb * restart
        delta = np.linalg.norm(Q - Q_new, 'fro')
        # print('Iter {}. Frobenius norm: {}'.format(i+1, delta))
        Q = Q_new
        if delta < 1e-6:
            # print('Converged.')
            break

    Q = Q - dia_I
    Q = np.where(Q<0,0,Q)
    Q =feature_nomalize(Q)
    Q = Q + dia_I
    return Q

