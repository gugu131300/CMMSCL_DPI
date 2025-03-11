import copy
import timeit
import argparse
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, roc_auc_score
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import *
from DTIDataset_cl2 import DTIDataset
from model.DTInet_cl2RWR_class import DTInet_cl2
from protein_embedding import protein_emb
import pandas as pd
import torch
import os
from dgl import load_graphs
import dgl

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model, train_loader, optimizer, args,compound_graph, protein_graph,
          protein_embedding, compoundsmiles, DPI_label,DTI_edge, RWR, transposed_RWR):
    model.train()
    
    for batch_idx, data in enumerate(train_loader):
        # torch.autograd.set_detect_anomaly(True)
        # model.train()
        # optimizer.zero_grad()
        # affinity = data[-1].to(args.device)
        # label = (affinity > 5).int()
        label = data[-1].to(args.device)
        compound_smiles_id, protein_seq_id = data[:-1]
        c_graph = copy.deepcopy(compound_graph.to(args.device))
        p_graph = copy.deepcopy(protein_graph.to(args.device))
        protein_embedding = protein_embedding.to(args.device)
        combined_fea_test, output, cl_loss = model(args,c_graph, p_graph, protein_embedding,
                    compound_smiles_id, args.batch, protein_seq_id, compoundsmiles,
                    DPI_label,DTI_edge, RWR, transposed_RWR)
        loss = criterion(output, label.view(-1, 1).float().to(args.device))
        loss = loss + args.alpha*cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, test_loader, args, compound_graph, protein_graph, protein_embedding,
         compoundsmiles, DPI_label, DTI_edge, RWR, transposed_RWR):
    model.eval()
    compound_smiles_ids = []
    protein_seq_ids = []
    total_labels = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_pred_scores = torch.Tensor()
    total_fea_test = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            # affinity = data[-1].to(device)
            # label = (affinity > 5).int()
            label = data[-1].to(device)
            compound_smiles_id, protein_seq_id = data[:-1]
            compound_graph = compound_graph.to(args.device)
            protein_graph = protein_graph.to(args.device)
            protein_embedding = protein_embedding.to(args.device)
            combined_fea_test, output, loss = model(args,compound_graph, protein_graph, protein_embedding, 
                                 compound_smiles_id,args.batch, protein_seq_id, 
                                 compoundsmiles,DPI_label,DTI_edge, RWR, transposed_RWR)
            predicted_test = (output > 0.5).int()
            compound_smiles_ids.extend(compound_smiles_id)
            protein_seq_ids.extend(protein_seq_id)
            total_pred_scores = torch.cat((total_pred_scores, output.cpu()), 0)
            total_pred_labels = torch.cat((total_pred_labels, predicted_test.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)
            total_fea_test = torch.cat((total_fea_test, combined_fea_test.cpu()), 0)

    compound_smiles_ids = np.array(compound_smiles_ids)
    protein_seq_ids = np.array(protein_seq_ids)
    total_labels = total_labels.numpy().flatten()
    total_pred_labels = total_pred_labels.numpy().flatten()
    total_pred_scores = total_pred_scores.numpy().flatten()
    # total_fea_test = total_fea_test.numpy().flatten()
    #total_fea_test = total_fea_test.flatten()
    
    # Save total_labels to a file
    np.save("/home/zqguxingyue/DTI/D84/D84_log/unlabel_total_pred_labels.npy", total_pred_labels)
    np.save("/home/zqguxingyue/DTI/D84/D84_log/unlabel_total_pred_scores.npy", total_pred_scores)
    np.save("/home/zqguxingyue/DTI/D84/D84_log/unlabel_compound_smiles_ids.npy", compound_smiles_ids)
    np.save("/home/zqguxingyue/DTI/D84/D84_log/unlabel_protein_seq_ids.npy", protein_seq_ids)
    prec = precision_score(total_labels, total_pred_labels)
    recall = recall_score(total_labels, total_pred_labels)
    roc_auc = roc_auc_score(total_labels, total_pred_scores)
    acc = accuracy_score(total_labels, total_pred_labels)
    prc = average_precision_score(total_labels, total_pred_scores)
    # tpr, fpr, _ = precision_recall_curve(total_labels, total_pred_scores)
    # AUPR = auc(fpr, tpr)

    if epoch % 1 == 0:
        np.savetxt("Davis/Davis_result_eps/unlabel_total_combined_fea_test_{}.txt".format(epoch + 1), total_fea_test.detach())
        np.savetxt("Davis/Davis_result_eps/unlabel_total_label_test_{}.txt".format(epoch + 1), total_labels)

        x_data = total_fea_test.detach()  # 需要可视化的数据
        y_data = total_labels  # 可视化的数据对应的label，label可以是true label，或者是分类or聚类后对应的label
        X = np.array(x_data)
        y = np.array(y_data).astype(int)
        '''t-SNE'''
        tsne = manifold.TSNE(n_components=2, init='random', random_state=1,
                             learning_rate=200.0)  # n_components=2降维为2维并且可视化
        X_tsne = tsne.fit_transform(X)
        '''空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = X_tsne

        plt.figure(figsize=(15, 15))
        plt.xlim([x_min[0] - 5, x_max[0] + 5])
        plt.ylim([x_min[1] - 5, x_max[1] + 5])
        # for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str('.'), color=c[y[i]], fontdict={'weight': 'bold', 'size': 40})
        # plt.show()
        save_dir = "Davis/Davis_result_eps"
        np.savetxt(os.path.join(save_dir, "Embeds_{}_epoch_{}.txt".format("DPI", epoch + 1)), X_norm)
        np.savetxt(os.path.join(save_dir, "labels_{}_epoch_{}.txt".format("DPI", epoch + 1)), y)

        # 通过不同的颜色表示不同的标签值
        #colors = ['orange', 'lightblue']  # 这里根据您的标签值设置颜色
        colors = [(117/255, 179/255, 113/255), (230/255, 50/255, 50/255)]

        # plt.figure(figsize=(8, 8))
        for i in range(len(X_norm)):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[int(y[i])], s=9)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Scatter Plot')
        plt.savefig(os.path.join(save_dir, "tsen_Random_features_{}_epoch_{}.eps".format("DPI", epoch + 1)))
        # plt.show()

    return recall, roc_auc, acc, prec, prc, total_labels, total_pred_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=100, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')  # 1e-4
    parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (1 - keep probability).')  # 0.1
    parser.add_argument('--seed', type=int, default=2024, help='Random Seed')
    parser.add_argument('--atom_dim', type=int, default=34, help='Dimension for Atom')
    parser.add_argument('--n_attentions', type=int, default=1, help='n_attentions')
    parser.add_argument('--gt_heads', type=int, default=8, help='gt_heads')
    parser.add_argument('--hidden-dim', type=int, default=64, help='hidden units of FC layers (default: 256)')
    parser.add_argument('--graph-dim', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--gt_layers', type=int, default=8, help='gt_layers')
    parser.add_argument('--compound_dim', type=int, default=34, help='compound_dim')
    parser.add_argument('--protein_dim', type=int, default=64, help='protein_dim')
    parser.add_argument('--out_dim', type=int, default=1, help='out_dim')
    parser.add_argument('--smile_vocab', type=int, default=63, help='smile_vocab')
    parser.add_argument('--rnn_dim', type=int, default=64, help='hidden unit/s of RNNs (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='rnn_dim')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--num_features_xt', type=int, default=26, help='num_features_xt')
    parser.add_argument('--embed_dim', type=int, default=128, help='embed_dim')
    parser.add_argument('--n_filters', type=int, default=64, help='n_filters')
    parser.add_argument('--output_dim', type=int, default=64, help='output_dim')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha')
    parser.add_argument('--objective', type=str, default='classification', help='Objective (classification / regression)')

    parser.add_argument('--dataset_path', default='/home/zqguxingyue/DTI/D84/D84_classification.csv', help='dataset path')
    parser.add_argument('--PPI_path', default='/home/zqguxingyue/DTI/D84/PPI/', help='PPI')
    parser.add_argument('--dataset', default='D84', help='dataset')
    parser.add_argument('--foldpath', default='/home/zqguxingyue/DTI/D84/PPI/fold_id/', help='foldpath')

    args = parser.parse_args()
    # 打印参数及其值
    print("********** D84 unlabel  **********")
    print("**********有struc + 分类任务 + transformer **********")
    print(args.__dict__)
    args.device = device

    # 读取CSV文件，将第一行作为数据的一部分，指定列名
    data = pd.read_csv(args.dataset_path)
    # 打乱数据集顺序
    data_shuffled = shuffle(data, random_state=42)
    # 初始化KFold对象
    kf = KFold(n_splits=5)

    start = timeit.default_timer()
    best_epoch = -1
    best_acc = 0
    best_AUPR = -1
    best_prec = 0
    best_recall = 0
    best_auc = 0

    """Start training."""
    # 交叉验证
    fold_index = 0
    # 检查数据集文件是否存在
    dataset_exists = True

    if dataset_exists:
        print("dataset_exists")
        protein_graph, _ = load_graphs(args.PPI_path + 'protein_graph_unique.bin')
        protein_graph = list(protein_graph)
        protein_graph = dgl.batch(protein_graph)

        compound_graph, _ = load_graphs(args.PPI_path + 'compound_graph_unique.bin')
        compound_graph = list(compound_graph)
        compound_graph = dgl.batch(compound_graph)
        proteinseq = np.load(args.PPI_path + 'protein_seq_unique.npy',allow_pickle=True)
        compoundsmiles = np.load(args.PPI_path + 'compound_smiles_unique.npy',allow_pickle=True)
        affinity = np.load(args.foldpath + str(fold_index + 1) + 'label_train.npy')
        # predicted_test = (pred_test > 0.5).int()
        DPI_label = np.where(affinity == 5, 0, 1)

        protein_embedding = protein_emb(proteinseq)
        protein_embedding = torch.tensor(np.array(protein_embedding)).to(args.device)
        
        ###求DPI的相似度网络特征
        # 构建相似度网络
        matrix = dpi_matrix(args.dataset_path).values
        matrix_similar = compute_similarity_matrix(matrix)
        # 使用Random Walk特征提取
        RWR = diffusionRWR(matrix_similar, 20, 0.50)
        RWR = torch.from_numpy(RWR).float().to(args.device)

        ###PDI的相似度网络特征
        # 构建相似度网络
        transposed_matrix = np.transpose(matrix)
        transposed_similar = compute_similarity_matrix(transposed_matrix)
        # 使用Random Walk特征提取
        transposed_RWR = diffusionRWR(transposed_similar, 20, 0.50)
        transposed_RWR = torch.from_numpy(transposed_RWR).float().to(args.device)

        model = DTInet_cl2(args)
        model = model.to(args.device)
        # print("elif dataset_exists device:{}".format(args.device))
        file_model = 'model_save/' + args.dataset + '/fold/' + str(fold_index) + '/'
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.8, patience=80,
            verbose=True, min_lr=1e-5)
        criterion = nn.BCELoss()

        trainDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'unproteinseq_train_id.npy',
                              compoundsmiles=args.foldpath + str(fold_index + 1) + 'uncompoundsmiles_train_id.npy',
                              label=args.foldpath + str(fold_index + 1) + 'unlabel_train.npy')
        testDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'unproteinseq_test_id.npy',
                             compoundsmiles=args.foldpath + str(fold_index + 1) + 'uncompoundsmiles_test_id.npy',
                             label=args.foldpath + str(fold_index + 1) + 'unlabel_test.npy')

        # todo 0613添加
        train_loader = DataLoader(trainDTI, batch_size=args.batch, shuffle=True,
                                      collate_fn=trainDTI.collate, drop_last=False)
        test_loader = DataLoader(testDTI, batch_size=args.batch, shuffle=False,
                                     collate_fn=testDTI.collate, drop_last=False)
        # todo 加载去重后的protein seq和graph
        # todo 加载去重后的compound smiles
        # todo 加载去重后的protein 氨基酸特征
        # todo 加载去重后的compound 原子特征

        # todo 加载batch中的protein id和compound id
        trainDTI_truple = trainDTI.collate(trainDTI)
        compound_idx = torch.tensor(trainDTI_truple[0],dtype=torch.int32)
        compound_idx = compound_idx.unsqueeze(0)
        protein_idx = torch.tensor(trainDTI_truple[1],dtype=torch.int32)
        protein_idx = protein_idx.unsqueeze(0)
        compound_idx = compound_idx + protein_embedding.shape[0]
        DPI_label = np.where(trainDTI_truple[2] >5,1,0)
        DTI_edge = torch.cat([protein_idx, compound_idx], dim=0).to(args.device)

        for epoch in range(args.epochs):
            train(model, train_loader, optimizer, args,
                  compound_graph, protein_graph, protein_embedding, compoundsmiles,
                  DPI_label,DTI_edge, RWR, transposed_RWR)
            recall, auc, acc, prec, prc, total_labels, total_scores = test(model, test_loader, args,
                 compound_graph, protein_graph, protein_embedding, compoundsmiles,
                        DPI_label,DTI_edge, RWR, transposed_RWR)
            scheduler.step(auc)
            # print("epoch:{} device:{}".format(epoch, args.device))
            end = timeit.default_timer()
            time = end - start
            print("epoch     time     acc    prec     recall    AUC     AUPR")
            ret = [epoch + 1, round(time, 2), round(acc, 5), round(prec, 5), round(recall, 5),
                    round(auc, 5), round(prc, 5)]
            print('\t\t'.join(map(str, ret)))
            if auc > best_auc:
                best_epoch = epoch + 1
                best_auc = auc
                np.save("total_labels_epoches_{}".format(epoch), total_labels)
                np.save("total_scores_epoches_{}".format(epoch), total_scores)
                print('**********************AUC improved at epoch ', best_epoch, ';\tbest_AUC:', best_auc, '**********************')
        print('***best_AUC: ', best_auc, '***best_epoch: ',best_epoch)