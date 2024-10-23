import copy
import timeit
import argparse
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, roc_auc_score
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import *
from DTIDataset_cl2 import DTIDataset
from model.DTInet_XR import DTInet_cl2
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
        affinity = data[-1].to(args.device)
        label = (affinity > 5).int()
        compound_smiles_id, protein_seq_id = data[:-1]
        c_graph = copy.deepcopy(compound_graph.to(args.device))
        p_graph = copy.deepcopy(protein_graph.to(args.device))
        protein_embedding = protein_embedding.to(args.device)
        output = model(args,c_graph, p_graph, protein_embedding,
                    compound_smiles_id, args.batch, protein_seq_id, compoundsmiles,
                    DPI_label,DTI_edge, RWR, transposed_RWR)
        loss = criterion(output, label.view(-1, 1).float().to(args.device))
        #loss = loss + args.alpha*cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, test_loader, args, compound_graph, protein_graph, protein_embedding,
         compoundsmiles, DPI_label,DTI_edge, RWR, transposed_RWR):
    model.eval()
    total_labels = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_pred_scores = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            affinity = data[-1].to(device)
            label = (affinity > 5).int()
            compound_smiles_id, protein_seq_id = data[:-1]
            compound_graph = compound_graph.to(args.device)
            protein_graph = protein_graph.to(args.device)
            protein_embedding = protein_embedding.to(args.device)
            output = model(args,compound_graph, protein_graph, protein_embedding, 
                                 compound_smiles_id,args.batch, protein_seq_id, 
                                 compoundsmiles,DPI_label,DTI_edge, RWR, transposed_RWR)
            predicted_test = (output > 0.5).int()
            total_pred_scores = torch.cat((total_pred_scores, output.cpu()), 0)
            total_pred_labels = torch.cat((total_pred_labels, predicted_test.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_pred_labels = total_pred_labels.numpy().flatten()
    total_pred_scores = total_pred_scores.numpy().flatten()

    prec = precision_score(total_labels, total_pred_labels)
    recall = recall_score(total_labels, total_pred_labels)
    roc_auc = roc_auc_score(total_labels, total_pred_scores)
    # tpr, fpr, _ = precision_recall_curve(total_labels, total_pred_scores)
    # AUPR = auc(fpr, tpr)
    acc = accuracy_score(total_labels, total_pred_labels)
    prc = average_precision_score(total_labels, total_pred_scores)

    return recall, roc_auc, acc, prec, prc ,total_labels, total_pred_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=100, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')  # 1e-4
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')  # 0.1
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

    parser.add_argument('--dataset_path', default='/home/zqguxingyue/DTI/Davis/PPI/Davis_classification.csv', help='dataset path')
    parser.add_argument('--PPI_path', default='/home/zqguxingyue/DTI/Davis/PPI/', help='PPI')
    parser.add_argument('--dataset', default='Davis', help='dataset')
    parser.add_argument('--foldpath', default='/home/zqguxingyue/DTI/Davis/PPI/fold_id/', help='foldpath')

    args = parser.parse_args()
    # 打印参数及其值
    print("**********消融实验 + 删除struct特征 + 得到AUC和PR曲线的值**********")
    print("**********2cl_RWR + 分类任务 + transformer **********")
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
    dataset_exists = os.path.exists(args.foldpath + str(fold_index + 1) + 'trainprotein_graph.bin') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'traincompound_graph.bin') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'proteinseq_train.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'compoundsmiles_train.csv') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'label_train.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'testprotein_graph.bin') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'testcompound_graph.bin') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'proteinseq_test.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'compoundsmiles_test.csv') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'label_test.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'proteinseq_train_id.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'proteinseq_test_id.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'compoundsmiles_train_id.npy') and \
                     os.path.exists(args.foldpath + str(fold_index + 1) + 'compoundsmiles_test_id.npy')

    if not dataset_exists:
        print("not dataset_exists")
        for train_index, test_index in kf.split(data_shuffled):
            print('Training on ' + args.dataset + ', fold:' + str(fold_index))
            fold_index += 1

            Kfold(train_index, test_index, args, data_shuffled, fold_index)
            # proteinemb_train = protein_emb(proteinseq_train)
            # proteinemb_test = protein_emb(proteinseq_test)
            proteinseq = np.load(args.PPI_path + 'protein_seq_unique.npy',allow_pickle=True)
            protein_embedding = protein_emb(proteinseq)
            protein_embedding = torch.tensor(np.array(protein_embedding)).to(args.device)
            protein_graph, _ = load_graphs(args.PPI_path + 'protein_graph_unique.bin')
            protein_graph = list(protein_graph)
            protein_graph = dgl.batch(protein_graph).to(args.device)
            compound_graph, _ = load_graphs(args.PPI_path + 'compound_graph_unique.bin')
            compound_graph = list(compound_graph)
            compound_graph = dgl.batch(compound_graph).to(args.device)
            affinity = np.load(args.foldpath + str(fold_index + 1) + 'label_train.npy')
            # predicted_test = (pred_test > 0.5).int()
            DPI_label = np.where(affinity==5,0,1)
            ###求DPI的相似度网络特征
            matrix = dpi_matrix(args.dataset_path).values
            matrix_similar = compute_similarity_matrix(matrix)
            RWR = diffusionRWR(matrix_similar, 20, 0.50)
            RWR = torch.from_numpy(RWR).float()
            ###PDI的相似度网络特征
            transposed_matrix = np.transpose(matrix)
            #transposed_matrix = metrics.dpi_matrix(args.dataset_path)
            transposed_similar = compute_similarity_matrix(transposed_matrix)
            transposed_RWR = diffusionRWR(transposed_similar, 20, 0.50)
            transposed_RWR = torch.from_numpy(transposed_RWR).float()

            model = DTInet_cl2(args)
            model = model.to(args.device)
            file_model = args.foldpath + str(fold_index+1) + '/'
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=0.8, patience=80,
                verbose=True, min_lr=1e-5)
            criterion = nn.BCELoss()

            trainDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'proteinseq_train_id.npy',
                                  compoundsmiles=args.foldpath + str(fold_index + 1) + 'compoundsmiles_train_id.csv',
                                  label=args.foldpath + str(fold_index + 1) + 'label_train.npy')
            testDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'proteinseq_test_id.npy',
                                 compoundsmiles=args.foldpath + str(fold_index + 1) + 'compoundsmiles_test_id.csv',
                                 label=args.foldpath + str(fold_index + 1) + 'label_test.npy')

            train_loader = DataLoader(trainDTI, batch_size=args.batch, shuffle=True,
                                      collate_fn=trainDTI.collate, drop_last=True)
            test_loader = DataLoader(testDTI, batch_size=args.batch, shuffle=False,
                                     collate_fn=testDTI.collate, drop_last=True)

            for epoch in range(args.epochs):

                train(model, train_loader, optimizer, args,
                      compound_graph, protein_graph, protein_embedding, DPI_label)
                mse_test, rmse_test, ci_test, rm2_test = test(model, test_loader, args,
                      compound_graph, protein_graph, protein_embedding, DPI_label)
                scheduler.step(mse_test)
                end = timeit.default_timer()
                time = end - start
                print("epoch     time     mse_test     rmse_test    ci_test    rm2_test")
                ret = [epoch + 1, round(time, 2), round(mse_test, 5), round(rmse_test, 5), round(ci_test, 5), round(rm2_test, 5)]
                print('\t\t'.join(map(str, ret)))
                if mse_test < best_mse:
                    if mse_test < 0.600:
                        torch.save(model.state_dict(), file_model + 'Epoch:' + str(epoch + 1) + '.pt')
                        print("model has been saved")
                    best_epoch = epoch + 1
                    best_mse = mse_test
                    print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)

    elif dataset_exists:
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
        file_model = 'model_save/' + args.dataset + '/fold/' + str(fold_index) + '/'
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.8, patience=80,
            verbose=True, min_lr=1e-5)
        criterion = nn.BCELoss()

        trainDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'proteinseq_train_id.npy',
                              compoundsmiles=args.foldpath + str(fold_index + 1) + 'compoundsmiles_train_id.npy',
                              label=args.foldpath + str(fold_index + 1) + 'label_train.npy')
        testDTI = DTIDataset(proteinseq=args.foldpath + str(fold_index + 1) + 'proteinseq_test_id.npy',
                             compoundsmiles=args.foldpath + str(fold_index + 1) + 'compoundsmiles_test_id.npy',
                             label=args.foldpath + str(fold_index + 1) + 'label_test.npy')

        train_loader = DataLoader(trainDTI, batch_size=args.batch, shuffle=True, collate_fn=trainDTI.collate,
                                  drop_last=True)
        test_loader = DataLoader(testDTI, batch_size=args.batch, shuffle=False, collate_fn=testDTI.collate,
                                 drop_last=True)
        # todo 加载去重后的protein seq和graph
        # todo 加载去重后的compound smiles
        # todo 加载去重后的protein 氨基酸特征
        # todo 加载去重后的compound 原子特征

        # todo 加载batch中的protein和compound id

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
            end = timeit.default_timer()
            time = end - start
            print("epoch     time     acc    prec     recall    AUC     AUPR")
            ret = [epoch + 1, round(time, 2), round(acc, 5), round(prec, 5), round(recall, 5),
                    round(auc, 5), round(prc, 5)]
            print('\t\t'.join(map(str, ret)))
            if auc > best_auc:
                best_epoch = epoch + 1
                best_auc = auc
                np.save("AUC/XR/XRGCN_cl_labels_epoches_{}".format(epoch), total_labels)
                np.save("AUC/XR/XRGCN_cl_scores_epoches_{}".format(epoch), total_scores)
                print('**********************AUC improved at epoch ', best_epoch, ';\tbest_AUC:', best_auc, '**********************')
        print('***best_AUC: ', best_auc, '***best_epoch: ',best_epoch)