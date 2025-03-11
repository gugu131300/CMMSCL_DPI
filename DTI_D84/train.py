import csv
import timeit
import argparse
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import *
from DTIDataset import DTIDataset
from model.DTInet import DTInet
from dgl import load_graphs
from protein_embedding import protein_emb
import dgl
import pandas as pd
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def train(model, device, train_loader, optimizer, batch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        protein_graph, compound_graph, protein_embedding, compound_smiles = data[:-1]
        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        output, loss = model(device, compound_graph, protein_graph, protein_embedding, compound_smiles, batch)
        loss = criterion(output, label.view(-1, 1).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, batch):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            label = data[-1]
            protein_graph, compound_graph, protein_embedding, compound_smiles = data[:-1]
            compound_graph = compound_graph.to(device)
            protein_graph = protein_graph.to(device)
            output, loss = model(device, compound_graph, protein_graph, protein_embedding, compound_smiles, batch)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()

    MSE = mse(total_labels, total_preds)
    RMSE = rmse(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)
    return MSE, RMSE, CI, RM2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=100, help='Number of batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')  # 1e-4
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate (1 - keep probability).')  # 0.1
    parser.add_argument('--seed', type=int, default=2024, help='Random Seed')
    parser.add_argument('--atom_dim', type=int, default=34, help='Dimension for Atom')
    parser.add_argument('--fold_num', type=int, default=5, help='fold_num')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu')
    parser.add_argument('--n_attentions', type=int, default=1, help='n_attentions')
    parser.add_argument('--gt_heads', type=int, default=8, help='gt_heads')
    parser.add_argument('--hidden-dim', type=int, default=128, help='hidden units of FC layers (default: 256)')
    parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--gt_layers', type=int, default=5, help='gt_layers')
    parser.add_argument('--compound_dim', type=int, default=34, help='compound_dim')
    parser.add_argument('--protein_dim', type=int, default=128, help='protein_dim')
    parser.add_argument('--out_dim', type=int, default=1, help='out_dim')
    parser.add_argument('--smile_vocab', type=int, default=63, help='smile_vocab')
    parser.add_argument('--rnn_dim', type=int, default=128, help='hidden unit/s of RNNs (default: 256)')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--num_features_xt', type=int, default=26, help='num_features_xt')
    parser.add_argument('--embed_dim', type=int, default=128, help='embed_dim')
    parser.add_argument('--n_filters', type=int, default=64, help='n_filters')
    parser.add_argument('--output_dim', type=int, default=128, help='output_dim')
    parser.add_argument('--objective', type=str, default='classification', help='Objective (classification / regression)')

    parser.add_argument('--dataset_path', default='/home/zqguxingyue/DTI/Davis/Davis.csv', help='dataset path')
    parser.add_argument('--dataset', default='Davis', help='dataset')
    parser.add_argument('--foldpath', default='/home/zqguxingyue/DTI/Davis/fold/', help='foldpath')

    args = parser.parse_args()
    # 打印参数及其值
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
    best_ci = 0
    best_mse = 100
    best_r2 = 0

    """Start training."""
    # 交叉验证
    fold_index = 0
    for train_index, test_index in kf.split(data_shuffled):
        fold_index += 1
        print("fold_index", fold_index)
        model = DTInet(args)
        train_len = len(train_index)
        test_len = len(test_index)
        print('Training on ' + args.dataset + ', fold:' + str(fold_index))
        train_data = data_shuffled.iloc[train_index]
        test_data = data_shuffled.iloc[test_index]

        label_train, label_test = [], []
        proteinseq_train, proteinseq_test = [], []
        compoundsmiles_train, compoundsmiles_test = [], []
        proteins_graph_train, proteins_graph_test = [], []
        compounds_graph_train, compounds_graph_test = [], []

        #############################################train、test的nfold protein graph
        for no, id in enumerate(train_index):
            protein_graph_train, _ = load_graphs('/home/zqguxingyue/DTI/' + args.dataset + '/protein_graph/' + data_shuffled.iloc[id]['PROTEIN_ID'] + '.bin')
            proteins_graph_train.append(protein_graph_train[0])
        dgl.save_graphs(args.foldpath+str(fold_index)+'trainprotein_graph.bin', proteins_graph_train)
        for no, id in enumerate(test_index):
            protein_graph_test, _ = load_graphs('/home/zqguxingyue/DTI/' + args.dataset + '/protein_graph/' + data_shuffled.iloc[id]['PROTEIN_ID'] + '.bin')
            proteins_graph_test.append(protein_graph_test[0])
        dgl.save_graphs(args.foldpath+str(fold_index)+'testprotein_graph.bin', proteins_graph_test)

        ###########################################train、test的nfold compound graph
        for no, id in enumerate(train_index):
            compound_graph_train, _ = load_graphs('/home/zqguxingyue/DTI/' + args.dataset + '/compound_graph/' + str(data_shuffled.iloc[id]['COMPOUND_ID']) + '.bin')
            compounds_graph_train.append(compound_graph_train[0])
        dgl.save_graphs(args.foldpath+str(fold_index)+'traincompound_graph.bin', compounds_graph_train)
        for no, id in enumerate(test_index):
            compound_graph_test, _ = load_graphs('/home/zqguxingyue/DTI/' + args.dataset + '/compound_graph/' + str(data_shuffled.iloc[id]['COMPOUND_ID']) + '.bin')
            compounds_graph_test.append(compound_graph_test[0])
        dgl.save_graphs(args.foldpath+str(fold_index)+'testcompound_graph.bin', compounds_graph_test)

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

        proteinemb_train = protein_emb(proteinseq_train)
        proteinemb_test = protein_emb(proteinseq_test)

        model = DTInet(args)
        model.to(device)
        file_model = '/home/zqguxingyue/DTI/' + args.dataset + '/' + str(fold_index) + '/'
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.8, patience=80,
            verbose=True, min_lr=1e-5)
        criterion = nn.MSELoss()

        trainDTI = DTIDataset(protein_graph = args.foldpath + str(fold_index) + 'trainprotein_graph.bin',
                                compound_graph = args.foldpath + str(fold_index) + 'traincompound_graph.bin',
                                proteinseq = args.foldpath + str(fold_index) + 'proteinseq_train.npy',
                                compoundsmiles = args.foldpath + str(fold_index) + 'compoundsmiles_train.csv',
                                label = args.foldpath + str(fold_index) + 'label_train.npy',
                                protein_embedding = proteinemb_train)
        testDTI = DTIDataset(protein_graph = args.foldpath + str(fold_index) + 'testprotein_graph.bin',
                              compound_graph = args.foldpath + str(fold_index) + 'testcompound_graph.bin',
                              proteinseq = args.foldpath + str(fold_index) + 'proteinseq_test.npy',
                              compoundsmiles = args.foldpath + str(fold_index) + 'compoundsmiles_test.csv',
                              label = args.foldpath + str(fold_index) + 'label_test.npy',
                              protein_embedding = proteinemb_test)
        
        train_loader = DataLoader(trainDTI, batch_size=args.batch, shuffle=True, collate_fn=trainDTI.collate, drop_last=True)
        test_loader = DataLoader(testDTI, batch_size=args.batch, shuffle=False, collate_fn=testDTI.collate, drop_last=True)

        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, args.batch)
            mse_test, rmse_test, ci_test, rm2_test = test(model, device, test_loader, args.batch)
            scheduler.step(mse_test)
            end = timeit.default_timer()
            time = end - start
            ret = [epoch + 1, round(time, 2), round(mse_test, 5), round(rmse_test, 5), round(ci_test, 5), round(rm2_test, 5)]
            print('\t\t'.join(map(str, ret)))
            if mse_test < best_mse:
                if mse_test < 0.600:
                    #torch.save(model.state_dict(), file_model + 'Epoch:' + str(epoch + 1) + '.pt')
                    print("model has been saved")
                best_epoch = epoch + 1
                best_mse = mse_test
                print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)
