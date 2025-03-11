import copy
import torch
import torch.nn as nn
from model import gt_net_compound, gt_net_protein
import torch.nn.functional as F
from protein_embedding import amino_acid
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class Model_Contrast(nn.Module):
    # 需要正样本pos、负样本neg，头尾所有节点的表示drug_embs、protein_embs
    def __init__(self, hidden_dim, args, tau=0.2):
        super(Model_Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.args = args
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, v1_embs, v2_embs):
        """
        未指定pos，使用对角矩阵替代
        """
        pos = torch.eye(v1_embs.shape[0], v1_embs.shape[0]).to(self.args.device)
        neg = torch.ones(v1_embs.shape[0], v1_embs.shape[0]).to(self.args.device)
        neg = neg - pos

        v1_embs_proj = self.proj(v1_embs)
        v2_embs_proj = self.proj(v2_embs)
        matrix_1to2 = self.sim(v1_embs_proj, v2_embs_proj)
        matrix_2to1 = self.sim(v2_embs_proj, v1_embs_proj)

        """view inter sim"""
        matrix_1to1 = self.sim(v1_embs, v1_embs)
        matrix_2to2 = self.sim(v2_embs, v2_embs)

        """
        sim_pos_1to2: the similarity score of pos pair for v1 to v2
        sim_neg_1to2: the similarity score of neg pair for v1 to v2
        sim_pos_2to1: the similarity score of pos pair for v2 to v1
        sim_neg_2to1: the similarity score of neg pair for v2 to v1
        """

        sim_pos_1to2 = matrix_1to2.mul(pos)
        sim_neg_1to2 = matrix_1to2.mul(neg)
        sim_pos_2to1 = matrix_2to1.mul(pos)
        sim_neg_2to1 = matrix_2to1.mul(neg)

        sim_pos_1to1 = matrix_1to1.mul(pos)
        sim_neg_1to1 = matrix_1to1.mul(neg)
        sim_pos_2to2 = matrix_2to2.mul(pos)
        sim_neg_2to2 = matrix_2to2.mul(neg)

        sum_1to2 = (sim_pos_1to2).sum() / (sim_pos_1to2 + sim_neg_1to2).sum()
        sum_2to1 = (sim_pos_2to1).sum() / (sim_pos_2to1 + sim_neg_2to1).sum()
        sum_1to1 = (sim_pos_1to1).sum() / (sim_pos_1to1 + sim_neg_1to1).sum()
        sum_2to2 = (sim_pos_2to2).sum() / (sim_pos_2to2 + sim_neg_2to2).sum()

        loss_1to2 = -torch.log(sum_1to2)
        loss_2to1 = -torch.log(sum_2to1)
        loss_1to1 = -torch.log(sum_1to1)
        loss_2to2 = -torch.log(sum_2to2)

        return loss_1to2 +loss_2to1+loss_1to1 +loss_2to2

class DTInet_cl2(nn.Module):
    def __init__(self, args):
        super(DTInet_cl2, self).__init__()

        # protein sequence branch (1d conv)
        self.conv_1D = nn.Conv1d(in_channels=400, out_channels=args.n_filters, kernel_size=7).to(args.device)
        self.fc_seq = nn.Linear(64, args.output_dim)
        self.amino_tensor = amino_acid()
        self.amino_tensor = self.amino_tensor.to(args.device)
        # protein graph
        self.protein_gt = gt_net_protein.GraphTransformer(args.device, n_layers=args.gt_layers,
         node_dim=41, edge_dim=5, hidden_dim=args.protein_dim, out_dim=args.protein_dim,
         n_heads=args.gt_heads, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)

        # compound sequence branch
        #self.smiles_vocab = args.smile_vocab
        #self.smiles_embed = nn.Embedding(args.smile_vocab + 1, 256, padding_idx=args.smile_vocab)
        self.rnn_layers = 2
        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(34, args.rnn_dim)
        self.smiles_lstm = nn.LSTM(args.rnn_dim, args.rnn_dim, self.rnn_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=0.2)
        self.smiles_out_fc = nn.Linear(args.rnn_dim * 2, args.rnn_dim)
        # compound graph
        self.compound_gt = gt_net_compound.GraphTransformer(args.device, n_layers=args.gt_layers,
                        node_dim=args.compound_dim, edge_dim=10, hidden_dim=args.rnn_dim,
                        out_dim=args.rnn_dim, n_heads=args.gt_heads, in_feat_dropout=0.0,
                        dropout=0.2, pos_enc_dim=8)

        self.relu = nn.ReLU()
        self.contrastive_protein = Model_Contrast(args.protein_dim, args)  # 蛋白质的输出维度
        self.contrastive_compound = Model_Contrast(args.protein_dim, args)  # 药物的输出维度

        # Heterogeneous Graph
        self.conv1 = GCNConv(78, 78)
        self.conv2 = GCNConv(78, 78 * 2)
        self.conv3 = GCNConv(78 * 2, 78 * 4)
        # self.RWR_linear1 = nn.Linear(365, args.hidden_dim)
        self.RWR_linear1 = nn.Linear(84, args.hidden_dim)
        # self.RWR_linear2 = nn.Linear(68, args.hidden_dim)
        self.RWR_linear2 = nn.Linear(105, args.hidden_dim)

        ### mlp
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, args.out_dim)
        )
        ### 可视化图MLP
        self.out_fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.out_fc2 = nn.Linear(64, 64)
        self.out_fc3 = nn.Linear(64, 64)
        self.out_fc4 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

        self.DTI_conv1 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.DTI_conv2 = GCNConv(args.hidden_dim, args.hidden_dim)
        self.CL_GCN_protein = Model_Contrast(args.hidden_dim, args)
        self.CL_GCN_compound = Model_Contrast(args.hidden_dim, args)
        self.device = args.device

    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())# 选择batch中最大的氨基酸数量
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in
                           zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, args, compound_graph, protein_graph, protein_embedding,compound_smiles_id,
                batch, protein_seq_id, compoundsmiles, DPI_label,DTI_edge, RWR, transposed_RWR):

        #################### protein ####################
        ## 1、cnn
        cnn_embedding = protein_embedding.clone().to(torch.int)
        protein_num = len(protein_embedding)
        embedded_xt = self.amino_tensor[cnn_embedding.to(torch.long)]
        conv_1d = self.conv_1D(embedded_xt)
        # flatten
        xt = conv_1d.view(protein_num, -1)
        xt_fc = self.fc_seq(xt)
        xt_relu = F.relu(xt_fc)

        ## 2、protein graph
        protein_feat = self.protein_gt(protein_graph)
        protein_feat_x = self.dgl_split(protein_graph, protein_feat)
        # 平均池化操作，将张量池化为 (100, 128) 的形状
        # pooled_protein = torch.max(protein_feat_x, dim=1).values
        pooled_protein = torch.mean(protein_feat_x, dim=1)
 
        #################### compound ####################
        ## 1、compound rnn
        all_atoms_emb = copy.deepcopy(compound_graph.ndata['atom'])
        compound_atom_feat = self.dgl_split(compound_graph, all_atoms_emb)
        smiles = compound_atom_feat.type(torch.float32)
        smiles = self.smiles_input_fc(smiles)
        smiles_out, _ = self.smiles_lstm(smiles)
        # pooled_smiles = torch.max(smiles_out, dim=1).values
        pooled_smiles = torch.mean(smiles_out, dim=1)
        smiles_fc = self.smiles_out_fc(pooled_smiles)

        ## 2、compound graph
        compound_feat = self.compound_gt(compound_graph)
        compound_feat_x = self.dgl_split(compound_graph, compound_feat)
        # pooled_compound = torch.max(compound_feat_x, dim=1).values
        pooled_compound = torch.mean(compound_feat_x, dim=1)

        #################### DPI Random Walk---GCN ####################
        RWR = self.RWR_linear2(RWR)
        transposed_RWR = self.RWR_linear1(transposed_RWR)
        joint_RWR = torch.cat([transposed_RWR, RWR], dim=0)
        # DTI GCN
        DP_gcn_RWR = self.DTI_conv1(joint_RWR, DTI_edge)
        P_gcn = DP_gcn_RWR[:len(xt_relu), :]
        D_gcn = DP_gcn_RWR[len(xt_relu):, :]
        # protein joint
        protein_joint = xt_relu + pooled_protein
        # compound joint
        compound_joint = smiles_fc + pooled_compound
        # DPI joint
        protein_joint_DPI = protein_joint + P_gcn
        compound_joint_DPI = compound_joint + D_gcn
        #############XR
        # protein_joint_DPI = protein_joint
        # compound_joint_DPI = compound_joint

        #################### MLP ####################
        ### protein-compound interaction
        batch_protein_joint = protein_joint_DPI[protein_seq_id]
        batch_compound_joint = compound_joint_DPI[compound_smiles_id]
        cp_joint = torch.cat([batch_protein_joint, batch_compound_joint], dim=1)
        # x0 = self.dropout(self.relu(self.out_fc1(cp_joint)))
        # x1 = self.out_fc2(x0)
        # x2 = self.out_fc3(x1)
        # x3 = self.out_fc4(x2)
        x0 = self.mlp(cp_joint)
        x = self.sigmoid(x0)

        #################### 对比损失 ####################
        # GCN后的RWR和structure对比
        loss_GCN_protein = self.CL_GCN_protein(P_gcn, protein_joint)
        loss_GCN_compound = self.CL_GCN_compound(D_gcn, compound_joint)
        cl_protein = self.contrastive_protein(xt, pooled_protein)
        cl_compound = self.contrastive_compound(smiles_fc, pooled_compound)
        loss_cl = cl_protein + cl_compound + loss_GCN_protein + loss_GCN_compound
        return x0, x, loss_cl