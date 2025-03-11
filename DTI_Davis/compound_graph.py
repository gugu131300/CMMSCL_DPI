###把compound转化为graph###
# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 11:05:03
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : compound.py
@Project    : X-DPI
@Description: 小分子相关特征
'''
import numpy as np
import torch
from mol2vec.features import (MolSentence, mol2alt_sentence, sentences2vec)
from rdkit import Chem
import dgl
from scipy import sparse as sp
import os
import pandas as pd
from dgl import load_graphs

PT = Chem.GetPeriodicTable()
ELEMENT_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce',
    'Gd', 'Ga', 'Cs', 'unknown'
]

ATOM_CLASS_TABLE = {}
NOBLE_GAS_ATOMIC_NUM = {2, 10, 18, 36, 54, 86}
OTHER_NON_METAL_ATOMIC_NUM = {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}
METALLOID_ATOMIC_NUM = {5, 14, 32, 33, 51, 52, 85}
POST_TRANSITION_METAL_ATOMIC_NUM = {13, 31, 49, 50, 81, 82, 83, 84, 114}
TRANSITION_METAL_ATOMIC_NUM = set(range(21, 30 + 1)) | set(range(39, 48 + 1)) | set(range(72, 80 + 1)) | set(
    range(104, 108 + 1)) | {112}
ALKALI_METAL_ATOMIC_NUM = {3, 11, 19, 37, 55, 87}
ALKALI_EARCH_METAL_ATOMIC_NUM = {4, 12, 20, 38, 56, 88}
LANTHANOID_ATOMIC_NUM = set(range(57, 71 + 1))
ACTINOID_ATOMIC_NUM = set(range(89, 103 + 1))
ATOM_CLASSES = [
    NOBLE_GAS_ATOMIC_NUM, OTHER_NON_METAL_ATOMIC_NUM, METALLOID_ATOMIC_NUM, POST_TRANSITION_METAL_ATOMIC_NUM,
    TRANSITION_METAL_ATOMIC_NUM, ALKALI_EARCH_METAL_ATOMIC_NUM, ALKALI_METAL_ATOMIC_NUM, LANTHANOID_ATOMIC_NUM,
    ACTINOID_ATOMIC_NUM
]
for class_index, atom_class in enumerate(ATOM_CLASSES):
    for num in atom_class:
        ATOM_CLASS_TABLE[num] = class_index + 1

ALLEN_NEGATIVITY_TABLE = {
}

ELECTRON_AFFINITY_TABLE = (
    (1, 0.75),
    (1, 0.75),
    (2, -0.52),
    (3, 0.62),
    (4, -0.52),
    (5, 0.28),
    (6, 1.26),
    (6, 1.26),
    (7, 0.00),
    (7, 0.01),
    (7, 0.01),
    (8, 1.46),
    (8, 1.46),
    (8, 1.46),
    (8, -7.71),
    (9, 3.40),
    (10, -1.20),
    (11, 0.55),
    (12, -0.41),
    (13, 0.43),
    (14, 1.39),
    (15, 0.75),
    (15, -4.85),
    (15, -9.18),
    (16, 2.08),
    (16, 2.08),
    (16, -4.72),
    (17, 3.61),
    (18, -1.00),
    (19, 0.50),
    (20, 0.02),
    (21, 0.19),
    (22, 0.08),
    (23, 0.53),
    (24, 0.68),
    (25, -0.52),
    (26, 0.15),
    (27, 0.66),
    (28, 1.16),
    (29, 1.24),
    (30, -0.62),
    (31, 0.43),
    (32, 1.23),
    (33, 0.80),
    (34, 2.02),
    (35, 3.36),
    (36, -0.62),
    (37, 0.49),
    (38, 0.05),
    (39, 0.31),
    (40, 0.43),
    (41, 0.92),
    (42, 0.75),
    (43, 0.55),
    (44, 1.05),
    (45, 1.14),
    (46, 0.56),
    (47, 1.30),
    (48, -0.72),
    (49, 0.30),
    (50, 1.11),
    (51, 1.05),
    (52, 1.97),
    (53, 3.06),
    (54, -0.83),
    (55, 0.47),
    (56, 0.14),
    (57, 0.47),
    (58, 0.65),
    (59, 0.96),
    (60, 1.92),
    (61, 0.13),
    (62, 0.16),
    (63, 0.86),
    (64, 0.14),
    (65, 1.17),
    (66, 0.35),
    (67, 0.34),
    (68, 0.31),
    (69, 1.03),
    (70, -0.02),
    (71, 0.35),
    (72, 0.02),
    (73, 0.32),
    (74, 0.82),
    (75, 0.06),
    (76, 1.10),
    (77, 1.56),
    (78, 2.13),
    (79, 2.31),
    (80, -0.52),
    (81, 0.38),
    (82, 0.36),
    (83, 0.94),
    (84, 1.90),
    (85, 2.30),
    (86, -0.72),
    (87, 0.49),
    (88, 0.10),
    (89, 0.35),
    (90, 1.17),
    (91, 0.55),
    (92, 0.53),
    (93, 0.48),
    (94, -0.50),
    (95, 0.10),
    (96, 0.28),
    (97, -1.72),
    (98, -1.01),
    (99, -0.30),
    (100, 0.35),
    (101, 0.98),
    (102, -2.33),
    (103, -0.31),
    (111, 1.57),
    (113, 0.69),
    (115, 0.37),
    (116, 0.78),
    (117, 1.72),
    (118, 0.06),
    (119, 0.66),
    (120, 0.02),
    (121, 0.57),
)
ELECTRON_AFFINITY_TABLE = {k: v for (k, v) in ELECTRON_AFFINITY_TABLE}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]  # 6-dim
    results = (one_of_k_encoding_unk(atom.GetSymbol(), symbol) + one_of_k_encoding(atom.GetDegree(), degree) +
               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
               one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
               )  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
                                                      ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    #A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    A = g.adjacency_matrix().to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(np.array(L))
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g

def bond_features(bond, use_chirality=True):
    """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)

def get_mol_features(smiles, atom_dim, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")
    g = dgl.DGLGraph()
    ###nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    ###atom
    atom_feat = np.zeros((mol.GetNumAtoms(), atom_dim))
    map_dict = dict()
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
        map_dict[atom.GetIdx()] = atom.GetSmarts()
    g.ndata["atom"] = torch.tensor(atom_feat)

    ###edge
    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = bond_features(bond, use_chirality=use_chirality)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)

    g.edata["edge"] = torch.tensor(np.array(bond_feats_all))
    g = laplacian_positional_encoding(g, pos_enc_dim=8)
    return g

def Compound_graph_construction(id, compound_values, dir_output):
    N = len(compound_values)
    for no, data in enumerate(id):
        compounds_g = list()
        print('/'.join(map(str, [no + 1, N])))
        smiles_data = compound_values[no]
        compound_graph = get_mol_features(smiles_data, atom_dim=34, use_chirality=True)
        compounds_g.append(compound_graph)
        dgl.save_graphs(dir_output + '/compound_graph/' + str(data) + '.bin', list(compounds_g))

def Compound_graph_process(dataset, dir_output, id_train, id_test):
    compounds_graph_train = []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        compound_graph_train, _ = load_graphs('E:/OneDrive/桌面/new_paper/dataset/' + dataset + '/processed' + '/compound_graph/' + str(id) + '.bin')
        compounds_graph_train.append(compound_graph_train[0])
    print(len(compounds_graph_train))
    dgl.save_graphs(dir_output + '/compound_graph.bin', compounds_graph_train)

def Compound_id_process(dataset, dir_output, id_train, id_test):
    compounds_id_train = []
    N = len(id_train)
    for no, id in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        compounds_id_train.append(id)
    np.save(dir_output + '/compound_id.npy', compounds_id_train)

def Label_process(dataset, dir_output, label_train, label_test):
    labels_train, labels_test = [], []
    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(dir_output + '/label.npy', labels_train)

if __name__ == '__main__':
    dataset = 'D84'
    file_path_compound = '/home/zqguxingyue/DTI/' + dataset + '/' + dataset + '.csv'
    dir_output = ('/home/zqguxingyue/DTI/' + dataset +'/')
    os.makedirs(dir_output, exist_ok=True)

    raw_data_compound = pd.read_csv(file_path_compound)
    compound_values = raw_data_compound['COMPOUND_SMILES'].drop_duplicates().values
    compound_id_unique = raw_data_compound['COMPOUND_ID'].drop_duplicates().values

    N = len(compound_values)
    compound_max_len = 100

    Compound_graph_construction(id=compound_id_unique, compound_values=compound_values, dir_output=dir_output)

    #train_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_train.csv')
    #test_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_test.csv')

    compound_id_train = raw_data_compound['COMPOUND_ID'].values
    #compound_id_test = test_data['COMPOUND_ID'].values

    label_train = raw_data_compound['REG_LABEL'].values
    #label_test = test_data['REG_LABEL'].values

    Compound_graph_process(dataset=dataset, id_train=compound_id_train, id_test=compound_id_train, dir_output=dir_output)
    Compound_id_process(dataset=dataset, id_train=compound_id_train, id_test=compound_id_train, dir_output=dir_output)
    Label_process(dataset=dataset, dir_output=dir_output, label_train=label_train, label_test=label_train)

    print('The preprocess of ' + dataset + ' dataset has finished!')
