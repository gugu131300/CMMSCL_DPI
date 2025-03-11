import dgl
import torch
import pandas as pd
from dgl import load_graphs
import numpy as np
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')

class Alphabets():
    def __init__(self, chars, encoding=None, missing=255):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding

    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]

class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        super(Smiles, self).__init__(chars)

class DTIDataset():
    def __init__(self,  protein_graph=None, compound_graph=None, proteinseq=None, compoundsmiles=None, label=None, protein_embedding=None):

        self.smilebet = Smiles()
        self.compound = pd.read_csv(compoundsmiles)
        compound_value = self.compound['COMPOUND_SMILES']
        smiles_cpy = copy.deepcopy(compound_value)
        smiles = [x.encode('utf-8').upper() for x in smiles_cpy]
        smiles_long = [torch.from_numpy(self.smilebet.encode(x)).long() for x in smiles]
        self.smiles_values = smiles_long  # smiles

        self.protein_embedding = protein_embedding
        self.compound_graph, _ = load_graphs(compound_graph)
        self.compound_graph = list(self.compound_graph)

        self.protein_graph, _ = load_graphs(protein_graph)
        self.protein_graph = list(self.protein_graph)

        self.label = np.load(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        compound_smiles = self.smiles_values[idx]

        return self.protein_graph[idx], self.compound_graph[idx], self.protein_embedding[idx], compound_smiles, self.label[idx]

    def collate(self, sample):

        protein_graph, compound_graph, protein_embedding, compound_smiles, label = map(list, zip(*sample))

        # todo 
        protein_graph = dgl.batch(protein_graph)
        compound_graph = dgl.batch(compound_graph)
        labels = torch.FloatTensor(label)
        protein_embedding = torch.FloatTensor(np.array(protein_embedding))

        return protein_graph, compound_graph, protein_embedding, compound_smiles, labels

