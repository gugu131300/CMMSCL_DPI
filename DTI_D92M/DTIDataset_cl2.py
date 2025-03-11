import torch
import pandas as pd
import numpy as np
import copy


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
    def __init__(self, proteinseq=None, compoundsmiles=None, label=None):

        # self.smilebet = Smiles()
        # self.compound = pd.read_csv(compoundsmiles)
        # compound_value = self.compound['COMPOUND_SMILES']
        # smiles_cpy = copy.deepcopy(compound_value)
        # smiles = [x.encode('utf-8').upper() for x in smiles_cpy]
        # smiles_long = [torch.from_numpy(self.smilebet.encode(x)).long() for x in smiles]
        # self.smiles_values = smiles_long  # smiles

        # self.protein = pd.read_csv(proteinseq)
        # self.protein_values = self.protein['PROTEIN_SEQUENCE'].values
        #self.protein_values = np.load(proteinseq)

        #self.compound_smiles = self.raw_data['COMPOUND_SMILES'].values
        self.compound_id = np.load(compoundsmiles)
        self.protein_id = np.load(proteinseq)
        self.label = np.load(label)
        #self.protein_seq = self.raw_data['PROTEIN_SEQUENCE'].values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        #protein_embedding = self.protein_embedding[idx]
        compound_smiles_id = self.compound_id[idx]
        protein_seq_id = self.protein_id[idx]
        #protein_seqs = self.protein_values[idx]

        return compound_smiles_id, protein_seq_id, self.label[idx]

    def collate(self, sample):

        compound_smiles_id, protein_seq_id, label = map(list, zip(*sample))
        labels = torch.FloatTensor(label)

        return compound_smiles_id, protein_seq_id, labels
