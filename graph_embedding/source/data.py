from torch.utils.data import Dataset
import torch
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import tqdm

class Data(Dataset):
    def __init__(self, data_path, max_atom):
        self.max_atom = max_atom
        with open(data_path,'r') as f:
            self.smiles = f.read().split('\n')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x, adj = self.gen_graph(idx)
        x, label = self.random_mask(x)
        output = {'x':x,'adj':adj,'label':label}
        return {key:torch.tensor(value) for key,value in output.items()}

    def gen_graph(self,idx):
        smi = self.smiles[idx]
        mol = MolFromSmiles(smi)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        feat = []
        for atom in mol.GetAtoms():
            feat.append(atom.GetAtomicNum())

        #cut and padding
        adj = adj[:self.max_atom,:self.max_atom]
        adj_pad = np.zeros((self.max_atom,self.max_atom))
        adj_pad[:len(adj),:len(adj)] = adj + np.eye(len(adj))

        feat = feat[:self.max_atom]
        padding = [0 for _ in range(self.max_atom-len(feat))]
        feat.extend(padding)

        return feat,adj_pad


    def random_mask(self, atoms):
        label = []
        for i,token in enumerate(atoms):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    atoms[i] = 119 # mask token. atoms number is 118

                # 10% randomly change token to random token
                elif prob < 0.9:
                    atoms[i] = random.randrange(120) #118+mask+pad
                label.append(token)

            else:
                label.append(0)

        return atoms, label


class Data2(Dataset):
    def __init__(self, data_path, max_atom):
        self.max_atom = max_atom

        with open(data_path,'r') as f:
            self.smiles =[line[:-1].split('\t') for line in tqdm.tqdm(f,desc='Loading dataset')]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x1,x2,next_label = self.random_pair(idx)
        x1, adj1 = self.gen_graph(x1)
        x2, adj2 = self.gen_graph(x2)
        x1, label1 = self.random_mask(x1)
        x2, label2 = self.random_mask(x2)

        output = {'x1':x1,'adj1':adj1,
                  'x2':x2,'adj2':adj2,
                  'label1':label1,
                  'label2':label2,
                  'next_label':next_label}
        return {key:torch.tensor(value) for key,value in output.items()}

    def gen_graph(self,smi):

        mol = MolFromSmiles(smi)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        feat = []
        for atom in mol.GetAtoms():
            feat.append(atom.GetAtomicNum())

        #cut and padding
        adj = adj[:self.max_atom,:self.max_atom]
        adj_pad = np.zeros((self.max_atom,self.max_atom))
        adj_pad[:len(adj),:len(adj)] = adj + np.eye(len(adj))

        feat = feat[:self.max_atom]
        padding = [0 for _ in range(self.max_atom-len(feat))]
        feat.extend(padding)

        return feat,adj_pad

    def random_pair(self, idx):
        x1, x2 = self.smiles[idx][0],self.smiles[idx][1]
        if random.random() > 0.5:
            return x1, x2, 1
        else:
            x2 = self.smiles[random.randrange(len(self.smiles))][1]
            return x1, x2, 0

    def random_mask(self, atoms):
        label = []
        for i,token in enumerate(atoms):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    atoms[i] = 119 # mask token. atoms number is 118

                # 10% randomly change token to random token
                elif prob < 0.9:
                    atoms[i] = random.randrange(120) #118+mask+pad
                label.append(token)

            else:
                label.append(0)

        return atoms, label

