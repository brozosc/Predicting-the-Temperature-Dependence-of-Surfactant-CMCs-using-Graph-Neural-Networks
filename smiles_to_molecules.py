# -*- coding: utf-8 -*-
"""
@author: BrozosCh
"""

import torch
import pandas as pd
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import Draw

class MyOwnDataset(InMemoryDataset):
    
    types = {'C' :0 , 'N' : 1, 'O' : 2, 'S' : 3, 'F' :4, 'Cl' : 5, 'Br' :6, 'Na' : 7, 'I': 8, 'B' :9, 'K' :10, 'H' :11, 'Li' :12} # atom types
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.AROMATIC: 2} # bond types
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'raw.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        df = pd.read_csv(self.raw_paths[0], sep = ';')
        data_list = []

        for _, row in df.iterrows():
            smiles, log_CMC, T_norm = row[0], row[1], row['T_norm']  # T_norm is the normalized temperature between [0,1]. On the GNN model, the normalized temperature is re-scaled between [0,10].
            print(_, smiles)
            mol = Chem.MolFromSmiles(smiles)
            
            
            N = mol.GetNumAtoms()
            type_idx = []
            ring = []
            aromatic = []
            sp2 = []
            sp3 = []
            unspecified = []
            cw = []
            ccw = []
            neutral, positive, negative = [], [], []
            num_hs = []
            num_neighbors = []
            
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                ring.append(1 if atom.IsInRing() else 0)
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)     
                unspecified.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED else 0)
                cw.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0)
                ccw.append(1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)
                negative.append(1 if atom.GetFormalCharge() == -1 else 0)   
                neutral.append(1 if atom.GetFormalCharge() == 0 else 0)
                positive.append(1 if atom.GetFormalCharge() == 1 else 0)
                num_neighbors.append(len(atom.GetNeighbors()))
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
                         

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([ring, aromatic, sp2, sp3, unspecified, cw, ccw, negative, neutral, positive], dtype=torch.float).t().contiguous()
            x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=5)
            x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
            x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float),x4.to(torch.float)], dim=-1)
                
            
            row, col, bond_idx, conj, ring, stereo = [], [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                conj.append(bond.GetIsConjugated())
                conj.append(bond.GetIsConjugated())
                ring.append(bond.IsInRing())
                ring.append(bond.IsInRing())
                stereo.append(bond.GetStereo())
                stereo.append(bond.GetStereo())

            edge_index = torch.tensor([row, col], dtype=torch.long)
            e1 = F.one_hot(torch.tensor(bond_idx),num_classes=len(self.bonds)).to(torch.float)
            e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
            e3 = F.one_hot(torch.tensor(stereo),num_classes=3).to(torch.float)
            edge_attr = torch.cat([e1, e2, e3], dim=-1)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            
            target  = []
            target.append([log_CMC])


            # Create PyTorch Geometric Data object
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr, T = T_norm, smiles_id = smiles,
                        y=torch.tensor(target, dtype=torch.float))
            

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

