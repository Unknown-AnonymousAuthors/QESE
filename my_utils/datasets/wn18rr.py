import torch
from torch_geometric.datasets import WordNet18RR
from torch.utils.data import Dataset
import random
from collections import Counter

class Wn18rrDataset(Dataset):
    def __init__(self, mode='train', debug=False, path='/Users/wangyingbo/Programming/CondaWorkplace/datasets/WN18RR'):
        wn = WordNet18RR(path)
        wn = wn.data
        self.summary = {
            'num_edges': wn.num_edges,
            'num_nodes': wn.num_nodes,
            'edge_type': dict(Counter(wn.edge_type.numpy().tolist()))
        }
        self.triples = torch.cat((wn.edge_index[0, :].reshape(1, -1), wn.edge_type.reshape(1, -1), wn.edge_index[1, :].reshape(1, -1))).T


        if mode == 'test':
            self.data = self.triples[wn.test_mask]
        elif mode == 'valid':
            self.data = self.triples[wn.val_mask]
        elif mode == 'train':
            num_nodes = wn.num_nodes
            train_len = torch.sum(wn.train_mask == True)
            correct_data = self.triples[wn.train_mask]
            correct_data_list = correct_data.tolist()

            corrupted_data_list = []
            count = 0

            while count < train_len:
                e1, r, e2 = self.triples[count].tolist()
                idx = random.randint(0, num_nodes - 1)
                n = random.randint(0, 1)
                if n == 0:
                    corrupt = [idx, r, e2]  # corrupt head
                else:
                    corrupt = [e1, r, idx]  # corrupt tail
                if corrupt not in correct_data_list and corrupt not in corrupted_data_list:
                    corrupted_data_list.append(corrupt)
                    count += 1
                    if debug and count % 100 == 0:
                        print(f'Corrupt progress: {count}/{train_len}.')

            self.data = torch.cat((correct_data, torch.tensor(corrupted_data_list)), 1)
        else:
            return None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)