import os
import math
import random
import pickle
from typing import *
import numpy as np
import torch
import networkx as nx
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.dataset import Dataset
from rawdata_process import get_data
from entropy import cal_node_centrality, cal_edge_centrality
from rawdata_process import get_task, Metric

class SepDataset(InMemoryDataset):
    '''
        Dataset for Graph level tasks, separately.
    '''
    def __init__(self, dataset_name: str, data_list: List[Data]):
        super().__init__()
        self.data_list: List[Data] = [data for data in data_list]
        self.data, _ = self.collate(self.data_list)
        self.dataset_name = dataset_name
        self.task_type = get_task(dataset_name)

    def get(self, idx: Union[int, slice]) -> Data:
        return self.data_list[idx]
    
    def __getitem__(self, idx: Union[int, List[bool]]) -> Dataset:
        if isinstance(idx, int) or isinstance(idx, slice):
            ret = self.get(idx)
        elif isinstance(idx, list) and all(isinstance(item, bool) for item in idx): # for bool masks
            assert len(idx) == self.len(), f"The length of mask is {len(idx)}, which is different to that of dataset {self.len()}."
            
            ret = []
            for i, item in enumerate(idx):
                if item:
                    ret.append(self.data_list[i])
        elif isinstance(idx, list) and all(isinstance(item, Tensor) for item in idx): # for long Tensor
            idx = idx
            ret = [self.get(id) for id in idx]
        elif isinstance(idx, Tensor): # for long Tensor
            ret = [self.get(id) for id in idx.numpy().tolist()]
        elif idx is None:
            ret = []
        else:
            if not isinstance(idx, list):
                raise ValueError(f"The index list of SepDataset must be slice or bool list.")
        
        return ret

    def len(self):
        return len(self.data_list)
    
    def shuffle(self) -> None:
        random.shuffle(self.data_list)
    
    def to(self, device) -> None:
        for d in self.data_list:
            d.to(device=device)

    def statistic_info(self, mode: str = 'simple') -> Dict:
        if mode == 'simple':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
            }
        elif mode == 'middle':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_graphs': len(self),
                'avg_num_nodes': self.avg_num_nodes,
                'avg_num_edges': self.avg_num_edges,
                'avg_node_deg': self.avg_degree,
            }
        elif mode == 'complex':
            ret = {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'num_graphs': len(self),
                'avg_num_nodes': self.avg_num_nodes,
                'avg_num_edges': self.avg_num_edges,
                'avg_node_deg': self.avg_degree,
                'avg_cluster': self.avg_cluster,
                'avg_diameter': self.avg_diameter,
                'max_num_nodes': self.max_num_nodes,
                'max_num_edges': self.max_num_edges,
            }
        else:
            raise ValueError('mode must be simple/middle/complex.')
        return ret
    
    @property
    def num_features(self) -> int:
        ret = self.data_list[0].x.shape[-1]
        return ret
    
    @property
    def num_classes(self) -> int:
        if self.task_type in [Metric.ACC]:
            label_l = [d.y.item() for d in self.data_list]
            num_class = np.array(label_l).max() + 1
        elif self.task_type in [Metric.AP, Metric.MAE, Metric.RMSE]:
            num_class = self.data_list[0].y.shape[-1]
        elif self.task_type in [Metric.AUROC]:
            num_class = 1
        elif self.task_type in [Metric.F1]:
            raise NotImplementedError()
        else:
            raise ValueError(f"The task type '{self.task_type}' is invalid.")
        return num_class
    
    @property
    def avg_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data_list]
        ret = np.array(num_nodes_l).mean()
        return ret
    
    @property
    def max_num_nodes(self) -> int:
        num_nodes_l = [d.x.shape[0] for d in self.data_list]
        ret = np.array(num_nodes_l).max()
        return ret
    
    @property
    def avg_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data_list]
        ret = np.array(num_edges_l).mean()
        return ret
    
    @property
    def max_num_edges(self) -> int:
        num_edges_l = [d.edge_index.shape[-1] / 2 for d in self.data_list]
        ret = np.array(num_edges_l).max()
        return ret
    
    @property
    def avg_degree(self) -> int:
        avg_deg_l = []
        for graph in self.data_list:
            graph = to_networkx(graph)
            avg_deg = sum(dict(graph.degree()).values()) / len(graph) / 2
            avg_deg_l.append(avg_deg)
        ret = np.array(avg_deg_l).mean()
        return ret
    
    @property
    def avg_cluster(self) -> int:
        avg_cluster_l = []
        for graph in self.data_list:
            graph = to_networkx(graph)
            clustering_coefficient = nx.average_clustering(graph)
            avg_cluster_l.append(clustering_coefficient)
        ret = np.array(avg_cluster_l).mean()
        return ret
    
    @property
    def avg_diameter(self) -> int:
        avg_diameter_l = []
        for graph in self.data_list:
            graph = to_networkx(graph)
            try:
                diameter = nx.diameter(graph)
            except:
                diameter = 0
            avg_diameter_l.append(diameter)
        ret = np.array(avg_diameter_l).mean()
        return ret

class KFoldIter():
    def __init__(self, num_data, folds=10) -> None:
        self.num_data = num_data
        self.folds = folds
    
    def __iter__(self):
        self.cursor = 0
        self.pos = [math.floor(fold * self.num_data / self.folds) for fold in range(self.folds + 1)]

        return self
    
    def __next__(self) -> tuple:
        if self.cursor == self.folds:
            raise StopIteration
    
        pos1 = self.pos[self.cursor]
        pos2 = self.pos[self.cursor + 1]
        train_mask = [True] * pos1 + [False] * (pos2-pos1) + [True] * (self.num_data - pos2)
        test_mask = [not m for m in train_mask]

        self.cursor += 1

        return train_mask, test_mask

def get_dataset(dataset_name: str, recal_rawdata: bool, recal_entro: bool, cal_entro: bool, nc_scale: int = 1, ec_scale: int = 1, eig: str = "appro_deg_ge0") -> SepDataset:
    dataset: Dataset = get_data(dataset_name, recal_rawdata)
    
    data_list: List = [] # assign centrality for graph obj doesn't work, thus create a new list
    if cal_entro:
        obj_file = f'./datasets/obj/{dataset_name}_nc_{eig}.pt'
        if not recal_entro and os.path.exists(obj_file):
            node_centralities = torch.load(obj_file)
        else:
            node_centralities = cal_node_centrality(dataset, eig=eig)
            torch.save(node_centralities, obj_file)

        obj_file = f'./datasets/obj/{dataset_name}_ec_{eig}.pt'
        if not recal_entro and os.path.exists(obj_file):
            edge_centralities = torch.load(obj_file)
        else:
            edge_centralities = cal_edge_centrality(dataset, eig=eig)
            torch.save(edge_centralities, obj_file)

        for graph, nc, ec in zip(dataset, node_centralities, edge_centralities):
            assert graph.num_nodes == len(nc)
            assert graph.num_edges == len(ec)
            if graph.num_nodes == 0 and len(nc) == 0: # manually filter graphs with zero nodes
                continue
            
            nc = torch.tensor(nc).float() * nc_scale
            ec = torch.tensor(ec).float() * ec_scale
            if nc_scale == -1.0:
                nc.fill_(1.0)
            if ec_scale == -1.0:
                ec.fill_(1.0)
            graph.node_centrality = nc
            graph.edge_centrality = ec
            data_list.append(graph)
    else:
        data_list = [graph for graph in dataset]

    dataset = SepDataset(dataset_name, data_list)
    
    return dataset

if __name__ == '__main__':
    ...