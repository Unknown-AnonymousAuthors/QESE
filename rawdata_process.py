import os
import csv
import pickle
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from enum import Enum
from tqdm import tqdm
from math import comb
from typing import *
from torch.utils.data import ConcatDataset
from torch import Tensor
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, QM9, ZINC, MoleculeNet, PCQM4Mv2, MalNetTiny, LRGBDataset
from torch_geometric.utils import to_undirected, to_networkx, from_networkx
import networkx as nx
from ogb.graphproppred import PygGraphPropPredDataset

SUBSETS: List[str] = ['train', 'test', 'val']

def get_data(dataset_name: str, recal: bool = False) -> Dataset:
    '''
        Get torch_geometric.data.Dataset obj from disk or recalculated from raw data
        Parameters:
            param1: Data obj
            param2: if recalculate the Data obj on disk
        Return:
            off-the-shelf Data obj or recalculated Data obj
    '''
    
    ### Note: Calling order of functions in the super class torch.Dataset. Really annoying! ###
    #   before saving to disk: 1.pre_filter() 2.pre_transform()
    #   from disk to memory:   1.transform()  2.filter()
    extra_args = {
        'pre_transform': get_transform_funcs(dataset_name),
        'force_reload': recal
    }
    mask: Dict[str: ...] = None # masks for train, test, val subsets (only for datasets with pre-splits)
    
    # AddRWPE = T.AddRandomWalkPE(walk_length=10, attr_name=None)
    # AddEgoRWPE = TransformEgoRW(walk_length=10, attr_name=None)
    # AddLEPE = T.AddLaplacianEigenvectorPE(k=3, attr_name=None, is_undirected=True)
    # T.Compose([AddRWPE, AddEgoRWPE]), AddLEPE
    # extra_args['transform'] = AddLEPE # AddRWPE AddLEPE
    # extra_args['transform'] = AddEgoRWPE
    # extra_args['transform'] = T.Compose([AddRWPE, AddEgoRWPE])
    
    if dataset_name in ['MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'PTC_MR'] \
                    + ['BZR', 'COX2'] \
                    + ['COLLAB', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        dataset = TUDataset('./datasets/raw/TUDataset/', dataset_name, **extra_args)
    elif dataset_name in ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']: # 'ogbg-ppa' is too large
        # The ogbg dataset API don't receive params 'force_reload', thus manually reloading processed files and then deleting it.
        if extra_args['force_reload']:
            root_dir = f'./datasets/raw/ogbg/{dataset_name.replace("-", "_")}/processed'
            if os.path.exists(root_dir):
                shutil.rmtree(root_dir)
        del extra_args['force_reload']
        
        dataset = PygGraphPropPredDataset(root='./datasets/raw/ogbg/', name=dataset_name, **extra_args)
        mask: Dict[str: Tensor] = dataset.get_idx_split()
        mask['val'] = mask['valid'] # be consistent for dict keys names: ['train', 'test', **'val'**]
        del mask['valid']
    elif dataset_name == "EXP":
        del extra_args['force_reload']
        dataset = EXPDataset(root="./datasets/raw/isomorphism/EXP/", **extra_args)
    elif dataset_name == "CSL":
        dataset = GNNBenchmarkDataset(root='./datasets/raw/isomorphism/', name='CSL')
        transform_funcs = extra_args['transform']
        transform_funcs = transform_funcs.transforms if transform_funcs is T.Compose else [transform_funcs]
        dataset = pre_iso_CSL(dataset, transform_funcs)
    elif dataset_name == "graph8c":
        del extra_args['force_reload']
        dataset = Grapg8cDataset(root="./datasets/raw/isomorphism/graph8c/", **extra_args)
    elif dataset_name == "SR25":
        del extra_args['force_reload']
        dataset = SRDataset(root="./datasets/raw/isomorphism/SR25/", **extra_args)
    elif dataset_name == "subgraphcount":
        dataset = GraphCountDataset(root='./datasets/raw/isomorphism/subgraphcount', **extra_args)
        mask = dataset.mask
    elif dataset_name in ['QM9']:
        dataset = QM9(f'./datasets/raw/QM9', **extra_args)
    elif dataset_name in ['ZINC_full', 'ZINC_subset']:
        subset: bool = True if dataset_name == 'ZINC_subset' else False
        dataset = [ZINC('./datasets/raw/ZINC/', subset=subset, split=split, **extra_args) for split in SUBSETS]
    elif dataset_name in ["ESOL", "FreeSolv", "Lipo", "MUV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]: # "PCBA" repeat
        dataset = MoleculeNet(root='./datasets/raw/MoleculeNet/', name=dataset_name, **extra_args)
    elif dataset_name == 'PCQM4Mv2':
        dataset = [PCQM4Mv2(root='./datasets/raw/PCQM4Mv2', split=split, **extra_args) for split in SUBSETS]
    elif dataset_name in ["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP"]:
        dataset = [GNNBenchmarkDataset(root='./datasets/raw/GNNBenchmark/', name=dataset_name, split=split, **extra_args) for split in SUBSETS]
    elif dataset_name == "MalNetTiny":
        dataset = [MalNetTiny(root='./datasets/raw/MalNetTiny/', split=split, **extra_args) for split in SUBSETS]
    elif dataset_name in ["Peptides-func", "Peptides-struct"]:
        dataset = [LRGBDataset(root='./datasets/raw/LRGB/', name=dataset_name, split=split, **extra_args) for split in SUBSETS]
    else:
        raise ValueError(f"The input name of dataset '{dataset_name}' is invalid.")
    
    # save masks for pre-split datasets
    if isinstance(dataset, list): # for pre split datasets
        assert mask is None
        assert len(dataset) == 3 # contains every subset in SUBSETS = ['train', 'test', 'val']
        mask: Dict[str: int] = {
            split: len(subset)
            for subset, split in zip(dataset, SUBSETS)
        }
        dataset = ConcatDataset(dataset)
    if mask is not None:
        torch.save(mask, f"./datasets/obj/{dataset_name}_split_mask.pt")
        
    # verify datasets
    verify_funcs: List[Callable] = get_verify_funcs(dataset_name)
    for data in tqdm(dataset, desc="Verifying dataset", ncols=80, leave=False, smoothing=1):
        for func in verify_funcs:
            # func(data)
            ...

    return dataset

class Metric(Enum):
    AUROC = 'AUROC'
    AP = 'AP'
    ACC = 'ACC'
    F1 = 'F1'
    RMSE = 'RMSE'
    MAE = 'MAE'
    ISO = 'ISO'

def get_task(dataset_name: str) -> Metric:
    '''
        Get the task type (actually, evaluation metric) given a dataset name.
        Return:
            One of ['AUROC', 'AP', 'ACC', 'F1', 'RMSE', 'MAE']
    '''
    task = None
    if dataset_name in ['MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'PTC_MR'] \
                    + ['BZR', 'COX2'] \
                    + ['COLLAB', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        task = Metric.ACC
    elif dataset_name in ['ogbg-molhiv']:
        task = Metric.AUROC
    elif dataset_name in ['ogbg-ppa']:
        task = None
    elif dataset_name in ['ogbg-molpcba']:
        task = Metric.AP
    elif dataset_name in ['ogbg-code2']:
        task = 'F1' # TODO ???
    elif dataset_name in ["EXP", "CSL", "graph8c", "SR25", "subgraphcount"]:
        task = Metric.ISO
    elif dataset_name in ['QM9']:
        task = Metric.MAE
    elif dataset_name in ['ZINC_full', 'ZINC_subset']:
        task = Metric.MAE
    elif dataset_name in ["ESOL", "FreeSolv", "Lipo"]:
        task = Metric.RMSE
    elif dataset_name in ["MUV"]:
        task = Metric.AP
    elif dataset_name in ["BACE", "BBBP"]:
        task = Metric.AUROC
    elif dataset_name in ["Tox21", "ToxCast", "SIDER", "ClinTox"]:
        task = Metric.AP # actually, multilabel AUROC
    elif dataset_name in ['PCQM4Mv2']:
        task = Metric.MAE
    elif dataset_name in ['MNIST']: # ["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"]:
        task = Metric.ACC
    elif dataset_name in ["MalNetTiny"]:
        task = None
    elif dataset_name in ["Peptides-func"]:
        task = Metric.AP
    elif dataset_name in ["Peptides-struct"]:
        task = Metric.MAE
    else:
        raise ValueError(f"The input name of dataset '{dataset_name}' is invalid.")
    
    # if task is None, it means that we are not ready to run experiments with that dataset (too large or sth else ...)
    
    return task

def get_transform_funcs(dataset_name: str) -> T.Compose:
    '''
        Compose transform functions given a dataset name.
    '''
    if dataset_name in ['EXP', 'SR25', 'graph8c']:
        return SpectralDesign()
    
    task = get_task(dataset_name)
    transform_compose: List[Callable] = [
        TransformEdgeIndex(), TransformFeatureOnehot(dataset_name), T.ToUndirected(),
        TransformFeature(torch.float) if dataset_name != 'ogbg-molhiv' else TransformFeature(torch.long),
    ]
    
    # if dataset_name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']: # TODO: for extremely sparse features, saving storage space
    #     transform_compose.append(transform_feature_sparse)
    
    if task in [Metric.ACC]:
        transform_compose.append(TransformLabel(torch.long, 1))
    elif task in [Metric.AUROC, Metric.AP, Metric.F1, Metric.RMSE, Metric.MAE]:
        transform_compose.append(TransformLabel(torch.float, 2))
    elif task in [Metric.ISO]:
        pass
    else:
        raise ValueError(f"The task type '{task}' is invalid.")
    transform_compose = T.Compose(transform_compose)
    return transform_compose

class TransformEdgeIndex:
    '''
        Ensure the begin value of edge_index is 0
    '''
    def __init__(self) -> None:
        pass
    
    def __call__(self, data: Data) -> Data:
        if data.edge_index.shape[1] == 0: # for graphs with only one node, which is invalid to call 'edge_index.min()'
            return data
        
        min_ei = data.edge_index.min().item()
        if min_ei != 0:
            data.edge_index -= min_ei
        return data    

class TransformFeature:
    '''
        Transform graph features into a desired data type
    '''
    def __init__(self, trans_type: torch.dtype) -> None:
        assert trans_type in [torch.long, torch.float]
        self.trans_type = trans_type
    
    def __call__(self, data: Data) -> Data:
        data.x = torch.tensor(data.x, dtype=self.trans_type)
        return data

class TransformFeatureSparse:
    '''
        Transform extremely sparse graph features into sparse tensors, e.g. REDDIT datasets.
        TODO: self-defined api or python lib api?
    '''
    def __init__(self) -> None:
        pass
    def __call__(self, data: Data) -> Data:
        return data

class TransformFeatureOnehot:
    '''
        Add one-hot features for the dataset if needed
    '''
    def __init__(self, dataset_name) -> None:
        len_one_hot: int = None
        if dataset_name == 'IMDB-BINARY':
            len_one_hot = 136
        elif dataset_name == 'IMDB-MULTI':
            len_one_hot = 89
        elif dataset_name == 'COLLAB':
            len_one_hot = 492
        elif dataset_name == 'REDDIT-BINARY':
            len_one_hot = 3782
        elif dataset_name == 'REDDIT-MULTI-5K':
            len_one_hot = 3648
        # elif dataset_name == 'ZINC':        # TODO
        #     len_one_hot = 28 # in G3N       ??????
        # elif dataset_name == 'ogbg-molhiv': ??????
        # elif dataset_name == '':
        else:
            len_one_hot = -1        # one-hot is not for every dataset
        self.len_one_hot = len_one_hot

    def __call__(self, data: Data) -> Data:
        if self.len_one_hot == -1:  # one-hot is not for every dataset
            return data
        
        num_nodes = data.num_nodes  # (data.edge_index.max() - data.edge_index.min() + 1).item()
        feature = torch.zeros(num_nodes, self.len_one_hot).float()
        for i in range(num_nodes):
            feature[i, i] = 1.0
        data.x = feature 
        return data

class TransformLabel:
    '''
        Ensure the dtype of label is long or float, when the task is classification or regression
    '''
    def __init__(self, trans_type: torch.dtype, dim: int) -> None:
        assert trans_type in [torch.long, torch.float]
        assert dim in [1, 2]
        self.trans_type = trans_type
        self.dim = dim
    
    def __call__(self, data: Data) -> Data:
        # 1.transform type
        label = data.y
        label = torch.tensor(label, dtype=self.trans_type)
        label_dim = label.dim()
        
        # 2.transform shape
        if self.dim == 1:
            if label_dim == 2:
                label = label.view(-1)
        elif self.dim == 2:
            if label.dim() == 1: # and label.shape[0] == 1:
                label = label.view(1, -1)
        data.y = label
        return data
    
class TransformEgoRW:
    '''
        Calculate the RWPE of node in its ego net, then set attribute or concat.
    '''
    def __init__(self, walk_length: int = 10, attr_name: Union[str, None] = 'ego_rw_pe') -> None:
        assert isinstance(attr_name, str) or attr_name is None
        self.walk_length = walk_length
        self.attr_name = attr_name
    
    def __call__(self, data: Data) -> Data:
        Trans = AddRandomWalkPE(walk_length=self.walk_length)
        G = to_networkx(data)
        node_ego_rwpe_l = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1, undirected=True)
            data_new = from_networkx(G_ego)
            data_new = Trans(data_new)
            node_idx = list(G_ego.nodes()).index(i)
            node_rwpe = data_new.random_walk_pe[node_idx] # .sum(0)
            node_rwpe = torch.cat((node_rwpe, data_new.random_walk_pe[1 if node_idx != 1 else 1]))
            node_ego_rwpe_l.append(node_rwpe)
        ego_rwpe = torch.stack(node_ego_rwpe_l)
        
        if self.attr_name: # is not None
            setattr(data, self.attr_name, ego_rwpe)
        else: # is None
            data.x = torch.cat((data.x, ego_rwpe), -1)
        
        return data

def get_verify_funcs(dataset_name: str) -> List[Callable]:
    '''
        Compose verify functions for dataset given a dataset name.
    '''
    task = get_task(dataset_name)
    verify_funcs: List[Callable] = [VerifyEdgeIndex()]
    
    if dataset_name == 'ogbg-molhiv':
        verify_funcs.append(VerifyFeature(torch.long))
    else:
        verify_funcs.append(VerifyFeature(torch.float))
    
    if dataset_name == 'ogbg-code2':
        pass # TODO: check how to deal with label y with List[str]
    elif task in [Metric.ACC]:
        verify_funcs.append(VerifyLabel(torch.long, 1))
    elif task in [Metric.AUROC, Metric.AP, Metric.F1, Metric.RMSE, Metric.MAE]:
        verify_funcs.append(VerifyLabel(torch.float, 2))
    elif task in [Metric.ISO]:
        pass
    else:
        raise ValueError(f"The task type '{task}' is invalid.")
    
    return verify_funcs

class VerifyFeature:
    '''
        Verify the shape and type of graph features
    '''
    def __init__(self, verify_type: torch.dtype) -> None:
        assert verify_type in [torch.long, torch.float]
        self.verify_type = verify_type
    
    def __call__(self, data: Data) -> None:
        feature = data.x
        assert isinstance(feature, torch.Tensor)
        assert feature.dim() == 2, f"Expect the feature dim to be 2, but get {feature.dim()}."
        assert feature.dtype == self.verify_type, f"Expect the feature type to be {self.verify_type}, but get {feature.dtype}."

class VerifyLabel:
    '''
        Verify the shape and type of graph label
    '''
    def __init__(self, verify_type: torch.dtype, dim: int) -> None:
        assert verify_type in [torch.long, torch.float]
        assert dim in [1, 2]
        self.verify_type = verify_type
        self.dim = dim
    
    def __call__(self, data: Data) -> None:
        labels = data.y
        assert isinstance(labels, torch.Tensor)
        assert labels.dim() == self.dim, f"Expect the dim of graph label to be {self.dim}, but get {labels.dim()}."
        assert labels.dtype == self.verify_type, f"Expect the type of label to be {self.verify_type}, but get {labels.dtype}."

class VerifyEdgeIndex:
    '''
        Verify shape and the begining index of graph edge index, and guarantee the symmetry without self-loops
    '''
    def __init__(self) -> None:
        pass
        
    def __call__(self, data: Data) -> None:
        edge_index = data.edge_index
        num_edges = data.num_edges
        num_nodes = data.num_nodes
        
        # 1.verify shape and the begining index
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.dim() == 2
        if data.edge_index.shape[1] != 0: # some graphs have only one node, which cause error at edge_index.min()
            assert edge_index.min() == 0  # node id in edge index begin with 0
        
        # 2.verify symmetry and no self-loops
        count1, count2 = 0, 0
        edge_dict = {i:[] for i in range(num_nodes)} # use adj list for fast varifying
        edge_index = edge_index.T.numpy().tolist()
        for edge in edge_index:
            if edge[1] not in edge_dict[edge[0]]:
                edge_dict[edge[0]].append(edge[1])
        edge_index_reverse = [[edge[1], edge[0]] for edge in edge_index]
        for edge_reverse in edge_index_reverse:
            if edge_reverse[1] in edge_dict[edge_reverse[0]]: # check asymmetric
                count1 += 1
            if edge_reverse[0] == edge_reverse[1]: # check self loop
                count2 += 1
        assert count1 == num_edges, f"Non-symmetric edge_index" # all edges have their reverse
        assert count2 == 0, f"The graph has self-loops"         # no self-loops

def dense2sparse(dense_tensor: torch.Tensor) -> torch.Tensor:
    '''
        Convert dense (strided) tensor to sparse tensor. (Manually without torch API)
    '''
    indices = torch.nonzero(dense_tensor)
    rows = indices[:, 0]
    cols = indices[:, 1]
    values = dense_tensor[rows, cols]
    sparse_tensor = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, dense_tensor.size())
    
    return sparse_tensor

def sparse2dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
    '''
        Convert sparse tensor to dense (strided) tensor. (Manually without torch API)
    '''
    dense_tensor = torch.zeros(sparse_tensor.size()).to(sparse_tensor.device)
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    dense_tensor[indices[0], indices[1]] = values
    
    return dense_tensor

### Dataset classes and functions for isomorphism datasets ###
class SpectralDesign(object):
    def __init__(self):
        ...

    def __call__(self, data):
        n = data.x.shape[0]
        data.x = data.x.type(torch.float32)
        A = np.zeros((n, n), dtype=np.float32)
        A[data.edge_index[0], data.edge_index[1]] = 1
        data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        return data

class EXPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["EXP.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/EXP.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def pre_iso_CSL(dataset: GNNBenchmarkDataset, transforms: Iterable[T.BaseTransform]) -> ConcatDataset:
    added = [False for _ in range(15)]
    refine_dataset = []
    for data in dataset:
        data.x = torch.zeros((data.num_nodes, 1))
            
        y = data.y[0]
        if not added[y]:
            added[y] = True
            for trans in iter(transforms):
                data = trans(data)
            refine_dataset.append(data)
    
    dataset = ConcatDataset([refine_dataset]) # being not List[], for not storing masks
    return dataset

class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.  
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for _, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for _, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform, force_reload=force_reload)
        self.get_mask()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def get_mask(self):
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # mask
        mask = {
            'train': torch.Tensor(a['train_idx'][0]).long(),
            'test': torch.Tensor(a['test_idx'][0]).long(),
            'val': torch.Tensor(a['val_idx'][0]).long(),
        }
        self.mask = mask
        return mask

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    ...