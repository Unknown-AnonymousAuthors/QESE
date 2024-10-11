import math
from tqdm import tqdm
import numpy as np
import scipy
import torch.nn as nn
import networkx as nx
from typing import *
from collections import Counter
from qiskit.quantum_info import entropy as v_entropy, DensityMatrix
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def _edgeindex2adj(edge_index: Tensor, num_nodes: int) -> np.ndarray:
    # nx.Graph(edges) will change the node order, thus manually implement this
    adj = np.zeros([num_nodes, num_nodes], dtype=np.int32)
    for i in range(edge_index.shape[-1]):
        node1 = edge_index[0][i].item()
        node2 = edge_index[1][i].item()
        adj[node1][node2] = 1

    return adj

def _adj2laplacian(adj: np.ndarray) -> np.ndarray:
    degree_matrix = np.diag(np.sum(adj, axis=1))
    laplacian = degree_matrix - adj
    return laplacian

def _get_egonet_idx(adj: List[list], idx: int, hop: int, ret: str) -> List[list]:
    G = nx.from_numpy_array(np.array(adj))
    ego_graph = nx.ego_graph(G, idx, radius=hop)
    
    if ret == 'node':
        egonet_idx = list(ego_graph.nodes())
    elif ret == 'edge':
        egonet_idx = list(ego_graph.edges())
    else:
        raise ValueError('param ret must be "node" or "edge"')

    return egonet_idx

def _rm_node(adj: np.ndarray, idx: int) -> np.ndarray:
    # if not isinstance(idx, int) and not isinstance(idx, list):
    #     raise ValueError("Param 'idx' must be int or list")

    # delete target rows and cols
    adj = np.delete(adj, idx, 0)
    adj = np.delete(adj, idx, 1)

    # remove zero-value rows and cols
    adj = adj[~np.all(adj == 0, axis=0)]
    adj = adj.T[~np.all(adj == 0, axis=0)].T
    
    return adj

def _rm_edge(adj: np.ndarray, idx: Tuple[int, int]) -> np.ndarray:
    # if not isinstance(idx, tuple):
    #     raise ValueError("Param 'idx' must be tuple")
    # if len(idx) != 2:
    #     raise ValueError("Tuple 'idx' must have 2 items.")
    
    adj_new = adj.copy()
    # delete target edges
    node0, node1 = idx[0], idx[1]
    adj_new[node0][node1] = 0
    adj_new[node1][node0] = 0

    # remove zero-value rows and cols
    adj_new = adj_new[~np.all(adj_new == 0, axis=0)]
    adj_new = adj_new.T[~np.all(adj_new == 0, axis=0)].T
    
    return adj_new

def vnge(adj: np.ndarray, eig: str = "np") -> float:
    if adj.shape[0] == 0:
        return 0
    
    volumn = adj.sum()
    laplacian = _adj2laplacian(adj)
    if eig == "np":
        eigenvalues, _ = np.linalg.eig(laplacian)
    elif eig == "scipy": # Calculate a part of eigenvalues only.
        eigenvalues, _ = scipy.sparse.linalg.eigs(laplacian.astype('f'), k=5) # the param 'k' means taking only top-k eigenvalues
    elif eig == 'appro_deg' or eig == 'appro_deg_ge0':
        # Approximate from degree distribution, from the paper https://arxiv.org/pdf/2102.09766.pdf (On the Similarity Between von Neumann Graph Entropy and Structural Information: Interpretation, Computation, and Applications)
        eigenvalues = adj.sum(axis=-1) # degrees as eigenvalues
    else:
        raise ValueError(f'args "eig" must be "np", "scipy", "appro_deg" or ...')
    
    entropy = 0
    for ev in eigenvalues:
        if ev <= 0:
            ev = 1e-17 # the computation inaccuracy may lead positive eigenvalues to be negative
        zz = ev / volumn # norm
        entropy -= zz * np.log2(zz)
    
    return entropy.real

def vnge_deg(degrees: List[int], volumn) -> np.float32:
    '''
        Specifical function for the approximated version VNGE
    '''
    if len(degrees) == 0:
        return 0.
    
    entropy = 0.
    entropy_items = [(deg / volumn) * np.log2(deg / volumn) for deg in degrees if deg != 0]
    entropy -= np.array(entropy_items).sum()
    
    return entropy

def cal_node_centrality(data_list: List[Data], eig: str, hop: int = 1) -> list:
    '''
        paper 'Measuring Vertex Centrality using the Holevo Quantity'
    '''
    # specific implemented functions for the approximated version QSE
    if eig in ['appro_deg', 'appro_deg_ge0']:
        node_centrality = cal_node_centrality_appro_deg(data_list=data_list, eig=eig, hop=hop)
        return node_centrality
    # Calculate traditional centrality for comparing with quantum mechanics
    if eig in ['betweenness', 'betweenness_norm', 'current_flow_betweenness', 'current_flow_betweenness_norm']:
        node_centrality = cal_traditional_node_centrality(data_list=data_list, method=eig)
        return node_centrality
    
    node_centrality = []
    for graph in tqdm(data_list, desc="Calculating node centrality", ncols=90, leave=False, smoothing=1):
        adj_origin = _edgeindex2adj(graph.edge_index, graph.x.shape[0])
        entropy_origin = vnge(adj_origin, eig=eig)
        
        # node centrality
        entro_list = []
        num_node = adj_origin.shape[0]
        for idx in range(num_node):
            ego_idx = _get_egonet_idx(adj_origin, idx, hop=hop, ret='node')
            ego_compl_idx = [idx for idx in list(range(num_node)) if idx not in ego_idx]
            ego_size = len(ego_idx)
            ego_compl_size = num_node - ego_size

            adj_ego = _rm_node(adj_origin, ego_compl_idx)
            entropy_ego = vnge(adj_ego, eig=eig)

            adj_compl = _rm_node(adj_origin, ego_idx)
            entropy_compl = vnge(adj_compl, eig=eig)
            
            entropy_gap = entropy_origin - (ego_compl_size / num_node * entropy_compl + ego_size / num_node * entropy_ego)
            print(entropy_gap, entropy_origin, entropy_ego, entropy_compl)
            if math.isclose(entropy_gap, 0.0, abs_tol = 1e-5):
                entropy_gap = np.log2(num_node) * 0.75 # any better value? TODO
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)
            
        node_centrality.append(entro_list)
    
    return node_centrality

def cal_edge_centrality(data_list, eig: str) -> list:
    '''
        paper 'Edge Centrality via the Holevo Quantity'
    '''
    # specific implemented functions for the approximated version QSE
    if eig in ['appro_deg', 'appro_deg_ge0']:
        edge_centrality = cal_edge_centrality_appro_deg(data_list=data_list, eig=eig)
        return edge_centrality
    # Calculate traditional centrality for comparing with quantum mechanics
    if eig in ['betweenness', 'betweenness_norm', 'current_flow_betweenness', 'current_flow_betweenness_norm']:
        edge_centrality = cal_traditional_edge_centrality(data_list=data_list, method=eig)
        return edge_centrality
    
    edge_centrality = []
    for graph in tqdm(data_list, desc="Calculating edge centrality", ncols=90, leave=False, smoothing=1):
        adj_origin = _edgeindex2adj(graph.edge_index, graph.x.shape[0])
        entropy_origin = vnge(adj_origin, eig=eig)

        # edge centrality
        entro_list = []
        num_edge = graph.edge_index.shape[-1]
        for edge in graph.edge_index.T:
            adj_complement = _rm_edge(adj_origin, (edge[0].item(), edge[1].item()))
            entropy_compl = vnge(adj_complement, eig=eig)
            # entropy_gap = entropy_compl - ((num_edge - 1) / num_edge) * entropy_origin # TODO: fault ???
            entropy_gap = entropy_origin - ((num_edge - 1) / num_edge) * entropy_compl # TODO: fault ???
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)
        edge_centrality.append(entro_list)
    
    return edge_centrality

def cal_node_centrality_appro_deg(data_list: List[Data], eig: str, hop: int = 1) -> list:
    '''
        Approximate by degree distribution, from the paper https://arxiv.org/pdf/2102.09766.pdf
        (On the Similarity Between von Neumann Graph Entropy and Structural Information: Interpretation, Computation, and Applications)
    '''
    node_centrality = []
    for graph in tqdm(data_list, desc="Calculating node centrality", ncols=90, leave=False, smoothing=1):
        entro_list = []
        
        G = nx.Graph()
        num_nodes = graph.num_nodes
        G.add_nodes_from(range(num_nodes)) # for isolated nodes
        edges = [edge for edge in graph.edge_index.T.numpy()]
        G.add_edges_from(edges)
        adjacency_list = {i: [] for i in range(num_nodes)}
        for edge in graph.edge_index.T:
            adjacency_list[edge[0].item()].append(edge[1].item())
        degrees = [len(deg) for deg in adjacency_list.values()]
        volume = sum(degrees)
        entropy_origin = vnge_deg(degrees, volume)
        for idx in range(num_nodes):
            ego_graph = nx.ego_graph(G, idx, radius=hop)
            del_edges = ego_graph.edges()
            if len(del_edges) == 0: # if the ego-graph only contains an isolated node, otherwise it will be an empty list
                del_edges = [idx]
            else:
                del_edges = [node for edge in del_edges for node in list(edge)]
            del_edges_counter = Counter(del_edges)
            volume_ego = len(del_edges)
            degrees_ego = list(del_edges_counter.values())
            
            degrees_compl = degrees.copy()
            flatten_nodes = [i for node_idx in ego_graph.nodes() for i in adjacency_list[node_idx]] # flatten edges into nodes, for deleted degrees
            flatten_nodes = Counter(flatten_nodes)
            for n in ego_graph.nodes(): # force deleleted nodes by its degrees (nodes in ego-graph) (all edges around it are deleted)
                flatten_nodes[n] = len(adjacency_list[n])
            for k, v in flatten_nodes.items(): # delete degree
                degrees_compl[k] -= v
            degrees_compl = [deg for deg in degrees_compl if deg != 0] # delete nodes with zero degrees
            volume_compl = sum(degrees_compl)
            
            entropy_ego = vnge_deg(degrees_ego, volume_ego)
            entropy_compl = vnge_deg(degrees_compl, volume_compl)
            ego_size = len(degrees_ego)
            ego_compl_size = num_nodes - ego_size
            entropy_gap = entropy_origin - (ego_compl_size / num_nodes * entropy_compl + ego_size / num_nodes * entropy_ego)
            if math.isclose(entropy_gap, 0.0, abs_tol = 1e-5):
                entropy_gap = np.log2(num_nodes) * 0.75 # any better value?
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)

        node_centrality.append(entro_list)
    
    return node_centrality

def cal_edge_centrality_appro_deg(data_list, eig: str) -> list:
    '''
        Approximate by degree distribution, from the paper https://arxiv.org/pdf/2102.09766.pdf
        (On the Similarity Between von Neumann Graph Entropy and Structural Information: Interpretation, Computation, and Applications)
    '''
    edge_centrality = []
    for graph in tqdm(data_list, desc="Calculating edge centrality", ncols=90, leave=False, smoothing=1):
        entro_list = []
        
        G = nx.Graph()
        num_nodes = graph.num_nodes
        G.add_nodes_from(range(num_nodes)) # for isolated nodes
        edges = [edge for edge in graph.edge_index.T.numpy()]
        G.add_edges_from(edges)

        degrees = list([deg for _, deg in G.degree()])
        volume = sum(degrees)
        entropy_origin = vnge_deg(degrees, volume)

        # edge centrality
        entro_list = []
        num_edge = graph.num_edges
        for edge in edges:
            node1, node2 = edge[0], edge[1]
            degrees[node1] -= 1 # modify directly, rather than degrees.copy(), and then will be restored
            degrees[node2] -= 1
            entropy_compl = vnge_deg(degrees, volume - 2)
            
            # entropy_gap = entropy_compl - ((num_edge - 1) / num_edge) * entropy_origin # TODO: fault ???
            entropy_gap = entropy_origin - ((num_edge - 1) / num_edge) * entropy_compl # TODO: fault ???
            if eig == 'appro_deg_ge0':
                entropy_gap = np.abs(entropy_gap)
            entro_list.append(entropy_gap)
            
            # restore degree sequence
            degrees[node1] += 1
            degrees[node2] += 1
        edge_centrality.append(entro_list)
    
    return edge_centrality

def cal_traditional_node_centrality(data_list: List[Data], method: str):
    if method == 'betweenness' or method == 'betweenness_norm':
        api_centrality = nx.betweenness_centrality
    elif method == 'current_flow_betweenness' or method == 'current_flow_betweenness_norm':
        api_centrality = nx.current_flow_betweenness_centrality
    else:
        raise ValueError('Traditional node centrality method error, which should be "betweenness" or "current_flow_betweenness"')
    
    norm = True if method == 'betweenness_norm' or method == 'current_flow_betweenness_norm' else False
        
    node_centrality = []
    for graph in tqdm(data_list, desc="Calculating traditional node centrality", ncols=110, leave=False, smoothing=1):
        g = to_networkx(graph, to_undirected=True)
        if nx.is_connected(g):
            nc = api_centrality(g, normalized=norm) # !
            nc = list(nc.values())
        else:
            nc = [1.0 / graph.num_nodes] * graph.num_nodes
        node_centrality.append(nc)
        
    return node_centrality

def cal_traditional_edge_centrality(data_list: List[Data], method: str):
    if method == 'betweenness' or method == 'betweenness_norm':
        api_centrality = nx.edge_betweenness_centrality
    elif method == 'current_flow_betweenness' or method == 'current_flow_betweenness_norm':
        api_centrality = nx.edge_current_flow_betweenness_centrality
    else:
        raise ValueError('Traditional edge centrality method error, which should be "betweenness" or "current_flow_betweenness"')
    
    norm = True if method == 'betweenness_norm' or method == 'current_flow_betweenness_norm' else False
        
    edge_centrality = []
    for graph in tqdm(data_list, desc="Calculating traditional edge centrality", ncols=110, leave=False, smoothing=1):
        g = to_networkx(graph, to_undirected=True)
        ec = [1.0 / graph.num_edges] * graph.num_edges # init, for unconnected graphs
        if nx.is_connected(g):
            ec_dict = api_centrality(g, normalized=norm) # dict, single direction edge
            for i_edge, edge in enumerate(graph.edge_index.T): # look up in dict
                idx = tuple(edge.numpy().tolist())
                try:
                    c = ec_dict[idx]
                except:
                    c = ec_dict[(idx[1], idx[0])]
                ec[i_edge] = c
        edge_centrality.append(ec)
        
    return edge_centrality

if __name__ == '__main__':
    ...