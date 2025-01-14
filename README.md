# Core codes of VNGE and QESE
```python
# VNGE and approximated VNGE (Line 74 and 79 in entropy.py)
def vnge(adj: np.ndarray, eig: str = "np") -> float:
    ...
def vnge_deg(degrees: List[int], volumn) -> np.float32:
    ...
```

```python
# node QESE and edge QESE (Line 112 and 157 in entropy.py)
def cal_node_centrality(data_list: List[Data], eig: str, hop: int = 1) -> list:
    ...
def cal_edge_centrality(data_list, eig: str) -> list:
    ...
```

# QESE: Plug-and-play structural encoding for Message Passing GNNs
Multiplying quantum entropy structrual encoding to each message for each GNN layer.
## Examples
### GCN+QESE
```python
# GCN+QESE layer (Line 66 in models.py)
def message(...):
    ...
    out = norm_deg.view(-1, 1) * norm_quantum.view(-1, 1) * x_j
    ...
```
### GIN+QESE
```python
# GIN+QESE layer (Line 126 and 131 in models.py)
def forward(...):
    ...
    out = out * node_centrality.view(-1, 1)
    ...
def message(...):
    out = x_j * norm_quantum.view(-1, 1)
```
### GAT+QESE
```python
# GAT+QESE layer (Line 234 and 239 in models.py)
def edge_update(...):
    ...
    norm_quantum = torch.cat([edge_centrality, node_centrality])
    ...
def message(...):
    ...
    out = norm_quantum.view(-1, 1, 1) * out
    ...
```

### SAGE+QESE
```python
# SAGE+QESE layer (Line 310 and 316 in models.py)
def forward(...):
    ...
    x_r = node_centrality.view(-1, 1) * x_r
    ...
def message(...):
    ...
    out = edge_centrality.view(-1, 1) * x_j
    ...
```

### G3N+QESE
```python
# G3N+QESE layer (Line 133 and 149 in /G3N-master/layers.py)
def forward(...):
    ...
    h_temp *= edge_centrality
    ...
    h_sum *= node_centrality
    ...
```

### GSN+QESE
```python
# GSN+QESE layer (Line 289 and 351 in /GSN-master/graph_filters/GSN_sparse.py)
def forward(...):
    ...
    out = self.update_fn(torch.cat((x * node_centrality.view(-1, 1), self.propagate(edge_index=edge_index, x=x, identifiers=identifiers, edge_centrality=edge_centrality)), -1))
    ...
def message(...):
    ...
    msg_j *= edge_centrality.view(-1, 1)
    ...
```

### k-gnn+QESE
```python
# k-gnn+QESE layer (Line 112 and 126 in k-gnn-master/k_gnn/graph_conv.py)
def forward(...):
    ...
    out_col *= edge_centrality.view(-1, 1)
    ...
    out = out + node_centrality.view(-1, 1) * torch.mm(x, self.root_weight)
    ...
```
