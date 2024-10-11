# Experiments for rebuttal
## Runtime of QESE and QESE*
|             |      | MUTAG | PROTEINS | PTC MR  | ENZYMES  | NCI109  | NCI1    | IMDB-M  | IMDB-B | BZR   | COX2   | molhiv  | EXP    | CSL    | graph8c | SR25  | ZINC    | BACE   | BBBP   | Tox21  | ToxCast | SIDER    | ClinTox | Peptides-func | Peptides-struct | COLLAB   | DD      | REDDIT-B |
|-------------|------|-------|----------|---------|----------|---------|---------|---------|--------|-------|--------|---------|--------|--------|---------|-------|---------|--------|--------|--------|---------|----------|---------|---------------|-----------------|----------|---------|----------|
| QESE        | node | 2.29s | 631.85s  | 1.91s   | 18.83s   | 159.48s | 159.21s | 20.12s  | 22.37s | 8.56s | 16.80s | 778.75s | 36.85s | 48.39s | 36.27s  | 0.66s | 135.81s | 34.75s | 25.77s | 88.18s | 95.63s  | 570.11s  | 37.33s  | -             | -               | -        | -       | -        |
|             | edge | 1.82s | 2231.83s | 1.55s   | 40.28s   | 171.58s | 196.08s | 35.99s  | 52.27s | 9.49s | 20.79s | 919.97s | 57.27s | 63.73s | 22.23s  | 0.92s | 117.66s | 37.83s | 28.19s | 80.07s | 95.64s  | 1070.88s | 43.87s  | -             | -               | -        | -       | -        |
| QESE*       | node | 0.72s | 14.75s   | 0.53s   | 5.78s    | 25.09s  | 26.16s  | 12.03s  | 13.35s | 1.88s | 2.94s  | 144.38s | 9.50s  | 0.56s  | 12.95s  | 0.31s | 44.48s  | 6.17s  | 9.80s  | 20.23s | 23.57s  | 11.85s   | 4.78s   | 656.73s       | 668.36s         | 2137.60s | 190.66s | 723.95s  |
|             | edge | 0.42s | 20.96s   | 0.26s   | 4.32s    | 14.48s  | 17.51s  | 5.05s   | 9.32s  | 1.87s | 2.58s  | 101.13s | 9.02s  | 0.74s  | 6.78s   | 0.20s | 17.64s  | 5.78s  | 5.41s  | 8.38s  | 9.58s   | 12.16s   | 2.78s   | 793.43s       | 756.60s         | 3594.15s | 585.55s | 1411.63s |
| random walk | node | 0.31s | 12.68s   | 0.32s   | 2.72s    | 13.46s  | 12.58s  | 2.41s   | 2.14s  | 1.98s | 2.12s  | 97.80s  | 0.53s  | 0.59s  | 5.07s   | 0.07s | 24.17s  | 6.32s  | 8.82s  | 13.38s | 10.57s  | 13.22s   | 4.28s   | 1496.63s      | 996.98s         | 656.98s  | 369.16s | 264.80s  |
|             | edge | 0.38s | 15.93s   | 0.41s   | 3.40s    | 15.33s  | 16.89s  | 2.93s   | 3.55s  | 2.36s | 2.94s  | 116.73s | 0.43s  | 0.71s  | 6.62s   | 0.11s | 30.51s  | 8.34s  | 9.27s  | 18.75s | 12.28s  | 16.26s   | 3.93s   | 1638.47s      | 1191.35s        | 830.93s  | 432.59s | 304.96s  |

## Larger-scale graphs
|            | COLLAB | DD    |
|------------|--------|-------|
| GCN        | 69.72±2.6  | 79.54±3.0 |
| GCN+QESE*  | 70.38±2.3  | 79.76±3.6 |
| GIN        | 75.28±1.4  | 78.35±3.4 |
| GIN+QESE*  | 74.52±1.5  | 80.90±5.2 |
| GAT        | 55.38±2.8  | 68.08±9.3 |
| GAT+QESE*  | 65.74±1.3  | 77.16±4.4 |
| SAGE       | 57.10±2.5  | 78.01±5.8 |
| SAGE+QESE* | 69.37±2.5  | 78.34±3.2 |
## Ablation study of node QESE and edge QESE
|      | node QESE | edge QESE | MUTAG       | PROTEINS   | PTC_MR     | ENZYMES    | NCI109     | NCI1       | IMDB-MULTI | IMDB-BINARY |
|------|-----------|-----------|-------------|------------|------------|------------|------------|------------|------------|-------------|
| SAGE |           |           | 85.61±9.75  | 74.66±3.16 | 56.36±4.22 | 28.33±4.65 | 63.34±2.24 | 63.75±2.15 | 37.73±2.29 | 53.90±3.45  |
|      |           | ✓         | 86.12±9.64  | 74.21±4.81 | 58.66±5.30 | 27.83±4.02 | 64.26±2.16 | 64.87±2.36 | 42.60±4.98 | 58.90±4.09  |
|      | ✓         |           | 85.61±9.47  | 76.19±2.74 | 61.87±4.76 | 28.17±3.69 | 64.07±1.97 | 64.62±1.86 | 45.33±4.00 | 65.30±5.90  |
|      | ✓         | ✓         | 89.30±5.45  | 76.90±3.37 | 64.78±5.83 | 30.83±4.61 | 64.91±3.10 | 64.99±1.93 | 45.27±4.78 | 71.40±3.20  |
| GCN  |           |           | 85.61±9.17  | 75.28±3.00 | 60.10±7.83 | 24.50±6.01 | 63.24±2.32 | 63.99±2.26 | 35.40±3.02 | 52.30±4.63  |
|      |           | ✓         | 86.70±13.17 | 76.19±2.77 | 61.24±7.55 | 26.50±3.53 | 63.93±4.62 | 63.26±4.72 | 40.27±3.60 | 69.30±7.25  |
|      | ✓         |           | 84.04±9.99  | 75.73±4.26 | 63.39±7.20 | 33.00±4.40 | 64.21±5.21 | 63.95±6.80 | 43.67±3.52 | 60.60±4.90  |
|      | ✓         | ✓         | 89.30±6.38  | 77.63±3.10 | 65.07±5.01 | 34.33±4.73 | 65.50±1.92 | 64.87±1.94 | 42.20±2.81 | 68.90±2.39  |
| GAT  |           |           | 85.61±9.17  | 74.57±7.27 | 57.54±3.18 | 26.67±5.77 | 63.29±2.31 | 62.85±3.48 | 36.00±3.00 | 51.80±3.52  |
|      |           | ✓         | 89.30±4.31  | 75.29±3.35 | 58.12±3.69 | 24.67±2.67 | 62.01±5.83 | 63.45±2.62 | 39.67±2.11 | 61.60±5.26  |
|      | ✓         |           | 90.38±5.78  | 75.20±3.84 | 58.41±3.37 | 24.67±2.87 | 59.29±6.38 | 61.36±5.21 | 40.93±2.08 | 60.30±3.20  |
|      | ✓         | ✓         | 88.27±7.42  | 76.46±3.57 | 64.82±5.45 | 26.00±4.16 | 64.14±2.10 | 64.62±2.09 | 41.47±3.62 | 65.90±4.61  |
| GIN  |           |           | 92.02±5.90  | 78.80±4.53 | 69.16±5.53 | 39.50±4.66 | 77.47±2.18 | 77.91±2.53 | 49.20±2.89 | 74.20±2.86  |
|      |           | ✓         | 94.09±4.50  | 78.61±3.19 | 71.79±5.77 | 41.33±5.49 | 75.96±2.87 | 78.73±2.33 | 48.93±3.60 | 72.10±3.88  |
|      | ✓         |           | 91.99±4.92  | 76.81±3.25 | 67.73±4.36 | 43.17±5.87 | 77.63±1.68 | 79.49±1.90 | 48.27±4.02 | 73.90±4.99  |
|      | ✓         | ✓         | 95.20±5.04  | 79.60±3.93 | 70.03±4.70 | 48.83±6.67 | 75.33±2.21 | 74.70±2.56 | 49.33±3.20 | 72.50±2.84  |

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
