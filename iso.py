import os
import torch
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.datasets import GNNBenchmarkDataset
import warnings
warnings.filterwarnings("ignore")

from dataset import get_dataset
from models import get_model

def main():
    parser = argparse.ArgumentParser(description='iso tests')
    parser.add_argument('--dataset', type=str, required=True, choices=['EXP', 'CSL', 'graph8c', 'SR25'])
    parser.add_argument('--model_name', type=str, required=True, choices=['GCN', 'QGCN', 'GIN', 'QGIN', 'GAT', 'QGAT', 'SAGE', 'QSAGE'])
    parser.add_argument('--eig', type=str, required=False, default='np', choices=['np', 'appro_deg_ge0', 'betweenness', 'current_flow_betweenness'])
    parser.add_argument('--nc_scale', type=float, required=False, default=1.)
    parser.add_argument('--ec_scale', type=float, required=False, default=10.)
    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers')
    parser.add_argument('--layer_size', type=int, default=40,
                        help='dimensionality of hidden units in GNNs')
    args = parser.parse_args()
    args = vars(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['device'])  
    device = torch.device(f"cuda:{args['device']}") if torch.cuda.is_available() else torch.device("cpu")
    
    dataset = get_dataset(args['dataset'], recal_rawdata=False, recal_entro=False, cal_entro=True, nc_scale=args['nc_scale'], ec_scale=args['ec_scale'], eig=args['eig'])
    
    train_loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    run(args, device, train_loader, tol=0.001)

def run(args, device, train_loader, tol):
    for data in train_loader:
        if data.x == None:
            args['num_features'] = 1
        else:
            args['num_features'] = data.x.shape[1]
        break
    args['num_classes'] = 10
    args['last_layer'] = 'lin'
    args['pool'] = 'add'
    args['dropout'] = 0.0

    M=0
    min_sm = 999999.
    with tqdm(range(1, 101)) as tq:
        for iter in tq:
            torch.manual_seed(iter)

            model = get_model(args)

            embeddings=[]
            model.eval()  # no learning
            for data in train_loader:
                if data.x == None: # ?? which dataset?
                    x = torch.zeros((data.num_nodes, 1)).to(device)
                else:
                    x = data.x.float().to(device)
                edge_index = data.edge_index.to(device)
                batch_idx = data.batch.to(device)
                node_centrality = data.node_centrality.to(device)
                edge_centrality = data.edge_centrality.to(device)
                pred = model(x, edge_index, batch_idx, node_centrality, edge_centrality)
                embeddings.append(pred)

            E = torch.cat(embeddings).cpu().detach().numpy()
            if args['dataset'] in ['EXP']:
                M += (np.abs(E[0::2] - E[1::2]).sum(1) > tol)
                sm = (M == 0).sum()
            else:
                M += (np.abs(np.expand_dims(E, 1) - np.expand_dims(E, 0))).sum(2) > tol
                sm = ((M == 0).sum() - M.shape[0]) / 2
            if sm < min_sm:
                min_sm = sm
            tq.set_description(f'iter:{iter}, similar: {sm}, min_sim: {min_sm}')
            if min_sm == 0:
                break
            
    result = f"{args['dataset']}, {args['model_name']}, {args['eig']}, Fail: {min_sm}"
    print(result)
    f = open('./result.txt', 'a')
    f.write(result + '\n')
    f.close()

if __name__ == "__main__":
    main()