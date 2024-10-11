import os
import time
import torch
import argparse
import scipy.io as sio
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

from dataset import get_dataset
from models import get_model

def train(epoch, model, train_loader, optimizer, device, ntask, trid):
    model.train()
    
    L=0
    for data in train_loader:
        optimizer.zero_grad()
        
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)
        y = data.y.to(device)
        node_centrality = data.node_centrality.to(device)
        edge_centrality = data.edge_centrality.to(device)
        pred = model(x, edge_index, batch_idx, node_centrality, edge_centrality)
        
        loss = torch.square(pred - y[:, ntask:ntask+1]).sum() 
        
        loss.backward()
        optimizer.step()  
        
        L += loss.item()

    return L / len(trid)

def test(model, val_loader, test_loader,device, ntask, vlid, tsid):
    model.eval()
    yhat=[]
    ygrd=[]
    L=0
    for data in test_loader:
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)
        y = data.y.to(device)
        node_centrality = data.node_centrality.to(device)
        edge_centrality = data.edge_centrality.to(device)
        pred = model(x, edge_index, batch_idx, node_centrality, edge_centrality)

        yhat.append(pred.cpu().detach())
        ygrd.append(y[:, ntask:ntask+1].cpu().detach())
        loss = torch.square(pred - y[:, ntask:ntask+1]).sum()         
        L += loss.item()
        
    yhat = torch.cat(yhat)
    ygrd = torch.cat(ygrd)
    testr2 = r2_score(ygrd.numpy(), yhat.numpy())

    Lv = 0
    for data in val_loader:
        x = data.x.float().to(device)
        edge_index = data.edge_index.to(device)
        batch_idx = data.batch.to(device)
        y = data.y.to(device)
        node_centrality = data.node_centrality.to(device)
        edge_centrality = data.edge_centrality.to(device)
        pred = model(x, edge_index, batch_idx, node_centrality, edge_centrality)

        loss = torch.square(pred - y[:, ntask:ntask+1]).sum()
        Lv += loss.item()
        
    return L / len(tsid), Lv / len(vlid), testr2

def main():
    parser = argparse.ArgumentParser(description='counting')
    parser.add_argument('--quantum', type=str, required=False, default='False', choices=['True', 'False'])
    parser.add_argument('--model_name', type=str, required=False, default='GCN', choices=['GCN', 'QGCN', 'GIN', 'QGIN', 'GAT', 'QGAT', 'SAGE', 'QSAGE'])
    parser.add_argument('--eig', type=str, required=False, default='np', choices=['np', 'appro_deg_ge0', 'betweenness', 'current_flow_betweenness'])
    parser.add_argument('--nc_scale', type=float, required=False, default=1.)
    parser.add_argument('--ec_scale', type=float, required=False, default=10.)
    parser.add_argument('--ntask', type=int, default=0, choices=[0, 1, 2, 3],
                        help='0: triangle, 1: tailed_triangle; 2: star; 3: 4-cycle')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=40,
                        help='dimensionality of hidden units in GNNs')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # get dataset
    dataset = get_dataset(
        dataset_name='subgraphcount', recal_rawdata=False, recal_entro=False, cal_entro=True,
        nc_scale=args.nc_scale, ec_scale=args.ec_scale, eig=args.eig # np appro_deg_ge0
    )
    mask = torch.load('./datasets/obj/subgraphcount_split_mask.pt') # get subset mask
    
    # data normalization
    data_y_norm = dataset.data.y.std(0)
    data_x_norm = dataset.data.x[:,0].max()
    for data in dataset:
        data.x /= data_x_norm
        data.y /= data_y_norm
    
    train_loader = DataLoader(dataset[mask['train']], batch_size=100, shuffle=True)
    valid_loader = DataLoader(dataset[mask['val']], batch_size=100, shuffle=False)
    test_loader = DataLoader(dataset[mask['test']], batch_size=100, shuffle=False)
        
    # select task, 0: triangle, 1: tailed_triangle 2: star  3: 4-cycle  4:custom
    ntask = args.ntask
    
    # init model
    params = {
        'model_name': args.model_name,
        'eig': args.eig,
        'quantum': True if args.quantum == 'True' else False,
        'num_features': dataset.num_features,
        'layer_size': args.emb_dim, 
        'num_classes': 1,
        'num_layers': args.num_layer,
    }
    args = params
    args['last_layer'] = 'lin'
    args['pool'] = 'add'
    args['dropout'] = 0.0
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    bval = 1000
    btest = 0
    btestr2 = 0
    tq = tqdm(range(500))
    for epoch in tq:
        trloss = train(epoch, model, train_loader, optimizer, device, ntask, mask['train'])
        test_loss, val_loss, testr2 = test(model, valid_loader, test_loader,device, ntask, mask['val'], mask['test'])
        if bval > val_loss:
            bval = val_loss
            btest = test_loss
            btestr2 = testr2
        
        tq.set_description('Epoch: {:02d}, trloss: {:.6f},  Valloss: {:.6f}, Testloss: {:.6f}, best test loss: {:.6f}, bestr2:{:.6f}'.format(epoch+1,trloss,val_loss,test_loss,btest,btestr2))
        
        if bval<1e-4:
            break
    
    result = f"{args['model_name']}, {args['quantum']}, {args['eig']}, Best test loss: {btest:.2e}"
    print(result)
    f = open('./result.txt', 'a')
    f.write(result + '\n')
    f.close()

if __name__ == "__main__":
    main()