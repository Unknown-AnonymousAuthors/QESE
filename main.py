import os, sys
import torch
import random
import numpy as np
import warnings
import torch.nn as nn

from models import get_model
from trainer import MyTrainer
from rawdata_process import get_task, Metric
from dataset import get_dataset, KFoldIter
from my_utils.utils.dict import Merge, DictIter

sys.path.append(os.getcwd() + '/..')
warnings.filterwarnings("ignore")

FOLDS = 10

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(opt):
    seed_everything(0)
    args_init = vars(opt)
    
    record_list = []
    args_iter = iter(DictIter(args_init))
    for args in args_iter:
        cal_entro = True # if args['model_name'][0] == 'Q' else False
        dataset = get_dataset(args['dataset'], recal_rawdata=False, recal_entro=False, cal_entro=cal_entro, nc_scale=args['nc_scale'], ec_scale=args['ec_scale'], eig=args['eig'])
        args['task_type'] = get_task(args['dataset'])
        
        # try to load pre-splited subdataset masks
        mask_file = f"./datasets/obj/{args['dataset']}_split_mask.pt"
        if os.path.exists(mask_file):
            split_mask = torch.load(mask_file)
            args['pre_split'] = True
            train_idx, test_idx, val_idx = split_mask['train'], split_mask['test'], split_mask['val']
            if isinstance(train_idx, int): # else: masks are Tensor
                assert train_idx + test_idx + val_idx == len(dataset)
                pre_train_mask = slice(0, train_idx)
                pre_test_mask = slice(train_idx, train_idx + test_idx)
                pre_val_mask = slice(train_idx + test_idx, train_idx + test_idx + val_idx)
            else:
                raise NotImplementedError()
        else:
            args['pre_split'] = False
            dataset.shuffle()
        
        data_info = dataset.statistic_info()
        args = Merge(args, data_info)
        device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        dataset.to(device)
        folds_record_list = []
        for fold, (train_mask, test_mask) in enumerate(iter(KFoldIter(len(dataset), folds=FOLDS))):
            ''' dataset '''
            val_mask = None
            if args['pre_split']: # force load masks for each fold
                train_mask, test_mask, val_mask = pre_train_mask, pre_test_mask, pre_val_mask
                seed_everything(fold) # different seeds for fixed-splited datasets
                
            ''' model '''
            model = get_model(args)
            
            ''' train '''
            model.to(device)
            if args['task_type'] in [Metric.ACC]:
                criterion = nn.CrossEntropyLoss()
            elif args['task_type'] in [Metric.AUROC, Metric.AP]:
                criterion = nn.BCEWithLogitsLoss()
            elif args['task_type'] in [Metric.MAE]:
                criterion = nn.L1Loss()
            elif args['task_type'] in [Metric.RMSE]:
                criterion = nn.MSELoss()
            elif args['task_type'] in [Metric.F1]:
                raise NotImplementedError()
            else:
                raise ValueError(f"The task type '{args['task_type']}' is invalid.")
            criterion.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            topic = f"{args['model_name']}_{args['dataset']}_{FOLDS}-folds"
            trainer = MyTrainer(args, model, dataset, criterion, optimizer, topic, device)
            record = trainer(train_mask=train_mask, test_mask=test_mask, val_mask=val_mask, scheduler=scheduler)
            folds_record_list.append(record)
            
        test_metric = np.array([metric[0] for metric in folds_record_list])
        val_metric = np.array([metric[1] for metric in folds_record_list]) if args['pre_split'] == True else None
        if args['task_type'] in [Metric.ACC, Metric.AUROC, Metric.AP]:
            test_metric *= 100
            if val_metric is not None:
                val_metric *= 100
            p = 2 # numerical precision for print
        else:
            p = 4
            
        if val_metric is None:
            print(f"Model {args['model_name']} achieve a test result {test_metric.mean():.{p}f}±{test_metric.std():.{p}f} on dataset {args['dataset']} with 10-fold setting.")
        else:
            print(f"Model {args['model_name']} achieve a test result {test_metric.mean():.{p}f}±{test_metric.std():.{p}f} and val result {val_metric.mean():.{p}f}±{val_metric.std():.{p}f} on dataset {args['dataset']} with pre-split.")
        
    return record_list

if __name__ == "__main__":
    from args import get_args_parser

    parser = get_args_parser()
    opt = parser.parse_args()

    main(opt)