import os
import csv
import torch
import torch.nn.functional as F
import torch_geometric
from typing import *
from torch_geometric.loader import DataLoader
from torchmetrics.functional import auroc

from my_utils.train.trainer import Trainer
from rawdata_process import sparse2dense, Metric

class MyTrainer(Trainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.interval = 25 # for checkpoint
        
        metric_max_or_min = 'max' if self.args['task_type'] in [Metric.ACC, Metric.AP, Metric.AUROC] else 'min'
        self.best_recorder.initial(
            ['train loss', 'test loss', 'test metric', 'val loss', 'val metric'] if self.args['pre_split'] else ['train loss', 'test loss', 'test metric'],
            ['min', 'min', metric_max_or_min, 'min', metric_max_or_min] if self.args['pre_split'] else ['min', 'min', metric_max_or_min]
        )

    def train(self, train_mask=None, test_mask=None, val_mask=None, scheduler=None) -> None:
        '''
            Train the model for several epochs
        '''
        for epoch in range(self.args['epochs']):
            train_loss = self.train_batch(self.dataset[train_mask])
            test_loss, test_metric = self.predict(self.dataset[test_mask])
            val_loss, val_metric = self.predict(self.dataset[val_mask])

            metric_name = self.args['task_type'].value
            if self.args['pre_split']:
                record_data = [train_loss, test_loss, test_metric, val_loss, val_metric]
                message = f'epoch:{epoch}, train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, test_{metric_name}:{test_metric:.4f}, val_loss:{val_loss:.4f}, val_{metric_name}:{val_metric:.4f}'
            else:
                record_data = [train_loss, test_loss, test_metric]
                message = f'epoch:{epoch}, train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, test_{metric_name}:{test_metric:.4f}'

            self.best_recorder.cal_best(epoch, record_data)
            self.logger.log(message, record_data)
            self.checkpoint(epoch)

            if self.args['pre_split'] == True:
                if self.early_stop(val_loss, 'min'):
                    break
            else:
                if self.early_stop(test_loss, 'min'):
                    break
            if scheduler is not None:
                scheduler.step()
    
    def train_batch(self, data_list: List[torch_geometric.data.Data]) -> float:
        '''
            Train the model for 1 batch
            Parameters:
                param1: Data obj
            Return:
                loss value
        '''
        self.model.train()

        train_loss = 0.
        data_list = DataLoader(data_list, self.args['batch_size'], shuffle=self.args['pre_split'])
        for data in data_list:
            self.optimizer.zero_grad()
            if data.x.is_sparse: # torch api to_dense() encounter some bugs
                data.x = sparse2dense(data.x)
            pred = self.model(data.x, data.edge_index, data.batch, data.node_centrality, data.edge_centrality)
            label = data.y
            if self.args['task_type'] in [Metric.AP]:
                non_nan: torch.Tensor[bool] = label == label # filter nan values
                if (non_nan == False).sum() != 0: # have nan values
                    # Tensor[Mask] will flatten the target tensor, but ok for BCE loss
                    pred = pred[non_nan].view(1, -1)
                    label = label[non_nan].view(1, -1)

            loss = self.criterion(pred, label)
            
            if self.args['task_type'] in [Metric.RMSE]:
                loss = torch.sqrt(loss)
            
            train_loss += loss.item() / len(data_list)
            loss.backward()
            self.optimizer.step()

        return train_loss
    
    def predict(self, data_list: List[torch_geometric.data.Data]) -> Tuple[float, float]:
        '''
            Inference the model, for test or valid
        '''
        if len(data_list) == 0:
            return 0.0, 0.0
        
        self.model.eval()

        predicts_l = []
        labels_l = []
        data_list = DataLoader(data_list, self.args['batch_size'], shuffle=False)
        with torch.no_grad():
            for data in data_list:
                # Actually, model.eval() will disable Dropout, which may allowing sparse tensors input without converting
                if data.x.is_sparse:
                    data.x = sparse2dense(data.x)
                pred = self.model(data.x, data.edge_index, data.batch, data.node_centrality, data.edge_centrality)
                label = data.y
                if self.args['task_type'] in [Metric.AP]:
                    non_nan: torch.Tensor[bool] = label == label # filter nan values
                    if (non_nan == False).sum() != 0: # have nan values
                        pred = pred[non_nan]
                        label = label[non_nan]
                predicts_l.append(pred)
                labels_l.append(label)

            predicts = torch.cat(predicts_l)
            labels = torch.cat(labels_l)
            loss = self.criterion(predicts, labels)
            
            if self.args['task_type'] in [Metric.ACC]:
                metric = (torch.argmax(predicts, dim=-1) == labels).sum() / predicts.shape[0]
            elif self.args['task_type'] in [Metric.AUROC]:
                metric = auroc(predicts, labels.long(), task='binary')
            elif self.args['task_type'] in [Metric.AP]:
                if self.args['dataset'] in ['MUV', 'Tox21', 'ToxCast']:
                    metric = auroc(predicts, labels.long(), task='binary')
                else:
                    metric = auroc(predicts, labels.long(), task='multilabel', num_labels=self.args['num_classes'])
            elif self.args['task_type'] in [Metric.MAE]:
                metric = loss
            elif self.args['task_type'] in [Metric.RMSE]:
                loss = torch.sqrt(loss)
                metric = loss
            elif self.args['task_type'] in [Metric.F1]:
                raise NotImplementedError()
            else:
                raise ValueError(f"The task type '{self.args['task_type']}' is invalid.")

        return loss.item(), metric.item()
    
    def test(self):
        return
    
    def valid(self):
        return

    def end_custom(self) -> List[float]:
        '''
            Record loss and acc.
        '''
        if self.args['task_type'] in [Metric.ACC, Metric.AUROC, Metric.AP]:
            test_best = self.best_recorder.get_best()[2]
            val_best = self.best_recorder.get_best()[4] if self.args['pre_split'] else None
        elif self.args['task_type'] in [Metric.MAE, Metric.RMSE]:
            test_best = self.best_recorder.get_best()[1]
            val_best = self.best_recorder.get_best()[3] if self.args['pre_split'] else None
        else:
            raise NotImplementedError
        test_best_name, test_best_score = test_best[0], test_best[1]
        if val_best is not None:
            val_best_name, val_best_score = val_best[0], val_best[1]
        else:
            val_best_name, val_best_score = '', 0
        best = [test_best_score, val_best_score]

        if not os.path.exists('records'):
            os.mkdir('records')
        try:
            file = open(self.args['record_path'], 'a')
        except:
            file = open('./records/record.csv', 'a')
        writer = csv.writer(file)
        records = best + list(self.args.values())
        writer.writerow(records)
        file.close()

        return best