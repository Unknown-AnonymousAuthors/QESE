from my_utils.utils.hidden_prints import HiddenPrints 
import pykeen.datasets.freebase as FB
import torch
import random
from torch.utils.data import Dataset

fb = FB.FB15k237()

class Fb15k237Dataset(Dataset):
    def __init__(self, mode='train', debug=False):
        train = get_datasets()
        test = get_datasets(mode='test')
        valid = get_datasets(mode='valid')
        num_edges = len(train) + len(test) + len(valid)
        num_eneities, num_relations = get_info()
        self.summary = {
            'num_eneities': num_eneities,
            'num_relations': num_relations,
            'num_edges': num_edges,
        }

        if mode == 'test':
            self.data = torch.tensor(test)
        elif mode == 'valid':
            self.data = torch.tensor(valid)
        elif mode == 'train':
            train_len = len(train)
            correct_data = train

            corrupted_data = []
            count = 0
            while count < train_len:
                e1, r, e2 = train[count]
                idx = random.randint(0, num_eneities - 1)
                n = random.randint(0, 1)
                if n == 0:
                    corrupt = [idx, r, e2]  # corrupt head
                else:
                    corrupt = [e1, r, idx]  # corrupt tail
                if corrupt not in correct_data and corrupt not in corrupted_data:
                    corrupted_data.append(corrupt)
                    count += 1
                    if debug and count % 100 == 0:
                        print(f'Corrupt progress: {count}/{train_len}.')
            self.data = torch.cat((torch.tensor(correct_data), torch.tensor(corrupted_data)), 1)
        else:
            return None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def get_summary():
    with HiddenPrints():
        s = fb.summary_str()
        return s

def get_info():
    with HiddenPrints():
        num_entities = fb.num_entities
        num_relations = fb.num_relations
        return num_entities, num_relations

def get_header():
    with HiddenPrints():
        header = fb.training.triples[0:5]
        return header

def get_dicts():
    with HiddenPrints():
        entity_to_id = fb.entity_to_id
        relation_to_id = fb.relation_to_id
        return entity_to_id, relation_to_id

def get_raw_data():
    with HiddenPrints():
        train_set = fb.training.triples
        test_set = fb.testing.triples
        valid_set = fb.validation.triples
        return train_set, test_set, valid_set

def get_datasets(mode='train'):
    entity_to_id, relation_to_id = get_dicts()
    train_set, test_set, valid_set = get_raw_data()

    dataset = None
    if mode == 'train':
        dataset = train_set
    elif mode == 'test':
        dataset = test_set
    elif mode == 'valid':
        dataset = valid_set
    else:
        return None

    ret_data = []
    for data in dataset:
        e1 = entity_to_id[data[0]]
        r = relation_to_id[data[1]]
        e2 = entity_to_id[data[2]]
        ret_data.append([e1, r, e2])

    return ret_data