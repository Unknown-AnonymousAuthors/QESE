import os
from my_utils.utils.dict import DictIter

def generate_shell(py_name: str, gpu_idxs: list, key: str) -> None:
    value = args[key]
    l1 = len(value) if isinstance(value, list) else 1
    l2 = len(gpu_idxs) if isinstance(gpu_idxs, list) else 1
    assert l1 == l2
    if not os.path.exists('shells'):
        os.mkdir('shells')

    for i, value in enumerate(value):
        filename = f'./shells/{i+1}.sh'
        f = open(filename, 'w')
        f.write('date -R\n')
        args_new = args
        args_new[key] = value
        for arg in DictIter(args_new):
            arg['gpu'] = f'{gpu_idxs[i]}' if isinstance(gpu_idxs, list) else f':{gpu_idxs}'
            arg['record_path'] = f'./records/record{i+1}.csv'
            arg['log_path'] = f'./logs/logs{i+1}'

            content = f"python {py_name}"
            for k, v in arg.items():
                if v is False:
                    pass
                elif v is True:
                    content += f' --{k}'
                else:
                    content += f' --{k} {v}'
            content += '\n'
            f.write(content)
        f.write('date -R\n')
        f.close()
        os.system(f'chmod 700 {filename}')

args = {
    'model_name': ['GCN'], # ['GCN', 'QGCN', 'GIN', 'QGIN', 'GAT', 'QGAT', 'SAGE', 'QSAGE'],
    
    'dataset': ["ESOL", "FreeSolv", "Lipo", "MUV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox", "Peptides-func", "Peptides-struct"],
    # ['MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI', 'PTC_MR']
    # ['BZR', 'COX2']
    # ['COLLAB', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    # ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']
    # ["EXP", "CSL", "graph8c", "SR25"]
    # ['QM9']
    # ['ZINC_full', 'ZINC_subset']
    # ["ESOL", "FreeSolv", "Lipo", "MUV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]
    # ["Peptides-func", "Peptides-struct"]
    
    'lr': 0.003, # [0.01 MUTAG, 0.003 Molecule, 0.0005 Peptides-func, 0.0003 Peptides-struct] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    'weight_decay': 5e-4, # [0, 5e-4]
    'layer_size': 64, # [32, 64]
    'num_layers': 5, # [2, 3, 4, 5]
    'last_layer': 'mlp', # ['lin', 'mlp']
    'pool': 'add', # ['mean', 'add']
    'nc_scale': 1.0, # [0.1, 0.3, 1.0, 3.0]
    'ec_scale': 1.0, # [0.1, 0.3, 1.0, 3.0]
    'eig': ['appro_deg_ge0'], # ['np', 'appro_deg_ge0', 'betweenness', 'current_flow_betweenness']
    'dropout': 0.5, # [0.3, 0.5]
    'epochs': 500,
    'early_stop': 50,
    'batch_size': 32,
}

if __name__ == '__main__':
    gpu_idxs = [0] # [0, 0, 1, 1, 2, 3, 2, 5, 4, 4]
    key = 'model_name' # model_name dataset
    py_name = 'main.py' # main.py
    generate_shell(py_name, gpu_idxs, key)