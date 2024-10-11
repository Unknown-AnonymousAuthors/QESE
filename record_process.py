import os
import pandas as pd
import numpy as np
from collections import Counter
from my_utils.utils.dict import Merge, DictIter
import warnings
warnings.filterwarnings("ignore")

def get_statistic():
    from dataset import get_dataset
    data_dic = {'': ['num_features','num_classes','num_graphs','avg_num_nodes','avg_num_edges','avg_node_deg','avg_cluster','avg_diameter','max_num_nodes','max_num_edges']}
    for d in ['ogbg-molhiv', 'ZINC_full', 'ZINC_subset']: # 
        dataset = get_dataset(d, recal=False, cal_entro=False)
        sta = dataset.statistic_info(mode='complex')
        data_dic[d] = list(sta.values())
    df = pd.DataFrame(data_dic)
    df.to_excel('./docs/sta.xlsx', index=False)
    return df

def collect_csv(dir, suffix):
    path = dir + f'/all{suffix}.xlsx'
    if os.path.exists(path):
        os.remove(path)
    df_l = []
    for filename in os.listdir(dir):
        file_extension = os.path.splitext(filename)[1]
        if file_extension in ['.csv']:
            df = pd.read_csv(f"{dir}/{filename}", header=None)
            df_l.append(df)
    df = pd.concat(df_l, ignore_index=True)
    df.to_excel(path, index=False)
    
    return df

def summerize_csv(dir, suffix, main_key_col=3, keys_cols=[2], skiprows=1):
    assert isinstance(main_key_col, int)
    assert isinstance(keys_cols, list)
    assert main_key_col not in keys_cols
    
    df = pd.read_excel(f"{dir}/all{suffix}.xlsx", header=None, skiprows=skiprows)
    # records = [row for _, row in df.iterrows()]
    col_name = list(Counter(list(df[main_key_col])).keys())
    col_name.reverse()
    row_name = {i: list(Counter(list(df[col_idx])).keys()) for i, col_idx in enumerate(keys_cols)}
    if len(keys_cols) == 1:
        row_name = [[n] for n in row_name[0]]
    else:
        row_name = [list(k.values()) for k in iter(DictIter(row_name))]
    
    results = [
        [[[], []] for _ in range(len(col_name))]
        for _ in range(len(row_name))
    ]
    
    for _, row in df.iterrows():
        main_key_value = row[main_key_col]
        row_values = [row[idx] for idx in keys_cols]
        for i, check_row in enumerate(row_name):
            same_count = 0
            for v1, v2 in zip(check_row, row_values):
                if v1 == v2:
                    same_count += 1
            if same_count == len(check_row):
                for j, check_col in enumerate(col_name):
                    if check_col == main_key_value:
                        results[i][j][0].append(row[0]) # !!! test or val
                        results[i][j][1].append(row[1]) # !!! test or val
                        break
                break
    
    results_new = [[] for _ in range(len(results))]
    for i, row in enumerate(results):
        for j, col in enumerate(row):
            # ??????????????????????
            test_score = np.array(col[0]) * 100
            val_score = np.array(col[1]) * 100
            # record = f"{test_score.mean():.2f}±{test_score.std():.2f}, {val_score.mean():.2f}±{val_score.std():.2f}"
            record = f"{test_score.mean():.2f}±{test_score.std():.2f}"
            results_new[i].append(record)
    
    records = [['' for _ in range(len(keys_cols))] + col_name]
    for i, row in enumerate(row_name):
        records.append(row + results_new[i])

    df = pd.DataFrame(records, columns=[i for i in range(len(col_name) + len(row_name[0]))])

    df.to_excel(f"{dir}/summerize{suffix}.xlsx", index=False, header=False)

if __name__ == '__main__':
    ...
    # get_statistic()
    
    suffix = '0516' # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    path = './docs/records/records' + suffix
    
    collect_csv(path, suffix)
    summerize_csv(
        path,
        suffix,
        main_key_col = 3, # begin with 0, and 3 is dataset
        keys_cols = [2, 10, 11, 12], # [2] [2, 10, 11] [2, 12] !!!!!!!!, begin with 0, and 2 is model ; 8 number layer
        skiprows = 1 # [0, 1, 2] !!!!!!!!!!!!!
    )