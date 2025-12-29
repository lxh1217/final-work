# create_csv_from_pkl.py
import pickle
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp

data_dir = "./data/lastfm/"
files = ['trnMat.pkl', 'tstMat.pkl']

for file in files:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        print(f"Processing {file}...")
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
        
        # 检查矩阵类型
        if sp.issparse(matrix):
            rows, cols = matrix.nonzero()
            data = pd.DataFrame({
                'user_id': rows,
                'item_id': cols,
                'time': 0 if 'trn' in file else 1
            })
            save_name = 'train.csv' if 'trn' in file else 'test.csv'
            data.to_csv(os.path.join(data_dir, save_name), index=False)
            print(f"  Saved {len(data)} records to {save_name}")
        else:
            print(f"  {file} is not a sparse matrix, skipping")

# 创建验证集（从训练集中抽取10%）
train_path = os.path.join(data_dir, 'train.csv')
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    valid_df = train_df.sample(frac=0.1, random_state=42)
    valid_df.to_csv(os.path.join(data_dir, 'valid.csv'), index=False)
    print(f"Created valid.csv with {len(valid_df)} records")