# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import torch
from collections import defaultdict
from tqdm import tqdm
import json

from utils import utils
from helpers.BaseReader import BaseReader

logging.basicConfig(level=logging.INFO)

class DiffKGReader(BaseReader):
    """
    DiffKG 数据读取器，扩展 BaseReader 以支持知识图谱数据
    """
    
    @staticmethod
    def parse_data_args(parser):
        parser = BaseReader.parse_data_args(parser)
        
        # KG 相关参数
        parser.add_argument('--kg_file', type=str, default='kg.txt',
                           help='Knowledge graph triplets file name')
        parser.add_argument('--rebuild_k', type=int, default=10,
                           help='Top-k items to rebuild KG edges for diffusion')
        parser.add_argument('--keep_rate', type=float, default=0.5,
                           help='Rate to keep edges after denoising')
        
        return parser

    def __init__(self, args):
        super().__init__(args)
        
        # KG 文件路径
        self.kg_file = args.kg_file
        self.rebuild_k = args.rebuild_k
        self.keep_rate = args.keep_rate
        
        # 加载和处理知识图谱
        self._load_knowledge_graph()
        
    def _load_knowledge_graph(self):
        """加载和处理知识图谱数据"""
        kg_path = os.path.join(self.prefix, self.dataset, self.kg_file)
        
        if not os.path.exists(kg_path):
            logging.warning('KG file not found: {}'.format(kg_path))
            self.has_kg = False
            return
        
        self.has_kg = True
        
        # 读取三元组数据
        self.triplets = self._read_triplets(kg_path)
        
        # 构建 KG 数据结构
        self.kg_edges, self.kg_dict = self._build_kg_graphs(self.triplets)
        
        # 构建关系字典
        self.relation_dict = self._build_relation_dict()
        
        logging.info('KG loaded: {} entities, {} relations, {} edges'.format(
            self.n_entities, self.n_relations, len(self.kg_edges)))
    
    def _read_triplets(self, file_name):
        """读取知识图谱三元组文件"""
        try:
            can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        except Exception as e:
            logging.error('Failed to load KG file {}: {}'.format(file_name, e))
            return np.zeros((0, 3), dtype=np.int32)
        
        # 去重
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        
        # 添加逆向关系（使图谱对称）
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + np.max(can_triplets_np[:, 1]) + 1
        
        # 合并原始和逆向三元组
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        
        # 计算统计信息
        self.n_relations = np.max(triplets[:, 1]) + 1
        self.n_entities = np.max(np.max(triplets[:, 0]), np.max(triplets[:, 2])) + 1
        
        # 确保实体数包含物品数
        if hasattr(self, 'n_items'):
            self.n_entities = max(self.n_entities, self.n_items)
        
        return triplets
    
    def _build_kg_graphs(self, triplets):
        """构建 KG 图数据结构"""
        kg_dict = defaultdict(list)
        kg_edges = []
        kg_counter_dict = {}
        
        logging.info('Building KG graphs...')
        
        for h_id, r_id, t_id in tqdm(triplets, desc='Processing KG edges', ascii=True):
            if h_id not in kg_counter_dict:
                kg_counter_dict[h_id] = set()
            
            # 去重
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            
            # 添加到数据结构
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))
        
        return kg_edges, kg_dict
    
    def _build_relation_dict(self):
        """构建关系字典"""
        relation_dict = {}
        
        for head in self.kg_dict:
            relation_dict[head] = {}
            for (relation, tail) in self.kg_dict[head]:
                relation_dict[head][tail] = relation
        
        return relation_dict

def load_pickle_matrix(file_path):
    """加载 pickle 格式的稀疏矩阵"""
    with open(file_path, 'rb') as f:
        matrix = pickle.load(f)
    return matrix

def load_kg_triplets(file_path):
    """加载知识图谱三元组"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append([h, r, t])
    return np.array(triples)

def convert_to_rechorus_format(data_dir, output_dir, dataset_name='mind'):
    """
    将 MIND 格式的数据转换为 ReChorus 格式
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 1. 加载交互数据
    logging.info("Loading interaction matrices...")
    train_mat_path = os.path.join(data_dir, 'trnMat.pkl')
    test_mat_path = os.path.join(data_dir, 'tstMat.pkl')
    
    if not os.path.exists(train_mat_path):
        raise FileNotFoundError(f"Train matrix not found: {train_mat_path}")
    
    # 加载训练矩阵
    train_mat = load_pickle_matrix(train_mat_path)
    n_users, n_items = train_mat.shape
    
    # 转换为 COO 格式
    train_coo = train_mat.tocoo()
    
    # 2. 创建时间戳（按交互顺序）
    logging.info("Creating timestamps...")
    
    # 为每个交互创建时间戳
    timestamps = list(range(len(train_coo.data)))
    
    # 3. 转换为 DataFrame
    logging.info("Converting to DataFrame...")
    
    # 训练集
    train_interactions = []
    for i in tqdm(range(len(train_coo.data)), desc="Processing train"):
        user_id = int(train_coo.row[i]) + 1  # ReChorus 从 1 开始
        item_id = int(train_coo.col[i]) + 1
        train_interactions.append([user_id, item_id, timestamps[i], 1])
    
    train_df = pd.DataFrame(train_interactions, columns=['user_id', 'item_id', 'time', 'label'])
    
    # 4. 分割数据为 train/dev/test (80/10/10)
    logging.info("Splitting data into train/dev/test...")
    
    # 按时间顺序分割（模拟时间序列）
    train_df = train_df.sort_values('time')
    total_size = len(train_df)
    
    train_size = int(0.8 * total_size)
    dev_size = int(0.1 * total_size)
    
    train_final = train_df.iloc[:train_size]
    dev_final = train_df.iloc[train_size:train_size + dev_size]
    test_final = train_df.iloc[train_size + dev_size:]
    
    # 5. 为测试集添加负样本
    logging.info("Adding negative samples for test set...")
    
    # 收集训练集中的点击
    user_clicked = defaultdict(set)
    for _, row in train_final.iterrows():
        user_clicked[row['user_id']].add(row['item_id'])
    
    # 为测试集的每个正样本添加 99 个负样本
    test_data_list = []
    np.random.seed(42)
    
    for idx, row in tqdm(test_final.iterrows(), total=len(test_final), desc="Adding neg samples"):
        user_id = row['user_id']
        pos_item = row['item_id']
        
        # 生成负样本
        neg_items = []
        clicked_set = user_clicked.get(user_id, set())
        
        while len(neg_items) < 99:
            neg_item = np.random.randint(1, n_items + 1)
            if neg_item not in clicked_set and neg_item != pos_item and neg_item not in neg_items:
                neg_items.append(neg_item)
        
        test_data_list.append({
            'user_id': user_id,
            'item_id': [pos_item] + neg_items,
            'time': row['time'],
            'label': 1
        })
    
    test_df = pd.DataFrame(test_data_list)
    
    # 6. 为开发集添加负样本（如果需要）
    dev_data_list = []
    for idx, row in tqdm(dev_final.iterrows(), total=len(dev_final), desc="Adding dev neg samples"):
        user_id = row['user_id']
        pos_item = row['item_id']
        
        neg_items = []
        clicked_set = user_clicked.get(user_id, set())
        
        while len(neg_items) < 99:
            neg_item = np.random.randint(1, n_items + 1)
            if neg_item not in clicked_set and neg_item != pos_item and neg_item not in neg_items:
                neg_items.append(neg_item)
        
        dev_data_list.append({
            'user_id': user_id,
            'item_id': [pos_item] + neg_items,
            'time': row['time'],
            'label': 1
        })
    
    dev_df = pd.DataFrame(dev_data_list)
    
    # 7. 保存数据
    logging.info("Saving data...")
    
    # 训练集
    train_final[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'train.csv'), sep='\t', index=False)
    logging.info(f"Train set saved: {len(train_final)} interactions")
    
    # 开发集
    dev_df['item_id'] = dev_df['item_id'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
    dev_df[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'dev.csv'), sep='\t', index=False)
    logging.info(f"Dev set saved: {len(dev_df)} users with pos/neg items")
    
    # 测试集
    test_df['item_id'] = test_df['item_id'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
    test_df[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'test.csv'), sep='\t', index=False)
    logging.info(f"Test set saved: {len(test_df)} users with pos/neg items")
    
    # 8. 处理知识图谱
    kg_path = os.path.join(data_dir, 'kg.txt')
    if os.path.exists(kg_path):
        logging.info("Processing knowledge graph...")
        kg_triplets = load_kg_triplets(kg_path)
        
        # 保存KG文件
        kg_output_path = os.path.join(dataset_path, 'kg.txt')
        np.savetxt(kg_output_path, kg_triplets, fmt='%d', delimiter='\t')
        logging.info(f"KG saved to {kg_output_path} ({len(kg_triplets)} triples)")
    else:
        logging.warning("KG file not found, will create empty KG")
        kg_output_path = os.path.join(dataset_path, 'kg.txt')
        with open(kg_output_path, 'w') as f:
            pass
    
    # 9. 创建数据集信息文件
    info = {
        'n_users': n_users,
        'n_items': n_items,
        'train_interactions': len(train_final),
        'dev_samples': len(dev_df),
        'test_samples': len(test_df),
        'train_users': train_final['user_id'].nunique(),
        'train_items': train_final['item_id'].nunique(),
        'has_kg': os.path.exists(kg_path)
    }
    
    info_path = os.path.join(dataset_path, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logging.info(f"Dataset saved to {dataset_path}")
    logging.info(f"Statistics: {info}")
    
    return dataset_path

def create_small_version(data_dir, output_dir, dataset_name='mind_small'):
    """
    创建小规模版本用于快速测试
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 加载原始数据
    train_mat = load_pickle_matrix(os.path.join(data_dir, 'trnMat.pkl'))
    
    # 取前 1000 个用户和前 2000 个物品
    small_train = train_mat[:1000, :2000].tocoo()
    
    # 转换为 DataFrame
    train_interactions = []
    for i in range(len(small_train.data)):
        user_id = int(small_train.row[i]) + 1
        item_id = int(small_train.col[i]) + 1
        train_interactions.append([user_id, item_id, i, 1])
    
    train_df = pd.DataFrame(train_interactions, columns=['user_id', 'item_id', 'time', 'label'])
    
    # 分割为 train/dev/test (80/10/10)
    train_df = train_df.sort_values('time')
    total_size = len(train_df)
    train_size = int(0.8 * total_size)
    dev_size = int(0.1 * total_size)
    
    train_final = train_df.iloc[:train_size]
    dev_final = train_df.iloc[train_size:train_size + dev_size]
    test_final = train_df.iloc[train_size + dev_size:]
    
    # 为测试集添加负样本
    n_items_small = 2000
    user_clicked = defaultdict(set)
    for _, row in train_final.iterrows():
        user_clicked[row['user_id']].add(row['item_id'])
    
    np.random.seed(42)
    test_data_list = []
    for idx, row in test_final.iterrows():
        user_id = row['user_id']
        pos_item = row['item_id']
        
        # 生成负样本
        neg_items = []
        clicked_set = user_clicked.get(user_id, set())
        
        while len(neg_items) < 99:
            neg_item = np.random.randint(1, n_items_small + 1)
            if neg_item not in clicked_set and neg_item != pos_item and neg_item not in neg_items:
                neg_items.append(neg_item)
        
        test_data_list.append({
            'user_id': user_id,
            'item_id': [pos_item] + neg_items,
            'time': row['time'],
            'label': 1
        })
    
    test_df = pd.DataFrame(test_data_list)
    test_df['item_id'] = test_df['item_id'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
    
    # 为开发集添加负样本
    dev_data_list = []
    for idx, row in dev_final.iterrows():
        user_id = row['user_id']
        pos_item = row['item_id']
        
        neg_items = []
        clicked_set = user_clicked.get(user_id, set())
        
        whiunzip -l ../rechorus_code.ziple len(neg_items) < 99:
            neg_item = np.random.randint(1, n_items_small + 1)
            if neg_item not in clicked_set and neg_item != pos_item and neg_item not in neg_items:
                neg_items.append(neg_item)
        
        dev_data_list.append({
            'user_id': user_id,
            'item_id': [pos_item] + neg_items,
            'time': row['time'],
            'label': 1
        })
    
    dev_df = pd.DataFrame(dev_data_list)
    dev_df['item_id'] = dev_df['item_id'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
    
    # 保存
    train_final[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'train.csv'), sep='\t', index=False)
    
    dev_df[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'dev.csv'), sep='\t', index=False)
    
    test_df[['user_id', 'item_id', 'time', 'label']].to_csv(
        os.path.join(dataset_path, 'test.csv'), sep='\t', index=False)
    
    # 处理KG（如果存在）
    kg_path = os.path.join(data_dir, 'kg.txt')
    if os.path.exists(kg_path):
        kg_triplets = load_kg_triplets(kg_path)
        # 只保留与物品相关的三元组
        item_ids = set(range(1, n_items_small + 1))
        filtered_triplets = []
        for h, r, t in kg_triplets:
            if (h + 1 in item_ids or t + 1 in item_ids):
                filtered_triplets.append([h + 1, r, t + 1])
        
        kg_output_path = os.path.join(dataset_path, 'kg.txt')
        np.savetxt(kg_output_path, filtered_triplets, fmt='%d', delimiter='\t')
    
    logging.info(f"Small dataset created at {dataset_path}")
    logging.info(f"Train: {len(train_final)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing kg.txt, trnMat.pkl, tstMat.pkl')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for ReChorus format data')
    parser.add_argument('--dataset_name', type=str, default='mind',
                       help='Name of the dataset')
    parser.add_argument('--create_small', action='store_true',
                       help='Create a small version for testing')
    
    args = parser.parse_args()
    
    if args.create_small:
        create_small_version(args.data_dir, args.output_dir, f"{args.dataset_name}_small")
    else:
        convert_to_rechorus_format(args.data_dir, args.output_dir, args.dataset_name)