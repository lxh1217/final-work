# helpers/DiffKGReader.py
import torch
import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict
import logging
from utils import utils

from helpers.BaseReader import BaseReader

class DiffKGReader(BaseReader):
    """DiffKG数据读取器 - 最终修复版"""
    
    @staticmethod
    def parse_data_args(parser):
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--kg_file', type=str, default='kg.txt',
                           help='Knowledge graph file.')
        parser.add_argument('--train_mat', type=str, default='trnMat.pkl',
                           help='Training matrix file.')
        parser.add_argument('--test_mat', type=str, default='tstMat.pkl',
                           help='Test matrix file.')
        return parser
    
    # helpers/DiffKGReader.py - 修改 __init__ 方法
    # helpers/DiffKGReader.py - 修改 __init__ 方法

    def __init__(self, args):
        print(f"DiffKGReader初始化开始...")
        
        # 保存 args
        self.args = args
        
        # 在调用父类之前设置KG相关属性
        self.kg_file = args.kg_file
        self.train_mat = args.train_mat
        self.test_mat = args.test_mat
        self.load_matrices = getattr(args, 'load_matrices', 0)
        
        # 初始化KG相关属性
        self.kg_dict = {}
        self.relation_dict = {}
        self.n_entities = 0
        self.n_relations = 0
        self.kg_edges = None
        
        # 调用父类初始化
        super().__init__(args)
        
        # 现在加载KG数据
        self._load_kg()
        if getattr(args, 'load_matrices', 0):
            self._load_matrices()
        
        print(f"DiffKGReader初始化完成")
        print(f"数据目录: {os.path.join(self.prefix, self.dataset)}")
        print(f"KG文件: {self.kg_file}")
        print(f"实体数: {self.n_entities}, 关系数: {self.n_relations}")
        print(f"用户数: {self.n_users}, 物品数: {self.n_items}")
    
    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        
        # 读取数据
        for key in ['train', 'dev', 'test']:
            file_path = os.path.join(self.prefix, self.dataset, key + '.csv')
            if os.path.exists(file_path):
                print(f"读取文件: {file_path}")
                
                # 尝试不同的分隔符
                try:
                    # 首先尝试默认分隔符
                    self.data_df[key] = pd.read_csv(file_path, sep=self.sep)
                    print(f"使用分隔符 '{self.sep}' 成功读取")
                except Exception as e1:
                    print(f"使用分隔符 '{self.sep}' 读取失败: {e1}")
                    
                    # 尝试空格分隔符
                    try:
                        self.data_df[key] = pd.read_csv(file_path, sep=' ')
                        print(f"使用空格分隔符成功读取")
                        self.sep = ' '  # 更新分隔符
                    except Exception as e2:
                        print(f"使用空格分隔符读取失败: {e2}")
                        
                        # 尝试制表符分隔符
                        try:
                            self.data_df[key] = pd.read_csv(file_path, sep='\t')
                            print(f"使用制表符分隔符成功读取")
                            self.sep = '\t'  # 更新分隔符
                        except Exception as e3:
                            print(f"使用制表符分隔符读取失败: {e3}")
                            
                            # 尝试逗号分隔符
                            try:
                                self.data_df[key] = pd.read_csv(file_path, sep=',')
                                print(f"使用逗号分隔符成功读取")
                                self.sep = ','  # 更新分隔符
                            except Exception as e4:
                                print(f"所有分隔符尝试都失败: {e4}")
                                raise
                
                print(f"读取后的列名: {list(self.data_df[key].columns)}")
                print(f"读取后的前几行:")
                print(self.data_df[key].head())
                
                # 重置索引并排序
                self.data_df[key] = self.data_df[key].reset_index(drop=True)
                
                # 确保列名正确
                if 'user_id' in self.data_df[key].columns:
                    # 进行排序
                    self.data_df[key] = self.data_df[key].sort_values(by=['user_id', 'time'])
                    
                    self.data_df[key] = utils.eval_list_columns(self.data_df[key])
                    print(f"已加载 {key}: {len(self.data_df[key])} 行")
                else:
                    print(f"错误: {key}.csv 不包含 user_id 列")
                    print(f"实际列名: {list(self.data_df[key].columns)}")
                    raise KeyError(f"CSV文件缺少 user_id 列")
            else:
                print(f"警告: 文件不存在 {file_path}")
                self.data_df[key] = pd.DataFrame()  # 空DataFrame

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id', 'item_id', 'time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        
        # 合并所有数据
        all_dfs = []
        for key in ['train', 'dev', 'test']:
            if key in self.data_df and not self.data_df[key].empty:
                all_dfs.append(self.data_df[key][key_columns])
        
        if all_dfs:
            self.all_df = pd.concat(all_dfs)
            
            # 用户和物品ID应该是从0开始的连续整数
            self.n_users = int(self.all_df['user_id'].max() + 1)
            self.n_items = int(self.all_df['item_id'].max() + 1)
            
            print(f"数据集统计:")
            print(f"  用户ID范围: [{self.all_df['user_id'].min()}, {self.all_df['user_id'].max()}]")
            print(f"  物品ID范围: [{self.all_df['item_id'].min()}, {self.all_df['item_id'].max()}]")
            print(f"  用户数 (n_users): {self.n_users}")
            print(f"  物品数 (n_items): {self.n_items}")
            print(f"  总交互数: {len(self.all_df)}")
            
            for key in ['dev', 'test']:
                if key in self.data_df and 'neg_items' in self.data_df[key]:
                    neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                    if neg_items.size > 0:
                        invalid_count = (neg_items >= self.n_items).sum()
                        if invalid_count > 0:
                            print(f"警告: {key}集中有 {invalid_count} 个负样本物品ID超出范围")
        else:
            print("错误: 没有加载到任何数据")
            self.n_users = 0
            self.n_items = 0
    
    def _load_kg(self):
        """加载知识图谱 - 修复版"""
        kg_path = os.path.join(self.prefix, self.dataset, self.kg_file)
        print(f"加载知识图谱: {kg_path}")
        
        if not os.path.exists(kg_path):
            print(f"警告: KG文件不存在: {kg_path}")
            self.n_entities = self.n_items
            self.n_relations = 1
            return
        
        kg_data = []
        try:
            with open(kg_path, 'r') as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 3:
                        print(f"警告: 第{line_idx}行格式不正确，跳过: {line}")
                        continue
                    
                    try:
                        h = int(parts[0])
                        r = int(parts[1])
                        t = int(parts[2])
                        kg_data.append([h, t, r])  # 注意：我们的格式是 [h, t, r]
                    except ValueError as e:
                        print(f"警告: 第{line_idx}行解析失败: {line} - {e}")
                        continue
            
            print(f"成功读取 {len(kg_data)} 个三元组")
            
            if len(kg_data) == 0:
                print("警告: KG文件为空，使用默认值")
                self.n_entities = self.n_items
                self.n_relations = 1
                return
                
        except Exception as e:
            print(f"读取KG文件失败: {e}")
            self.n_entities = self.n_items
            self.n_relations = 1
            return
        
        # 统计所有实体和关系
        all_entities = set()
        all_relations = set()
        
        for h, t, r in kg_data:
            all_entities.add(h)
            all_entities.add(t)
            all_relations.add(r)
        
        print(f"实体集合: {len(all_entities)} 个唯一实体")
        print(f"关系集合: {len(all_relations)} 个唯一关系")
        
        # 计算实体和关系的数量
        self.n_entities = max(all_entities) + 1 if all_entities else self.n_items
        self.n_relations = max(all_relations) + 1 if all_relations else 1
        
        # 如果物品数大于实体数，使用物品数
        if self.n_items > self.n_entities:
            print(f"警告: 物品数({self.n_items}) > 实体数({self.n_entities})，使用物品数")
            self.n_entities = self.n_items
        
        print(f"实体ID范围: {min(all_entities)} - {max(all_entities)}")
        print(f"关系ID范围: {min(all_relations)} - {max(all_relations)}")
        print(f"设置: n_entities = {self.n_entities}, n_relations = {self.n_relations}")
        
        # 构建数据结构
        self.kg_dict = defaultdict(list)
        self.relation_dict = defaultdict(dict)
        
        for h, t, r in kg_data:
            self.kg_dict[h].append((r, t))
            self.relation_dict[h][t] = r
        
        # 创建边张量
        if kg_data:
            edges = torch.tensor(kg_data, dtype=torch.long)
            self.kg_edges = (edges[:, :2].t().long(), edges[:, 2].long())
            print(f"KG边张量形状: 索引={self.kg_edges[0].shape}, 类型={self.kg_edges[1].shape}")
        
        # 验证一些数据
        if len(kg_data) > 5:
            print(f"前5个三元组示例:")
            for i in range(min(5, len(kg_data))):
                h, t, r = kg_data[i]
                print(f"  {i+1}: ({h}, {r}, {t})")
    
    def _load_matrices(self):
        """加载训练和测试矩阵"""
        train_path = os.path.join(self.prefix, self.dataset, self.train_mat)
        test_path = os.path.join(self.prefix, self.dataset, self.test_mat)
        
        print(f"训练矩阵: {train_path}")
        print(f"测试矩阵: {test_path}")
        
        if os.path.exists(train_path):
            try:
                with open(train_path, 'rb') as f:
                    self.train_matrix = pickle.load(f)
                print(f"训练矩阵形状: {self.train_matrix.shape}")
            except Exception as e:
                print(f"加载训练矩阵失败: {e}")
        
        if os.path.exists(test_path):
            try:
                with open(test_path, 'rb') as f:
                    self.test_matrix = pickle.load(f)
                print(f"测试矩阵形状: {self.test_matrix.shape}")
            except Exception as e:
                print(f"加载测试矩阵失败: {e}")
    
    def _append_user_history(self, data_df, phase):
        """附加用户历史信息"""
        data_df = super()._append_user_history(data_df, phase)
        
        # 添加KG信息到语料库
        self.corpus.kg_dict = self.kg_dict
        self.corpus.relation_dict = self.relation_dict
        self.corpus.n_entities = self.n_entities
        self.corpus.n_relations = self.n_relations
        self.corpus.kg_edges = self.kg_edges
        
        return data_df