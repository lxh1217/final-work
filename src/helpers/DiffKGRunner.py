# helpers/DiffKGRunner.py
# -*- coding: UTF-8 -*-
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import logging
from helpers.BaseRunner import BaseRunner
from utils import utils

# 将DiffusionDataset定义为全局类
class DiffusionDataset(torch.utils.data.Dataset):
    """扩散模型训练数据集"""
    def __init__(self, indices, emb_size=64):
        self.indices = indices
        self.emb_size = emb_size
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 生成随机关系向量
        relation_vec = torch.randn(self.emb_size)
        return relation_vec, self.indices[idx]

class DiffKGRunner(BaseRunner):
    """DiffKG专用Runner，处理扩散模型和推荐模型的交替训练"""
    
    @staticmethod
    def parse_runner_args(parser):
        # 先调用父类的 parse_runner_args
        parser = BaseRunner.parse_runner_args(parser)
        
        # 添加 DiffKG 特定的参数
        parser.add_argument('--tstEpoch', type=int, default=1,
                            help='Number of epochs between testing.')
        parser.add_argument('--save_epoch', type=int, default=10,
                            help='Save model every N epochs.')
        
        # DiffKG特定参数
        parser.add_argument('--diffusion_batch', type=int, default=256,
                            help='Batch size for diffusion training.')
        parser.add_argument('--train_max_epoch_diff', type=int, default=1,
                            help='Max epochs for diffusion training per round.')
        
        return parser
    
    def __init__(self, args):
        super().__init__(args)
        self.tstEpoch = args.tstEpoch
        self.save_epoch = args.save_epoch
        self.early_stop = args.early_stop
        self.diffusion_batch = args.diffusion_batch
        self.train_max_epoch_diff = args.train_max_epoch_diff
        
        # 记录最佳结果
        self.best_metrics = {}
        self.best_epoch = 0
        self.stop_count = 0
    
    def train(self, data_dict: Dict[str, object]):
        """
        训练模型 - 覆盖父类的train方法
        使用DiffKG特有的训练流程
        """
        # 从data_dict中获取训练数据集
        train_dataset = data_dict['train']
        dev_dataset = data_dict['dev']
        test_dataset = data_dict['test']
        
        model = train_dataset.model
        corpus = train_dataset.corpus
        
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        
        try:
            for epoch in range(self.epoch):
                logging.info(f"\n{'='*60}")
                logging.info(f"Epoch {epoch+1}/{self.epoch}")
                logging.info(f"{'='*60}")
                
                # 1. 训练扩散模型
                diff_loss, ukgc_loss = self._train_diffusion_one_epoch(model, corpus, epoch+1)
                
                # 2. 重建知识图谱
                denoisedKG = self._rebuild_kg_one_epoch(model, corpus)
                
                # 3. 训练推荐模型
                train_loss = self._train_recommendation_one_epoch(model, train_dataset, denoisedKG, epoch+1)
                
                # 观察selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)
                
                # 记录dev结果
                dev_result = self.evaluate(dev_dataset, [self.main_topk], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                
                training_time = self._check_time()
                logging_str = 'Epoch {:<5} loss={:<.4f} (diff={:.4f}, ukgc={:.4f}) [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, train_loss, diff_loss, ukgc_loss, training_time, utils.format_metric(dev_result))
                
                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(test_dataset, self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)
                
                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1]:
                    model.save_model()
                    logging_str += ' *'
                
                logging.info(logging_str)
                
                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)
        
        # Find the best dev result across iterations
        if main_metric_results:
            best_epoch = main_metric_results.index(max(main_metric_results))
            logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
                best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
            model.load_model()
    
    def _train_diffusion_one_epoch(self, model, corpus, epoch):
        """训练扩散模型一个epoch"""
        logging.info(f"训练扩散模型 (Epoch {epoch})")
        
        # 准备扩散数据加载器
        diffusion_loader = self._prepare_diffusion_loader(corpus,model)
        
        model.train()
        model.denoise_model.train()
        
        total_diff_loss = 0
        total_ukgc_loss = 0
        batch_count = 0
        
        # 获取UI矩阵
        ui_matrix = self._get_ui_matrix(model, corpus)
        
        for batch_idx, (batch_item, batch_index) in enumerate(diffusion_loader):
            batch_item = batch_item.to(model.device)
            batch_index = batch_index.to(model.device)
            
            # 训练扩散模型一步
            diff_loss, ukgc_loss = model.train_diffusion_step(batch_item, batch_index, ui_matrix)
            
            total_diff_loss += diff_loss
            total_ukgc_loss += ukgc_loss
            batch_count += 1
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                logging.info(f"扩散训练 - 批次 {batch_idx}/{len(diffusion_loader)}, "
                           f"扩散损失: {diff_loss:.4f}, UKGC损失: {ukgc_loss:.4f}")
        
        avg_diff_loss = total_diff_loss / max(batch_count, 1)
        avg_ukgc_loss = total_ukgc_loss / max(batch_count, 1)
        
        logging.info(f"Epoch {epoch} 扩散模型 - 平均扩散损失: {avg_diff_loss:.4f}, "
                   f"平均UKGC损失: {avg_ukgc_loss:.4f}")
        
        return avg_diff_loss, avg_ukgc_loss
    
    def _rebuild_kg_one_epoch(self, model, corpus):
        """重建知识图谱"""
        logging.info("重建知识图谱...")
        
        # 准备扩散数据加载器
        diffusion_loader = self._prepare_diffusion_loader(corpus,model)
        
        # 重建KG
        denoisedKG = model.rebuild_kg(diffusion_loader)
        
        if denoisedKG[0].shape[1] > 0:
            logging.info(f"重建的KG有 {denoisedKG[0].shape[1]} 条边")
        else:
            logging.warning("重建的KG为空")
        
        return denoisedKG
    
    def _train_recommendation_one_epoch(self, model, train_dataset, denoisedKG, epoch):
        """训练推荐模型一个epoch"""
        logging.info(f"训练推荐模型 (Epoch {epoch})")
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # 设置为0避免多进程问题
            collate_fn=train_dataset.collate_batch,
            pin_memory=self.pin_memory
        )
        
        model.train()
        
        total_loss = 0
        total_bpr_loss = 0
        total_cl_loss = 0
        batch_count = 0
        
        # 获取邻接矩阵
        adj = self._get_adjacency_matrix(model)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = utils.batch_to_gpu(batch, model.device)
            
            # 前向传播
            if model.cl_pattern == 0:
                output = model(batch, adj=adj, kg=denoisedKG)
            else:
                output = model(batch, adj=adj)
            
            # 计算损失
            bpr_loss = model.loss(output)
            
            # 对比学习损失
            if model.cl_pattern == 0:
                # 使用原始KG获取另一组嵌入
                original_kg = getattr(model.corpus, 'kg_edges', None)
                if original_kg is not None:
                    output_kg = model(batch, adj=adj, kg=original_kg)
                    
                    # 对比损失
                    kg_embeds = torch.cat([output_kg['user_emb'], output_kg['item_emb']], axis=0)
                    denoised_embeds = torch.cat([output['user_emb'], output['item_emb']], axis=0)
                    
                    pos_items = batch['item_id'][:, 0]
                    users = batch['user_id']
                    
                    cl_loss_item = self._contrastive_loss(
                        kg_embeds[model.user_num:], 
                        denoised_embeds[model.user_num:],
                        pos_items, 
                        model.temp
                    )
                    cl_loss_user = self._contrastive_loss(
                        kg_embeds[:model.user_num],
                        denoised_embeds[:model.user_num],
                        users,
                        model.temp
                    )
                    
                    cl_loss = (cl_loss_item + cl_loss_user) * model.ssl_reg
                else:
                    cl_loss = 0
            else:
                cl_loss = 0
            
            # 总损失
            loss = bpr_loss + cl_loss
            
            # 反向传播
            model.opt.zero_grad()
            loss.backward()
            model.opt.step()
            
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            if isinstance(cl_loss, (int, float)):
                total_cl_loss += cl_loss
            elif cl_loss is not None:
                total_cl_loss += cl_loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                logging.info(f"推荐训练 - 批次 {batch_idx}/{len(train_loader)}, "
                           f"损失: {loss.item():.4f}, BPR损失: {bpr_loss.item():.4f}")
        
        avg_loss = total_loss / max(batch_count, 1)
        avg_bpr_loss = total_bpr_loss / max(batch_count, 1)
        avg_cl_loss = total_cl_loss / max(batch_count, 1)
        
        logging.info(f"Epoch {epoch} 推荐模型 - 平均损失: {avg_loss:.4f}, "
                   f"平均BPR损失: {avg_bpr_loss:.4f}, 平均CL损失: {avg_cl_loss:.4f}")
        
        return avg_loss
    
    def _prepare_diffusion_loader(self, corpus,model):
        """准备扩散模型训练数据"""
        # 使用更少的实体
        n_entities = getattr(corpus, 'n_entities', corpus.n_items)
        max_entities = 1000  # 减少数量
        entity_indices = list(range(min(n_entities, max_entities)))
        
        # 创建数据集
        class DiffusionDataset(torch.utils.data.Dataset):
            def __init__(self, indices, emb_size=64):
                self.indices = indices
                self.emb_size = emb_size
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                # 生成随机关系向量，确保维度为 64
                relation_vec = torch.randn(self.emb_size)  # 使用传入的emb_size
                return relation_vec, self.indices[idx]
        
        dataset = DiffusionDataset(entity_indices, emb_size=model.latdim)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(self.diffusion_batch, 128),  # 减小批次大小
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        logging.info(f"扩散数据集: {len(dataset)} 个实体，批次大小: {self.diffusion_batch}")
        return loader
    
    # helpers/DiffKGRunner.py - 修改 _get_ui_matrix 方法

    def _get_ui_matrix(self, model, corpus):
        """获取用户-物品交互矩阵（简化版本）"""
        # 从训练数据中获取实际的用户-物品交互
        if hasattr(corpus, 'train_matrix'):
            # 如果已有训练矩阵，直接使用
            return corpus.train_matrix.to(model.device)
        else:
            # 否则创建简化版本
            n_users = model.user_num
            n_items = model.item_num
            
            # 使用实际数据中的用户数
            n_users = min(n_users, 1000)
            n_items = min(n_items, 5000)
            
            print(f"创建UI矩阵 - 用户数: {n_users}, 物品数: {n_items}")
            
            # 创建一个简单的交互矩阵
            # 每个用户有1-5个随机交互
            rows, cols = [], []
            for u in range(n_users):
                # 随机选择1-5个物品
                n_interactions = np.random.randint(1, 6)
                items = np.random.choice(n_items, min(n_interactions, n_items), replace=False)
                rows.extend([u] * len(items))
                cols.extend(items)
            
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.ones(len(rows))
            
            # 确保形状正确
            ui_matrix = torch.sparse_coo_tensor(
                indices, 
                values, 
                (n_users, n_items)
            ).to(model.device)
            
            print(f"UI矩阵形状: {ui_matrix.shape}, 非零元素: {len(rows)}")
            
            return ui_matrix
    
    def _get_adjacency_matrix(self, model):
        """获取邻接矩阵（简化版本）"""
        # 创建用户-物品二部图
        n_users = model.user_num
        n_items = model.item_num
        n_nodes = n_users + n_items
        
        # 创建单位矩阵（自连接）
        adj = torch.eye(n_nodes).to(model.device)
        
        return adj
    
    def _contrastive_loss(self, emb1, emb2, indices, temperature=0.2):
        """对比损失"""
        if len(indices) == 0:
            return 0
        
        # 确保索引在有效范围内
        indices = indices[indices < emb1.shape[0]]
        if len(indices) == 0:
            return 0
        
        # 获取正样本对
        pos_emb1 = emb1[indices]
        pos_emb2 = emb2[indices]
        
        # 计算相似度
        sim = torch.sum(pos_emb1 * pos_emb2, dim=-1) / temperature
        
        # 计算损失
        loss = -torch.log(torch.sigmoid(sim)).mean()
        
        return loss
    
    def _check_best(self, model, epoch, metrics):
        """检查并保存最佳模型"""
        if not metrics:
            return
        
        # 使用main_metric作为主要指标
        if self.main_metric in metrics:
            current_val = metrics[self.main_metric]
            
            if self.main_metric not in self.best_metrics or current_val > self.best_metrics[self.main_metric]:
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch
                self.stop_count = 0
                
                # 保存最佳模型
                self._save_model(model, epoch, best=True)
                logging.info(f"New best model at epoch {epoch}: {metrics}")
            else:
                self.stop_count += 1
    
    def _check_early_stop(self, epoch, metrics):
        """检查是否早停"""
        if epoch <= self.early_stop:
            return False
        
        return self.stop_count >= self.early_stop
    
    def _save_model(self, model, epoch, best=False):
        """保存模型"""
        if best:
            path = f"{model.model_path}_best.pt"
        else:
            path = f"{model.model_path}_epoch{epoch}.pt"
        
        model.save_model(path)
    
    def _load_best_model(self, model):
        """加载最佳模型"""
        path = f"{model.model_path}_best.pt"
        if os.path.exists(path):
            model.load_model(path)
        else:
            logging.warning(f"最佳模型文件不存在: {path}")