# models/FinalFixModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.BaseModel import GeneralModel

class MinimalModel(GeneralModel):
    """最终修复模型，解决所有NaN问题"""
    reader = 'BaseReader'
    runner = 'BaseRunner'
    
    @staticmethod
    def parse_model_args(parser):
        parser = GeneralModel.parse_model_args(parser)
        parser.add_argument('--emb_dim', type=int, default=32,
                           help='Embedding dimension.')
        parser.add_argument('--init_range', type=float, default=0.01,
                           help='Initialization range.')
        parser.add_argument('--clip_value', type=float, default=10.0,
                           help='Value clipping threshold.')
        parser.add_argument('--safe_loss', type=int, default=1,
                           help='Use safe loss function (1=yes, 0=no).')
        return parser
    
    def __init__(self, args, corpus):
        # 必须保存args！
        self.args = args
        
        super().__init__(args, corpus)
        
        self.emb_dim = args.emb_dim
        self.init_range = args.init_range
        self.clip_value = args.clip_value
        self.safe_loss = args.safe_loss
        
        print(f"\n{'='*60}")
        print("FinalFixModel - 最终修复版本")
        print(f"{'='*60}")
        print(f"用户数: {self.user_num}")
        print(f"物品数: {self.item_num}")
        print(f"嵌入维度: {self.emb_dim}")
        print(f"初始化范围: ±{self.init_range}")
        print(f"值裁剪: {self.clip_value}")
        print(f"安全损失: {self.safe_loss}")
        
        # 创建嵌入层
        self.user_emb = nn.Embedding(self.user_num, self.emb_dim)
        self.item_emb = nn.Embedding(self.item_num, self.emb_dim)
        
        # 安全的初始化
        self._safe_init_weights()
        
        # 创建优化器
        self.opt = torch.optim.Adam(
            self.parameters(), 
            lr=args.lr,
            weight_decay=args.l2,
            eps=1e-8
        )
        
        # 总参数统计
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数: {total_params:,}")
        print(f"{'='*60}\n")
    
    def _safe_init_weights(self):
        """绝对安全的权重初始化"""
        # 使用均匀分布，范围很小
        nn.init.uniform_(self.user_emb.weight, -self.init_range, self.init_range)
        nn.init.uniform_(self.item_emb.weight, -self.init_range, self.init_range)
        
        # 确保没有NaN/Inf
        with torch.no_grad():
            self.user_emb.weight.data = torch.nan_to_num(
                self.user_emb.weight.data, nan=0.0, posinf=self.init_range, neginf=-self.init_range
            )
            self.item_emb.weight.data = torch.nan_to_num(
                self.item_emb.weight.data, nan=0.0, posinf=self.init_range, neginf=-self.init_range
            )
    
    def forward(self, feed_dict):
        """绝对安全的前向传播"""
        # 获取输入
        user_ids = feed_dict['user_id']
        item_ids = feed_dict['item_id']
        
        # 确保类型正确
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids)
        if not isinstance(item_ids, torch.Tensor):
            item_ids = torch.tensor(item_ids)
        
        user_ids = user_ids.long()
        item_ids = item_ids.long()
        
        # 限制索引范围
        user_ids = torch.clamp(user_ids, 0, self.user_num - 1)
        item_ids = torch.clamp(item_ids, 0, self.item_num - 1)
        
        # 移动到设备
        user_ids = user_ids.to(self.device)
        item_ids = item_ids.to(self.device)
        
        # 获取嵌入
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)
        
        # 归一化嵌入（关键！）
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        
        # 计算预测分数
        scores = (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
        
        # 裁剪分数范围
        scores = torch.clamp(scores, -self.clip_value, self.clip_value)
        
        # 最后检查
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("警告: 前向传播产生NaN/Inf，使用替代值")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        
        return {'prediction': scores}
    
    def loss(self, output):
        """多种安全的损失函数选项"""
        predictions = output['prediction']
        
        # 检查输入
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("损失计算: 输入包含NaN/Inf，修复中...")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 正负样本
        pos_scores = predictions[:, 0]
        neg_scores = predictions[:, 1:].mean(dim=1)
        
        if self.safe_loss == 1:
            # 选项1: MSE损失（最稳定）
            target_pos = torch.ones_like(pos_scores)
            target_neg = torch.zeros_like(neg_scores)
            loss = F.mse_loss(pos_scores, target_pos) + F.mse_loss(neg_scores, target_neg)
            
        elif self.safe_loss == 2:
            # 选项2: 带裁剪的BPR损失
            diff = pos_scores - neg_scores
            diff = torch.clamp(diff, -10.0, 10.0)
            loss = -F.logsigmoid(diff).mean()
            
        elif self.safe_loss == 3:
            # 选项3: BCE with logits
            diff = pos_scores - neg_scores
            diff = torch.clamp(diff, -10.0, 10.0)
            loss = F.binary_cross_entropy_with_logits(
                diff,
                torch.ones_like(diff),
                reduction='mean'
            )
        else:
            # 选项4: 原始BPR（不安全）
            diff = pos_scores - neg_scores
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        # 最后检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"损失计算返回NaN/Inf: {loss.item()}, 使用替代值")
            loss = torch.tensor(0.001, requires_grad=True).to(self.device)
        
        return loss
    
    def apply_gradient_clipping(self, max_norm=1.0):
        """应用梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)