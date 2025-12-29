# models/DiffKGReChorus.py
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from torch_scatter import scatter_sum, scatter_softmax

from models.BaseModel import GeneralModel
from utils import utils
from torch_sparse import SparseTensor
try:
    from torch_scatter import scatter_sum, scatter_softmax
    HAS_SCATTER = True
except ImportError:
    HAS_SCATTER = False
    print("Warning: torch_scatter not installed. Some functionality may be limited.")

try:
    import torch_sparse
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False
    print("Warning: torch_sparse not installed. Some functionality may be limited.")

class DiffKGReChorus(GeneralModel):
    """DiffKG适配ReChorus框架的版本"""
    reader = 'DiffKGReader'  # 改为DiffKGReader
    runner = 'DiffKGRunner'  # 改为DiffKGRunner
    
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--latdim', type=int, default=64,
                            help='Embedding dimension.')
        parser.add_argument('--gnn_layer', type=int, default=2,
                            help='Number of GCN layers.')
        parser.add_argument('--layer_num_kg', type=int, default=2,
                            help='Number of RGAT layers.')
        parser.add_argument('--mess_dropout_rate', type=float, default=0.1,
                            help='Dropout rate for message passing.')
        parser.add_argument('--triplet_num', type=int, default=-1,
                            help='Number of triplets to sample from KG.')
        parser.add_argument('--res_lambda', type=float, default=0.0,
                            help='Residual connection weight.')
        
        # 扩散模型参数
        parser.add_argument('--noise_scale', type=float, default=1.0,
                            help='Diffusion noise scale.')
        parser.add_argument('--noise_min', type=float, default=0.0001,
                            help='Min noise for diffusion.')
        parser.add_argument('--noise_max', type=float, default=0.02,
                            help='Max noise for diffusion.')
        parser.add_argument('--diffusion_steps', type=int, default=1000,
                            help='Number of diffusion steps.')
        parser.add_argument('--d_emb_size', type=int, default=64,
                            help='Denoise embedding size.')
        parser.add_argument('--dims', type=str, default="[64, 128, 64]",
                            help='Denoise network dimensions.')
        parser.add_argument('--norm', type=int, default=1,
                            help='Whether to use normalization in denoise.')
        
        # 训练参数
        parser.add_argument('--e_loss', type=float, default=0.1,
                            help='Weight for knowledge graph loss.')
        parser.add_argument('--cl_pattern', type=int, default=0,
                            help='Contrastive learning pattern.')
        parser.add_argument('--ssl_reg', type=float, default=1.0,
                            help='SSL regularization weight.')
        parser.add_argument('--temp', type=float, default=0.2,
                            help='Temperature for contrastive loss.')
        parser.add_argument('--keepRate', type=float, default=0.1,
                            help='Keep rate for KG edges.')
        parser.add_argument('--rebuild_k', type=int, default=10,
                            help='Top-k for KG rebuilding.')
        parser.add_argument('--sampling_steps', type=int, default=1000,
                            help='Sampling steps for diffusion.')
        
        return GeneralModel.parse_model_args(parser)
    
    # models/DiffKGReChorus.py - 修改 __init__ 方法
    def __init__(self, args, corpus):
        # 确保从 corpus 获取必要的 KG 信息
        self.n_entities = getattr(corpus, 'n_entities', corpus.n_items)
        self.n_relations = getattr(corpus, 'n_relations', 1)  # 改为默认1而不是10
        
        # 确保实体数至少等于物品数
        if self.n_entities < corpus.n_items:
            print(f"警告: 实体数({self.n_entities}) < 物品数({corpus.n_items})，将实体数设置为物品数")
            self.n_entities = corpus.n_items
        
        # 确保关系数至少为1
        if self.n_relations < 1:
            print(f"警告: 关系数({self.n_relations}) < 1，将关系数设置为1")
            self.n_relations = 1
        
        print(f"模型初始化 - 实体数: {self.n_entities}, 关系数: {self.n_relations}")
        print(f"用户数: {corpus.n_users}, 物品数: {corpus.n_items}")
        
        # 保存到 args 以便其他方法访问
        args.n_entities = self.n_entities
        args.n_relations = self.n_relations
        self.corpus = corpus
         # 保存args和关键参数
        self.args = args
        self.lr = args.lr
        super().__init__(args, corpus)
        
        # 基础参数
        self.latdim = args.latdim
        self.gnn_layer = args.gnn_layer
        self.layer_num_kg = args.layer_num_kg
        self.mess_dropout_rate = args.mess_dropout_rate
        self.triplet_num = args.triplet_num
        self.res_lambda = args.res_lambda
        
        # 扩散模型参数
        self.noise_scale = args.noise_scale
        self.noise_min = args.noise_min
        self.noise_max = args.noise_max
        self.diffusion_steps = args.diffusion_steps
        self.d_emb_size = args.d_emb_size
        self.norm = args.norm
        
        # 训练参数
        self.e_loss = args.e_loss
        self.cl_pattern = args.cl_pattern
        self.ssl_reg = args.ssl_reg
        self.temp = args.temp
        self.keepRate = args.keepRate
        self.rebuild_k = args.rebuild_k
        self.sampling_steps = args.sampling_steps
        
        # 解析维度
        self.dims = eval(args.dims) + [corpus.n_entities if hasattr(corpus, 'n_entities') else corpus.n_items]
        
        # 确保最后一个维度至少为1
        if self.dims[-1] <= 0:
            print(f"警告: 输出维度为 {self.dims[-1]}，使用物品数 {self.item_num} 替代")
            self.dims[-1] = max(self.item_num, 1)
        
        # 初始化嵌入
        self._init_embeddings()
        
        # 初始化模型组件
        self._init_components()
        
        # 初始化优化器
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        
        # 扩散模型优化器（单独）
        self.diffusion_opt = None
        
        self.apply(self.init_weights)
        
        # 重建的KG
        self.generatedKG = None
    
    def _init_embeddings(self):
        """初始化所有嵌入"""
        print(f"初始化嵌入 - 用户: {self.user_num}, 物品: {self.item_num}")
        print(f"实体: {self.n_entities}, 关系: {self.n_relations}")
        
        # 用户和物品嵌入
        self.uEmbeds = nn.Embedding(self.user_num, self.latdim)
        self.iEmbeds = nn.Embedding(self.item_num, self.latdim)
        
        # 确保实体数至少为1
        self.n_entities = max(self.n_entities, 1)
        self.n_relations = max(self.n_relations, 1)
        
        # 知识图谱嵌入（实体和关系）
        self.eEmbeds = nn.Embedding(self.n_entities, self.latdim)
        self.rEmbeds = nn.Embedding(self.n_relations, self.latdim)
        
        print(f"嵌入维度 - uEmbeds: {self.uEmbeds.weight.shape}")
        print(f"嵌入维度 - iEmbeds: {self.iEmbeds.weight.shape}")
        print(f"嵌入维度 - eEmbeds: {self.eEmbeds.weight.shape}")
        print(f"嵌入维度 - rEmbeds: {self.rEmbeds.weight.shape}")
    def _init_components(self):
        """初始化模型组件"""
        # GCN层
        self.gcnLayers = nn.ModuleList([self.GCNLayer() for _ in range(self.gnn_layer)])
        
        # RGAT层，传递res_lambda
        self.rgat = self.RGAT(self.latdim, self.layer_num_kg, self.mess_dropout_rate, self.res_lambda)
        
        # 扩散模型
        self.diffusion_model = self.GaussianDiffusion(
            self.noise_scale, self.noise_min, self.noise_max, self.diffusion_steps
        )
        
        # 去噪网络
        in_dims = self.dims[::-1]
        out_dims = self.dims
        self.denoise_model = self.Denoise(in_dims, out_dims, self.d_emb_size, 
                                  norm=self.norm, input_dim=self.latdim)
    
    def forward(self, feed_dict, adj=None, kg=None, mess_dropout=True):
        """
        前向传播
        对应原始Model中的forward
        """
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, 1+neg]
        
        # 确保索引为 long 类型且在有效范围内
        user_ids = user_ids.long().clamp(0, self.user_num - 1)
        
        # 处理item_ids，确保形状正确
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
        item_ids = item_ids.long().clamp(0, self.item_num - 1)
        
        # 处理知识图谱
        if kg is not None:
            # 确保kg在正确的设备上
            edge_index, edge_type = kg
            if edge_index.device != self.device:
                kg = (edge_index.to(self.device), edge_type.to(self.device))
            hids_KG = self.rgat.forward(self.eEmbeds.weight, self.rEmbeds.weight, kg, mess_dropout)
        else:
            # 如果没有提供KG，使用原始嵌入
            hids_KG = self.eEmbeds.weight
        
        # 获取所有用户和物品的嵌入
        all_u_embeds = self.uEmbeds.weight  # [user_num, latdim]
        all_i_embeds = hids_KG[:self.item_num]  # [item_num, latdim]
        
        # 确保物品嵌入维度正确
        if all_i_embeds.shape[0] < self.item_num:
            print(f"警告: 物品嵌入维度不匹配: {all_i_embeds.shape[0]} < {self.item_num}")
            # 使用零填充
            padding = torch.zeros(self.item_num - all_i_embeds.shape[0], 
                                all_i_embeds.shape[1], device=all_i_embeds.device)
            all_i_embeds = torch.cat([all_i_embeds, padding], dim=0)
        
        # 融合用户和物品的KG增强表示
        all_embeds = torch.cat([all_u_embeds, all_i_embeds], dim=0)  # [user_num + item_num, latdim]
        
        # GCN处理
        embedsLst = [all_embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1]) if adj is not None else embedsLst[-1]
            embedsLst.append(embeds)
        
        # 残差连接
        embeds = torch.stack(embedsLst, dim=0).mean(dim=0)  # 平均所有层
        
        # 分离用户和物品嵌入
        user_final = embeds[:self.user_num]  # [user_num, latdim]
        item_final = embeds[self.user_num:self.user_num + self.item_num]  # [item_num, latdim]
        
        # 获取当前batch的嵌入 - 添加安全检查
        try:
            batch_u_embeds = user_final[user_ids]  # [batch_size, latdim]
            batch_i_embeds = item_final[item_ids]  # [batch_size, 1+neg, latdim]
        except IndexError as e:
            print(f"索引错误: user_ids范围: [{user_ids.min().item()}, {user_ids.max().item()}] / {self.user_num}")
            print(f"索引错误: item_ids范围: [{item_ids.min().item()}, {item_ids.max().item()}] / {self.item_num}")
            raise e
        
        # 计算预测分数
        predictions = (batch_u_embeds.unsqueeze(1) * batch_i_embeds).sum(dim=-1)
        
        return {
            'prediction': predictions,
            'user_emb': user_final,
            'item_emb': item_final,
            'u_emb_batch': batch_u_embeds,
            'i_emb_batch': batch_i_embeds
        }
    def _clamp_indices(self, indices, max_val):
        """限制索引在有效范围内"""
        if indices.numel() == 0:
            return indices
        
        # 确保索引非负
        indices = torch.clamp(indices, min=0)
        
        # 确保索引不超过最大值
        if max_val > 0:
            indices = torch.clamp(indices, max=max_val-1)
        
        return indices
    
# models/DiffKGReChorus.py - 修改 train_diffusion_step 方法开头部分

    def train_diffusion_step(self, batch_item, batch_index, ui_matrix):
        """训练扩散模型的一步，增强错误处理"""
        # 检查 batch_item 维度
        print(f"训练扩散模型 - batch_item维度: {batch_item.shape}, latdim: {self.latdim}")
        
        # 维度检查
        if batch_item.shape[1] != self.latdim:
            print(f"调整batch_item维度: {batch_item.shape[1]} -> {self.latdim}")
            if batch_item.shape[1] > self.latdim:
                batch_item = batch_item[:, :self.latdim]
            else:
                padding = torch.zeros(batch_item.shape[0], self.latdim - batch_item.shape[1], 
                                    device=batch_item.device)
                batch_item = torch.cat([batch_item, padding], dim=1)
        
        # 初始化优化器
        if self.diffusion_opt is None:
            print("初始化扩散模型优化器...")
            self.diffusion_opt = torch.optim.Adam(self.denoise_model.parameters(), 
                                                lr=self.lr,
                                                weight_decay=0)
        
        self.diffusion_opt.zero_grad()
        
        # 确保所有张量都在正确的设备上
        batch_item = batch_item.to(self.device)
        batch_index = batch_index.to(self.device)
        
        # 检查 batch_index 的范围
        batch_index = torch.clamp(batch_index, 0, self.item_num - 1)
        
        # 获取当前实体和用户嵌入
        try:
            iEmbeds = self.eEmbeds.weight.detach()[:self.item_num].to(self.device)
            uEmbeds = self.uEmbeds.weight.detach().to(self.device)
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 创建替代嵌入
            iEmbeds = torch.randn(self.item_num, self.latdim, device=self.device)
            uEmbeds = torch.randn(self.user_num, self.latdim, device=self.device)
        
        # 确保 ui_matrix 在正确的设备上
        ui_matrix = ui_matrix.to(self.device)
        
        # 确保扩散模型已初始化
        if not hasattr(self.diffusion_model, '_initialized') or not self.diffusion_model._initialized:
            print("初始化扩散模型参数...")
            try:
                self.diffusion_model._initialize_on_device(self.device)
            except Exception as e:
                print(f"扩散模型初始化失败: {e}")
                # 返回虚拟损失
                return torch.tensor(0.0), torch.tensor(0.0)
        
        # 计算扩散损失
        try:
            diff_loss, ukgc_loss = self.diffusion_model.training_losses(
                self.denoise_model, 
                batch_item, 
                ui_matrix, 
                uEmbeds, 
                iEmbeds, 
                batch_index
            )
            
            # 检查损失是否有效
            if torch.isnan(diff_loss).any() or torch.isinf(diff_loss).any():
                print("警告: diff_loss包含NaN或Inf，使用替代值")
                diff_loss = torch.zeros_like(diff_loss).mean()
            
            if torch.isnan(ukgc_loss).any() or torch.isinf(ukgc_loss).any():
                print("警告: ukgc_loss包含NaN或Inf，使用替代值")
                ukgc_loss = torch.zeros_like(ukgc_loss).mean()
            
            # 总损失
            loss = diff_loss.mean() * (1 - self.e_loss) + ukgc_loss.mean() * self.e_loss
            
            # 反向传播
            loss.backward()
            self.diffusion_opt.step()
            
            return diff_loss.mean().item(), ukgc_loss.mean().item()
            
        except Exception as e:
            print(f"扩散训练失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0
    
    def rebuild_kg(self, diffusion_loader):
        """重建知识图谱"""
        with torch.no_grad():
            denoised_edges = []
            h_list = []
            t_list = []
            
            # 限制处理的批次数量，避免内存问题
            max_batches = 10  # 只处理前10个批次
            
            for batch_idx, batch in enumerate(diffusion_loader):
                if batch_idx >= max_batches:
                    break
                    
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
                
                # 确保batch_index在有效范围内
                if hasattr(self, 'item_num'):
                    batch_index = torch.clamp(batch_index, 0, self.item_num - 1)
                
                # 采样去噪后的关系向量
                try:
                    denoised_batch = self.diffusion_model.p_sample(
                        self.denoise_model, batch_item, min(self.sampling_steps, 50)  # 限制采样步数
                    )
                except Exception as e:
                    print(f"扩散采样失败: {e}")
                    continue
                
                # 获取top-k个相关实体
                k = min(self.rebuild_k, denoised_batch.shape[1])
                top_item, indices_ = torch.topk(denoised_batch, k=k)
                
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])
            
            # 如果没有边，返回空KG
            if len(h_list) == 0:
                print("警告: 没有生成任何边")
                return (torch.zeros(2, 0).long().to(self.device), 
                        torch.zeros(0).long().to(self.device))
            
            # 构建边集合（去重）
            edge_set = set()
            for idx in range(len(h_list)):
                h_val = int(h_list[idx].cpu().numpy())
                t_val = int(t_list[idx].cpu().numpy())
                if h_val >= 0 and t_val >= 0:
                    edge_set.add((h_val, t_val))
            
            # 获取关系类型
            relation_dict = getattr(self.corpus, 'relation_dict', {})
            for h, t in edge_set:
                try:
                    r = relation_dict.get(h, {}).get(t, 0)
                    denoised_edges.append([h, t, r])
                except:
                    denoised_edges.append([h, t, 0])  # 默认关系类型为0
            
            # 转换为tensor
            if len(denoised_edges) > 0:
                graph_tensor = torch.tensor(denoised_edges)
                index_ = graph_tensor[:, :2]
                type_ = graph_tensor[:, -1]
                denoisedKG = (index_.t().long().to(self.device), type_.long().to(self.device))
            else:
                denoisedKG = (torch.zeros(2, 0).long().to(self.device), 
                            torch.zeros(0).long().to(self.device))
            
            # 随机丢弃一些边（keepRate）
            index_, type_ = denoisedKG
            if index_.shape[1] > 0:
                mask = ((torch.rand(type_.shape[0]) + self.keepRate).floor()).type(torch.bool)
                denoisedKG = (index_[:, mask], type_[mask])
            
            self.generatedKG = denoisedKG
        
        return self.generatedKG
    
    def get_embeddings_for_test(self, adj, mess_dropout=False):
        """获取测试用的嵌入"""
        with torch.no_grad():
            if self.cl_pattern == 0:
                denoisedKG = self.generatedKG
                usrEmbeds, itmEmbeds = self.forward(
                    {'user_id': torch.arange(self.user_num).to(self.device),
                     'item_id': torch.arange(self.item_num).to(self.device)},
                    adj=adj, kg=denoisedKG, mess_dropout=mess_dropout
                )
            else:
                usrEmbeds, itmEmbeds = self.forward(
                    {'user_id': torch.arange(self.user_num).to(self.device),
                     'item_id': torch.arange(self.item_num).to(self.device)},
                    adj=adj, mess_dropout=mess_dropout
                )
        
        return usrEmbeds['user_emb'], itmEmbeds['item_emb']
    
    # ========== 内部组件定义 ==========
    class GCNLayer(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, adj, embeds):
            return torch.spmm(adj, embeds)
    
    class RGAT(nn.Module):
        def __init__(self, latdim, n_hops, mess_dropout_rate=0.4,res_lambda=0.0):
            super().__init__()
            self.mess_dropout_rate = mess_dropout_rate
            self.W = nn.Parameter(torch.empty(size=(2*latdim, latdim)))
            nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
            
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.n_hops = n_hops
            self.dropout = nn.Dropout(p=mess_dropout_rate)
            self.res_lambda = res_lambda 
        
        def agg(self, entity_emb, relation_emb, kg):
            edge_index, edge_type = kg
            head, tail = edge_index
            
            # 确保head, tail, edge_type与entity_emb在同一个设备上
            if head.device != entity_emb.device:
                head = head.to(entity_emb.device)
                tail = tail.to(entity_emb.device)
            if edge_type.device != entity_emb.device:
                edge_type = edge_type.to(entity_emb.device)
            
            # 确保是long类型
            head = head.long()
            tail = tail.long()
            edge_type = edge_type.long()
            
            a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
            e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
            e = self.leakyrelu(e_input)
            e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
            agg_emb = entity_emb[tail] * e.view(-1, 1)
            agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
            agg_emb = agg_emb + entity_emb
            return agg_emb
        
        def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
            entity_res_emb = entity_emb
            for _ in range(self.n_hops):
                entity_emb = self.agg(entity_emb, relation_emb, kg)
                if mess_dropout:
                    entity_emb = self.dropout(entity_emb)
                entity_emb = F.normalize(entity_emb)
                entity_res_emb = self.res_lambda * entity_res_emb + entity_emb
            return entity_res_emb
    
    class Denoise(nn.Module):
        def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, input_dim=None):  # 添加input_dim参数
            super().__init__()
            self.time_emb_dim = emb_size
            self.norm = norm
            
            print(f"初始化 Denoise 网络:")
            print(f"  in_dims 参数: {in_dims}")
            print(f"  out_dims 参数: {out_dims}")
            print(f"  emb_size: {emb_size}")
            
            # 动态设置input_dim，如果没提供则使用默认值
            self.input_dim = input_dim if input_dim is not None else 64
            self.hidden_dim = max(self.input_dim * 2, 128)  # 根据input_dim动态调整
            
            print(f"  实际网络: 输入={self.input_dim + self.time_emb_dim} -> {self.hidden_dim} -> {self.input_dim}")
            print(f"  输入维度: {self.input_dim}")
            
            self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
            
            # 动态的网络结构
            self.network = nn.Sequential(
                nn.Linear(self.input_dim + self.time_emb_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.input_dim)  # 输出维度与输入相同
            )
            
            self.drop = nn.Dropout(dropout)
        
        # models/DiffKGReChorus.py - 修改 Denoise.forward 方法
        def forward(self, x, timesteps, mess_dropout=True):
            # 1. 时间步编码
            device = x.device
            
            # 创建频率参数
            freqs = torch.exp(-math.log(10000) * torch.arange(
                start=0, end=self.time_emb_dim//2, 
                dtype=torch.float32, device=device
            ) / (self.time_emb_dim//2))
            
            # 确保 timesteps 是 2D 张量
            if timesteps.dim() == 1:
                timesteps = timesteps.unsqueeze(1)
            
            temp = timesteps.float() * freqs[None]
            time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
            
            if self.time_emb_dim % 2:
                time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
            
            # 2. 时间嵌入变换
            emb = self.emb_layer(time_emb)
            
            # 3. 输入归一化
            if self.norm:
                x = F.normalize(x, dim=-1)
            
            # 4. Dropout
            if mess_dropout:
                x = self.drop(x)
            
            # 5. 特征拼接
            # 确保 x 的维度正确
            if x.shape[1] > self.input_dim:
                x = x[:, :self.input_dim]
            elif x.shape[1] < self.input_dim:
                padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=device)
                x = torch.cat([x, padding], dim=1)
            
            h = torch.cat([x, emb], dim=-1)
            
            # 6. 通过网络
            output = self.network(h)
            
            return output

    class GaussianDiffusion(nn.Module):
        def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
            super().__init__()
            self.noise_scale = noise_scale
            self.noise_min = noise_min
            self.noise_max = noise_max
            self.steps = steps
            self.beta_fixed = beta_fixed
            
            # 先创建基本属性，稍后在to()方法中初始化
            self.betas = None
            self.alphas_cumprod = None
            self.sqrt_alphas_cumprod = None
            self.sqrt_one_minus_alphas_cumprod = None
            
            # 标记是否已初始化
            self._initialized = False
        
        def _initialize_on_device(self, device):
            """在特定设备上初始化扩散参数，添加更多安全检查"""
            if self._initialized:
                print("扩散模型参数已初始化，跳过...")
                return
            
            print(f"在设备 {device} 上初始化扩散模型参数...")
            
            # 检查steps参数
            if self.steps <= 0:
                print(f"警告: diffusion_steps={self.steps} <= 0，设置为默认值1000")
                self.steps = 1000
            
            # 检查noise参数
            if self.noise_scale <= 0:
                print(f"警告: noise_scale={self.noise_scale} <= 0，设置为默认值1.0")
                self.noise_scale = 1.0
            
            if self.noise_min <= 0:
                print(f"警告: noise_min={self.noise_min} <= 0，设置为默认值0.0001")
                self.noise_min = 0.0001
            
            if self.noise_max <= self.noise_min:
                print(f"警告: noise_max={self.noise_max} <= noise_min={self.noise_min}，调整noise_max")
                self.noise_max = self.noise_min * 20  # 默认20倍关系
            
            # 计算betas
            try:
                if self.noise_scale != 0:
                    betas_np = self.get_betas()
                    if len(betas_np) == 0:
                        raise ValueError("get_betas返回空数组")
                    
                    self.betas = torch.tensor(betas_np, dtype=torch.float64).to(device)
                    
                    # 检查betas值
                    if (self.betas <= 0).any() or (self.betas >= 1).any():
                        print(f"警告: betas值异常[{self.betas.min().item()}, {self.betas.max().item()}]，进行裁剪")
                        self.betas = torch.clamp(self.betas, min=1e-6, max=1-1e-6)
                    
                    if self.beta_fixed:
                        self.betas[0] = 0.0001
                    
                    self.calculate_for_diffusion(device)
                else:
                    print("noise_scale=0，跳过扩散参数初始化")
                    
            except Exception as e:
                print(f"扩散参数初始化失败: {e}")
                print("使用默认参数...")
                # 使用安全的默认参数
                self.steps = 1000
                self.betas = torch.linspace(1e-4, 0.02, self.steps, dtype=torch.float64).to(device)
                self.calculate_for_diffusion(device)
            
            self._initialized = True
            print("扩散模型参数初始化完成")
        
        def get_betas(self):
            """计算beta schedule，修复空数组问题"""
            # 检查参数有效性
            if self.steps <= 0:
                print(f"警告: diffusion_steps={self.steps}，使用默认值1000")
                self.steps = 1000
            
            if self.noise_scale <= 0:
                print(f"警告: noise_scale={self.noise_scale}，使用默认值1.0")
                self.noise_scale = 1.0
            
            if self.noise_min >= self.noise_max:
                print(f"警告: noise_min={self.noise_min} >= noise_max={self.noise_max}，调整参数")
                self.noise_min = 0.0001
                self.noise_max = 0.02
            
            # 计算variance
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            
            print(f"扩散参数: steps={self.steps}, noise_scale={self.noise_scale}")
            print(f"噪声范围: {start:.6f} -> {end:.6f}")
            
            try:
                # 确保steps是正整数
                steps = max(1, int(self.steps))
                variance = np.linspace(start, end, steps, dtype=np.float64)
                
                # 计算alpha_bar
                alpha_bar = 1 - variance
                
                # 安全检查
                if len(alpha_bar) == 0:
                    print("错误: alpha_bar为空，使用默认beta schedule")
                    # 返回默认的线性schedule
                    return np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
                
                # 计算betas
                betas = []
                betas.append(1 - alpha_bar[0])
                
                for i in range(1, len(alpha_bar)):
                    # 确保数值稳定
                    ratio = alpha_bar[i] / alpha_bar[i-1]
                    if ratio > 0 and ratio < 1:
                        betas.append(min(1 - ratio, 0.999))
                    else:
                        # 数值异常，使用退火策略
                        betas.append(0.02)
                
                betas_array = np.array(betas)
                print(f"beta schedule计算完成: {len(betas_array)}步, 范围[{betas_array.min():.6f}, {betas_array.max():.6f}]")
                
                return betas_array
        
            except Exception as e:
                print(f"计算beta schedule失败: {e}")
                # 返回安全的默认值
                return np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
        
        def calculate_for_diffusion(self, device=None):
            """计算扩散参数，添加安全检查"""
            # 检查betas是否有效
            if self.betas is None or len(self.betas) == 0:
                print("警告: betas为空，重新计算...")
                self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
            
            if device is not None:
                self.betas = self.betas.to(device)
            
            device = self.betas.device
            
            # 计算alphas
            alphas = 1.0 - self.betas
            
            # 检查数值有效性
            if (alphas <= 0).any() or (alphas >= 1).any():
                print(f"警告: alphas值异常，范围[{alphas.min().item():.6f}, {alphas.max().item():.6f}]，进行裁剪")
                alphas = torch.clamp(alphas, min=1e-6, max=1-1e-6)
                self.betas = 1.0 - alphas
            
            # 计算累积乘积
            self.alphas_cumprod = torch.cumprod(alphas, axis=0)
            
            # 检查累积乘积是否有效
            if torch.isnan(self.alphas_cumprod).any() or torch.isinf(self.alphas_cumprod).any():
                print("警告: alphas_cumprod包含NaN或Inf，重新初始化")
                # 使用简单的线性schedule
                self.betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64, device=device)
                alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(alphas, axis=0)
            
            # 继续计算其他参数
            self.alphas_cumprod_prev = torch.cat([
                torch.tensor([1.0], device=device, dtype=torch.float64), 
                self.alphas_cumprod[:-1]
            ])
            
            self.alphas_cumprod_next = torch.cat([
                self.alphas_cumprod[1:], 
                torch.tensor([0.0], device=device, dtype=torch.float64)
            ])
            
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
            self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
            self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
            
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            
            # 避免除零
            self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-6)
            
            self.posterior_log_variance_clipped = torch.log(
                torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
            )
            
            self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
            )
            
            print(f"扩散参数初始化完成:")
            print(f"  betas形状: {self.betas.shape}, 范围: [{self.betas.min().item():.6f}, {self.betas.max().item():.6f}]")
            print(f"  alphas_cumprod范围: [{self.alphas_cumprod.min().item():.6f}, {self.alphas_cumprod.max().item():.6f}]")
        
        def p_sample(self, model, x_start, steps):
            # 确保steps在有效范围内
            steps = min(steps, self.steps - 1)
            
            if steps == 0:
                x_t = x_start
            else:
                # 确保t在有效范围内
                t = torch.clamp(torch.tensor([steps-1] * x_start.shape[0]), 0, self.steps-1).to(x_start.device)
                x_t = self.q_sample(x_start, t)
            
            indices = list(range(self.steps))[::-1]
            
            for i in indices:
                # 确保i在有效范围内
                if i >= self.steps:
                    continue
                    
                t = torch.clamp(torch.tensor([i] * x_t.shape[0]), 0, self.steps-1).to(x_start.device)
                model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
                x_t = model_mean
            
            return x_t
        
        def q_sample(self, x_start, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_start)
            
            # 确保t在有效范围内
            if isinstance(t, torch.Tensor):
                t = torch.clamp(t, 0, self.steps - 1)
            else:
                t = min(t, self.steps - 1)
            
            device = x_start.device
            
            # 检查扩散参数是否已初始化
            if self.sqrt_alphas_cumprod is None:
                print("警告: 扩散参数未初始化，正在初始化...")
                self._initialize_on_device(device)
            
            # 确保张量在正确设备上
            sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(device)
            
            return (self._extract_into_tensor(sqrt_alphas, t, x_start.shape) * x_start + 
                    self._extract_into_tensor(sqrt_one_minus, t, x_start.shape) * noise)
        def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
            # 确保timesteps在有效范围内
            if isinstance(timesteps, torch.Tensor):
                timesteps = torch.clamp(timesteps, 0, arr.shape[0] - 1)
            else:
                timesteps = min(timesteps, arr.shape[0] - 1)
            
            res = arr[timesteps].float()
            while len(res.shape) < len(broadcast_shape):
                res = res[..., None]
            return res.expand(broadcast_shape)
        
        def p_mean_variance(self, model, x, t):
            model_output = model(x, t, False)
            
            model_variance = self.posterior_variance
            model_log_variance = self.posterior_log_variance_clipped
            
            model_variance = self._extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
            
            model_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + 
                self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
            )
            
            return model_mean, model_log_variance
        
        # models/DiffKGReChorus.py - 修改 GaussianDiffusion 类中的 training_losses 方法

        def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_index):
            batch_size = x_start.size(0)
            ts = torch.randint(0, self.steps, (batch_size,)).long().to(x_start.device)
            noise = torch.randn_like(x_start)
            
            # 确保扩散参数已初始化
            if not hasattr(self, 'sqrt_alphas_cumprod') or self.sqrt_alphas_cumprod is None:
                self._initialize_on_device(x_start.device)
            
            if self.noise_scale != 0:
                x_t = self.q_sample(x_start, ts, noise)
            else:
                x_t = x_start
            
            model_output = model(x_t, ts)
            
            # 1. 扩散损失 (MSE)
            mse = self.mean_flat((x_start - model_output) ** 2)
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = torch.where((ts == 0), 1.0, weight)
            diff_loss = weight * mse
            
            # 2. UKGC损失 - 修复维度问题
            try:
                # 确保输入有效
                if ui_matrix is None or userEmbeds is None or itmEmbeds is None:
                    raise ValueError("缺少必要的输入")
                
                # model_output 应该是 [batch_size, d_emb_size]
                d_emb_size = model_output.shape[1]
                n_users = userEmbeds.shape[0]
                n_items = itmEmbeds.shape[0]
                
                print(f"UKGC计算 - model_output形状: {model_output.shape}, d_emb_size: {d_emb_size}")
                print(f"UKGC计算 - userEmbeds形状: {userEmbeds.shape}, n_users: {n_users}")
                print(f"UKGC计算 - itmEmbeds形状: {itmEmbeds.shape}, n_items: {n_items}")
                print(f"UKGC计算 - ui_matrix形状: {ui_matrix.shape}")
                print(f"UKGC计算 - batch_index形状: {batch_index.shape}")
                
                # 简化版本的 UKGC 损失
                # 方法1: 直接计算相似度损失
                if d_emb_size == itmEmbeds.shape[1]:
                    # 直接计算余弦相似度
                    target_embeds = itmEmbeds[batch_index]
                    
                    # 确保维度匹配
                    if model_output.shape[1] != target_embeds.shape[1]:
                        # 调整维度
                        if model_output.shape[1] < target_embeds.shape[1]:
                            padding = torch.zeros(model_output.shape[0], 
                                                target_embeds.shape[1] - model_output.shape[1],
                                                device=model_output.device)
                            model_output = torch.cat([model_output, padding], dim=1)
                        else:
                            model_output = model_output[:, :target_embeds.shape[1]]
                    
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(model_output, target_embeds, dim=-1)
                    ukgc_loss = (1 - similarity).mean()
                    
                    print(f"UKGC损失 (相似度): {ukgc_loss.item():.4f}")
                
                else:
                    # 方法2: 简单的 MSE 损失
                    # 将 model_output 投影到正确的维度
                    if not hasattr(self, 'ukgc_projection'):
                        self.ukgc_projection = nn.Linear(d_emb_size, itmEmbeds.shape[1]).to(model_output.device)
                        nn.init.xavier_uniform_(self.ukgc_projection.weight)
                    
                    projected_output = self.ukgc_projection(model_output)
                    target_embeds = itmEmbeds[batch_index]
                    
                    # 确保维度匹配
                    if projected_output.shape[1] != target_embeds.shape[1]:
                        projected_output = projected_output[:, :target_embeds.shape[1]]
                    
                    ukgc_loss = self.mean_flat((projected_output - target_embeds) ** 2)
                    
                    print(f"UKGC损失 (MSE): {ukgc_loss.mean().item():.4f}")
                
            except Exception as e:
                print(f"⚠️ UKGC损失计算失败，使用替代方法: {e}")
                
                # 替代方案: 使用扩散损失的一部分
                ukgc_loss = diff_loss.detach() * 0.1
            
            return diff_loss, ukgc_loss
        
        def mean_flat(self, tensor):
            return tensor.mean(dim=list(range(1, len(tensor.shape))))
        
        def SNR(self, t):
            return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
        
class Dataset(GeneralModel.Dataset):
    def __init__(self, model, corpus, phase):
        super().__init__(model, corpus, phase)
        print(f"初始化 {phase} 数据集，样本数: {len(self)}")
        
    def _get_feed_dict(self, index):
        """
        确保训练和测试都生成正确数量的候选物品
        """
        # 获取用户ID和正样本物品ID
        user_id = self.data['user_id'][index]
        pos_item = self.data['item_id'][index]
        
        feed_dict = {
            'user_id': user_id,
        }
        
        if self.phase == 'train':
            # 训练阶段：使用模型指定的负样本数量
            num_neg = self.model.num_neg
            num_candidates = num_neg + 1  # 正样本 + 负样本
            
            # 获取用户点击过的物品集合（用于排除）
            clicked_set = self.corpus.train_clicked_set.get(user_id, set())
            
            # 生成负样本
            neg_items = []
            attempts = 0
            max_attempts = num_neg * 100
            
            while len(neg_items) < num_neg and attempts < max_attempts:
                attempts += 1
                neg_item = np.random.randint(0, self.model.item_num)
                
                # 检查是否有效
                if neg_item != pos_item and neg_item not in clicked_set:
                    neg_items.append(neg_item)
            
            # 如果生成不足，补充随机物品
            while len(neg_items) < num_neg:
                neg_item = np.random.randint(0, self.model.item_num)
                if neg_item != pos_item:
                    neg_items.append(neg_item)
            
            # 合并正负样本
            item_list = [pos_item] + neg_items
            feed_dict['item_id'] = item_list
            
            if index < 3:  # 调试信息
                print(f"训练样本 {index}: 用户={user_id}, 正样本={pos_item}, 负样本数={len(neg_items)}")
            
        else:
            # 测试/验证阶段：生成固定数量的负样本
            num_neg_test = 100  # 测试时使用100个负样本
            num_candidates = num_neg_test + 1
            
            # 获取用户点击过的所有物品（包括训练、验证、测试）
            clicked_set = set()
            if hasattr(self.corpus, 'train_clicked_set'):
                clicked_set.update(self.corpus.train_clicked_set.get(user_id, set()))
            if hasattr(self.corpus, 'residual_clicked_set'):
                clicked_set.update(self.corpus.residual_clicked_set.get(user_id, set()))
            
            # 生成负样本
            neg_items = []
            attempts = 0
            max_attempts = num_neg_test * 100
            
            # 使用固定种子确保可重复性
            seed = index * 1000
            rng = np.random.RandomState(seed)
            
            while len(neg_items) < num_neg_test and attempts < max_attempts:
                attempts += 1
                neg_item = rng.randint(0, self.model.item_num)
                
                # 检查是否有效
                if neg_item != pos_item and neg_item not in clicked_set:
                    neg_items.append(neg_item)
            
            # 如果生成不足，补充随机物品
            while len(neg_items) < num_neg_test:
                neg_item = rng.randint(0, self.model.item_num)
                if neg_item != pos_item and neg_item not in clicked_set:
                    neg_items.append(neg_item)
            
            # 合并正负样本
            item_list = [pos_item] + neg_items
            feed_dict['item_id'] = item_list
            
            if index < 3:  # 调试信息
                print(f"测试样本 {index}: 用户={user_id}, 正样本={pos_item}, 候选总数={len(item_list)}")
        
        return feed_dict
    
    def actions_before_epoch(self):
        """
        在每个训练epoch之前调用的方法
        这里可以预先生成负样本，但我们的_get_feed_dict已经实时生成
        """
        if self.phase == 'train':
            print(f"准备训练epoch，样本数: {len(self)}")

DiffKGReChorus.Dataset = Dataset