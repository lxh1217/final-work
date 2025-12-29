# DiffKG-ReChorus 配置与运行指南

> 基于ReChorus框架的DiffKG模型完整配置与参数说明文档

---

## 目录

1. [环境配置](#环境配置)
2. [数据准备](#数据准备)
3. [参数配置](#参数配置)
4. [运行命令示例](#运行命令示例)

---

## 项目简介

DiffKG-ReChorus是基于ReChorus框架实现的DiffKG（Diffusion-based Knowledge Graph）推荐模型。该模型结合了知识图谱和扩散模型，用于提升推荐系统的性能。


## 环境配置

### 1. Python环境要求

- **Python版本**: 3.10.4（推荐）
- **操作系统**: Windows 10/11 或 Linux

### 2. 依赖包安装

#### 基础依赖
```bash
pip install torch==1.12.1
pip install numpy==1.22.3
pip install pandas==1.4.4
pip install scikit-learn==1.1.3
pip install scipy==1.7.3
pip install tqdm==4.66.1
pip install pyyaml
```

#### 关键依赖（必须安装）
```bash
# torch_scatter - 用于图神经网络操作
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu102.html

# torch_sparse - 用于稀疏张量操作
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
```

**注意**: 
- 如果使用CUDA，请根据你的CUDA版本调整安装命令
- 对于CUDA 10.2，使用上面的链接
- 对于CUDA 11.x，需要相应调整版本号

#### 可选依赖
```bash
pip install ipython==8.10.0
pip install jupyter==1.0.0
```

### 3. CUDA配置（如果使用GPU）

- 确保已安装对应版本的CUDA Toolkit
- 设置环境变量（Windows）:
  ```powershell
  $env:CUDA_LAUNCH_BLOCKING = "1"
  ```
- 或在代码中已设置（main.py第4行）

---

## 数据准备

### 1. 数据文件要求

每个数据集需要以下文件，放在 `data/<dataset_name>/` 目录下：

#### 必需文件

1. **train.csv** - 训练集
   - 格式: `user_id, item_id, time, label`
   - 分隔符: 逗号（`,`）或制表符（`\t`）
   - 示例:
     ```
     user_id,item_id,time,label
     1,6633,0,1
     1,28211,1,1
     ```

2. **dev.csv** - 验证集
   - 格式同train.csv

3. **test.csv** - 测试集
   - 格式同train.csv

4. **kg.txt** - 知识图谱三元组文件
   - 格式: `头实体 关系 尾实体` 或 `头实体\t关系\t尾实体`
   - 每行一个三元组
   - 示例:
     ```
     0 0 1
     1 1 2
     2 0 3
     ```
   - **注意**: 实体ID必须与item_id对应


### 2. 目录结构

```
项目根目录/
├── src/                          # ReChorus源码目录
│   ├── main.py                   # 主入口文件
│   ├── models/                   # 模型目录
│   │   └── DiffKGReChorus.py     # DiffKG模型
│   ├── helpers/                  # 辅助类目录
│   │   ├── DiffKGReader.py       # DiffKG数据读取器
│   │   └── DiffKGRunner.py       # DiffKG训练器
│   └── data/                      # 数据目录
│       ├── mind/                 # MIND数据集
│       │   ├── train.csv
│       │   ├── dev.csv
│       │   ├── test.csv
│       │   └── kg.txt
│       └── ...
├── log/                           # 日志目录
└── model/                         # 模型保存目录
```

---


## 参数配置

###  必需参数

| 参数 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `--model_name` | str | 模型名称 | `DiffKGReChorus` |
| `--dataset` | str | 数据集名称 | `mind`, `amazon`, `lastfm` |
| `--path` | str |数据根目录 | `./data/`, `../data/` |

### 模型架构参数

#### 嵌入和维度

| 参数 | 类型 | 默认值 | 小规模 | 大规模 | 说明 |
|------|------|--------|--------|--------|------|
| `--latdim` | int | 64 | 32 | 128 | 嵌入维度 |
| `--gnn_layer` | int | 2 | 1 | 3 | GCN层数 |
| `--layer_num_kg` | int | 2 | 1 | 3 | RGAT层数 |
| `--d_emb_size` | int | 64 | 32 | 128 | 去噪嵌入维度 |
| `--dims` | str | `"[64,128,64]"` | `"[32,64,32]"` | `"[128,256,128]"` | 去噪网络维度（JSON格式字符串） |

#### Dropout和正则化

| 参数 | 类型 | 默认值 | 推荐范围 | 说明 |
|------|------|--------|----------|------|
| `--mess_dropout_rate` | float | 0.1 | 0.0-0.3 | 消息传递dropout率 |
| `--dropout` | float | 0.2 | 0.1-0.5 | 通用dropout率 |
| `--l2` | float | 1e-5 | 1e-6 ~ 1e-4 | L2正则化系数 |

### 扩散模型参数

| 参数 | 类型 | 默认值 | 快速测试 | 完整训练 | 说明 |
|------|------|--------|----------|----------|------|
| `--diffusion_steps` | int | 1000 | 100 | 1000 | 扩散步数 |
| `--sampling_steps` | int | 1000 | 50 | 1000 | 采样步数 |
| `--noise_scale` | float | 1.0 | 0.5 | 1.0 | 扩散噪声尺度 |
| `--noise_min` | float | 0.0001 | 0.0001 | 0.0001 | 最小噪声值 |
| `--noise_max` | float | 0.02 | 0.01 | 0.02 | 最大噪声值 |
| `--norm` | int | 1 | 0 | 1 | 是否使用归一化（0=否，1=是） |

###  训练参数

#### 基础训练

| 参数 | 类型 | 默认值 | 快速测试 | 完整训练 | 说明 |
|------|------|--------|----------|----------|------|
| `--epoch` | int | 50 | 5 | 50-100 | 训练轮数 |
| `--lr` | float | 0.001 | 0.001 | 0.0001-0.01 | 学习率 |
| `--batch_size` | int | 256 | 32 | 256-512 | 训练批次大小 |
| `--eval_batch_size` | int | 256 | 32 | 256-512 | 测试批次大小 |
| `--num_neg` | int | 1 | 2 | 1-5 | 负采样数量 |

#### 损失函数权重

| 参数 | 类型 | 默认值 | 推荐范围 | 说明 |
|------|------|--------|----------|------|
| `--e_loss` | float | 0.01 | 0.001-0.1 | 知识图谱损失权重 |
| `--ssl_reg` | float | 1.0 | 0.1-2.0 | SSL正则化权重 |
| `--temp` | float | 0.2 | 0.1-0.5 | 对比损失温度 |

#### 知识图谱相关

| 参数 | 类型 | 默认值 | 推荐范围 | 说明 |
|------|------|--------|----------|------|
| `--keepRate` | float | 0.1 | 0.05-0.2 | KG边保留率 |
| `--rebuild_k` | int | 10 | 5-20 | KG重建top-k |
| `--triplet_num` | int | -1 | -1或10000 | 采样三元组数量（-1=全部） |
| `--cl_pattern` | int | 0 | 0或1 | 对比学习模式 |

### Runner参数

| 参数 | 类型 | 默认值 | 快速测试 | 完整训练 | 说明 |
|------|------|--------|----------|----------|------|
| `--tstEpoch` | int | 1 | 1 | 1 | 每N个epoch测试一次 |
| `--save_epoch` | int | 10 | 5 | 10 | 每N个epoch保存一次模型 |
| `--early_stop` | int | 20 | 3 | 20-50 | 早停轮数（验证集无提升） |
| `--diffusion_batch` | int | 256 | 16 | 256 | 扩散训练批次大小 |
| `--train_max_epoch_diff` | int | 1 | 1 | 1-3 | 每轮扩散训练最大epoch数 |
| `--topk` | str | `10,20` | `10` | `10,20,50` | Top-K评估指标（逗号分隔） |
| `--test_all` | int | 0 | 0 | 0或1 | 是否测试所有物品（0=否，1=是） |

### 系统参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpu` | str | `0` | GPU设备ID，`''`表示使用CPU |
| `--random_seed` | int | 0 | 随机种子 |
| `--load` | int | 0 | 是否加载已保存模型（0=否，1=是） |
| `--train` | int | 1 | 是否训练（0=仅测试，1=训练） |
| `--regenerate` | int | 0 | 是否重新生成中间文件（0=否，1=是） |
| `--save_final_results` | int | 1 | 是否保存最终结果（0=否，1=是） |

---

## 运行命令示例

### 1. 标准训练命令

```bash
python main.py \
    --model_name DiffKGReChorus \
    --dataset mind \
    --path ./data/ \
    --kg_file kg.txt \
    --train_mat trnMat.pkl \
    --test_mat tstMat.pkl \
    --gpu 0 \
    --random_seed 0 \
    --epoch 50 \
    --lr 0.001 \
    --l2 1e-5 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --latdim 64 \
    --gnn_layer 2 \
    --layer_num_kg 2 \
    --mess_dropout_rate 0.2 \
    --noise_scale 1.0 \
    --diffusion_steps 1000 \
    --d_emb_size 64 \
    --dims "[64, 128, 64]" \
    --e_loss 0.01 \
    --ssl_reg 1.0 \
    --temp 0.2 \
    --keepRate 0.1 \
    --rebuild_k 10 \
    --sampling_steps 1000 \
    --tstEpoch 1 \
    --save_epoch 10 \
    --early_stop 20 \
    --diffusion_batch 256 \
    --train_max_epoch_diff 1 \
    --num_neg 1 \
    --dropout 0.2 \
    --test_all 0 \
    --topk 10,20
```

### 2. 快速测试命令

```bash
python main.py \
    --model_name DiffKGReChorus \
    --dataset mind \
    --path ./data/ \
    --kg_file kg.txt \
    --gpu 0 \
    --epoch 5 \
    --lr 0.001 \
    --batch_size 32 \
    --latdim 32 \
    --gnn_layer 1 \
    --diffusion_steps 100 \
    --d_emb_size 32 \
    --dims "[32, 64, 32]" \
    --early_stop 3 \
    --topk 10
```

### 3. 从已保存模型继续训练

```bash
python main.py \
    --model_name DiffKGReChorus \
    --dataset mind \
    --path ./data/ \
    --load 1 \
    --train 1 \
    --epoch 50 \
    --gpu 0
```

### 4. 仅测试（不训练）

```bash
python main.py \
    --model_name DiffKGReChorus \
    --dataset mind \
    --path ./data/ \
    --load 1 \
    --train 0 \
    --gpu 0
```

#### 推荐训练配置
```bash
--epoch 50 \
--batch_size 256 \
--latdim 64 \
--gnn_layer 2 \
--diffusion_steps 1000 \
--d_emb_size 64 \
--dims "[64, 128, 64]" \
--early_stop 20 \
--diffusion_batch 256
```
