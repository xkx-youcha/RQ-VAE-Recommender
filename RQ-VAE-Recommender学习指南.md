# RQ-VAE 推荐系统学习指南

## 📖 项目概述

RQ-VAE-Recommender 是一个基于生成式检索的推荐系统实现，使用语义ID（Semantic IDs）和残差量化变分自编码器（Residual Quantized Variational Autoencoder, RQ-VAE）。该项目是论文《Recommender Systems with Generative Retrieval》的PyTorch实现。

### 🎯 核心思想

传统的推荐系统通常使用双塔模型（Dual-Tower）或交叉编码器（Cross-Encoder）进行检索和排序。而RQ-VAE采用了一种全新的生成式检索方法：

1. **语义ID映射**：将物品映射为语义ID元组
2. **序列生成**：使用Transformer模型生成下一个语义ID序列

## 🏗️ 系统架构

### 两阶段训练流程

```
第一阶段：RQ-VAE Tokenizer训练
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   物品特征      │ -> │   RQ-VAE编码器  │ -> │   语义ID元组    │
│   (768维)       │    │   (3层量化)     │    │   (3个ID)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘

第二阶段：检索模型训练
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户序列      │ -> │   Transformer   │ -> │   下一个语义ID  │
│   (语义ID序列)  │    │   (解码器)      │    │   (预测)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 代码结构分析

### 核心模块

#### 1. 数据处理模块 (`data/`)
- **`processed.py`**: 数据集处理核心
  - `ItemData`: 物品数据处理
  - `SeqData`: 序列数据处理
  - 支持Amazon、MovieLens 1M、MovieLens 32M数据集

- **`amazon.py`**: Amazon评论数据集
- **`ml1m.py`**: MovieLens 1M数据集  
- **`ml32m.py`**: MovieLens 32M数据集

#### 2. RQ-VAE模块 (`modules/`)
- **`rqvae.py`**: RQ-VAE核心实现
  - 多层残差量化
  - 编码器-解码器架构
  - 语义ID生成

- **`quantize.py`**: 量化模块
  - Gumbel-Softmax量化
  - Rotation Trick量化
  - K-means初始化

- **`encoder.py`**: MLP编码器
- **`loss.py`**: 损失函数定义

#### 3. 检索模型模块 (`modules/`)
- **`model.py`**: 编码器-解码器检索模型
  - Transformer架构
  - 语义ID嵌入
  - 用户ID嵌入

- **`transformer/`**: Transformer实现
  - `attention.py`: 注意力机制
  - `model.py`: Transformer模型

- **`embedding/`**: 嵌入层
  - `id_embedder.py`: ID嵌入器

#### 4. 训练脚本
- **`train_rqvae.py`**: RQ-VAE训练脚本
- **`train_decoder.py`**: 检索模型训练脚本

#### 5. 配置文件 (`configs/`)
- **`rqvae_amazon.gin`**: Amazon数据集RQ-VAE配置
- **`decoder_amazon.gin`**: Amazon数据集解码器配置
- **`rqvae_ml32m.gin`**: MovieLens 32M数据集RQ-VAE配置
- **`decoder_ml32m.gin`**: MovieLens 32M数据集解码器配置

## 🔄 详细工作流程

### 第一阶段：RQ-VAE训练

#### 1. 数据准备
```python
# 物品数据处理
train_dataset = ItemData(
    root=dataset_folder, 
    dataset=dataset, 
    force_process=force_dataset_process, 
    train_test_split="train" if do_eval else "all", 
    split=dataset_split
)
```

#### 2. 模型初始化
```python
model = RqVae(
    input_dim=vae_input_dim,        # 768 (物品特征维度)
    embed_dim=vae_embed_dim,        # 32 (嵌入维度)
    hidden_dims=vae_hidden_dims,    # [512, 256, 128]
    codebook_size=vae_codebook_size, # 256 (码本大小)
    n_layers=vae_n_layers,          # 3 (量化层数)
    commitment_weight=commitment_weight # 0.25
)
```

#### 3. 训练过程
- **前向传播**: 物品特征 → 编码器 → 多层量化 → 语义ID
- **损失计算**: 重构损失 + 量化损失
- **反向传播**: 更新编码器、解码器和码本参数

### 第二阶段：检索模型训练

#### 1. 数据准备
```python
# 序列数据处理
train_dataset = SeqData(
    root=dataset_folder, 
    dataset=dataset, 
    is_train=True, 
    subsample=train_data_subsample, 
    split=dataset_split
)
```

#### 2. 模型初始化
```python
model = EncoderDecoderRetrievalModel(
    embedding_dim=decoder_embed_dim,    # 128
    attn_dim=attn_embed_dim,           # 512
    dropout=dropout_p,                  # 0.3
    num_heads=attn_heads,              # 8
    n_layers=attn_layers,              # 8
    num_embeddings=vae_codebook_size,  # 256
    sem_id_dim=vae_n_layers            # 3
)
```

#### 3. 训练过程
- **输入**: 用户历史序列（语义ID序列）
- **编码**: 序列编码 + 位置编码
- **解码**: 生成下一个语义ID
- **损失**: 交叉熵损失

## 🎛️ 关键配置参数

### RQ-VAE配置 (Amazon Beauty)
```gin
train.iterations=400000              # 训练迭代次数
train.learning_rate=0.0005           # 学习率
train.batch_size=64                  # 批次大小
train.vae_input_dim=768              # 输入维度
train.vae_embed_dim=32               # 嵌入维度
train.vae_codebook_size=256          # 码本大小
train.vae_n_layers=3                 # 量化层数
train.commitment_weight=0.25         # 承诺权重
```

### 检索模型配置 (Amazon Beauty)
```gin
train.iterations=200000              # 训练迭代次数
train.learning_rate=0.0003           # 学习率
train.batch_size=256                 # 批次大小
train.attn_heads=8                   # 注意力头数
train.attn_embed_dim=512             # 注意力维度
train.attn_layers=8                  # Transformer层数
train.decoder_embed_dim=128          # 解码器嵌入维度
```

## 🚀 使用方法

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 训练RQ-VAE
```bash
# Amazon Beauty数据集
python train_rqvae.py configs/rqvae_amazon.gin

# MovieLens 32M数据集
python train_rqvae.py configs/rqvae_ml32m.gin
```

### 3. 训练检索模型
```bash
# Amazon Beauty数据集
python train_decoder.py configs/decoder_amazon.gin

# MovieLens 32M数据集
python train_decoder.py configs/decoder_ml32m.gin
```

## 🔍 核心算法详解

### 1. 残差量化 (Residual Quantization)

RQ-VAE使用多层量化来生成语义ID：

```python
# 第一层量化
z1 = encoder(x)
q1, loss1 = quantize_layer1(z1)
residual1 = z1 - q1

# 第二层量化
z2 = encoder(residual1)
q2, loss2 = quantize_layer2(z2)
residual2 = z2 - q2

# 第三层量化
z3 = encoder(residual2)
q3, loss3 = quantize_layer3(z3)

# 语义ID: [id1, id2, id3]
semantic_ids = [q1_idx, q2_idx, q3_idx]
```

### 2. Gumbel-Softmax量化

在训练时使用Gumbel-Softmax进行可微分量化：

```python
def gumbel_softmax(logits, temperature=1.0):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    return F.softmax(gumbels, dim=-1)
```

### 3. 序列生成

检索模型使用Transformer解码器生成下一个语义ID：

```python
def generate_next_sem_id(self, batch, temperature=1, top_k=True):
    # 编码用户序列
    encoded = self.encode_sequence(batch)
    
    # 生成下一个ID
    logits = self.transformer_decoder(encoded)
    
    # 采样下一个语义ID
    next_id = self.sample_next_id(logits, temperature, top_k)
    
    return next_id
```

## 📊 评估指标

### 1. 重构质量
- **重构损失**: 衡量RQ-VAE重构物品特征的能力
- **量化损失**: 衡量码本使用的效率

### 2. 推荐质量
- **Top-K准确率**: 预测的语义ID是否在真实物品的语义ID中
- **召回率**: 推荐物品的覆盖率
- **多样性**: 推荐结果的多样性

## 🎯 技术亮点

### 1. 生成式检索
- 不同于传统的检索-排序范式
- 直接生成下一个物品的语义ID
- 支持序列推荐和会话推荐

### 2. 语义ID表示
- 将物品映射为离散的语义ID
- 保持物品的语义相似性
- 支持高效的索引和检索

### 3. 残差量化
- 多层量化提高表达能力
- 渐进式特征提取
- 平衡压缩率和重构质量

### 4. 可扩展性
- 支持大规模数据集
- 模块化设计
- 易于扩展和修改

## 🔗 相关论文

1. **Recommender Systems with Generative Retrieval** - 主要论文
2. **Categorical Reparametrization with Gumbel-Softmax** - Gumbel-Softmax技术
3. **Restructuring Vector Quantization with the Rotation Trick** - Rotation Trick技术

## 💡 学习建议

### 1. 理论基础
- 理解变分自编码器(VAE)原理
- 学习残差量化技术
- 掌握Transformer架构

### 2. 代码实践
- 从配置文件开始理解参数设置
- 逐步调试训练流程
- 分析模型输出和中间结果

### 3. 实验探索
- 尝试不同的数据集
- 调整模型参数
- 比较不同量化策略的效果

### 4. 扩展应用
- 应用到其他推荐场景
- 集成到现有推荐系统
- 优化推理性能

---

*这份学习指南涵盖了RQ-VAE推荐系统的核心概念、架构设计、实现细节和使用方法，希望能帮助你深入理解这个创新的推荐系统方法。* 