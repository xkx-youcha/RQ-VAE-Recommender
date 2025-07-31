# RQ-VAE 推荐系统流程与代码详解

## 1. RQ-VAE 推荐系统简介

RQ-VAE（Residual Quantized Variational Autoencoder）是一种结合残差量化和变分自编码器思想的离散表示学习方法，适用于推荐系统中的高效物品编码与生成式检索。

本项目基于论文《Recommender Systems with Generative Retrieval》，实现了 RQ-VAE 在推荐系统中的完整训练与推理流程。

---

## 2. 整体流程概览

### 2.1 两阶段训练流程

1. **阶段一：RQ-VAE Tokenizer 训练**
   - 目标：将物品特征编码为多层离散语义ID（如 [id1, id2, id3]）。
   - 流程：物品特征 → 编码器 → 多层残差量化 → 语义ID
2. **阶段二：检索模型训练**
   - 目标：基于用户历史语义ID序列，生成下一个物品的语义ID，实现生成式推荐。
   - 流程：用户序列（语义ID）→ Transformer → 预测下一个语义ID

---

## 3. 关键细节与算法

### 3.1 残差量化（Residual Quantization）

- 多层量化，每层编码残差信息，提升表达能力。
- 每层输出一个离散ID，最终物品表示为ID元组。

**核心代码片段（modules/quantize.py）：**
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

### 3.2 Gumbel-Softmax 可微分量化

- 训练时采用 Gumbel-Softmax 技术，使离散量化过程可微。

**核心代码片段（distributions/gumbel.py）：**
```python
def gumbel_softmax(logits, temperature=1.0):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    return F.softmax(gumbels, dim=-1)
```

### 3.3 RQ-VAE 前向传播流程

**核心代码片段（modules/rqvae.py）：**
```python
def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
    x = batch.x  # [batch_size, input_dim]
    encoded = self.encoder(x)  # [batch_size, embed_dim]
    embeddings = []
    residuals = []
    sem_ids = []
    quantize_loss = 0
    current_input = encoded
    for i, layer in enumerate(self.layers):
        quantized, loss = layer(current_input, gumbel_t)
        embeddings.append(quantized)
        residuals.append(current_input - quantized)
        sem_id = layer.get_codebook_indices(current_input)
        sem_ids.append(sem_id)
        quantize_loss += loss
        if i < len(self.layers) - 1:
            current_input = residuals[-1]
    reconstructed = self.decoder(embeddings[-1])
    reconstruction_loss = self.reconstruction_loss(reconstructed, x)
    total_loss = reconstruction_loss + quantize_loss
    return RqVaeComputedLosses(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        rqvae_loss=quantize_loss,
        embs_norm=embeddings[-1].norm(dim=-1).mean(),
        p_unique_ids=self._compute_unique_ids_ratio(sem_ids)
    )
```

---

## 4. 训练与推理流程

### 4.1 RQ-VAE 训练脚本（train_rqvae.py）

- 读取物品特征（如融合了元数据和海报特征的向量）。
- 配置参数通过 gin 文件指定。
- 训练主循环包括前向传播、损失计算、反向传播和模型保存。

**伪代码片段：**
```python
@gin.configurable
def train(..., dataset_folder, poster_feature_path, ...):
    train_dataset = ML25MItemData(dataset_folder, poster_feature_path)
    for iteration in range(iterations):
        batch = next_batch(train_dataloader)
        losses = model(batch, gumbel_t=gumbel_t)
        loss = losses.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 日志与模型保存
```

### 4.2 检索模型训练脚本（train_decoder.py）

- 读取用户历史序列（语义ID序列）。
- 使用 Transformer 结构预测下一个语义ID。
- 损失函数为交叉熵。

**伪代码片段：**
```python
@gin.configurable
def train(..., dataset_folder, poster_feature_path, ...):
    train_dataset = SeqData(...)
    for iteration in range(iterations):
        batch = next_batch(train_dataloader)
        tokenized_batch = tokenizer.tokenize_sequences(batch)
        model_output = model(tokenized_batch)
        loss = model_output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. 配置与数据融合

- 配置文件（如 configs/rqvae_ml25m.gin）指定特征维度、数据路径、训练参数等。
- 数据融合通过 ML25MItemData 类实现，将电影元数据和海报特征拼接为最终输入。

**代码片段（experiments/ml25m.py）：**
```python
class ML25MItemData:
    def __init__(self, root, poster_feature_path=None):
        self.movies = pd.read_csv(os.path.join(root, 'movies.csv'))
        genres = self.movies['genres'].str.get_dummies(sep='|').values
        self.meta_features = genres.astype(np.float32)
        if poster_feature_path:
            poster_features = np.load(poster_feature_path)
            self.features = np.concatenate([self.meta_features, poster_features], axis=1)
        else:
            self.features = self.meta_features
```

---

## 6. 评估与应用

- 评估指标包括重构损失、量化损失、Top-K 推荐准确率、召回率、多样性等。
- 训练好的模型可用于高效的生成式推荐、冷启动推荐等场景。

---

## 7. 学习建议

1. 先理解 VAE、残差量化、Gumbel-Softmax 等基础理论。
2. 结合本项目代码，重点关注数据流、模型结构和训练主循环。
3. 多调试、可视化中间结果，理解每一步的输入输出。
4. 尝试不同的数据集和特征融合方式，观察模型表现。

---

如需进一步细化某一部分代码或原理，欢迎随时提问！