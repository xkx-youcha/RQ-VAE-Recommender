# experiments 文件夹说明与复现路径

本文件夹包含 MovieLens-25M + 海报特征 RQ-VAE 推荐系统复现的所有关键脚本。

## 目录结构

- download_posters.py  —— 批量下载电影海报，失败项记录到 failed_posters.txt
- poster_features.py   —— 提取所有电影海报的图像特征，保存为 numpy 文件
- ml25m.py             —— MovieLens-25M 数据处理与特征融合类
- ml25m_seq.py         —— 端到端序列推荐数据处理，构建用户-物品交互序列（支持ID映射、特征、评分、时间等）
- failed_posters.txt   —— 下载失败的海报信息（movieId, tmdbId, 错误信息）

## 推荐复现流程

1. **下载 MovieLens-25M 数据集**
   - 放在 dataset/ml-25m/ 目录下
2. **批量下载电影海报**
   ```bash
   python experiments/download_posters.py
   ```
   - 下载失败的海报会被即时记录到 `./experiments/failed_posters.txt`
3. **提取海报特征**
   ```bash
   python experiments/poster_features.py
   ```
   - 输出特征文件：dataset/ml25m_poster_features.npy
4. **融合元数据和海报特征**
   - 见 ml25m.py，或在主训练脚本中调用 ML25MItemData
5. **端到端序列推荐数据处理**
   - 见 ml25m_seq.py，或如下用法：
   ```python
   from experiments.ml25m_seq import ML25MSeqDataset
   train_set = ML25MSeqDataset('dataset/ml-25m', split='train')
   valid_set = ML25MSeqDataset('dataset/ml-25m', split='valid')
   test_set = ML25MSeqDataset('dataset/ml-25m', split='test')
   seq, target, user_id, rating, time = train_set[0]
   ```
   - 每个样本为 (历史ID序列, 目标物品ID, 用户ID, 评分, 时间戳)，可直接用于序列推荐模型训练。
   - 支持用户/物品ID映射（便于embedding）、物品特征（类型one-hot、标题）、评分/时间等信息。
   - 支持最大序列长度、滑动窗口、训练/验证/测试划分。
   - 可通过 `with_features` 参数控制是否加载物品特征。
   - 可通过 `window_stride` 参数控制滑动窗口步长。
   - 适用于Transformer、RNN等序列推荐模型的输入。
6. **配置 gin 文件，指定特征维度和路径**
7. **运行主训练脚本**
   ```bash
   python train_rqvae.py configs/rqvae_ml25m.gin
   python train_decoder.py configs/decoder_ml25m.gin
   ```

## 其他说明
- 你可以多次运行 download_posters.py，已下载的图片会自动跳过，失败项会持续追加到 failed_posters.txt。
- 可根据 failed_posters.txt 单独重试下载失败的海报。
- 如需进一步自动化或扩展，欢迎随时咨询。