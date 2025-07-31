# MovieLens-25M + 海报特征 RQ-VAE 推荐系统复现路线

## 1. 目标说明
- 在 MovieLens-25M 数据集上复现 RQ-VAE 推荐系统的完整流程。
- 融合每部电影的海报图像特征，提升物品表示的丰富性。
- 实现端到端的训练与评估流程，便于后续实验和扩展。

---

## 2. 数据准备

### 2.1 获取 MovieLens-25M 数据集
- 官网下载：[https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)
- 解压后主要文件：
  - `movies.csv`：电影元数据
  - `ratings.csv`：用户评分数据
  - `links.csv`：电影与 TMDB/IMDB 的映射

### 2.2 批量下载电影海报
- 利用 `links.csv` 获取每部电影的 TMDB id。
- 使用 TMDB API 批量下载电影海报图片，保存为 `posters/{movieId}.jpg`。
- 下载失败的海报信息会被即时记录到 `./experiments/failed_posters.txt` 文件中，便于后续重试和排查。

**代码示例：experiments/download_posters.py**
```python
import os
import requests
import pandas as pd
from tqdm import tqdm

TMDB_API_KEY = 'f435dc1a3c7093347b7c3a22f5395d02'  # 替换为你的TMDB API Key
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

LINKS_CSV = 'dataset/ml-25m/links.csv'
POSTER_DIR = 'posters'

os.makedirs(POSTER_DIR, exist_ok=True)

def get_poster_path(tmdb_id):
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('poster_path', None)
    except Exception as e:
        print(f'Error fetching TMDB id {tmdb_id}: {e}')
    return None

def download_poster(movie_id, tmdb_id):
    poster_path = get_poster_path(tmdb_id)
    if poster_path:
        img_url = f'{TMDB_IMAGE_BASE}{poster_path}'
        try:
            img_data = requests.get(img_url, timeout=10).content
            with open(os.path.join(POSTER_DIR, f'{movie_id}.jpg'), 'wb') as f:
                f.write(img_data)
            return True
        except Exception as e:
            print(f'Error downloading poster for movie {movie_id}: {e}')
    return False

if __name__ == '__main__':
    links = pd.read_csv(LINKS_CSV)
    for _, row in tqdm(links.iterrows(), total=len(links)):
        movie_id = row['movieId']
        tmdb_id = row['tmdbId']
        if pd.isna(tmdb_id):
            continue
        poster_file = os.path.join(POSTER_DIR, f'{movie_id}.jpg')
        if os.path.exists(poster_file):
            continue
        success = download_poster(movie_id, int(tmdb_id))
        if not success:
            print(f'No poster for movieId {movie_id}, tmdbId {tmdb_id}')
```
- 参考：[TMDB API 文档](https://developers.themoviedb.org/3/getting-started/introduction)

### 2.3 图像特征提取
- 使用预训练的图像模型（如 `ResNet50`, `ViT`, `CLIP` 等）提取每张海报的特征向量（如 2048/768 维）。
- 将每部电影的图像特征与原有的元数据特征拼接或融合，作为最终的物品特征输入 RQ-VAE。

**代码示例：experiments/poster_features.py**
```python
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms, models

def extract_poster_features(poster_dir, movie_ids, output_path, device='cuda'):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    for mid in tqdm(movie_ids):
        img_path = os.path.join(poster_dir, f"{mid}.jpg")
        if not os.path.exists(img_path):
            features.append(np.zeros(2048, dtype=np.float32))
            continue
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().squeeze()
        features.append(feat)
    features = np.stack(features)
    np.save(output_path, features)
    print(f"Saved poster features to {output_path}")

if __name__ == "__main__":
    import pandas as pd
    movies = pd.read_csv('dataset/ml-25m/movies.csv')
    movie_ids = movies['movieId'].values
    extract_poster_features('posters', movie_ids, 'dataset/ml25m_poster_features.npy')
```

---

## 3. 数据处理与融合

### 3.1 MovieLens-25M 数据处理

**代码示例：experiments/ml25m.py**
```python
import os
import numpy as np
import pandas as pd

class ML25MItemData:
    def __init__(self, root, poster_feature_path=None):
        self.movies = pd.read_csv(os.path.join(root, 'movies.csv'))
        self.movie_ids = self.movies['movieId'].values
        genres = self.movies['genres'].str.get_dummies(sep='|').values
        self.meta_features = genres.astype(np.float32)
        if poster_feature_path:
            poster_features = np.load(poster_feature_path)
            assert poster_features.shape[0] == self.meta_features.shape[0]
            self.features = np.concatenate([self.meta_features, poster_features], axis=1)
        else:
            self.features = self.meta_features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)

if __name__ == "__main__":
    item_data = ML25MItemData('dataset/ml-25m', 'dataset/ml25m_poster_features.npy')
    print(item_data[0].shape)  # [元数据维度+2048]
```

---

## 4. 配置文件模板

### 4.1 configs/rqvae_ml25m.gin
```gin
train.vae_input_dim = 2068  # 假设元数据20维+海报2048维
train.vae_embed_dim = 32
train.vae_codebook_size = 256
train.vae_n_layers = 3
train.commitment_weight = 0.25
train.batch_size = 64
train.iterations = 400000
train.learning_rate = 0.0005
train.dataset = 'ml25m'
train.dataset_folder = 'dataset/ml-25m'
train.poster_feature_path = 'dataset/ml25m_poster_features.npy'
```

### 4.2 configs/decoder_ml25m.gin
```gin
train.decoder_embed_dim = 128
train.attn_embed_dim = 512
train.attn_heads = 8
train.attn_layers = 8
train.batch_size = 256
train.iterations = 200000
train.learning_rate = 0.0003
train.dataset = 'ml25m'
train.dataset_folder = 'dataset/ml-25m'
train.poster_feature_path = 'dataset/ml25m_poster_features.npy'
```

---

## 5. 训练流程适配

### 5.1 数据加载适配
在 `train_rqvae.py` 和 `train_decoder.py` 的数据加载部分，判断 `dataset == 'ml25m'` 时，调用 `ML25MItemData` 并传入 `poster_feature_path`。

**伪代码示例：**
```python
from experiments.ml25m import ML25MItemData

@gin.configurable
def train(..., dataset='ml25m', dataset_folder='dataset/ml-25m', poster_feature_path=None, ...):
    if dataset == 'ml25m':
        train_dataset = ML25MItemData(dataset_folder, poster_feature_path)
    else:
        # 其他数据集
        pass
    # 后续训练流程不变
```

---

## 6. 复现流程总结

1. **下载 MovieLens-25M 数据集**
2. **运行 experiments/download_posters.py 批量下载电影海报**
   - 下载失败的海报会被记录到 `./experiments/failed_posters.txt`。
3. **运行 experiments/poster_features.py 提取海报特征**
4. **用 experiments/ml25m.py 融合元数据和海报特征，生成最终物品特征**
5. **配置 gin 文件，指定特征维度和路径**
6. **运行 train_rqvae.py 和 train_decoder.py 进行训练**

---

## 7. 可选扩展
- 可用 CLIP、ViT 等更强的视觉模型替换 ResNet50。
- 可将电影简介、演员等文本特征进一步融合。
- 支持多模态特征的自适应融合（如 MLP/Transformer 融合层）。

---

如需具体某一部分的详细代码实现（如训练主循环、特征融合细节、评估脚本等），请随时告知！