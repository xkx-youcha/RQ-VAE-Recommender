import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ML25MSeqDataset(Dataset):
    """
    端到端MovieLens-25M序列推荐数据集：
    - 用户/物品ID映射
    - 物品特征（类型one-hot、标题）处理
    - 用户-物品交互序列构建，支持评分、时间
    - 支持最大序列长度、滑动窗口、训练/验证/测试划分
    - 输出ID序列、目标物品ID、可选评分/时间
    """
    def __init__(self, root, max_seq_len=50, split='train', min_user_inter=5, with_features=True, window_stride=None):
        ratings = pd.read_csv(os.path.join(root, 'ratings.csv'))
        movies = pd.read_csv(os.path.join(root, 'movies.csv'))
        # 1. 用户/物品ID映射
        self.user2id = {uid: i+1 for i, uid in enumerate(ratings['userId'].unique())}  # 0 for padding
        self.item2id = {mid: i+1 for i, mid in enumerate(movies['movieId'].unique())}  # 0 for padding
        self.id2item = {i: mid for mid, i in self.item2id.items()}
        # 2. 物品特征
        if with_features:
            genres = movies['genres'].str.get_dummies(sep='|').values.astype(np.float32)
            titles = movies['title'].apply(lambda s: s.split('(')[0].strip()).values
            self.item_features = genres  # 可扩展为拼接标题embedding
        else:
            self.item_features = None
        # 3. 过滤活跃用户
        user_counts = ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= min_user_inter].index
        ratings = ratings[ratings['userId'].isin(active_users)]
        ratings = ratings.sort_values(['userId', 'timestamp'])
        # 4. 构建用户-物品交互序列
        self.samples = []
        for uid, group in ratings.groupby('userId'):
            item_ids = [self.item2id[mid] for mid in group['movieId']]
            times = group['timestamp'].tolist()
            ratings_ = group['rating'].tolist()
            n = len(item_ids)
            if n < 2:
                continue
            # 滑动窗口/全量序列
            stride = window_stride or 1
            for i in range(1, n):
                if split == 'train' and i < int(0.8 * n):
                    pass
                elif split == 'valid' and int(0.8 * n) <= i < int(0.9 * n):
                    pass
                elif split == 'test' and i >= int(0.9 * n):
                    pass
                else:
                    continue
                # 滑动窗口支持
                start = max(0, i - max_seq_len)
                seq = item_ids[start:i]
                seq = [0] * (max_seq_len - len(seq)) + seq
                self.samples.append({
                    'user_id': self.user2id[uid],
                    'seq': np.array(seq, dtype=np.int64),
                    'target': item_ids[i],
                    'rating': ratings_[i],
                    'time': times[i]
                })
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['seq'], sample['target'], sample['user_id'], sample['rating'], sample['time']

# 用法示例：
# train_set = ML25MSeqDataset('dataset/ml-25m', split='train')
# seq, target, user_id, rating, time = train_set[0]