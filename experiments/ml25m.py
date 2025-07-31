import os
import numpy as np
import pandas as pd

class ML25MItemData:
    def __init__(self, root, poster_feature_path=None):
        self.movies = pd.read_csv(os.path.join(root, 'movies.csv'))
        self.movie_ids = self.movies['movieId'].values
        # 电影类型 one-hot
        genres = self.movies['genres'].str.get_dummies(sep='|').values
        self.meta_features = genres.astype(np.float32)
        # 加载海报特征
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