import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms, models

def extract_poster_features(poster_dir, movie_ids, output_path, device='cuda'):
    """
    使用ResNet50提取每部电影海报的特征
    """
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # 去掉最后分类层
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
    # 假设你已下载好movies.csv和海报图片
    movies = pd.read_csv('dataset/ml-25m/movies.csv')
    movie_ids = movies['movieId'].values
    extract_poster_features('dataset/ml-25m/posters', movie_ids, 'dataset/ml25m_poster_features.npy')