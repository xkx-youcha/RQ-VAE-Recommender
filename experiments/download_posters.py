import os
import requests
import pandas as pd
from tqdm import tqdm

TMDB_API_KEY = 'f435dc1a3c7093347b7c3a22f5395d02'  # 替换为你的TMDB API Key
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

LINKS_CSV = 'dataset/ml-25m/links.csv'
POSTER_DIR = 'dataset/ml-25m/posters'

os.makedirs(POSTER_DIR, exist_ok=True)

def get_poster_path(tmdb_id):
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}'
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('poster_path', None), None  # 始终返回两个值
        else:
            return None, f'HTTP {resp.status_code}'
    except Exception as e:
        return None, f'Error fetching TMDB id {tmdb_id}: {e}'

def download_poster(movie_id, tmdb_id):
    poster_path, error_msg = get_poster_path(tmdb_id)
    if poster_path:
        img_url = f'{TMDB_IMAGE_BASE}{poster_path}'
        try:
            img_data = requests.get(img_url, timeout=10).content
            with open(os.path.join(POSTER_DIR, f'{movie_id}.jpg'), 'wb') as f:
                f.write(img_data)
            return True, None
        except Exception as e:
            return False, f'Error downloading poster for movie {movie_id}: {e}'
    else:
        if error_msg:
            return False, error_msg
        else:
            return False, 'Poster path not found'

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
        try:
            success, error_msg = download_poster(movie_id, int(tmdb_id))
        except Exception as e:
            success = False
            error_msg = f'Exception: {str(e)}'
        if not success:
            print(f'No poster for movieId {movie_id}, tmdbId {tmdb_id}, error: {error_msg}')
            with open('./experiments/failed_posters.txt', 'a', encoding='utf-8') as f:
                f.write(f'{movie_id},{tmdb_id},{error_msg}\n')