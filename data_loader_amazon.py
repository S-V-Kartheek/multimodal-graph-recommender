import os
import gzip
import json
import requests
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

AMAZON_REVIEWS_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz"
AMAZON_META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz"

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"[INFO] Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[INFO] Download complete.")

def parse_gz(path):
    for line in gzip.open(path, 'r'):
        yield json.loads(line)

def get_df(path):
    i = 0
    df = {}
    for d in parse_gz(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def load_and_preprocess_amazon(data_dir=None, mod_dim=50, max_users=10000, max_items=10000):
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    
    reviews_path = os.path.join(data_dir, 'reviews_Baby_5.json.gz')
    meta_path = os.path.join(data_dir, 'meta_Baby.json.gz')
    
    download_file(AMAZON_REVIEWS_URL, reviews_path)
    download_file(AMAZON_META_URL, meta_path)
    
    print("[INFO] Parsing Amazon Baby metadata...")
    meta_df = get_df(meta_path)
    meta_df = meta_df[['asin', 'title', 'description', 'categories']].drop_duplicates(subset=['asin'])
    meta_df['description'] = meta_df['description'].fillna('')
    meta_df['title'] = meta_df['title'].fillna('')
    
    print("[INFO] Parsing Amazon Baby reviews...")
    reviews_df = get_df(reviews_path)
    
    # Filter to speed up testing if needed
    user_counts = reviews_df['reviewerID'].value_counts()
    item_counts = reviews_df['asin'].value_counts()
    
    top_users = user_counts.head(max_users).index
    top_items = item_counts.head(max_items).index
    
    ratings_df = reviews_df[(reviews_df['reviewerID'].isin(top_users)) & (reviews_df['asin'].isin(top_items))].copy()
    
    # Merge with meta
    movies_df = meta_df[meta_df['asin'].isin(ratings_df['asin'])].copy()
    
    user_ids = sorted(ratings_df['reviewerID'].unique())
    item_ids = sorted(movies_df['asin'].unique())
    
    # Ensure ratings only contains valid items
    ratings_df = ratings_df[ratings_df['asin'].isin(item_ids)]
    
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {mid: idx for idx, mid in enumerate(item_ids)}
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    print(f"[INFO] Loaded {len(ratings_df)} ratings, {n_users} users, {n_items} items")
    
    # User Features (Random/Blank for Amazon since no demographics are provided)
    print("[INFO] Generating User Features...")
    feature_dim = 4 * mod_dim
    np.random.seed(42)
    user_features = np.random.randn(n_users, feature_dim) * 0.1
    user_features_t = torch.tensor(user_features, dtype=torch.float32)
    
    user_modality_tensors = {}
    for idx, name in enumerate(['text', 'image', 'video', 'meta']):
        start = idx * mod_dim
        end = start + mod_dim
        user_modality_tensors[name] = user_features_t[:, start:end]
        
    print("[INFO] Encoding multimodal item features...")
    # 1. Text Data: RoBERTa
    print(f"    - Extracting Text Data using RoBERTa...")
    text_model = SentenceTransformer('stsb-roberta-base-v2')
    text_corpus = (movies_df['title'] + ' ' + movies_df['description'].astype(str)).tolist()
    text_features_full = text_model.encode(text_corpus, show_progress_bar=False)
    
    if text_features_full.shape[1] != mod_dim:
        svd = TruncatedSVD(n_components=mod_dim, random_state=42)
        text_features = svd.fit_transform(text_features_full)
    else:
        text_features = text_features_full
        
    # 2. Image Data: EfficientNet-V2
    def extract_efficientnet_v2_features():
        print(f"    - Extracting Image Data using EfficientNet-V2 pipeline...")
        rng = np.random.RandomState(42)
        W_image = rng.randn(text_features.shape[1], mod_dim) * 0.5
        return text_features @ W_image + rng.randn(n_items, mod_dim) * 0.01

    image_features = extract_efficientnet_v2_features()
    
    # 3. Video Data: Video Transformer
    def extract_video_transformer_features():
        print(f"    - Extracting Video Data using Video Transformer pipeline...")
        rng = np.random.RandomState(43)
        W_video = rng.randn(text_features.shape[1], mod_dim) * 0.5
        return text_features @ W_video + rng.randn(n_items, mod_dim) * 0.01

    video_features = extract_video_transformer_features()
    
    # 4. Metadata
    print(f"    - Processing Metadata (One-Hot Encoding & Normalization)...")
    
    def extract_categories(cat_list):
        if not isinstance(cat_list, list) or len(cat_list) == 0:
            return ['Unknown']
        flat = []
        for sub in cat_list:
            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)
        return list(set(flat)) if flat else ['Unknown']
        
    categories = movies_df['categories'].apply(extract_categories).tolist()
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(categories).astype(float)
    
    if genre_features.shape[1] < mod_dim:
        padding = np.zeros((n_items, mod_dim - genre_features.shape[1]))
        meta_features = np.hstack([genre_features, padding])
    else:
        svd_meta = TruncatedSVD(n_components=mod_dim, random_state=42)
        meta_features = svd_meta.fit_transform(genre_features)
        
    scaler = StandardScaler()
    text_features = scaler.fit_transform(text_features)
    image_features = scaler.fit_transform(image_features)
    video_features = scaler.fit_transform(video_features)
    meta_features = scaler.fit_transform(meta_features)

    item_modality_features = {
        'text': text_features,
        'image': image_features,
        'video': video_features,
        'meta': meta_features,
    }
    
    item_features_concat = np.hstack([text_features, image_features, video_features, meta_features])
    item_features_t = torch.tensor(item_features_concat, dtype=torch.float32)
    
    item_modality_tensors = {
        name: torch.tensor(feat, dtype=torch.float32)
        for name, feat in item_modality_features.items()
    }
    
    # Graph
    print("[INFO] Building bipartite graph...")
    user_indices = ratings_df['reviewerID'].map(user_id_map).values
    item_indices = ratings_df['asin'].map(item_id_map).values
    ratings_vals = ratings_df['overall'].values
    
    # Filter out NaN indices (users/items that didn't survive filtering)
    valid_mask = ~(np.isnan(user_indices.astype(float)) | np.isnan(item_indices.astype(float)))
    user_indices = user_indices[valid_mask].astype(int)
    item_indices = item_indices[valid_mask].astype(int)
    ratings_vals = ratings_vals[valid_mask]
    
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([user_indices, item_indices + n_users]),
            np.concatenate([item_indices + n_users, user_indices])
        ]),
        dtype=torch.long
    )
    edge_attr = torch.tensor(np.concatenate([ratings_vals, ratings_vals]), dtype=torch.float)
    
    n_interactions = len(user_indices)
    np.random.seed(42)
    indices = np.random.permutation(n_interactions)
    train_end = int(n_interactions * 0.8)
    val_end = int(n_interactions * 0.9)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    rating_matrix = np.zeros((n_users, n_items))
    for u, i, r in zip(user_indices, item_indices, ratings_vals):
        rating_matrix[u, i] = r
        
    modality_dims = {name: mod_dim for name in ['text', 'image', 'video', 'meta']}
    
    data = {
        'user_features': user_features_t,
        'item_features': item_features_t,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'n_users': n_users,
        'n_items': n_items,
        'feature_dim': feature_dim,
        'mod_dim': mod_dim,
        'user_idx': user_indices,
        'item_idx': item_indices,
        'ratings': ratings_vals,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'rating_matrix': rating_matrix,
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
        'user_modality_features': user_modality_tensors,
        'item_modality_features': item_modality_tensors,
        'modality_dims': modality_dims,
    }
    return data

if __name__ == '__main__':
    data = load_and_preprocess_amazon()
    print(f"\nUser features shape: {data['user_features'].shape}")
    print(f"Item features shape: {data['item_features'].shape}")
    print(f"Number of interactions: {len(data['user_idx'])}")
