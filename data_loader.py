"""
Data Loader for MovieLens 1M dataset — MM-CLightRec version.

Handles downloading, parsing, feature extraction (with per-modality separation),
graph construction, and train/val/test splitting.

Changes from base paper:
- Added encode_item_features_multimodal() returning 4 separate modality feature vectors
- Data dict now includes 'modality_features' for per-modality LightGCN channels
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(data_dir=None):
    """Download and extract MovieLens 1M dataset."""
    if data_dir is None:
        data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    
    ml_dir = os.path.join(data_dir, 'ml-1m')
    if os.path.exists(ml_dir):
        print(f"[INFO] MovieLens 1M already exists at {ml_dir}")
        return ml_dir
    
    zip_path = os.path.join(data_dir, 'ml-1m.zip')
    if not os.path.exists(zip_path):
        print("[INFO] Downloading MovieLens 1M dataset...")
        response = requests.get(ML1M_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[INFO] Download complete.")
    
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("[INFO] Extraction complete.")
    
    return ml_dir


def load_ratings(ml_dir):
    """Load ratings.dat → DataFrame."""
    ratings_path = os.path.join(ml_dir, 'ratings.dat')
    ratings = pd.read_csv(
        ratings_path, sep='::', header=None, engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    return ratings


def load_users(ml_dir):
    """Load users.dat → DataFrame."""
    users_path = os.path.join(ml_dir, 'users.dat')
    users = pd.read_csv(
        users_path, sep='::', header=None, engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
        encoding='latin-1'
    )
    return users


def load_movies(ml_dir):
    """Load movies.dat → DataFrame."""
    movies_path = os.path.join(ml_dir, 'movies.dat')
    movies = pd.read_csv(
        movies_path, sep='::', header=None, engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    return movies


def encode_user_features(users_df):
    """
    Encode user demographic features into numerical vectors.
    - gender: binary (M=1, F=0)
    - age: normalized continuous
    - occupation: one-hot encoded
    """
    gender_enc = (users_df['gender'] == 'M').astype(float).values.reshape(-1, 1)
    
    scaler = StandardScaler()
    age_norm = scaler.fit_transform(users_df['age'].values.reshape(-1, 1))
    
    le = LabelEncoder()
    occ_labels = le.fit_transform(users_df['occupation'])
    n_occ = len(le.classes_)
    occ_onehot = np.zeros((len(users_df), n_occ))
    occ_onehot[np.arange(len(users_df)), occ_labels] = 1.0
    
    user_features = np.hstack([gender_enc, age_norm, occ_onehot])
    return user_features


def encode_item_features(movies_df, text_dim=100, visual_dim=2048):
    """
    Encode movie features (legacy — concatenated):
    - genres: multi-hot
    - text: TF-IDF on title+genres (SVD reduced)
    - visual: synthetic random features
    """
    all_genres = set()
    genre_lists = []
    for g in movies_df['genres']:
        glist = g.split('|')
        genre_lists.append(glist)
        all_genres.update(glist)
    
    mlb = MultiLabelBinarizer(classes=sorted(all_genres))
    genre_features = mlb.fit_transform(genre_lists).astype(float)
    
    text_corpus = (movies_df['title'] + ' ' + movies_df['genres'].str.replace('|', ' ', regex=False)).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_corpus)
    
    actual_text_dim = min(text_dim, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=actual_text_dim, random_state=42)
    text_features = svd.fit_transform(tfidf_matrix)
    
    if text_features.shape[1] < text_dim:
        padding = np.zeros((text_features.shape[0], text_dim - text_features.shape[1]))
        text_features = np.hstack([text_features, padding])
    
    np.random.seed(42)
    visual_features = np.random.randn(len(movies_df), visual_dim) * 0.1
    
    item_features = np.hstack([genre_features, text_features, visual_features])
    return item_features


def encode_item_features_multimodal(movies_df, text_dim=100, image_dim=64, video_dim=20, meta_dim=18):
    """
    Encode movie features into 4 SEPARATE modality vectors for MM-CLightRec.

    Note: this implementation is dependency-light and deterministic:
    - text: TF-IDF(title+genres) + SVD → (n_items, text_dim)
    - image: synthetic proxy (seeded) → (n_items, image_dim)
    - video: synthetic proxy (seeded) → (n_items, video_dim)
    - meta: genre multi-hot (padded/truncated) → (n_items, meta_dim)
    """
    n_items = len(movies_df)
    
    # =========================================================================
    # 1. Text Data: TF-IDF + SVD (lightweight proxy)
    # =========================================================================
    print("    - Extracting Text Data using TF-IDF + SVD...")
    text_corpus = (movies_df['title'] + ' ' + movies_df['genres'].str.replace('|', ' ', regex=False)).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text_corpus)
    svd = TruncatedSVD(n_components=min(text_dim, tfidf_matrix.shape[1]), random_state=42)
    text_features = svd.fit_transform(tfidf_matrix)
    if text_features.shape[1] < text_dim:
        text_features = np.hstack([text_features, np.zeros((n_items, text_dim - text_features.shape[1]))])

    # =========================================================================
    # 2. Image Data: load real TMDB features if available, else synthetic proxy
    # =========================================================================
    # image_feat.npy is saved by extract_tmdb_images.py with shape (max_movie_id+1, dim)
    # i.e. row[i] = embedding for raw movie_id=i (row 0 is empty — IDs start at 1).
    # We must look up each movie by its actual movie_id to avoid off-by-one errors.
    image_feat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml-1m', 'image_feat.npy')
    if os.path.exists(image_feat_path):
        print(f"    - Loading REAL Image Features from {image_feat_path}...")
        loaded = np.load(image_feat_path).astype(np.float32)
        raw_dim = loaded.shape[1]

        # --- FIX: align by actual movie_id, not by positional index ---
        # movies_df['movie_id'] gives the original IDs (1-based, non-contiguous)
        movie_ids_ordered = movies_df['movie_id'].values   # shape: (n_items,)
        valid_mask = movie_ids_ordered < loaded.shape[0]
        if valid_mask.all():
            aligned = loaded[movie_ids_ordered]            # correct per-movie lookup
        else:
            # Some movie_ids exceed the array — use mean for those
            aligned = np.zeros((n_items, raw_dim), dtype=np.float32)
            aligned[valid_mask]  = loaded[movie_ids_ordered[valid_mask]]
            mean_emb = loaded[loaded.any(axis=1)].mean(axis=0)
            aligned[~valid_mask] = mean_emb
            print(f"    - WARNING: {(~valid_mask).sum()} movies had IDs beyond array range → filled with mean")

        # Compress to image_dim if needed
        if raw_dim == image_dim:
            image_features = aligned
            print(f"    - Image features: {n_items} movies × {image_dim}D (direct load) ✅")
        else:
            print(f"    - Compressing image features from {raw_dim}D → {image_dim}D via SVD...")
            svd_img = TruncatedSVD(n_components=image_dim, random_state=42)
            image_features = svd_img.fit_transform(aligned)
            print(f"    - Image features: {n_items} movies × {image_dim}D (SVD compressed) ✅")
    else:
        print("    - Generating Image Data (synthetic proxy)...")
        rng_img = np.random.RandomState(42)
        image_features = rng_img.randn(n_items, image_dim) * 0.1


    # =========================================================================
    # 3. Video Data: synthetic proxy (seeded) — no real video for ML-1M
    # =========================================================================
    print("    - Generating Video Data (synthetic proxy)...")
    rng_vid = np.random.RandomState(43)
    video_features = rng_vid.randn(n_items, video_dim) * 0.1


    # =========================================================================
    # 4. Metadata: One-Hot Encoding & Normalization
    # =========================================================================
    print(f"    - Processing Metadata (One-Hot Encoding & Normalization)...")
    all_genres = sorted(set('|'.join(movies_df['genres']).split('|')))
    mlb = MultiLabelBinarizer(classes=all_genres)
    
    genre_features = mlb.fit_transform(movies_df['genres'].str.split('|')).astype(float)
    
    if genre_features.shape[1] < meta_dim:
        padding = np.zeros((n_items, meta_dim - genre_features.shape[1]))
        meta_features = np.hstack([genre_features, padding])
    else:
        meta_features = genre_features[:, :meta_dim]
    
    # Normalize continuous/transformed variables
    scaler = StandardScaler()
    text_features = scaler.fit_transform(text_features)
    image_features = scaler.fit_transform(image_features)
    video_features = scaler.fit_transform(video_features)
    meta_features = scaler.fit_transform(meta_features)
    
    modality_features = {
        'text': text_features,
        'image': image_features,
        'video': video_features,
        'meta': meta_features,
    }
    
    concatenated = np.hstack([text_features, image_features, video_features, meta_features])
    
    return modality_features, concatenated


def build_bipartite_graph(ratings_df, n_users, n_items, user_id_map, item_id_map):
    """
    Build user-item bipartite graph from ratings.
    Items are offset by n_users in the node index.
    """
    user_indices = ratings_df['user_id'].map(user_id_map).values
    item_indices = ratings_df['movie_id'].map(item_id_map).values
    
    valid_mask = ~(np.isnan(user_indices) | np.isnan(item_indices))
    user_indices = user_indices[valid_mask].astype(int)
    item_indices = item_indices[valid_mask].astype(int)
    ratings = ratings_df['rating'].values[valid_mask]
    
    # Bidirectional edges
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([user_indices, item_indices + n_users]),
            np.concatenate([item_indices + n_users, user_indices])
        ]),
        dtype=torch.long
    )
    
    edge_attr = torch.tensor(
        np.concatenate([ratings, ratings]), dtype=torch.float
    )
    
    labels = torch.ones(len(user_indices), dtype=torch.float)
    
    return edge_index, edge_attr, labels, user_indices, item_indices


def create_train_val_test_split(n_interactions, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split interaction indices into train/val/test (8:1:1)."""
    np.random.seed(seed)
    indices = np.random.permutation(n_interactions)
    
    train_end = int(n_interactions * train_ratio)
    val_end = int(n_interactions * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return train_idx, val_idx, test_idx


def create_user_temporal_split(ratings_df, user_idx, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Per-user temporal split (MGRS-HFA-style protocol) with overall 80/10/10:
    - Sort each user's interactions by timestamp.
    - Allocate first ~80% to train, next ~10% to val, last ~10% to test.
    - Enforces at least 1 val + 1 test interaction when user has enough history.
    """
    rng = np.random.RandomState(seed)
    n = len(ratings_df)
    timestamps = ratings_df['timestamp'].values

    user_idx = np.asarray(user_idx)
    order = np.arange(n)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for u in np.unique(user_idx):
        u_pos = order[user_idx == u]
        if len(u_pos) < 5:
            # Too few interactions: keep all in train (avoids unstable tiny val/test)
            train_mask[u_pos] = True
            continue
        # Sort by timestamp; break ties deterministically
        u_ts = timestamps[u_pos]
        tie = rng.permutation(len(u_pos))
        u_sorted = u_pos[np.lexsort((tie, u_ts))]

        m = len(u_sorted)
        train_end = int(np.floor(m * train_ratio))
        val_end = int(np.floor(m * (train_ratio + val_ratio)))

        # Ensure at least 1 in each split when possible
        train_end = max(1, min(train_end, m - 2))
        val_end = max(train_end + 1, min(val_end, m - 1))

        train_mask[u_sorted[:train_end]] = True
        val_mask[u_sorted[train_end:val_end]] = True
        test_mask[u_sorted[val_end:]] = True

    train_idx = order[train_mask]
    val_idx = order[val_mask]
    test_idx = order[test_mask]
    return train_idx, val_idx, test_idx


def select_cold_users_by_first_timestamp(ratings_df, user_idx, cold_fraction=0.2, seed=42):
    """
    Select cold users based on FIRST interaction timestamp (realistic cold-start).
    Picks the latest `cold_fraction` users by first_ts.
    """
    rng = np.random.RandomState(seed)
    user_idx = np.asarray(user_idx)
    ts = ratings_df['timestamp'].values
    users = np.unique(user_idx)

    first_ts = np.full(users.max() + 1, np.inf, dtype=np.float64)
    for u, t in zip(user_idx, ts):
        if t < first_ts[u]:
            first_ts[u] = t

    user_first = np.array([(u, first_ts[u]) for u in users], dtype=np.float64)
    # Break ties deterministically
    tie = rng.permutation(len(user_first))
    order = np.lexsort((tie, user_first[:, 1]))
    sorted_users = user_first[order][:, 0].astype(int)

    n_cold = int(np.ceil(len(sorted_users) * cold_fraction))
    cold_users = sorted_users[-n_cold:]
    warm_users = sorted_users[:-n_cold]
    return warm_users, cold_users


def load_and_preprocess_ml1m(data_dir=None, text_dim=100, image_dim=64, video_dim=20, meta_dim=18):
    """
    Full pipeline: download, load, encode features, build graph, split data.
    
    Returns:
        data dict with all necessary data, including 'modality_features'
    """
    ml_dir = download_movielens_1m(data_dir)
    
    # Load raw data
    ratings_df = load_ratings(ml_dir)
    users_df = load_users(ml_dir)
    movies_df = load_movies(ml_dir)
    
    print(f"[INFO] Loaded {len(ratings_df)} ratings, {len(users_df)} users, {len(movies_df)} movies")
    
    # Create ID mappings
    user_ids = sorted(users_df['user_id'].unique())
    movie_ids = sorted(movies_df['movie_id'].unique())
    
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    n_users = len(user_ids)
    n_items = len(movie_ids)
    
    print(f"[INFO] Users: {n_users}, Items: {n_items}")
    
    # Encode user features
    print("[INFO] Encoding user features...")
    user_features_raw = encode_user_features(users_df)
    
    # Encode item features — MULTIMODAL (4 separate modalities)
    # Note: image_feat_path is resolved once inside encode_item_features_multimodal
    image_feat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml-1m', 'image_feat.npy')
    print("[INFO] Encoding multimodal item features...")
    item_modality_features, item_features_concat = encode_item_features_multimodal(
        movies_df, text_dim=text_dim, image_dim=image_dim, video_dim=video_dim, meta_dim=meta_dim
    )

    # ── Feature source summary ─────────────────────────────────────────────
    print("")
    print("=" * 60)
    if os.path.exists(image_feat_path):
        print("  IMAGE SOURCE : ✅  REAL TMDB Poster Features Loaded")
        print(f"               (from {image_feat_path})")
    else:
        print("  IMAGE SOURCE : ⚠️   Synthetic Random Noise (proxy)")
        print("               Run extract_tmdb_images.py to get real features")
    print("=" * 60)
    print("")
    # The total feature_dim
    feature_dim = text_dim + image_dim + video_dim + meta_dim

    # Project user features to match feature_dim
    np.random.seed(42)
    if user_features_raw.shape[1] < feature_dim:
        W_user = np.random.randn(user_features_raw.shape[1], feature_dim) * 0.01
        user_features = user_features_raw @ W_user
    else:
        user_features = user_features_raw[:, :feature_dim]
    
    # Normalize user features
    scaler_u = StandardScaler()
    user_features = scaler_u.fit_transform(user_features)
    
    # Item features are already normalized per-modality in encode_item_features_multimodal
    item_features = item_features_concat
    
    # Convert to tensors
    user_features_t = torch.tensor(user_features, dtype=torch.float32)
    item_features_t = torch.tensor(item_features, dtype=torch.float32)
    
    # Per-modality tensors
    item_modality_tensors = {
        name: torch.tensor(feat, dtype=torch.float32)
        for name, feat in item_modality_features.items()
    }
    
    # User modality features: split projected user features according to dimensions
    user_modality_tensors = {}
    modality_dims = {'text': text_dim, 'image': image_dim, 'video': video_dim, 'meta': meta_dim}
    
    current_idx = 0
    for name, dim in modality_dims.items():
        user_modality_tensors[name] = user_features_t[:, current_idx:current_idx+dim]
        current_idx += dim
    
    # Build bipartite graph (ALL interactions; we'll derive train edges from splits)
    print("[INFO] Building bipartite graph...")
    edge_index_all, edge_attr_all, labels, user_idx, item_idx = build_bipartite_graph(
        ratings_df, n_users, n_items, user_id_map, item_id_map
    )
    
    # -------------------------------------------------------------------------
    # Cold-start user holdout protocol (CHECK 5)
    # -------------------------------------------------------------------------
    warm_users, cold_users = select_cold_users_by_first_timestamp(
        ratings_df, user_idx, cold_fraction=0.2, seed=42
    )
    warm_set = set(warm_users.tolist())
    cold_set = set(cold_users.tolist())

    is_warm_inter = np.array([u in warm_set for u in user_idx], dtype=bool)
    warm_inter_idx_all = np.where(is_warm_inter)[0]
    cold_inter_idx_all = np.where(~is_warm_inter)[0]

    # Remap warm users to contiguous ids [0..n_warm-1] for true exclusion
    warm_user_map = {u: j for j, u in enumerate(sorted(warm_set))}
    user_idx_warm = np.array([warm_user_map[int(u)] for u in user_idx[warm_inter_idx_all]], dtype=int)
    item_idx_warm = item_idx[warm_inter_idx_all].astype(int)
    ratings_warm = ratings_df['rating'].values[warm_inter_idx_all].astype(float)
    ts_warm = ratings_df['timestamp'].values[warm_inter_idx_all].astype(int)

    # Per-user temporal 80/10/10 split on warm interactions only
    ratings_df_warm = pd.DataFrame({
        "user_id": user_idx_warm,
        "movie_id": item_idx_warm,
        "rating": ratings_warm,
        "timestamp": ts_warm,
    })
    # Bug 3 fix: column names already match — no redundant rename/assign needed
    train_idx, val_idx, test_idx = create_user_temporal_split(
        ratings_df_warm, user_idx_warm, train_ratio=0.8, val_ratio=0.1, seed=42
    )

    # Build TRAIN edge_index only (warm train interactions only)
    n_users_warm = len(warm_user_map)
    train_users = user_idx_warm[train_idx]
    train_items = item_idx_warm[train_idx]
    edge_index_train = torch.tensor(
        np.stack([
            np.concatenate([train_users, train_items + n_users_warm]),
            np.concatenate([train_items + n_users_warm, train_users])
        ]),
        dtype=torch.long
    )
    edge_attr_train = torch.tensor(
        np.concatenate([ratings_warm[train_idx], ratings_warm[train_idx]]),
        dtype=torch.float32
    )

    # Cold-start test-time data: for each cold user, reveal K=5 interactions (earliest), rest is GT
    cold_user_ids_sorted = np.array(sorted(cold_set), dtype=int)
    cold_user_features_full = user_features_t[cold_user_ids_sorted]
    cold_user_modality_full = {k: v[cold_user_ids_sorted] for k, v in user_modality_tensors.items()}

    cold_known = {}
    cold_gt = {}
    for u in cold_user_ids_sorted:
        inter = cold_inter_idx_all[user_idx[cold_inter_idx_all] == u]
        if len(inter) == 0:
            continue
        # sort by timestamp
        inter_ts = ratings_df['timestamp'].values[inter]
        order = np.argsort(inter_ts)
        inter = inter[order]
        known = inter[:5]
        remaining = inter[5:]
        known_items = item_idx[known].astype(int)
        gt_items = item_idx[remaining][ratings_df['rating'].values[remaining] >= 4].astype(int)
        cold_known[int(u)] = known_items
        cold_gt[int(u)] = gt_items
    
    print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Rating matrix for WARM users only (for warm eval)
    rating_matrix = np.zeros((n_users_warm, n_items), dtype=np.float32)
    # fill from warm interactions (all warm interactions)
    for u, i, r in zip(user_idx_warm, item_idx_warm, ratings_warm):
        rating_matrix[u, i] = r
    
    # Modality dimensions dict
    modality_dims = {'text': text_dim, 'image': image_dim, 'video': video_dim, 'meta': meta_dim}
    
    data = {
        # Warm-only training/eval tensors
        'user_features': user_features_t[sorted(warm_set)],
        'item_features': item_features_t,
        'edge_index': edge_index_train,
        'edge_attr': edge_attr_train,
        'edge_index_all': edge_index_all,
        'edge_attr_all': edge_attr_all,
        'n_users': n_users_warm,
        'n_items': n_items,
        'feature_dim': feature_dim,
        'user_idx': user_idx_warm,
        'item_idx': item_idx_warm,
        'ratings': ratings_warm,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'rating_matrix': rating_matrix,
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
        # NEW: per-modality features
        'user_modality_features': {k: v[sorted(warm_set)] for k, v in user_modality_tensors.items()},
        'item_modality_features': item_modality_tensors,
        'modality_dims': modality_dims,
        # Cold-start holdout info
        'cold_users_orig': cold_user_ids_sorted,
        'warm_users_orig': np.array(sorted(warm_set), dtype=int),
        'warm_user_map': warm_user_map,
        'cold_user_features': cold_user_features_full,
        'cold_user_modality_features': cold_user_modality_full,
        'cold_known_items': cold_known,
        'cold_gt_items': cold_gt,
    }
    
    return data


if __name__ == '__main__':
    data = load_and_preprocess_ml1m()
    print(f"\nUser features shape: {data['user_features'].shape}")
    print(f"Item features shape: {data['item_features'].shape}")
    print(f"Edge index shape: {data['edge_index'].shape}")
    print(f"Number of interactions: {len(data['user_idx'])}")
    print(f"\nPer-modality item features:")
    for name, feat in data['item_modality_features'].items():
        print(f"  {name}: {feat.shape}")
    print(f"\nPer-modality user features:")
    for name, feat in data['user_modality_features'].items():
        print(f"  {name}: {feat.shape}")
