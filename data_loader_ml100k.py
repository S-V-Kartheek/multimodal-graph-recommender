"""
Data Loader for MovieLens 100K dataset.

Creates the same data dictionary contract used by train.py:
- train-only edge_index (prevents train/val/test leakage)
- user/item modality features compatible with MM-CLightRec
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ml-100k")


def _load_users(data_dir):
    users_path = os.path.join(data_dir, "u.user")
    return pd.read_csv(
        users_path,
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )


def _load_items(data_dir):
    items_path = os.path.join(data_dir, "u.item")
    cols = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    return pd.read_csv(items_path, sep="|", header=None, names=cols, encoding="latin-1")


def _load_ratings(data_dir):
    ratings_path = os.path.join(data_dir, "u.data")
    return pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def _encode_user_features(users_df):
    gender = (users_df["gender"] == "M").astype(float).values.reshape(-1, 1)
    age = StandardScaler().fit_transform(users_df["age"].values.reshape(-1, 1))
    occ_le = LabelEncoder()
    occ = occ_le.fit_transform(users_df["occupation"])
    occ_onehot = np.zeros((len(users_df), len(occ_le.classes_)), dtype=np.float32)
    occ_onehot[np.arange(len(users_df)), occ] = 1.0
    return np.hstack([gender, age, occ_onehot]).astype(np.float32)


def _encode_item_modalities(items_df, text_dim=100, image_dim=64, video_dim=20, meta_dim=18, seed=42):
    genre_cols = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    genres_text = []
    for _, row in items_df.iterrows():
        active = [g for g in genre_cols if int(row[g]) == 1]
        genres_text.append(" ".join(active) if active else "unknown")

    corpus = (items_df["title"].fillna("") + " " + pd.Series(genres_text)).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_x = tfidf.fit_transform(corpus)
    n_comp = min(text_dim, tfidf_x.shape[1])
    text = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(tfidf_x)
    if text.shape[1] < text_dim:
        text = np.hstack([text, np.zeros((text.shape[0], text_dim - text.shape[1]))])

    rng = np.random.RandomState(seed)
    image = rng.randn(len(items_df), image_dim).astype(np.float32) * 0.1
    video = rng.randn(len(items_df), video_dim).astype(np.float32) * 0.1

    meta_full = items_df[genre_cols].values.astype(np.float32)
    if meta_full.shape[1] < meta_dim:
        meta = np.hstack([meta_full, np.zeros((meta_full.shape[0], meta_dim - meta_full.shape[1]))])
    else:
        meta = meta_full[:, :meta_dim]

    scaler = StandardScaler()
    text = scaler.fit_transform(text).astype(np.float32)
    image = scaler.fit_transform(image).astype(np.float32)
    video = scaler.fit_transform(video).astype(np.float32)
    meta = scaler.fit_transform(meta).astype(np.float32)

    modalities = {"text": text, "image": image, "video": video, "meta": meta}
    return modalities, np.hstack([text, image, video, meta]).astype(np.float32)


def _per_user_temporal_split(ratings_df, user_indices, train_ratio=0.8, val_ratio=0.1):
    train_mask = np.zeros(len(ratings_df), dtype=bool)
    val_mask = np.zeros(len(ratings_df), dtype=bool)
    test_mask = np.zeros(len(ratings_df), dtype=bool)

    for u in np.unique(user_indices):
        idxs = np.where(user_indices == u)[0]
        idxs_sorted = idxs[np.argsort(ratings_df.iloc[idxs]["timestamp"].values)]
        n = len(idxs_sorted)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        train_ids = idxs_sorted[:n_train]
        val_ids = idxs_sorted[n_train:n_train + n_val]
        test_ids = idxs_sorted[n_train + n_val:]
        train_mask[train_ids] = True
        val_mask[val_ids] = True
        test_mask[test_ids] = True

    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def _build_edge_index(user_indices, item_indices, n_users):
    return torch.tensor(
        np.stack(
            [
                np.concatenate([user_indices, item_indices + n_users]),
                np.concatenate([item_indices + n_users, user_indices]),
            ]
        ),
        dtype=torch.long,
    )


def load_and_preprocess_ml100k(data_dir=None, text_dim=100, image_dim=64, video_dim=20, meta_dim=18, seed=42):
    if data_dir is None:
        data_dir = DATA_DIR

    users_df = _load_users(data_dir)
    items_df = _load_items(data_dir)
    ratings_df = _load_ratings(data_dir)

    user_ids = sorted(users_df["user_id"].unique())
    item_ids = sorted(items_df["movie_id"].unique())
    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    item_id_map = {iid: i for i, iid in enumerate(item_ids)}

    ratings_df = ratings_df[ratings_df["user_id"].isin(user_id_map) & ratings_df["movie_id"].isin(item_id_map)].copy()
    ratings_df["u"] = ratings_df["user_id"].map(user_id_map).astype(int)
    ratings_df["i"] = ratings_df["movie_id"].map(item_id_map).astype(int)

    n_users = len(user_ids)
    n_items = len(item_ids)

    users_df = users_df.sort_values("user_id")
    items_df = items_df.sort_values("movie_id")
    user_feat_raw = _encode_user_features(users_df)
    item_modalities, item_feat_concat = _encode_item_modalities(
        items_df, text_dim=text_dim, image_dim=image_dim, video_dim=video_dim, meta_dim=meta_dim, seed=seed
    )
    feature_dim = text_dim + image_dim + video_dim + meta_dim

    rng = np.random.RandomState(seed)
    if user_feat_raw.shape[1] < feature_dim:
        W = rng.randn(user_feat_raw.shape[1], feature_dim).astype(np.float32) * 0.01
        user_feat = user_feat_raw @ W
    else:
        user_feat = user_feat_raw[:, :feature_dim]
    user_feat = StandardScaler().fit_transform(user_feat).astype(np.float32)

    user_features_t = torch.tensor(user_feat, dtype=torch.float32)
    item_features_t = torch.tensor(item_feat_concat, dtype=torch.float32)
    item_modality_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in item_modalities.items()}

    modality_dims = {"text": text_dim, "image": image_dim, "video": video_dim, "meta": meta_dim}
    user_modality_tensors = {}
    pos = 0
    for name, dim in modality_dims.items():
        user_modality_tensors[name] = user_features_t[:, pos:pos + dim]
        pos += dim

    user_idx = ratings_df["u"].values
    item_idx = ratings_df["i"].values
    ratings = ratings_df["rating"].values.astype(np.float32)

    train_idx, val_idx, test_idx = _per_user_temporal_split(ratings_df, user_idx)
    edge_index_train = _build_edge_index(user_idx[train_idx], item_idx[train_idx], n_users)
    edge_index_all = _build_edge_index(user_idx, item_idx, n_users)
    edge_attr = torch.tensor(np.concatenate([ratings[train_idx], ratings[train_idx]]), dtype=torch.float32)

    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in zip(user_idx, item_idx, ratings):
        rating_matrix[u, i] = r

    print(f"[INFO] ML-100K loaded: users={n_users}, items={n_items}, interactions={len(user_idx)}")
    print(f"[INFO] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return {
        "user_features": user_features_t,
        "item_features": item_features_t,
        "edge_index": edge_index_train,
        "edge_index_all": edge_index_all,
        "edge_attr": edge_attr,
        "n_users": n_users,
        "n_items": n_items,
        "feature_dim": feature_dim,
        "user_idx": user_idx,
        "item_idx": item_idx,
        "ratings": ratings,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "rating_matrix": rating_matrix,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "user_modality_features": user_modality_tensors,
        "item_modality_features": item_modality_tensors,
        "modality_dims": modality_dims,
    }

