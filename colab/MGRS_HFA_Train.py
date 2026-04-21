"""
MGRS-HFA: Multimodal Graph-based Recommendation System using Hybrid Filtering Approach
======================================================================================
Google Colab GPU Training Script

INSTRUCTIONS:
1. Upload this file to Google Colab
2. Set Runtime > Change runtime type > GPU (T4 or better)
3. Run all cells or execute: !python MGRS_HFA_Train.py
4. After training completes, download the 'results/' folder
5. Place the downloaded 'results/' folder into your local project at:
   c:\\code playground\\MGRS1\\results\\

This script is self-contained - it includes all model code, data loading,
training, evaluation, and visualization.
"""

# ============================================================================
# SECTION 0: Install Dependencies (run this cell first in Colab)
# ============================================================================
"""
# Run these commands in a Colab cell BEFORE running this script:

!pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -q
!pip install pandas scikit-learn matplotlib tqdm requests -q
"""

import os
import sys
import time
import math
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, MixHopConv
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: Configuration
# ============================================================================

CONFIG = {
    # Dataset
    'data_dir': './data',
    'feature_dim': 200,
    'text_dim': 100,
    'visual_dim': 2048,
    
    # Model Architecture
    'cf_hidden_dim': 128,
    'cf_out_dim': 32,
    'cbf_hidden_dim': 60,
    'cbf_out_dim': 32,
    'n_user_clusters': 20,
    'n_item_clusters': 15,
    'vgae_hidden_dim': 100,
    'vgae_latent_dim': 50,
    
    # Training
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 4096,
    'n_neg': 1,
    'user_sim_threshold': 0.1,
    'item_sim_threshold': 0.1,
    'user_top_k': 20,
    'item_top_k': 20,
    
    # Evaluation
    'k': 10,
    'eval_every': 10,
    'n_eval_users': 500,
    
    # Other
    'seed': 42,
    'results_dir': './results',
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# SECTION 2: Data Loading & Preprocessing
# ============================================================================

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    ml_dir = os.path.join(data_dir, 'ml-1m')
    if os.path.exists(ml_dir):
        print(f"[DATA] MovieLens 1M already at {ml_dir}")
        return ml_dir

    zip_path = os.path.join(data_dir, 'ml-1m.zip')
    if not os.path.exists(zip_path):
        print("[DATA] Downloading MovieLens 1M...")
        resp = requests.get(ML1M_URL, stream=True)
        resp.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        print("[DATA] Download complete.")

    print("[DATA] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    print("[DATA] Extraction complete.")
    return ml_dir


def load_and_preprocess(config):
    ml_dir = download_movielens_1m(config['data_dir'])

    # Load raw files
    ratings = pd.read_csv(os.path.join(ml_dir, 'ratings.dat'), sep='::', header=None,
                          engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          encoding='latin-1')
    users = pd.read_csv(os.path.join(ml_dir, 'users.dat'), sep='::', header=None,
                        engine='python', names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                        encoding='latin-1')
    movies = pd.read_csv(os.path.join(ml_dir, 'movies.dat'), sep='::', header=None,
                         engine='python', names=['movie_id', 'title', 'genres'],
                         encoding='latin-1')

    print(f"[DATA] Loaded {len(ratings)} ratings, {len(users)} users, {len(movies)} movies")

    # ID mappings
    user_ids = sorted(users['user_id'].unique())
    movie_ids = sorted(movies['movie_id'].unique())
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
    n_users, n_items = len(user_ids), len(movie_ids)

    # --- User features ---
    gender_enc = (users['gender'] == 'M').astype(float).values.reshape(-1, 1)
    age_norm = StandardScaler().fit_transform(users['age'].values.reshape(-1, 1))
    occ_labels = LabelEncoder().fit_transform(users['occupation'])
    occ_onehot = np.zeros((len(users), occ_labels.max() + 1))
    occ_onehot[np.arange(len(users)), occ_labels] = 1.0
    user_feat_raw = np.hstack([gender_enc, age_norm, occ_onehot])

    # --- Item features ---
    genre_lists = [g.split('|') for g in movies['genres']]
    all_genres = sorted(set(g for gl in genre_lists for g in gl))
    mlb = MultiLabelBinarizer(classes=all_genres)
    genre_feat = mlb.fit_transform(genre_lists).astype(float)

    text_corpus = (movies['title'] + ' ' + movies['genres'].str.replace('|', ' ', regex=False)).tolist()
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_mat = tfidf.fit_transform(text_corpus)
    text_dim = min(config['text_dim'], tfidf_mat.shape[1])
    text_feat = TruncatedSVD(n_components=text_dim, random_state=42).fit_transform(tfidf_mat)
    if text_feat.shape[1] < config['text_dim']:
        text_feat = np.hstack([text_feat, np.zeros((len(movies), config['text_dim'] - text_feat.shape[1]))])

    np.random.seed(42)
    visual_feat = np.random.randn(len(movies), config['visual_dim']) * 0.1
    item_feat_raw = np.hstack([genre_feat, text_feat, visual_feat])

    # --- Project to common feature dim ---
    fd = config['feature_dim']
    np.random.seed(42)
    if user_feat_raw.shape[1] < fd:
        W = np.random.randn(user_feat_raw.shape[1], fd) * 0.01
        user_feat = user_feat_raw @ W
    else:
        user_feat = user_feat_raw[:, :fd]

    if item_feat_raw.shape[1] > fd:
        item_feat = TruncatedSVD(n_components=fd, random_state=42).fit_transform(item_feat_raw)
    else:
        W = np.random.randn(item_feat_raw.shape[1], fd) * 0.01
        item_feat = item_feat_raw @ W

    user_feat = StandardScaler().fit_transform(user_feat)
    item_feat = StandardScaler().fit_transform(item_feat)

    user_feat_t = torch.tensor(user_feat, dtype=torch.float32)
    item_feat_t = torch.tensor(item_feat, dtype=torch.float32)

    # --- Bipartite graph ---
    u_idx = ratings['user_id'].map(user_id_map).values.astype(int)
    i_idx = ratings['movie_id'].map(item_id_map).values.astype(int)
    edge_index = torch.tensor(
        np.stack([np.concatenate([u_idx, i_idx + n_users]),
                  np.concatenate([i_idx + n_users, u_idx])]),
        dtype=torch.long
    )

    # --- Train/val/test split (8:1:1) ---
    np.random.seed(42)
    perm = np.random.permutation(len(u_idx))
    tr_end = int(len(u_idx) * 0.8)
    va_end = int(len(u_idx) * 0.9)
    train_idx, val_idx, test_idx = perm[:tr_end], perm[tr_end:va_end], perm[va_end:]

    # Rating matrix
    rating_matrix = np.zeros((n_users, n_items))
    for u, i, r in zip(u_idx, i_idx, ratings['rating'].values):
        rating_matrix[u, i] = r

    print(f"[DATA] Users={n_users}, Items={n_items}, Interactions={len(u_idx)}")
    print(f"[DATA] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return {
        'user_features': user_feat_t, 'item_features': item_feat_t,
        'edge_index': edge_index, 'n_users': n_users, 'n_items': n_items,
        'feature_dim': fd, 'user_idx': u_idx, 'item_idx': i_idx,
        'ratings': ratings['rating'].values,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'rating_matrix': rating_matrix,
    }


# ============================================================================
# SECTION 3: Model — Collaborative Filtering (UserGCN + ItemGAT)
# ============================================================================

def build_sim_graph(features, threshold=0.1, top_k=20):
    """Build similarity graph via cosine similarity (batched for memory)."""
    n = features.shape[0]
    norms = torch.norm(features, dim=1, keepdim=True).clamp(min=1e-8)
    normed = features / norms
    
    src, dst = [], []
    bs = 512
    for i in range(0, n, bs):
        end = min(i + bs, n)
        sim = torch.mm(normed[i:end], normed.t())
        for j in range(i, end):
            sim[j - i, j] = 0  # no self-loops
        k = min(top_k, n - 1)
        vals, indices = torch.topk(sim, k=k, dim=1)
        for li in range(end - i):
            gi = i + li
            mask = vals[li] > threshold
            nbrs = indices[li][mask]
            if len(nbrs) > 0:
                src.extend([gi] * len(nbrs))
                dst.extend(nbrs.tolist())
    
    if not src:
        # fallback: connect to top-5 regardless of threshold
        for i in range(0, n, bs):
            end = min(i + bs, n)
            sim = torch.mm(normed[i:end], normed.t())
            for j in range(i, end):
                sim[j - i, j] = -1
            _, indices = torch.topk(sim, k=min(5, n-1), dim=1)
            for li in range(end - i):
                gi = i + li
                nbrs = indices[li]
                src.extend([gi] * len(nbrs))
                dst.extend(nbrs.tolist())
    
    return torch.tensor([src, dst], dtype=torch.long)


class UserGCN(nn.Module):
    def __init__(self, in_ch, hid=128, out_ch=32):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid)
        self.conv2 = GCNConv(hid, hid)
        self.fc = nn.Linear(hid, out_ch)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), 0.5, self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.fc(h)


class ItemGAT(nn.Module):
    def __init__(self, in_ch, hid=128, out_ch=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid // heads, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hid, hid, heads=1, concat=False, dropout=0.6)
        self.fc = nn.Linear(hid, out_ch)

    def forward(self, x, edge_index):
        h = F.dropout(F.elu(self.conv1(x, edge_index)), 0.6, self.training)
        h = F.elu(self.conv2(h, edge_index))
        return self.fc(h)


# ============================================================================
# SECTION 4: Model — Content Filtering (K-Means + MixHopConv)
# ============================================================================

def cluster_and_build_graph(user_feat, item_feat, n_uc=20, n_ic=15):
    """K-means cluster users & items, build cluster similarity graph."""
    uf = user_feat.detach().cpu().numpy()
    itf = item_feat.detach().cpu().numpy()
    
    n_uc = min(n_uc, len(uf))
    n_ic = min(n_ic, len(itf))

    km_u = KMeans(n_clusters=n_uc, n_init=10, random_state=42).fit(uf)
    km_i = KMeans(n_clusters=n_ic, n_init=10, random_state=42).fit(itf)

    u_labels, u_cents = km_u.labels_, km_u.cluster_centers_
    i_labels, i_cents = km_i.labels_, km_i.cluster_centers_

    # Cosine sim between cluster centroids
    uc = torch.tensor(u_cents, dtype=torch.float32)
    ic = torch.tensor(i_cents, dtype=torch.float32)
    uc_n = F.normalize(uc, dim=1)
    ic_n = F.normalize(ic, dim=1)
    sim = torch.mm(uc_n, ic_n.t())

    src, dst, wt = [], [], []
    for j in range(n_uc):
        for k in range(n_ic):
            s = sim[j, k].item()
            if s > 0:
                src.extend([j, n_uc + k])
                dst.extend([n_uc + k, j])
                wt.extend([s, s])
    
    if not src:  # fallback
        for j in range(n_uc):
            for k in range(n_ic):
                s = max(sim[j, k].item(), 0.01)
                src.extend([j, n_uc + k])
                dst.extend([n_uc + k, j])
                wt.extend([s, s])

    edge_idx = torch.tensor([src, dst], dtype=torch.long)
    cluster_feat = torch.tensor(np.vstack([u_cents, i_cents]), dtype=torch.float32)
    
    return edge_idx, cluster_feat, u_labels, i_labels, n_uc, n_ic


class ContentFilteringGNN(nn.Module):
    def __init__(self, in_ch, hid=60, out_ch=32):
        super().__init__()
        mh_out = hid * 3  # 180
        self.mh1 = MixHopConv(in_ch, hid, powers=[0, 1, 2])
        self.bn1 = nn.BatchNorm1d(mh_out)
        self.mh2 = MixHopConv(mh_out, hid, powers=[0, 1, 2])
        self.bn2 = nn.BatchNorm1d(mh_out)
        self.mh3 = MixHopConv(mh_out, hid, powers=[0, 1, 2])
        self.bn3 = nn.BatchNorm1d(mh_out)
        self.gcn = GCNConv(mh_out, mh_out)
        self.fc = nn.Linear(mh_out, out_ch)

    def forward(self, x, edge_index):
        h = F.dropout(x, 0.7, self.training)
        h = F.dropout(F.relu(self.bn1(self.mh1(h, edge_index))), 0.9, self.training)
        h = F.dropout(F.relu(self.bn2(self.mh2(h, edge_index))), 0.9, self.training)
        h = F.dropout(F.relu(self.bn3(self.mh3(h, edge_index))), 0.9, self.training)
        h = F.relu(self.gcn(h, edge_index))
        return self.fc(h)


class ContentFilteringModule(nn.Module):
    def __init__(self, feat_dim, n_uc=20, n_ic=15, hid=60, out=32):
        super().__init__()
        self.n_uc, self.n_ic = n_uc, n_ic
        self.gnn = ContentFilteringGNN(feat_dim, hid, out)
        self.u_proj = nn.Linear(out + feat_dim, out)
        self.i_proj = nn.Linear(out + feat_dim, out)

    def forward(self, user_feat, item_feat):
        device = user_feat.device
        edge_idx, cluster_feat, u_labels, i_labels, n_uc, n_ic = \
            cluster_and_build_graph(user_feat, item_feat, self.n_uc, self.n_ic)
        edge_idx = edge_idx.to(device)
        cluster_feat = cluster_feat.to(device)

        cluster_emb = self.gnn(cluster_feat, edge_idx)
        u_cluster_emb = cluster_emb[:n_uc]
        i_cluster_emb = cluster_emb[n_uc:]

        u_lab = torch.tensor(u_labels, dtype=torch.long, device=device)
        i_lab = torch.tensor(i_labels, dtype=torch.long, device=device)

        h_u = self.u_proj(torch.cat([u_cluster_emb[u_lab], user_feat], dim=1))
        h_i = self.i_proj(torch.cat([i_cluster_emb[i_lab], item_feat], dim=1))
        return h_u, h_i


# ============================================================================
# SECTION 5: Model — Cross-Attention
# ============================================================================

class CrossAttention(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.scale = math.sqrt(d)
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)

    def forward(self, H_U, H_I, H_UI_u, H_UI_i):
        Q = self.Wq(H_U)
        K = self.Wk(H_I)
        Vu = self.Wv(H_UI_u)
        Vi = self.Wv(H_UI_i)

        A = F.softmax(torch.mm(Q, K.t()) / self.scale, dim=-1)

        Zu = self.Wo(torch.mm(A, Vi)) + H_U + H_UI_u
        Zi = self.Wo(torch.mm(A.t(), Vu)) + H_I + H_UI_i
        return Zu, Zi


# ============================================================================
# SECTION 6: Model — VGAE
# ============================================================================

class VGAEEncoder(nn.Module):
    def __init__(self, in_ch, hid=100, lat=50):
        super().__init__()
        self.base = GCNConv(in_ch, hid)
        self.mu_conv = GCNConv(hid, lat)
        self.logvar_conv = GCNConv(hid, lat)

    def forward(self, x, edge_index):
        h = F.relu(self.base(x, edge_index))
        return self.mu_conv(h, edge_index), self.logvar_conv(h, edge_index)


class VGAEModule(nn.Module):
    def __init__(self, in_ch, hid=100, lat=50):
        super().__init__()
        self.encoder = VGAEEncoder(in_ch, hid, lat)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def encode(self, x, ei):
        mu, lv = self.encoder(x, ei)
        return self.reparameterize(mu, lv), mu, lv

    def decode_pairs(self, z, u_idx, i_idx):
        return (z[u_idx] * z[i_idx]).sum(dim=1)

    def forward(self, x, ei, u_idx, i_idx):
        z, mu, lv = self.encode(x, ei)
        logits = self.decode_pairs(z, u_idx, i_idx)
        return logits, mu, lv, z

    @staticmethod
    def loss(logits, labels, mu, logvar):
        recon = F.binary_cross_entropy_with_logits(logits, labels)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon + kl, recon, kl


# ============================================================================
# SECTION 7: Full MGRS-HFA Model
# ============================================================================

class MGRS_HFA(nn.Module):
    def __init__(self, feat_dim, cf_hid=128, cf_out=32, cbf_hid=60, cbf_out=32,
                 n_uc=20, n_ic=15, vgae_hid=100, vgae_lat=50):
        super().__init__()
        self.user_gcn = UserGCN(feat_dim, cf_hid, cf_out)
        self.item_gat = ItemGAT(feat_dim, cf_hid, cf_out)
        self.cbf = ContentFilteringModule(feat_dim, n_uc, n_ic, cbf_hid, cbf_out)
        self.cross_attn = CrossAttention(cf_out)
        self.vgae = VGAEModule(cf_out, vgae_hid, vgae_lat)
        self.cf_out = cf_out

    def forward(self, user_feat, item_feat, bi_ei, u_idx, i_idx,
                user_ei, item_ei):
        n_users = user_feat.shape[0]

        # 1. Collaborative Filtering
        H_U = self.user_gcn(user_feat, user_ei)
        H_I = self.item_gat(item_feat, item_ei)

        # 2. Content-Based Filtering
        H_UI_u, H_UI_i = self.cbf(user_feat, item_feat)

        # 3. Cross-Attention
        Z_u, Z_i = self.cross_attn(H_U, H_I, H_UI_u, H_UI_i)

        # 4. VGAE
        combined = torch.cat([Z_u, Z_i], dim=0)
        i_idx_offset = i_idx + n_users
        logits, mu, lv, z = self.vgae(combined, bi_ei, u_idx, i_idx_offset)
        return logits, mu, lv, z, Z_u, Z_i


# ============================================================================
# SECTION 8: Evaluation Metrics
# ============================================================================

def precision_at_k(pred, actual, k=10):
    return len(set(pred[:k]) & set(actual)) / k if k else 0

def recall_at_k(pred, actual, k=10):
    return len(set(pred[:k]) & set(actual)) / len(actual) if actual else 0

def ndcg_at_k(pred, actual, k=10):
    s = set(actual)
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(pred[:k]) if x in s)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg else 0

def f1_at_k(pred, actual, k=10):
    p, r = precision_at_k(pred, actual, k), recall_at_k(pred, actual, k)
    return 2 * p * r / (p + r) if (p + r) else 0

def evaluate_recommendations(score_mat, rating_mat, train_mask, k=10):
    n_users = score_mat.shape[0]
    prec, rec, ndcg, f1, acc = [], [], [], [], []

    for u in range(n_users):
        actual = np.where((rating_mat[u] >= 4) & (~train_mask[u]))[0]
        if len(actual) == 0:
            continue
        scores = score_mat[u].copy()
        scores[train_mask[u]] = -np.inf
        pred = np.argsort(-scores)[:k]

        prec.append(precision_at_k(pred, actual, k))
        rec.append(recall_at_k(pred, actual, k))
        ndcg.append(ndcg_at_k(pred, actual, k))
        f1.append(f1_at_k(pred, actual, k))
        acc.append(len(set(pred[:k]) & set(actual)) / k)

    # RMSE on test ratings
    test_mask = ~train_mask
    test_ratings = rating_mat[test_mask]
    test_scores = score_mat[test_mask] * 5  # scale to 1-5
    valid = test_ratings > 0
    rmse = np.sqrt(np.mean((test_scores[valid] - test_ratings[valid]) ** 2)) if valid.any() else 0

    return {
        'Precision@K': np.mean(prec) if prec else 0,
        'Recall@K': np.mean(rec) if rec else 0,
        'NDCG@K': np.mean(ndcg) if ndcg else 0,
        'F1-Score@K': np.mean(f1) if f1 else 0,
        'Accuracy@K': np.mean(acc) if acc else 0,
        'RMSE': rmse,
    }


# ============================================================================
# SECTION 9: Visualization
# ============================================================================

def plot_training_loss(losses, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses)+1), losses, alpha=0.3, color='brown', label='Training Loss')
    w = max(5, len(losses) // 20)
    if len(losses) >= w:
        ma = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax.plot(range(w//2+1, w//2+1+len(ma)), ma, color='darkred', lw=2, label='Mean Training Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss - MGRS-HFA on MovieLens 1M')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[VIS] Saved {path}")


def plot_metrics_bar(metrics, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [k for k in metrics if k != 'RMSE']
    vals = [metrics[k] for k in names]
    colors = ['#8B4513', '#D2691E', '#DAA520', '#2E8B57', '#4682B4']
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor='black', lw=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{v:.4f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('MGRS-HFA Performance on MovieLens 1M')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[VIS] Saved {path}")


def plot_epoch_metrics(history, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metric_names = ['Precision@K', 'Recall@K', 'NDCG@K', 'F1-Score@K', 'Accuracy@K', 'RMSE']
    colors = ['#8B4513', '#D2691E', '#DAA520', '#2E8B57', '#4682B4', '#6A5ACD']
    for idx, (mn, c) in enumerate(zip(metric_names, colors)):
        ax = axes[idx // 3, idx % 3]
        if mn in history and history[mn]:
            ax.plot(range(1, len(history[mn])+1), history[mn], color=c, lw=1.5, marker='o', ms=3)
        ax.set_title(mn); ax.set_xlabel('Eval Step'); ax.grid(alpha=0.3)
    plt.suptitle('Metrics Over Training', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[VIS] Saved {path}")


# ============================================================================
# SECTION 10: Training Loop
# ============================================================================

def train(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[TRAIN] Device: {device}")
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load data
    data = load_and_preprocess(config)
    user_feat = data['user_features'].to(device)
    item_feat = data['item_features'].to(device)
    bi_ei = data['edge_index'].to(device)
    n_users, n_items = data['n_users'], data['n_items']
    fd = data['feature_dim']
    u_idx, i_idx = data['user_idx'], data['item_idx']
    train_idx, val_idx, test_idx = data['train_idx'], data['val_idx'], data['test_idx']
    rating_mat = data['rating_matrix']

    # Build collaboration graphs
    print("[TRAIN] Building collaboration graphs...")
    with torch.no_grad():
        user_ei = build_sim_graph(user_feat, config['user_sim_threshold'], config['user_top_k']).to(device)
        item_ei = build_sim_graph(item_feat, config['item_sim_threshold'], config['item_top_k']).to(device)
    print(f"[TRAIN] User graph edges: {user_ei.shape[1]}, Item graph edges: {item_ei.shape[1]}")

    # Model
    model = MGRS_HFA(fd, config['cf_hidden_dim'], config['cf_out_dim'],
                     config['cbf_hidden_dim'], config['cbf_out_dim'],
                     config['n_user_clusters'], config['n_item_clusters'],
                     config['vgae_hidden_dim'], config['vgae_latent_dim']).to(device)
    print(f"[TRAIN] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Training mask
    train_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in train_idx:
        train_mask[u_idx[idx], i_idx[idx]] = True

    losses = []
    history = {k: [] for k in ['Precision@K', 'Recall@K', 'NDCG@K', 'F1-Score@K', 'Accuracy@K', 'RMSE']}
    best_f1 = 0
    best_state = None
    k = config['k']
    bs = config['batch_size']
    n_neg = config['n_neg']

    print(f"\n[TRAIN] Starting {config['epochs']} epochs...")
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        t0 = time.time()

        # Positive samples from training set
        pos_u = u_idx[train_idx]
        pos_i = i_idx[train_idx]

        # Negative sampling
        neg_i = np.random.randint(0, n_items, len(pos_u) * n_neg)
        neg_u = np.repeat(pos_u, n_neg)
        all_u = np.concatenate([pos_u, neg_u])
        all_i = np.concatenate([pos_i, neg_i])
        all_labels = np.concatenate([np.ones(len(pos_u)), np.zeros(len(neg_i))])

        perm = np.random.permutation(len(all_u))
        all_u, all_i, all_labels = all_u[perm], all_i[perm], all_labels[perm]

        total_loss, n_batch = 0, 0
        for start in range(0, len(all_u), bs):
            end = min(start + bs, len(all_u))
            bu = torch.tensor(all_u[start:end], dtype=torch.long, device=device)
            bi = torch.tensor(all_i[start:end], dtype=torch.long, device=device)
            bl = torch.tensor(all_labels[start:end], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits, mu, lv, z, _, _ = model(user_feat, item_feat, bi_ei, bu, bi, user_ei, item_ei)
            loss, rl, kl = VGAEModule.loss(logits, bl, mu, lv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)
        losses.append(avg_loss)
        dt = time.time() - t0

        # Evaluation
        do_eval = (epoch % config['eval_every'] == 0) or epoch == 1 or epoch == config['epochs']
        if do_eval:
            model.eval()
            with torch.no_grad():
                # Compute full score matrix efficiently
                _, _, _, _, Z_u, Z_i = model(
                    user_feat, item_feat, bi_ei,
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.zeros(1, dtype=torch.long, device=device),
                    user_ei, item_ei
                )
                # Get VGAE latent
                combined = torch.cat([Z_u, Z_i], dim=0)
                z, mu_all, lv_all = model.vgae.encode(combined, bi_ei)
                z_users = z[:n_users]
                z_items = z[n_users:]

                # Score matrix: sigmoid(z_u^T * z_i)
                n_eval = min(config['n_eval_users'], n_users)
                eval_users = np.random.RandomState(42).choice(n_users, n_eval, replace=False)
                score_mat = np.zeros((n_users, n_items))
                
                # Batch scoring
                for us in range(0, n_eval, 200):
                    ue = min(us + 200, n_eval)
                    u_batch = eval_users[us:ue]
                    scores = torch.sigmoid(torch.mm(z_users[u_batch], z_items.t()))
                    score_mat[u_batch] = scores.cpu().numpy()

                metrics = evaluate_recommendations(score_mat, rating_mat, train_mask, k)

            for mn in history:
                history[mn].append(metrics.get(mn, 0))

            if metrics['F1-Score@K'] > best_f1:
                best_f1 = metrics['F1-Score@K']
                best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}

            print(f"Epoch {epoch:3d}/{config['epochs']} | Loss: {avg_loss:.4f} | {dt:.1f}s | "
                  f"P@{k}: {metrics['Precision@K']:.4f} | R@{k}: {metrics['Recall@K']:.4f} | "
                  f"NDCG@{k}: {metrics['NDCG@K']:.4f} | F1@{k}: {metrics['F1-Score@K']:.4f} | "
                  f"Acc@{k}: {metrics['Accuracy@K']:.4f} | RMSE: {metrics['RMSE']:.4f}")
        else:
            print(f"Epoch {epoch:3d}/{config['epochs']} | Loss: {avg_loss:.4f} | {dt:.1f}s")

    # Load best model
    if best_state:
        model.load_state_dict({k_: v.to(device) for k_, v in best_state.items()})

    # ===== FINAL TEST EVALUATION =====
    print("\n" + "=" * 80)
    print("  FINAL TEST EVALUATION")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        _, _, _, _, Z_u, Z_i = model(
            user_feat, item_feat, bi_ei,
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            user_ei, item_ei
        )
        combined = torch.cat([Z_u, Z_i], dim=0)
        z, _, _ = model.vgae.encode(combined, bi_ei)
        z_users = z[:n_users]
        z_items = z[n_users:]

        # Full score matrix for all users
        score_mat = np.zeros((n_users, n_items))
        for us in tqdm(range(0, n_users, 500), desc="Scoring"):
            ue = min(us + 500, n_users)
            scores = torch.sigmoid(torch.mm(z_users[us:ue], z_items.t()))
            score_mat[us:ue] = scores.cpu().numpy()

        # test mask = train + val
        test_mask = np.zeros((n_users, n_items), dtype=bool)
        for idx in np.concatenate([train_idx, val_idx]):
            test_mask[u_idx[idx], i_idx[idx]] = True

        final = evaluate_recommendations(score_mat, rating_mat, test_mask, k)

    # Print results
    paper_targets = {
        'Precision@K': 0.8269, 'Recall@K': 0.8718, 'NDCG@K': 0.6844,
        'F1-Score@K': 0.8484, 'Accuracy@K': 0.5182, 'RMSE': 0.8496
    }
    print(f"\n{'Metric':<20} {'Our Result':<15} {'Paper Target':<15}")
    print("-" * 50)
    for mn, val in final.items():
        tgt = paper_targets.get(mn, '-')
        print(f"{mn:<20} {val:<15.4f} {f'{tgt:.4f}' if isinstance(tgt, float) else tgt:<15}")

    # ===== SAVE RESULTS =====
    rdir = config['results_dir']
    
    # Plots
    plot_training_loss(losses, os.path.join(rdir, 'training_loss_ml1m.png'))
    plot_metrics_bar(final, os.path.join(rdir, 'metrics_ml1m.png'))
    plot_epoch_metrics(history, os.path.join(rdir, 'metrics_over_epochs_ml1m.png'))

    # Save model
    torch.save(model.state_dict(), os.path.join(rdir, 'mgrs_hfa_ml1m.pth'))
    print(f"[SAVE] Model saved to {rdir}/mgrs_hfa_ml1m.pth")

    # Save metrics to text file
    with open(os.path.join(rdir, 'final_metrics.txt'), 'w') as f:
        f.write("MGRS-HFA Final Test Results on MovieLens 1M\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Metric':<20} {'Our Result':<15} {'Paper Target':<15}\n")
        f.write("-" * 50 + "\n")
        for mn, val in final.items():
            tgt = paper_targets.get(mn, '-')
            f.write(f"{mn:<20} {val:<15.4f} {f'{tgt:.4f}' if isinstance(tgt, float) else tgt:<15}\n")
        f.write(f"\nTraining Epochs: {config['epochs']}\n")
        f.write(f"Final Training Loss: {losses[-1]:.4f}\n")
        f.write(f"Device: {device}\n")
    print(f"[SAVE] Metrics saved to {rdir}/final_metrics.txt")

    # Save training loss history
    np.save(os.path.join(rdir, 'training_losses.npy'), np.array(losses))
    print(f"[SAVE] Loss history saved to {rdir}/training_losses.npy")

    print(f"\n[DONE] All results saved to: {rdir}/")
    print("Download this folder and place it at: c:\\code playground\\MGRS1\\results\\")


# ============================================================================
# SECTION 11: Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("  MGRS-HFA: Multimodal Graph-based Recommendation System")
    print("  using Hybrid Filtering Approach")
    print("  Dataset: MovieLens 1M")
    print("=" * 80)
    train(CONFIG)
