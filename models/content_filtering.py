"""
Content-Based Filtering Module — MM-CLightRec.

Replaces the base paper's single MixHopConv (treating all modalities as one fused
vector) with **4 independent modality-specific LightGCN channels** on the cluster
similarity graph, followed by **learnable adaptive fusion** via softmax weights.

Per-modality channels:
    e_text  = LightGCN(G_cluster, f_text)
    e_image = LightGCN(G_cluster, f_image)
    e_video = LightGCN(G_cluster, f_video)
    e_meta  = LightGCN(G_cluster, f_meta)

Adaptive fusion:
    [α, β, γ, δ] = softmax(W_fuse · [e_text, e_image, e_video, e_meta])
    H_UI = α·e_text + β·e_image + γ·e_video + δ·e_meta

K-means clustering is RETAINED from the base paper (unchanged).

Change 2 from MM-CLightRec architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from sklearn.cluster import KMeans
import numpy as np


def cluster_features(features, n_clusters=10, random_state=42):
    """
    Apply K-means clustering to features (UNCHANGED from base paper).
    
    Args:
        features: numpy array or tensor of shape (n, d)
        n_clusters: number of clusters
    
    Returns:
        labels: cluster assignments (n,)
        centroids: cluster centroids (n_clusters, d)
    """
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = features
    
    n_clusters = min(n_clusters, len(features_np))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_np)
    centroids = kmeans.cluster_centers_
    
    return labels, centroids


def build_cluster_similarity_graph(user_labels, item_labels, user_centroids, item_centroids,
                                    threshold=0.0):
    """
    Build a cluster similarity graph between user and item clusters (UNCHANGED from base paper).
    
    Nodes: user clusters (0..n_uc-1) + item clusters (n_uc..total-1)
    Edges: connect cluster j to cluster k if cosine_sim > threshold
    
    Returns:
        edge_index, similarity_weights, n_user_clusters, n_item_clusters
    """
    user_centroids_t = torch.tensor(user_centroids, dtype=torch.float32)
    item_centroids_t = torch.tensor(item_centroids, dtype=torch.float32)
    
    u_norm = F.normalize(user_centroids_t, dim=1)
    i_norm = F.normalize(item_centroids_t, dim=1)
    
    sim_matrix = torch.mm(u_norm, i_norm.t())
    
    n_user_clusters = user_centroids_t.shape[0]
    n_item_clusters = item_centroids_t.shape[0]
    
    src_list, dst_list, weight_list = [], [], []
    
    for j in range(n_user_clusters):
        for k in range(n_item_clusters):
            sim = sim_matrix[j, k].item()
            if sim > threshold:
                src_list.extend([j, n_user_clusters + k])
                dst_list.extend([n_user_clusters + k, j])
                weight_list.extend([sim, sim])
    
    if not src_list:
        for j in range(n_user_clusters):
            for k in range(n_item_clusters):
                sim = sim_matrix[j, k].item()
                src_list.extend([j, n_user_clusters + k])
                dst_list.extend([n_user_clusters + k, j])
                weight_list.extend([max(sim, 0.01), max(sim, 0.01)])
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    similarity_weights = torch.tensor(weight_list, dtype=torch.float32)
    
    return edge_index, similarity_weights, n_user_clusters, n_item_clusters


class ModalityLightGCN(nn.Module):
    """
    Lightweight LightGCN for a single modality channel on the cluster graph.
    
    No learnable weight matrices per layer, no activations.
    Pure neighborhood aggregation with symmetric normalization + layer combination.
    
    Args:
        in_dim: Input feature dimension for this modality
        out_dim: Output embedding dimension
        n_layers: Number of propagation layers (default 2)
    """
    
    def __init__(self, in_dim, out_dim, n_layers=2):
        super(ModalityLightGCN, self).__init__()
        self.n_layers = n_layers
        # Single linear projection: input features → embedding space
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Graph edges (2, num_edges)
        
        Returns:
            embeddings: (n_nodes, out_dim)
        """
        n_nodes = x.shape[0]
        
        # Project to embedding space
        h = self.proj(x)
        
        # Compute symmetric normalization
        row, col = edge_index
        deg = degree(row, num_nodes=n_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Layer combination: store all layer outputs
        layer_outputs = [h]
        
        for _ in range(self.n_layers):
            messages = h[col] * norm.unsqueeze(1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, messages)
            h = agg
            layer_outputs.append(h)
        
        # Mean of all layers
        embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return embeddings


class AdaptiveFusion(nn.Module):
    """
    Learnable adaptive fusion of modality embeddings via softmax weights.
    
    [α, β, γ, δ] = softmax(W_fuse · [e_text, e_image, e_video, e_meta])
    H_fused = α·e_text + β·e_image + γ·e_video + δ·e_meta
    
    Args:
        embed_dim: Dimension of each modality embedding
        n_modalities: Number of modalities (default 4)
    """
    
    def __init__(self, embed_dim, n_modalities=4):
        super(AdaptiveFusion, self).__init__()
        self.n_modalities = n_modalities
        # Fusion weight predictor: concat of all modality embeddings → n_modalities weights
        self.W_fuse = nn.Linear(embed_dim * n_modalities, n_modalities)
    
    def forward(self, modality_embeddings):
        """
        Args:
            modality_embeddings: list of (n_nodes, embed_dim) tensors, one per modality
        
        Returns:
            fused: (n_nodes, embed_dim)
            weights: (n_nodes, n_modalities) — the learned α, β, γ, δ
        """
        # Concatenate all modality embeddings
        concat = torch.cat(modality_embeddings, dim=1)  # (n_nodes, embed_dim * n_modalities)
        
        # Compute softmax weights
        weights = F.softmax(self.W_fuse(concat), dim=1)  # (n_nodes, n_modalities)
        
        # Weighted combination
        stacked = torch.stack(modality_embeddings, dim=1)  # (n_nodes, n_modalities, embed_dim)
        fused = (stacked * weights.unsqueeze(2)).sum(dim=1)  # (n_nodes, embed_dim)
        
        return fused, weights


class ContentFilteringModule(nn.Module):
    """
    Content-Based Filtering using modality-specific LightGCN channels + adaptive fusion.
    
    Pipeline:
    1. K-means clustering of users and items (UNCHANGED)
    2. Build cluster similarity graph (UNCHANGED)
    3. Run 4 independent LightGCN channels on cluster graph — one per modality
    4. Adaptively fuse with learned weights [α, β, γ, δ]
    5. Map cluster embeddings back to individual user/item nodes
    
    Args:
        modality_dims: dict of {modality_name: feature_dim} for each modality
        n_user_clusters: Number of user clusters for K-means
        n_item_clusters: Number of item clusters for K-means
        out_dim: Output embedding dimension
        n_layers: Number of LightGCN layers per modality channel
    """
    
    def __init__(self, modality_dims, n_user_clusters=20, n_item_clusters=15,
                 out_dim=32, n_layers=2):
        super(ContentFilteringModule, self).__init__()
        self.n_user_clusters = n_user_clusters
        self.n_item_clusters = n_item_clusters
        self.out_dim = out_dim
        self.modality_names = list(modality_dims.keys())
        
        # Per-modality LightGCN channels
        self.modality_channels = nn.ModuleDict({
            name: ModalityLightGCN(dim, out_dim, n_layers)
            for name, dim in modality_dims.items()
        })
        
        # Adaptive fusion
        self.fusion = AdaptiveFusion(out_dim, n_modalities=len(modality_dims))
        
        # Projection layers to map cluster embeddings + features → node embeddings
        total_feat_dim = sum(modality_dims.values())
        self.user_proj = nn.Linear(out_dim + total_feat_dim, out_dim)
        self.item_proj = nn.Linear(out_dim + total_feat_dim, out_dim)
    
    def build_cluster_graph(self, user_features, item_features, threshold=0.0):
        print(f"[CBF] Running K-means: {self.n_user_clusters} user clusters, {self.n_item_clusters} item clusters...")
        # Step 1: Cluster users and items
        user_labels, user_centroids = cluster_features(
            user_features, n_clusters=self.n_user_clusters
        )
        item_labels, item_centroids = cluster_features(
            item_features, n_clusters=self.n_item_clusters
        )
        
        # Step 2: Ensure centroids have same dimension
        feature_dim = user_features.shape[1]
        if user_centroids.shape[1] != feature_dim:
            user_centroids = np.pad(user_centroids,
                ((0, 0), (0, max(0, feature_dim - user_centroids.shape[1]))))[:, :feature_dim]
        if item_centroids.shape[1] != feature_dim:
            item_centroids = np.pad(item_centroids,
                ((0, 0), (0, max(0, feature_dim - item_centroids.shape[1]))))[:, :feature_dim]
        
        # Step 3: Build cluster similarity graph
        cluster_edge_index, cluster_weights, n_uc, n_ic = build_cluster_similarity_graph(
            user_labels, item_labels, user_centroids, item_centroids, threshold
        )
        
        self.user_labels = user_labels
        self.item_labels = item_labels
        self.user_centroids = user_centroids
        self.item_centroids = item_centroids
        self.cluster_edge_index = cluster_edge_index
        self.cluster_weights = cluster_weights
        self.n_uc = n_uc
        self.n_ic = n_ic
        
        return self.cluster_edge_index, self.user_labels, self.item_labels

    def forward(self, user_features, item_features, user_modality_features=None, 
                item_modality_features=None):
        """
        Args:
            user_features: (n_users, total_feature_dim) — concatenated user features
            item_features: (n_items, total_feature_dim) — concatenated item features
            user_modality_features: dict of {modality_name: (n_users, mod_dim)} per-modality features
            item_modality_features: dict of {modality_name: (n_items, mod_dim)} per-modality features
        
        Returns:
            H_UI: (n_users + n_items, out_dim)
            H_UI_users: (n_users, out_dim)
            H_UI_items: (n_items, out_dim)
        """
        device = user_features.device
        
        # Extract precomputed cluster info
        user_labels = self.user_labels
        item_labels = self.item_labels
        user_centroids = self.user_centroids
        item_centroids = self.item_centroids
        n_uc = self.n_uc
        n_ic = self.n_ic
        
        feature_dim = user_features.shape[1]
        
        cluster_edge_index = self.cluster_edge_index.to(device)
        cluster_weights = self.cluster_weights.to(device)
        
        # Step 4: Prepare per-modality cluster node features
        all_centroids = np.vstack([user_centroids, item_centroids])
        n_cluster_nodes = n_uc + n_ic
        
        # If per-modality features are available, split centroids accordingly
        if user_modality_features is not None and item_modality_features is not None:
            modality_cluster_features = {}
            for mod_name in self.modality_names:
                # Get per-modality features and compute cluster centroids for this modality
                u_mod = user_modality_features[mod_name].detach().cpu().numpy()
                i_mod = item_modality_features[mod_name].detach().cpu().numpy()
                
                # User cluster centroids for this modality
                u_mod_centroids = np.zeros((n_uc, u_mod.shape[1]))
                for c in range(n_uc):
                    mask = user_labels == c
                    if mask.sum() > 0:
                        u_mod_centroids[c] = u_mod[mask].mean(axis=0)
                
                # Item cluster centroids for this modality
                i_mod_centroids = np.zeros((n_ic, i_mod.shape[1]))
                for c in range(n_ic):
                    mask = item_labels == c
                    if mask.sum() > 0:
                        i_mod_centroids[c] = i_mod[mask].mean(axis=0)
                
                mod_cluster_feat = torch.tensor(
                    np.vstack([u_mod_centroids, i_mod_centroids]),
                    dtype=torch.float32
                ).to(device)
                modality_cluster_features[mod_name] = mod_cluster_feat
        else:
            # Fallback: split concatenated centroids evenly among modalities
            dim_per_mod = feature_dim // len(self.modality_names)
            modality_cluster_features = {}
            all_centroids_t = torch.tensor(all_centroids, dtype=torch.float32).to(device)
            for idx, mod_name in enumerate(self.modality_names):
                start = idx * dim_per_mod
                end = start + dim_per_mod if idx < len(self.modality_names) - 1 else feature_dim
                modality_cluster_features[mod_name] = all_centroids_t[:, start:end]
        
        # Step 5: Run per-modality LightGCN channels on cluster graph
        modality_embeddings = []
        for mod_name in self.modality_names:
            mod_feat = modality_cluster_features[mod_name].to(device)
            mod_embed = self.modality_channels[mod_name](mod_feat, cluster_edge_index)
            modality_embeddings.append(mod_embed)
        
        # Step 6: Adaptive fusion
        fused_embeddings, fusion_weights = self.fusion(modality_embeddings)
        
        # Step 7: Map cluster embeddings back to individual nodes
        user_cluster_embeddings = fused_embeddings[:n_uc]
        item_cluster_embeddings = fused_embeddings[n_uc:]
        
        user_labels_t = torch.tensor(user_labels, dtype=torch.long, device=device)
        item_labels_t = torch.tensor(item_labels, dtype=torch.long, device=device)
        
        user_cluster_embeds = user_cluster_embeddings[user_labels_t]
        item_cluster_embeds = item_cluster_embeddings[item_labels_t]
        
        # Step 8: Combine with original features and project
        user_combined = torch.cat([user_cluster_embeds, user_features], dim=1)
        item_combined = torch.cat([item_cluster_embeds, item_features], dim=1)
        
        H_UI_users = self.user_proj(user_combined)
        H_UI_items = self.item_proj(item_combined)
        
        H_UI = torch.cat([H_UI_users, H_UI_items], dim=0)
        
        return H_UI, H_UI_users, H_UI_items
