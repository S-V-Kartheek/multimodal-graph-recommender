"""
Contrastive Loss Functions — MM-CLightRec.

Three contrastive learning objectives:

L₁ — Inter-Modal Contrastive Loss (Change 3):
    Aligns modality embedding spaces before graph propagation using InfoNCE.
    Same-item cross-modal pairs are positives, different-item pairs are negatives.

L₂ — Structural Graph Contrastive Loss (Change 4):
    Creates two augmented views of the bipartite graph (edge dropout + feature masking),
    runs LightGCN on each, and pulls same-node representations together via InfoNCE.
    Addresses data sparsity.

L₃ — Cluster-Aware Cold-Start Contrastive Loss (Change 5):
    Simulates cold-start users during training, pulls their embeddings toward warm users
    in the same K-means cluster. Journal-only (gated by flag).

Changes 3, 4, 5 from MM-CLightRec architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
#  L₁ — Inter-Modal Contrastive Loss (Change 3)
# ============================================================================

class ModalityProjectionHead(nn.Module):
    """
    2-layer MLP projection head mapping modality features to a shared contrastive space.
    
    z_mod = MLP(f_mod)
    
    Applied BEFORE graph propagation so that modalities speak the same language
    when LightGCN propagates signals.
    
    Args:
        in_dim: Input modality feature dimension
        proj_dim: Output projection dimension (shared contrastive space)
        hidden_dim: Hidden layer dimension (default: 2x proj_dim)
    """
    
    def __init__(self, in_dim, proj_dim=64, hidden_dim=None):
        super(ModalityProjectionHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = proj_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x):
        """. Project features to contrastive space and L2-normalize."""
        z = self.mlp(x)
        z = F.normalize(z, dim=1)
        return z


def inter_modal_contrastive_loss(z_mod1, z_mod2, temperature=0.2):
    """
    InfoNCE contrastive loss between two modality projections.
    
    For each item i:
        Positive: (z_mod1_i, z_mod2_i) — same item, different modalities
        Negative: (z_mod1_i, z_mod2_j) where j≠i — different items
    
    L₁ = -log [ exp(sim(z1_i, z2_i)/τ) / Σ_k exp(sim(z1_i, z2_k)/τ) ]
    
    Args:
        z_mod1: (n_items, proj_dim) — L2-normalized projection of modality 1
        z_mod2: (n_items, proj_dim) — L2-normalized projection of modality 2
        temperature: InfoNCE temperature τ
    
    Returns:
        loss: scalar contrastive loss
    """
    # Similarity matrix: (n, n) 
    sim_matrix = torch.mm(z_mod1, z_mod2.t()) / temperature  # (n, n)
    
    # Positive pairs are on the diagonal
    n = z_mod1.shape[0]
    labels = torch.arange(n, device=z_mod1.device)
    
    # Cross-entropy: row i should have max at column i
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.t(), labels)
    
    loss = (loss_12 + loss_21) / 2.0
    return loss


def compute_inter_modal_loss(projection_heads, modality_features, temperature=0.2):
    """
    Compute the full inter-modal contrastive loss across all modality pairs.
    
    Args:
        projection_heads: dict of {modality_name: ModalityProjectionHead}
        modality_features: dict of {modality_name: (n_items, mod_dim)}
        temperature: InfoNCE temperature
    
    Returns:
        total_loss: average contrastive loss across all modality pairs
    """
    mod_names = list(modality_features.keys())
    projections = {}
    
    for name in mod_names:
        if name in projection_heads:
            projections[name] = projection_heads[name](modality_features[name])
    
    total_loss = 0.0
    n_pairs = 0
    
    for i in range(len(mod_names)):
        for j in range(i + 1, len(mod_names)):
            name_i, name_j = mod_names[i], mod_names[j]
            if name_i in projections and name_j in projections:
                loss = inter_modal_contrastive_loss(
                    projections[name_i], projections[name_j], temperature
                )
                total_loss += loss
                n_pairs += 1
    
    return total_loss / max(n_pairs, 1)


# ============================================================================
#  L₂ — Structural Graph Contrastive Loss (Change 4)
# ============================================================================

def edge_dropout(edge_index, drop_rate=0.1):
    """
    Randomly remove edges from the graph.
    
    View 1 augmentation: randomly remove `drop_rate` fraction of edges.
    
    Args:
        edge_index: (2, num_edges)
        drop_rate: fraction of edges to drop
    
    Returns:
        augmented_edge_index: (2, num_kept_edges)
    """
    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    return edge_index[:, keep_mask]


def feature_masking(x, mask_rate=0.2):
    """
    Randomly zero out feature dimensions.
    
    View 2 augmentation: randomly zero `mask_rate` fraction of feature dimensions.
    
    Args:
        x: (n_nodes, feat_dim)
        mask_rate: fraction of dimensions to zero
    
    Returns:
        masked_x: (n_nodes, feat_dim) with some dimensions zeroed
    """
    feat_dim = x.shape[1]
    mask = torch.rand(feat_dim, device=x.device) > mask_rate
    return x * mask.unsqueeze(0)


def structural_contrastive_loss(z_view1, z_view2, temperature=0.2):
    """
    InfoNCE structural contrastive loss between two augmented graph views.
    
    Same node should produce similar embeddings regardless of augmentation.
    
    L₂ = -log [ exp(sim(e_u^G', e_u^G'')/τ) / Σ_v exp(sim(e_u^G', e_v^G'')/τ) ]
    
    Args:
        z_view1: (n_nodes, embed_dim) — embeddings from View 1 (edge dropout)
        z_view2: (n_nodes, embed_dim) — embeddings from View 2 (feature masking)
        temperature: InfoNCE temperature τ
    
    Returns:
        loss: scalar contrastive loss
    """
    # L2 normalize
    z1 = F.normalize(z_view1, dim=1)
    z2 = F.normalize(z_view2, dim=1)
    
    # Similarity
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    n = z1.shape[0]
    labels = torch.arange(n, device=z1.device)
    
    loss_12 = F.cross_entropy(sim_matrix, labels)
    loss_21 = F.cross_entropy(sim_matrix.t(), labels)
    
    return (loss_12 + loss_21) / 2.0


def compute_structural_contrastive_loss(lightgcn, edge_index, 
                                         edge_drop_rate=0.1, feat_mask_rate=0.2,
                                         temperature=0.2, max_nodes=2048):
    """
    Full L₂ computation: augment graph two ways, run LightGCN, compute InfoNCE.
    
    Args:
        lightgcn: LightGCN model (with forward_with_augmented_graph method)
        edge_index: Original bipartite graph edges
        edge_drop_rate: Edge dropout rate for View 1
        feat_mask_rate: Feature masking rate for View 2
        temperature: InfoNCE temperature
        max_nodes: Max number of nodes to sample for contrastive loss (memory control)
    
    Returns:
        loss: structural contrastive loss
    """
    # View 1: edge dropout
    edge_index_v1 = edge_dropout(edge_index, edge_drop_rate)
    z_v1 = lightgcn.forward_with_augmented_graph(edge_index_v1)
    
    # View 2: feature masking
    all_embeddings = torch.cat([
        lightgcn.user_embedding.weight,
        lightgcn.item_embedding.weight
    ], dim=0)
    x_v2 = feature_masking(all_embeddings, feat_mask_rate)
    z_v2 = lightgcn.forward_with_augmented_graph(edge_index, x=x_v2)
    
    # Sample nodes if too many (memory efficiency)
    n_nodes = z_v1.shape[0]
    if n_nodes > max_nodes:
        sample_idx = torch.randperm(n_nodes, device=z_v1.device)[:max_nodes]
        z_v1 = z_v1[sample_idx]
        z_v2 = z_v2[sample_idx]
    
    loss = structural_contrastive_loss(z_v1, z_v2, temperature)
    return loss


# ============================================================================
#  L₃ — Cluster-Aware Cold-Start Contrastive Loss (Change 5)
# ============================================================================

def simulate_cold_start(user_idx, item_idx, n_users, keep_k=5, cold_ratio=0.2, seed=None):
    """
    Simulate cold-start users by hiding most of their interactions.
    
    Randomly select `cold_ratio` of users and keep only `keep_k` interactions.
    
    Args:
        user_idx: Array of user indices from all interactions
        item_idx: Array of item indices from all interactions
        n_users: Total number of users
        keep_k: Number of interactions to keep for cold-start users (K-shot)
        cold_ratio: Fraction of users to designate as cold-start
        seed: Random seed for reproducibility
    
    Returns:
        cold_user_ids: array of cold-start user IDs
        cold_mask: boolean mask — True for interactions belonging to cold users
                   that should be HIDDEN (not kept)
        warm_mask: boolean mask — True for interactions to keep for training
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Select cold-start users
    all_users = np.unique(user_idx)
    n_cold = max(1, int(len(all_users) * cold_ratio))
    cold_user_ids = rng.choice(all_users, size=n_cold, replace=False)
    cold_set = set(cold_user_ids)
    
    # For each cold user, keep only keep_k interactions
    cold_mask = np.zeros(len(user_idx), dtype=bool)
    warm_mask = np.ones(len(user_idx), dtype=bool)
    
    for uid in cold_user_ids:
        user_interactions = np.where(user_idx == uid)[0]
        if len(user_interactions) > keep_k:
            # Randomly select interactions to HIDE
            hide_idx = rng.choice(user_interactions, 
                                   size=len(user_interactions) - keep_k, 
                                   replace=False)
            cold_mask[hide_idx] = True
            warm_mask[hide_idx] = False
    
    return cold_user_ids, cold_mask, warm_mask


def cold_start_contrastive_loss(cold_embeddings, warm_embeddings, 
                                 cold_cluster_labels, warm_cluster_labels,
                                 temperature=0.2):
    """
    Cluster-aware cold-start contrastive loss (L₃).
    
    Positive: cold user vs. warm user in SAME cluster  
    Negative: cold user vs. warm users in DIFFERENT clusters
    
    L₃ = -log [ exp(sim(e_cold, e_warm_same)/τ) / Σ_k exp(sim(e_cold, e_warm_k)/τ) ]
    
    Args:
        cold_embeddings: (n_cold, embed_dim) — embeddings of cold-start users
        warm_embeddings: (n_warm, embed_dim) — embeddings of warm users
        cold_cluster_labels: (n_cold,) — cluster assignments for cold users
        warm_cluster_labels: (n_warm,) — cluster assignments for warm users
        temperature: InfoNCE temperature τ
    
    Returns:
        loss: cold-start contrastive loss
    """
    if cold_embeddings.shape[0] == 0 or warm_embeddings.shape[0] == 0:
        return torch.tensor(0.0, device=cold_embeddings.device)
    
    # Normalize embeddings
    cold_norm = F.normalize(cold_embeddings, dim=1)
    warm_norm = F.normalize(warm_embeddings, dim=1)
    
    # Similarity: (n_cold, n_warm)
    sim_matrix = torch.mm(cold_norm, warm_norm.t()) / temperature
    
    # For each cold user, the positive targets are warm users in the same cluster
    total_loss = 0.0
    valid_count = 0
    
    for i in range(cold_embeddings.shape[0]):
        cold_cluster = cold_cluster_labels[i].item() if isinstance(cold_cluster_labels, torch.Tensor) else cold_cluster_labels[i]
        
        # Find warm users in the same cluster
        if isinstance(warm_cluster_labels, torch.Tensor):
            same_cluster_mask = (warm_cluster_labels == cold_cluster)
        else:
            same_cluster_mask = torch.tensor(warm_cluster_labels == cold_cluster, 
                                              device=cold_embeddings.device)
        
        if same_cluster_mask.sum() == 0:
            continue
        
        # Log-sum-exp denominator (all warm users)
        log_denom = torch.logsumexp(sim_matrix[i], dim=0)
        
        # Log-sum-exp numerator (same cluster warm users only)
        same_cluster_sims = sim_matrix[i][same_cluster_mask]
        log_numer = torch.logsumexp(same_cluster_sims, dim=0)
        
        total_loss += -(log_numer - log_denom)
        valid_count += 1
    
    if valid_count == 0:
        return torch.tensor(0.0, device=cold_embeddings.device, requires_grad=True)
    
    return total_loss / valid_count
