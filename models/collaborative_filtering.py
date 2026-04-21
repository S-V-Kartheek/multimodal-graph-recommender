"""
Collaborative Filtering Module — MM-CLightRec.

Replaces the base paper's separate UserGCN (2-layer GCN on user-user graph) and
ItemGAT (2-layer GAT on item-item graph) with a **single unified 3-layer LightGCN**
operating directly on the bipartite user-item interaction graph.

LightGCN (He et al., SIGIR 2020):
  - No learnable weight matrix W per layer
  - No nonlinear activation σ
  - Pure symmetric-normalized neighborhood aggregation
  - Final embedding = mean of all layer embeddings (layer combination)

Uses sparse matrix multiplication for efficient CPU/GPU propagation.

Change 1 from MM-CLightRec architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


def _build_sparse_adj(edge_index, n_nodes):
    """
    Build sparse adjacency matrix with symmetric normalization D^{-1/2} A D^{-1/2}.
    Returns a torch.sparse_coo_tensor for efficient matrix multiplication.
    """
    row, col = edge_index
    deg = degree(row, num_nodes=n_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    
    # Edge weights = D^{-1/2}[row] * D^{-1/2}[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    # Build sparse matrix
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(n_nodes, n_nodes)
    ).coalesce()
    
    return adj


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    
    Propagation rule (no W, no σ):
        e_u^(l+1) = Σ_{i∈N(u)} 1/√(|N(u)|·|N(i)|) × e_i^(l)
        e_i^(l+1) = Σ_{u∈N(i)} 1/√(|N(i)|·|N(u)|) × e_u^(l)
    
    Final embedding (layer combination):
        e_u* = (1/(L+1)) × Σ_{l=0}^{L} e_u^(l)
    
    Uses sparse matrix multiplication (torch.sparse.mm) for O(|E|·d) propagation,
    which is dramatically faster than scatter-based message passing on CPU.
    
    Args:
        n_users: Number of user nodes
        n_items: Number of item nodes
        embed_dim: Embedding dimension (default 32)
        n_layers: Number of propagation layers (default 3)
    """
    
    def __init__(self, n_users, n_items, embed_dim=32, n_layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        
        # Learnable initial embeddings (ID embeddings)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Cached sparse adjacency (built on first forward pass)
        self._cached_adj = None
        self._cached_edge_index_id = None
    
    def _get_sparse_adj(self, edge_index, n_nodes):
        """Get or build cached sparse adjacency matrix."""
        eid = id(edge_index)
        if self._cached_adj is not None and self._cached_edge_index_id == eid:
            return self._cached_adj
        
        adj = _build_sparse_adj(edge_index, n_nodes)
        self._cached_adj = adj
        self._cached_edge_index_id = eid
        return adj
    
    def forward(self, edge_index, user_features=None, item_features=None):
        """
        Forward pass: L layers of LightGCN propagation + layer combination.
        
        Args:
            edge_index: Bipartite graph edge_index (2, num_edges).
                        Node indices: users=[0..n_users-1], items=[n_users..n_users+n_items-1]
            user_features: Optional initial user features (n_users, feat_dim).
                          If provided, projected to embed_dim and added to ID embeddings.
            item_features: Optional initial item features (n_items, feat_dim).
                          If provided, projected to embed_dim and added to ID embeddings.
        
        Returns:
            H_U: User embeddings (n_users, embed_dim)
            H_I: Item embeddings (n_items, embed_dim)
        """
        n_nodes = self.n_users + self.n_items
        
        # Initial embeddings
        e_u = self.user_embedding.weight
        e_i = self.item_embedding.weight
        
        # Add projected external features if available
        if user_features is not None and hasattr(self, 'user_feat_proj'):
            e_u = e_u + self.user_feat_proj(user_features)
        if item_features is not None and hasattr(self, 'item_feat_proj'):
            e_i = e_i + self.item_feat_proj(item_features)
        
        all_embeddings = torch.cat([e_u, e_i], dim=0)  # (n_nodes, embed_dim)
        
        # Build normalized sparse adjacency
        adj = self._get_sparse_adj(edge_index, n_nodes)
        adj = adj.to(all_embeddings.device)
        
        # Layer combination: store all layer outputs
        layer_embeddings = [all_embeddings]
        
        # L-layer propagation using sparse matrix multiplication
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            layer_embeddings.append(all_embeddings)
        
        # Mean of all layers: e* = (1/(L+1)) × Σ e^(l)
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        
        # Split into user and item embeddings
        H_U = final_embeddings[:self.n_users]
        H_I = final_embeddings[self.n_users:]
        
        return H_U, H_I
    
    def forward_with_augmented_graph(self, edge_index, x=None, n_nodes=None):
        """
        Forward pass on an augmented/perturbed graph (for L₂ structural contrastive loss).
        
        Does NOT use caching since the edge_index changes every call.
        
        Args:
            edge_index: Augmented bipartite graph edge_index
            x: Optional pre-augmented embeddings (n_nodes, embed_dim)
            n_nodes: Total number of nodes (if None, uses n_users + n_items)
        
        Returns:
            all_embeddings: (n_nodes, embed_dim)
        """
        if n_nodes is None:
            n_nodes = self.n_users + self.n_items
        
        if x is not None:
            all_embeddings = x
        else:
            all_embeddings = torch.cat([
                self.user_embedding.weight,
                self.item_embedding.weight
            ], dim=0)
        
        # Build fresh sparse adj (no caching for augmented graphs)
        adj = _build_sparse_adj(edge_index, n_nodes).to(all_embeddings.device)
        
        layer_embeddings = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            layer_embeddings.append(all_embeddings)
        
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return final_embeddings


class CollaborativeFilteringModule(nn.Module):
    """
    Collaborative Filtering Module using unified LightGCN on bipartite graph.
    
    Replaces the base paper's separate UserGCN + ItemGAT approach.
    """
    
    def __init__(self, n_users, n_items, user_in_dim, item_in_dim,
                 embed_dim=32, n_layers=3):
        super(CollaborativeFilteringModule, self).__init__()
        self.out_dim = embed_dim
        
        # Core LightGCN
        self.lightgcn = LightGCN(n_users, n_items, embed_dim, n_layers)
        
        # Feature projection layers
        self.lightgcn.user_feat_proj = nn.Linear(user_in_dim, embed_dim)
        self.lightgcn.item_feat_proj = nn.Linear(item_in_dim, embed_dim)
    
    def forward(self, user_features, item_features, bipartite_edge_index):
        """
        Args:
            user_features: (n_users, user_in_dim)
            item_features: (n_items, item_in_dim)
            bipartite_edge_index: Bipartite graph edges
        
        Returns:
            H_U: (n_users, embed_dim) user embeddings
            H_I: (n_items, embed_dim) item embeddings
        """
        H_U, H_I = self.lightgcn(bipartite_edge_index, user_features, item_features)
        return H_U, H_I
