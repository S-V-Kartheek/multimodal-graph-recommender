"""
MGRS-HFA: Multimodal Graph-based Recommendation System using Hybrid Filtering Approach.
Full pipeline combining all modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .collaborative_filtering import CollaborativeFilteringModule, build_collaboration_graph
from .content_filtering import ContentFilteringModule
from .cross_attention import CrossAttention
from .vgae import VGAE


class MGRS_HFA(nn.Module):
    """
    Complete MGRS-HFA model.
    
    Pipeline:
    1. Collaborative Filtering: UserGCN + ItemGAT on collaboration graphs
    2. Content-Based Filtering: K-means clustering + MixHopConv GNN
    3. Cross-Attention: Fuse CF and CBF outputs
    4. VGAE: Link prediction for recommendation generation
    """
    
    def __init__(self, user_feature_dim, item_feature_dim, 
                 cf_hidden_dim=128, cf_out_dim=32,
                 cbf_hidden_dim=60, cbf_out_dim=32,
                 n_user_clusters=20, n_item_clusters=15,
                 vgae_hidden_dim=100, vgae_latent_dim=50):
        super(MGRS_HFA, self).__init__()
        
        # Collaborative Filtering Module
        self.cf_module = CollaborativeFilteringModule(
            user_in_dim=user_feature_dim,
            item_in_dim=item_feature_dim,
            hidden_dim=cf_hidden_dim,
            out_dim=cf_out_dim
        )
        
        # Content-Based Filtering Module
        self.cbf_module = ContentFilteringModule(
            feature_dim=user_feature_dim,  # Using same dim after projection
            n_user_clusters=n_user_clusters,
            n_item_clusters=n_item_clusters,
            hidden_dim=cbf_hidden_dim,
            out_dim=cbf_out_dim
        )
        
        # Cross-Attention Module
        self.cross_attention = CrossAttention(feature_dim=cf_out_dim)
        
        # VGAE for Link Prediction
        # Input: cross-attention output dimension
        self.vgae = VGAE(
            in_channels=cf_out_dim,
            hidden_channels=vgae_hidden_dim,
            latent_channels=vgae_latent_dim
        )
        
        # Store dimensions
        self.cf_out_dim = cf_out_dim
        self.cbf_out_dim = cbf_out_dim
        
        # Collaboration graph caches
        self._user_edge_index = None
        self._item_edge_index = None
    
    def build_collaboration_graphs(self, user_features, item_features, 
                                     user_threshold=0.3, item_threshold=0.3,
                                     user_top_k=30, item_top_k=30):
        """Build user-user and item-item collaboration graphs."""
        self._user_edge_index = build_collaboration_graph(
            user_features, threshold=user_threshold, top_k=user_top_k
        ).to(user_features.device)
        
        self._item_edge_index = build_collaboration_graph(
            item_features, threshold=item_threshold, top_k=item_top_k
        ).to(item_features.device)
        
        return self._user_edge_index, self._item_edge_index
    
    def forward(self, user_features, item_features, 
                bipartite_edge_index, user_idx, item_idx,
                user_edge_index=None, item_edge_index=None):
        """
        Full forward pass.
        
        Args:
            user_features: (n_users, user_feature_dim)
            item_features: (n_items, item_feature_dim)
            bipartite_edge_index: user-item bipartite graph edges
            user_idx: user indices for link prediction pairs
            item_idx: item indices for link prediction pairs (offset by n_users for VGAE)
            user_edge_index: user collaboration graph (optional, cached)
            item_edge_index: item collaboration graph (optional, cached)
        
        Returns:
            link_logits: Predicted link logits for (user_idx, item_idx) pairs
            mu: VGAE mean
            logvar: VGAE log-variance
            z: VGAE latent variables
        """
        device = user_features.device
        n_users = user_features.shape[0]
        
        # Use cached or provided collaboration graphs
        if user_edge_index is None:
            user_edge_index = self._user_edge_index
        if item_edge_index is None:
            item_edge_index = self._item_edge_index
        
        # 1. Collaborative Filtering
        H_U, H_I = self.cf_module(
            user_features, item_features,
            user_edge_index, item_edge_index
        )
        
        # 2. Content-Based Filtering
        H_UI, H_UI_users, H_UI_items = self.cbf_module(
            user_features, item_features
        )
        
        # 3. Cross-Attention
        Z_users, Z_items, attn_weights = self.cross_attention(
            H_U, H_I, H_UI_users, H_UI_items
        )
        
        # 4. Combine features for VGAE
        # Stack user and item features: [Z_users; Z_items]
        combined_features = torch.cat([Z_users, Z_items], dim=0)  # (n_users + n_items, cf_out_dim)
        
        # 5. VGAE - Link Prediction
        # Adjust item indices to be offset by n_users
        item_idx_offset = item_idx + n_users
        
        link_logits, mu, logvar, z = self.vgae(
            combined_features, bipartite_edge_index,
            user_idx, item_idx_offset
        )
        
        return link_logits, mu, logvar, z
    
    def predict(self, user_features, item_features, bipartite_edge_index,
                user_idx, item_idx, user_edge_index=None, item_edge_index=None):
        """
        Predict link probabilities (with sigmoid).
        """
        self.eval()
        with torch.no_grad():
            link_logits, mu, logvar, z = self.forward(
                user_features, item_features, bipartite_edge_index,
                user_idx, item_idx, user_edge_index, item_edge_index
            )
            link_probs = torch.sigmoid(link_logits)
        return link_probs
    
    def get_all_scores(self, user_features, item_features, bipartite_edge_index,
                       user_edge_index=None, item_edge_index=None):
        """
        Get all user-item scores for recommendation generation.
        Returns: (n_users, n_items) score matrix
        """
        self.eval()
        device = user_features.device
        n_users = user_features.shape[0]
        
        with torch.no_grad():
            # Use cached graphs
            if user_edge_index is None:
                user_edge_index = self._user_edge_index
            if item_edge_index is None:
                item_edge_index = self._item_edge_index
            
            # 1. CF
            H_U, H_I = self.cf_module(
                user_features, item_features,
                user_edge_index, item_edge_index
            )
            
            # 2. CBF
            H_UI, H_UI_users, H_UI_items = self.cbf_module(
                user_features, item_features
            )
            
            # 3. Cross-Attention
            Z_users, Z_items, _ = self.cross_attention(
                H_U, H_I, H_UI_users, H_UI_items
            )
            
            # 4. VGAE encode
            combined_features = torch.cat([Z_users, Z_items], dim=0)
            z, mu, logvar = self.vgae.encode(combined_features, bipartite_edge_index)
            
            # 5. Compute all user-item scores
            z_users = z[:n_users]
            z_items = z[n_users:]
            scores = torch.sigmoid(torch.mm(z_users, z_items.t()))
        
        return scores
