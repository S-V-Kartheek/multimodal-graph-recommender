"""
MM-CLightRec: Contrastive Multimodal LightGCN for Recommendation.

Main model pipeline combining all modules:
1. Collaborative Filtering: Unified 3-layer LightGCN on bipartite graph (Change 1)
2. Content-Based Filtering: 4 modality-specific LightGCN channels + adaptive fusion (Change 2)
3. Inter-Modal Contrastive Loss L₁ (Change 3)
4. Structural Graph Contrastive Loss L₂ (Change 4)
5. Cold-Start Contrastive Loss L₃ (Change 5 — journal only)
6. Unified hierarchical loss (Change 6)

Cross-Attention + VGAE are RETAINED from the base paper (unchanged).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .collaborative_filtering import CollaborativeFilteringModule, LightGCN
from .content_filtering import ContentFilteringModule
from .cross_attention import CrossAttention
from .vgae import VGAE
from .contrastive_losses import (
    ModalityProjectionHead, compute_inter_modal_loss,
    compute_structural_contrastive_loss,
    simulate_cold_start, cold_start_contrastive_loss
)


class MM_CLightRec(nn.Module):
    """
    MM-CLightRec: Contrastive Multimodal LightGCN Recommendation Model.
    
    Pipeline:
    1. LightGCN on bipartite graph → H_U, H_I
    2. Per-modality LightGCN on cluster graph + adaptive fusion → H_UI
    3. Cross-Attention (Q=H_U, K=H_I, V=H_UI) → Z (UNCHANGED)
    4. VGAE link prediction on Z (UNCHANGED)
    5. Contrastive losses: L₁ + L₂ + L₃ computed during training
    
    Args:
        n_users: Number of users
        n_items: Number of items
        user_feature_dim: Dimension of concatenated user features
        item_feature_dim: Dimension of concatenated item features
        modality_dims: dict {modality_name: dim} for per-modality features
        cf_embed_dim: LightGCN embedding dimension (default 32)
        cf_n_layers: Number of LightGCN propagation layers (default 3)
        cbf_out_dim: Content filtering output dimension (default 32)
        cbf_n_layers: Per-modality LightGCN layers (default 2)
        n_user_clusters: K-means user clusters (default 20)
        n_item_clusters: K-means item clusters (default 15)
        vgae_hidden_dim: VGAE encoder hidden dim (default 100)
        vgae_latent_dim: VGAE latent dim (default 50)
        contrastive_proj_dim: Projection dimension for contrastive learning (default 64)
        temperature: InfoNCE temperature τ (default 0.2)
        include_cold_start: Whether to include L₃ (default False — conference version)
    """
    
    def __init__(self, n_users, n_items, user_feature_dim, item_feature_dim,
                 modality_dims=None,
                 cf_embed_dim=32, cf_n_layers=3,
                 cbf_out_dim=32, cbf_n_layers=2,
                 n_user_clusters=20, n_item_clusters=15,
                 vgae_hidden_dim=100, vgae_latent_dim=50,
                 contrastive_proj_dim=64, temperature=0.2,
                 include_cold_start=False):
        super(MM_CLightRec, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.temperature = temperature
        self.include_cold_start = include_cold_start
        
        # Default modality dimensions if not specified
        if modality_dims is None:
            dim_per_mod = item_feature_dim // 4
            modality_dims = {
                'text': dim_per_mod,
                'image': dim_per_mod,
                'video': dim_per_mod,
                'meta': item_feature_dim - 3 * dim_per_mod  # Remainder
            }
        self.modality_dims = modality_dims
        
        # ---- Module 1: Collaborative Filtering (LightGCN) ----
        self.cf_module = CollaborativeFilteringModule(
            n_users=n_users,
            n_items=n_items,
            user_in_dim=user_feature_dim,
            item_in_dim=item_feature_dim,
            embed_dim=cf_embed_dim,
            n_layers=cf_n_layers
        )
        
        # ---- Module 2: Content-Based Filtering (Per-Modality LightGCN) ----
        self.cbf_module = ContentFilteringModule(
            modality_dims=modality_dims,
            n_user_clusters=n_user_clusters,
            n_item_clusters=n_item_clusters,
            out_dim=cbf_out_dim,
            n_layers=cbf_n_layers
        )
        
        # ---- Module 3: Cross-Attention (UNCHANGED from base) ----
        self.cross_attention = CrossAttention(feature_dim=cf_embed_dim)
        
        # ---- Module 4: VGAE Link Prediction (UNCHANGED from base) ----
        self.vgae = VGAE(
            in_channels=cf_embed_dim,
            hidden_channels=vgae_hidden_dim,
            latent_channels=vgae_latent_dim
        )
        
        # ---- Contrastive Components ----
        
        # L₁: Inter-modal projection heads (one per modality)
        self.projection_heads = nn.ModuleDict({
            name: ModalityProjectionHead(dim, contrastive_proj_dim)
            for name, dim in modality_dims.items()
        })
        
        # Store dimensions
        self.cf_embed_dim = cf_embed_dim
        self.cbf_out_dim = cbf_out_dim
    
    def forward(self, user_features, item_features, bipartite_edge_index,
                user_idx, item_idx,
                user_modality_features=None, item_modality_features=None):
        """
        Full forward pass.
        
        Args:
            user_features: (n_users, user_feature_dim)
            item_features: (n_items, item_feature_dim)
            bipartite_edge_index: Bipartite graph edges (user ↔ item)
            user_idx: User indices for link prediction pairs
            item_idx: Item indices for link prediction pairs
            user_modality_features: dict {mod_name: (n_users, mod_dim)} — optional
            item_modality_features: dict {mod_name: (n_items, mod_dim)} — optional
        
        Returns:
            link_logits: Predicted link logits for (user_idx, item_idx) pairs
            mu: VGAE mean
            logvar: VGAE log-variance
            z: VGAE latent variables
        """
        n_users = user_features.shape[0]
        
        # 1. Collaborative Filtering — LightGCN on bipartite graph
        H_U, H_I = self.cf_module(user_features, item_features, bipartite_edge_index)
        
        # 2. Content-Based Filtering — Per-modality LightGCN + adaptive fusion
        H_UI, H_UI_users, H_UI_items = self.cbf_module(
            user_features, item_features,
            user_modality_features, item_modality_features
        )
        
        # 3. Cross-Attention (UNCHANGED)
        Z_users, Z_items, attn_weights = self.cross_attention(
            H_U, H_I, H_UI_users, H_UI_items
        )
        
        # 4. VGAE — Link Prediction (UNCHANGED)
        combined_features = torch.cat([Z_users, Z_items], dim=0)
        item_idx_offset = item_idx + n_users
        
        link_logits, mu, logvar, z = self.vgae(
            combined_features, bipartite_edge_index,
            user_idx, item_idx_offset
        )
        
        return link_logits, mu, logvar, z
    
    def compute_contrastive_losses(self, bipartite_edge_index,
                                    item_modality_features=None,
                                    user_cluster_labels=None,
                                    cold_user_ids=None,
                                    cold_embeddings=None,
                                    warm_embeddings=None,
                                    warm_cluster_labels=None):
        """
        Compute all contrastive loss terms.
        
        Args:
            bipartite_edge_index: for L₂ structural contrastive
            item_modality_features: dict for L₁ inter-modal contrastive
            user_cluster_labels: for L₃ cold-start contrastive
            cold_user_ids, cold_embeddings, warm_embeddings, warm_cluster_labels: for L₃
        
        Returns:
            losses: dict with 'L_inter', 'L_struct', 'L_cold'
        """
        losses = {}
        device = bipartite_edge_index.device
        
        # L₁: Inter-modal contrastive loss
        if item_modality_features is not None:
            losses['L_inter'] = compute_inter_modal_loss(
                self.projection_heads, item_modality_features, self.temperature
            )
        else:
            losses['L_inter'] = torch.tensor(0.0, device=device)
        
        # L₂: Structural graph contrastive loss
        losses['L_struct'] = compute_structural_contrastive_loss(
            self.cf_module.lightgcn, bipartite_edge_index,
            edge_drop_rate=0.1, feat_mask_rate=0.2,
            temperature=self.temperature
        )
        
        # L₃: Cold-start contrastive loss (journal only)
        if self.include_cold_start and cold_embeddings is not None:
            cold_cluster_labels_t = torch.tensor(
                user_cluster_labels[cold_user_ids], 
                dtype=torch.long, device=device
            ) if user_cluster_labels is not None else None
            
            if cold_cluster_labels_t is not None and warm_cluster_labels is not None:
                losses['L_cold'] = cold_start_contrastive_loss(
                    cold_embeddings, warm_embeddings,
                    cold_cluster_labels_t, warm_cluster_labels,
                    self.temperature
                )
            else:
                losses['L_cold'] = torch.tensor(0.0, device=device)
        else:
            losses['L_cold'] = torch.tensor(0.0, device=device)
        
        return losses
    
    def predict(self, user_features, item_features, bipartite_edge_index,
                user_idx, item_idx,
                user_modality_features=None, item_modality_features=None):
        """Predict link probabilities (with sigmoid)."""
        self.eval()
        with torch.no_grad():
            link_logits, mu, logvar, z = self.forward(
                user_features, item_features, bipartite_edge_index,
                user_idx, item_idx,
                user_modality_features, item_modality_features
            )
            link_probs = torch.sigmoid(link_logits)
        return link_probs
    
    def get_all_scores(self, user_features, item_features, bipartite_edge_index,
                       user_modality_features=None, item_modality_features=None):
        """
        Get all user-item scores for recommendation generation.
        Matches the training pipeline (CF -> CBF -> Attention -> VGAE).
        Returns: (n_users, n_items) score matrix
        """
        self.eval()
        n_users = user_features.shape[0]
        
        with torch.no_grad():
            # 1. Collaborative Filtering
            H_U, H_I = self.cf_module(user_features, item_features, bipartite_edge_index)
            
            # 2. Content-Based Filtering
            H_UI, H_UI_users, H_UI_items = self.cbf_module(
                user_features, item_features,
                user_modality_features, item_modality_features
            )
            
            # 3. Cross-Attention
            Z_users, Z_items, _ = self.cross_attention(
                H_U, H_I, H_UI_users, H_UI_items
            )
            
            # 4. VGAE encode (eval mode returns mu as z)
            combined_features = torch.cat([Z_users, Z_items], dim=0)
            z, mu, logvar = self.vgae.encode(combined_features, bipartite_edge_index)
            
            # 5. All user-item scores via inner product
            z_users = z[:n_users]
            z_items = z[n_users:]
            # Score(u,i) = z_u @ z_i^T
            scores = torch.sigmoid(torch.mm(z_users, z_items.t()))
        
        return scores
