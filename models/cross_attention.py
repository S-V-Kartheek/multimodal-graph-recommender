"""
Cross-Attention Mechanism for Hybrid Filtering.
Combines collaborative filtering outputs (H_U, H_I) with content-based filtering output (H_UI).

Equation: Z = Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
Where Q = H_U, K = H_I, V = H_UI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for combining collaborative and content-based features.
    
    Q (Query) = H_U (user features from CF)
    K (Key) = H_I (item features from CF)
    V (Value) = H_UI (mixed features from CBF)
    
    Output Z = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, feature_dim=32, value_dim=None):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.scale = math.sqrt(feature_dim)
        self.value_dim = value_dim if value_dim is not None else feature_dim
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(self.value_dim, feature_dim)
        
        # Output projection
        self.W_o = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, H_U, H_I, H_UI_users, H_UI_items):
        """
        Args:
            H_U: User features from CF (n_users, feature_dim)
            H_I: Item features from CF (n_items, feature_dim)
            H_UI_users: User features from CBF (n_users, feature_dim)
            H_UI_items: Item features from CBF (n_items, feature_dim)
        
        Returns:
            Z_users: Attention-weighted user representations (n_users, feature_dim)
            Z_items: Attention-weighted item representations (n_items, feature_dim)
            attention_weights: The attention matrix
        """
        # Project Q, K, V
        Q = self.W_q(H_U)       # (n_users, d)
        K = self.W_k(H_I)       # (n_items, d)
        V_users = self.W_v(H_UI_users)  # (n_users, d)
        V_items = self.W_v(H_UI_items)  # (n_items, d)
        
        # Compute attention weights: softmax(QK^T / sqrt(d_k))
        # Shape: (n_users, n_items)
        attention_scores = torch.mm(Q, K.t()) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        # Z = A * V (attention-weighted combination)
        # For users: aggregate item-side content features weighted by attention
        Z_users = torch.mm(attention_weights, V_items)  # (n_users, d)
        
        # For items: aggregate user-side content features weighted by transposed attention
        Z_items = torch.mm(attention_weights.t(), V_users)  # (n_items, d)
        
        # Output projection
        Z_users = self.W_o(Z_users)
        Z_items = self.W_o(Z_items)
        
        # Add residual connections
        projected_users = self.W_v(H_UI_users)
        projected_items = self.W_v(H_UI_items)
        Z_users = Z_users + H_U + projected_users
        Z_items = Z_items + H_I + projected_items
        
        return Z_users, Z_items, attention_weights
