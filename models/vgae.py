"""
Variational Graph Autoencoder (VGAE) for Link Prediction.

Encoder: GCN layers to produce mean (mu) and log-variance (logvar)
Reparameterization: Z = mu + eps * sigma
Decoder: Inner product → sigmoid for link probability
Loss: Reconstruction (BCE) + KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _degree(index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if index.numel() == 0:
        return torch.zeros(num_nodes, device=index.device, dtype=torch.float32)
    return torch.bincount(index, minlength=num_nodes).to(dtype=torch.float32)


def _build_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    r"""
    Build symmetric-normalized adjacency as a sparse COO tensor:
    \hat{A} = D^{-1/2} A D^{-1/2}
    """
    row, col = edge_index
    deg = _degree(row, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()


class GraphConv(nn.Module):
    """Pure PyTorch GCN-style layer using sparse adjacency + linear transform."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        return self.lin(x)


class VGAEEncoder(nn.Module):
    """
    VGAE Encoder using GCN layers.
    
    Shared base GCN layer, then two separate GCN layers for mu and logvar.
    Architecture: input_dim → hidden_dim → (mu_dim, logvar_dim)
    """
    
    def __init__(self, in_channels, hidden_channels=100, latent_channels=50):
        super(VGAEEncoder, self).__init__()
        
        # Shared base GCN layer
        self.base_gcn = GraphConv(in_channels, hidden_channels)
        
        # Mean and log-variance GCN layers
        self.gcn_mu = GraphConv(hidden_channels, latent_channels)
        self.gcn_logvar = GraphConv(hidden_channels, latent_channels)

        # Critical init to prevent KL collapse
        nn.init.normal_(self.gcn_logvar.lin.weight, std=0.01)
        if self.gcn_logvar.lin.bias is not None:
            nn.init.constant_(self.gcn_logvar.lin.bias, -2.0)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Graph edges
        Returns:
            mu: Mean of latent distribution (n_nodes, latent_channels)
            logvar: Log-variance of latent distribution (n_nodes, latent_channels)
        """
        n_nodes = x.shape[0]
        adj = _build_norm_adj(edge_index, num_nodes=n_nodes).to(x.device)

        # Shared base layer
        h = self.base_gcn(x, adj)
        h = F.relu(h)
        
        # Separate mu and logvar
        mu = self.gcn_mu(h, adj)
        logvar = self.gcn_logvar(h, adj).clamp(-4, 4)
        
        return mu, logvar


class VGAE(nn.Module):
    """
    Variational Graph Autoencoder for link prediction.
    
    Encoder: GCN-based, produces mu and logvar
    Decoder: Inner product decoder → sigmoid
    Loss: Reconstruction (BCE) + KL divergence
    """
    
    def __init__(self, in_channels, hidden_channels=100, latent_channels=50):
        super(VGAE, self).__init__()
        self.encoder = VGAEEncoder(in_channels, hidden_channels, latent_channels)
        self.latent_channels = latent_channels
    
    def reparameterize(self, mu, logvar):
        """Sample Z from N(mu, sigma^2) using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def encode(self, x, edge_index):
        """
        Encode node features to latent space.
        Returns: z, mu, logvar
        """
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z, edge_index=None, user_idx=None, item_idx=None):
        """
        Decode latent variables to link probabilities.
        
        If user_idx and item_idx provided, compute specific pair probabilities.
        Otherwise, compute all-pairs probabilities.
        
        p(A_uv = 1 | Z) = sigmoid(z_u^T * z_v)
        """
        if user_idx is not None and item_idx is not None:
            z_u = z[user_idx]
            z_v = z[item_idx]
            return torch.sigmoid((z_u * z_v).sum(dim=1))
        else:
            # Full adjacency reconstruction
            return torch.sigmoid(torch.mm(z, z.t()))
    
    def decode_logits(self, z, user_idx, item_idx):
        """Decode without sigmoid (for BCEWithLogitsLoss)."""
        z_u = z[user_idx]
        z_v = z[item_idx]
        return (z_u * z_v).sum(dim=1)
    
    def forward(self, x, edge_index, user_idx=None, item_idx=None):
        """
        Full forward pass: encode → sample → decode.
        
        Args:
            x: Node features
            edge_index: Graph edges for encoding
            user_idx: User indices for specific predictions
            item_idx: Item indices for specific predictions
        
        Returns:
            link_probs: Predicted link probabilities
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            z: Sampled latent variables
        """
        z, mu, logvar = self.encode(x, edge_index)
        
        if user_idx is not None and item_idx is not None:
            link_logits = self.decode_logits(z, user_idx, item_idx)
        else:
            link_logits = None
        
        return link_logits, mu, logvar, z
    
    @staticmethod
    def kl_divergence(mu, logvar):
        """
        KL divergence: KL(q(Z|G) || p(Z))
        where q is the approximate posterior and p(Z) = N(0, I)
        """
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    @staticmethod
    def reconstruction_loss(logits, labels):
        """Binary cross-entropy reconstruction loss."""
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def loss(self, logits, labels, mu, logvar):
        """
        Total VGAE loss = reconstruction loss + KL divergence.
        L = E_q[log p(A|Z)] - KL(q(Z|G) || p(Z))
        """
        recon_loss = self.reconstruction_loss(logits, labels)
        kl_loss = self.kl_divergence(mu, logvar)
        return recon_loss + kl_loss, recon_loss, kl_loss
