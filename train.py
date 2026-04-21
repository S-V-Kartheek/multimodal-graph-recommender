"""
Training module for MM-CLightRec model.

Unified hierarchical loss (Change 6):
    L_total = L_BPR + lambda1*L_inter + lambda2*L_struct + lambda3*L_cold + lambda4*L_KL

Where:
- L_BPR:   Bayesian Personalized Ranking loss (ranking supervision)
- L_inter: Inter-modal contrastive loss L1 (modality alignment)  
- L_struct: Structural graph contrastive loss L2 (sparsity compensation)
- L_cold:  Cluster-aware cold-start contrastive loss L3 (journal only)
- L_KL:    VGAE KL divergence (regularization)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.mm_clightrec import MM_CLightRec
from models.contrastive_losses import simulate_cold_start, cold_start_contrastive_loss
from models.content_filtering import cluster_features
from evaluate import evaluate_model, evaluate_link_prediction, evaluate_cold_start
from visualize import (plot_training_loss, plot_metrics_bar, plot_metrics_over_epochs, 
                        plot_link_prediction_metrics, plot_loss_components)


def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking loss.
    L_BPR = -Σ log σ(ŷ_ui - ŷ_uj)
    
    Directly optimizes the relative ranking of observed vs unobserved interactions.
    
    Args:
        pos_scores: Predicted scores for positive (observed) interactions
        neg_scores: Predicted scores for negative (unobserved) interactions
    
    Returns:
        loss: BPR loss scalar
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def select_best_threshold(logits, labels):
    """
    Pick threshold on validation logits by maximizing F1.
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))
    labels = labels.astype(int)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probs > t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def negative_sampling_bpr(user_idx, item_idx, n_items, n_neg=1, popularity=None, user_pos_items=None):
    """
    Generate BPR training triplets: (user, positive_item, negative_item).
    If popularity is provided, 50% of negatives are sampled from the top 20% most popular items
    to create "hard" negatives for better ranking precision.
    """
    n_pos = len(user_idx)
    
    users = np.repeat(user_idx, n_neg)
    pos_items = np.repeat(item_idx, n_neg)

    if popularity is not None:
        thresh = np.percentile(popularity, 80)
        popular_items = np.where(popularity >= thresh)[0]
    else:
        popular_items = np.array([], dtype=np.int64)

    neg_items = np.empty(n_pos * n_neg, dtype=np.int64)
    for t, u in enumerate(users):
        blocked = user_pos_items.get(int(u), set()) if user_pos_items is not None else set()
        for _ in range(30):
            if len(popular_items) > 0 and np.random.rand() > 0.5:
                cand = int(np.random.choice(popular_items))
            else:
                cand = int(np.random.randint(0, n_items))
            if cand not in blocked:
                neg_items[t] = cand
                break
        else:
            # fallback: guaranteed unseen item
            if user_pos_items is not None and len(blocked) < n_items:
                available = np.setdiff1d(np.arange(n_items), np.fromiter(blocked, dtype=np.int64), assume_unique=False)
                neg_items[t] = int(np.random.choice(available))
            else:
                neg_items[t] = int(np.random.randint(0, n_items))

    return users, pos_items, neg_items


def train_mm_clightrec(data, config=None):
    """
    Train the MM-CLightRec model.
    
    Args:
        data: dict from data_loader containing all preprocessed data
        config: dict of hyperparameters
    
    Returns:
        model: trained model
        results: dict of training history and final metrics
    """
    if config is None:
        config = {}
    
    # Hyperparameters
    epochs = config.get('epochs', 100)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    batch_size = config.get('batch_size', 4096)
    cf_embed_dim = config.get('cf_embed_dim', 32)
    cf_n_layers = config.get('cf_n_layers', 3)
    cbf_out_dim = config.get('cbf_out_dim', cf_embed_dim)
    cbf_n_layers = config.get('cbf_n_layers', 2)
    n_user_clusters = config.get('n_user_clusters', 20)
    n_item_clusters = config.get('n_item_clusters', 15)
    vgae_hidden_dim = config.get('vgae_hidden_dim', 100)
    vgae_latent_dim = config.get('vgae_latent_dim', 50)
    contrastive_proj_dim = config.get('contrastive_proj_dim', 64)
    temperature = config.get('temperature', 0.2)
    k = config.get('k', 10)
    eval_every = config.get('eval_every', 10)
    n_neg = config.get('n_neg', 1)
    include_cold_start = config.get('include_cold_start', False)
    
    # Loss weights (λ values)
    lambda_1 = config.get('lambda_1', 0.1)   # L_inter
    lambda_2 = config.get('lambda_2', 0.1)   # L_struct
    lambda_3 = config.get('lambda_3', 0.05)  # L_cold
    lambda_4 = config.get('lambda_4', 0.01)  # L_KL
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Extract data
    user_features = data['user_features'].to(device)
    item_features = data['item_features'].to(device)
    edge_index = data['edge_index'].to(device)
    n_users = data['n_users']
    n_items = data['n_items']
    feature_dim = data['feature_dim']
    user_idx = data['user_idx']
    item_idx = data['item_idx']
    ratings = data['ratings']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    rating_matrix = data['rating_matrix']
    modality_dims = data['modality_dims']
    
    # Per-modality features
    user_modality_features = {
        name: feat.to(device) for name, feat in data['user_modality_features'].items()
    }
    item_modality_features = {
        name: feat.to(device) for name, feat in data['item_modality_features'].items()
    }
    
    print(f"[INFO] Feature dim: {feature_dim}, Users: {n_users}, Items: {n_items}")
    print(f"[INFO] Modality dims: {modality_dims}")
    
    # Initialize model
    model = MM_CLightRec(
        n_users=n_users,
        n_items=n_items,
        user_feature_dim=feature_dim,
        item_feature_dim=feature_dim,
        modality_dims=modality_dims,
        cf_embed_dim=cf_embed_dim,
        cf_n_layers=cf_n_layers,
        cbf_out_dim=cbf_out_dim,
        cbf_n_layers=cbf_n_layers,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        vgae_hidden_dim=vgae_hidden_dim,
        vgae_latent_dim=vgae_latent_dim,
        contrastive_proj_dim=contrastive_proj_dim,
        temperature=temperature,
        include_cold_start=include_cold_start
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters:     {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Loss weights: lambda1={lambda_1}, lambda2={lambda_2}, lambda3={lambda_3}, lambda4={lambda_4}")
    print(f"[INFO] Include cold-start (L3): {include_cold_start}")
    
    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    # KL Annealing (beta warmup)
    # Linearly increase beta from 0 to lambda_4 over first 20 epochs
    warmup_epochs = 20
    
    # Training history
    train_losses = []
    loss_components_history = {
        'L_BPR': [], 'L_inter': [], 'L_struct': [], 'L_KL': [], 'L_cold': [], 'L_total': []
    }
    epoch_metrics_history = {
        'Precision@K': [], 'Recall@K': [], 'NDCG@K': [],
        'F1-Score@K': [], 'Accuracy@K': [], 'RMSE': []
    }
    cold_start_metrics_history = {
        'ColdStart-Hit@K': [], 'ColdStart-NDCG@K': [], 'ColdStart-Recall@K': []
    }
    
    # Build split-specific graph edge indices to avoid evaluation leakage
    def build_edge_index_from_indices(indices):
        users = user_idx[indices]
        items = item_idx[indices]
        return torch.tensor(
            np.stack([
                np.concatenate([users, items + n_users]),
                np.concatenate([items + n_users, users])
            ]),
            dtype=torch.long,
            device=device
        )

    edge_index_train = build_edge_index_from_indices(train_idx)
    edge_index_train_val = build_edge_index_from_indices(np.concatenate([train_idx, val_idx]))

    # User->positive-item sets for safe negative sampling
    user_pos_train = {}
    user_pos_all = {}
    for u, i in zip(user_idx[train_idx], item_idx[train_idx]):
        user_pos_train.setdefault(int(u), set()).add(int(i))
    for u, i in zip(user_idx, item_idx):
        user_pos_all.setdefault(int(u), set()).add(int(i))

    # Training mask for evaluation
    train_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in train_idx:
        u, i = user_idx[idx], item_idx[idx]
        train_mask[u, i] = True

    val_interaction_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in val_idx:
        u, i = user_idx[idx], item_idx[idx]
        val_interaction_mask[u, i] = True

    test_interaction_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in test_idx:
        u, i = user_idx[idx], item_idx[idx]
        test_interaction_mask[u, i] = True
    
    # Training loop
    # Calculate item popularity for hard negative sampling
    item_counts = np.bincount(item_idx[train_idx], minlength=n_items)
    # Build cluster similarity graph ONCE before the logic
    print("[INIT] Building cluster similarity graph (runs once)...")
    cluster_edge_index, user_labels, item_labels = model.cbf_module.build_cluster_graph(
        user_features.cpu(), item_features.cpu()
    )
    print(f"[INIT] Cluster graph ready. Starting training for {epochs} epochs...")
    best_f1 = 0.0
    best_lp_threshold = 0.5
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        
        # Get training interactions
        train_users = user_idx[train_idx]
        train_items = item_idx[train_idx]
        
        # BPR negative sampling with popularity bias (Hard Negatives)
        bpr_users, bpr_pos_items, bpr_neg_items = negative_sampling_bpr(
            train_users, train_items, n_items, n_neg=n_neg, popularity=item_counts,
            user_pos_items=user_pos_train
        )
        
        # Shuffle
        perm = np.random.permutation(len(bpr_users))
        bpr_users = bpr_users[perm]
        bpr_pos_items = bpr_pos_items[perm]
        bpr_neg_items = bpr_neg_items[perm]
        
        # Epoch accumulators
        epoch_loss_bpr = 0.0
        epoch_loss_kl = 0.0
        epoch_loss_inter = 0.0
        epoch_loss_struct = 0.0
        epoch_loss_cold = 0.0
        n_batches = 0
        
        # Mini-batch training
        for start in range(0, len(bpr_users), batch_size):
            end = min(start + batch_size, len(bpr_users))
            
            b_users = torch.tensor(bpr_users[start:end], dtype=torch.long, device=device)
            b_pos_items = torch.tensor(bpr_pos_items[start:end], dtype=torch.long, device=device)
            b_neg_items = torch.tensor(bpr_neg_items[start:end], dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass for POSITIVE pairs
            pos_logits, mu, logvar, z = model(
                user_features, item_features, edge_index_train,
                b_users, b_pos_items,
                user_modality_features, item_modality_features
            )
            
            # Forward pass for NEGATIVE pairs — reuse same z, just decode
            n_users_total = user_features.shape[0]
            neg_item_offset = b_neg_items + n_users_total
            neg_logits = model.vgae.decode_logits(z, b_users, neg_item_offset)
            
            # L_BPR: ranking loss
            loss_bpr = bpr_loss(pos_logits, neg_logits)
            
            # L_KL: VGAE regularization
            loss_kl = model.vgae.kl_divergence(mu, logvar)
            
            # Batch loss = L_BPR + λ₄·L_KL
            # Apply KL annealing (beta warmup)
            beta = lambda_4 * min(1.0, epoch / warmup_epochs)
            
            # --- Contrastive losses (Move inside batch for unified gradients) ---
            # Sampling a subset of contrastive losses per batch to keep it fast
            contrastive_losses = model.compute_contrastive_losses(
                bipartite_edge_index=edge_index_train,
                item_modality_features=item_modality_features
            )
            
            loss_inter = contrastive_losses['L_inter']
            loss_struct = contrastive_losses['L_struct']
            
            # L3 Cold-start loss (if enabled and applicable)
            loss_cold = torch.tensor(0.0, device=device)
            if include_cold_start:
                # Simulate cold start using batch users/items to keep it dynamic and differentiable
                cold_user_ids, _, _ = simulate_cold_start(
                    b_users.cpu().numpy(), b_pos_items.cpu().numpy(), n_users,
                    keep_k=1, cold_ratio=0.3, seed=epoch + start
                )
                
                if len(cold_user_ids) > 0:
                        # CF embeddings (with gradients!)
                    H_U, H_I = model.cf_module(user_features, item_features, edge_index_train)
                    
                    cold_user_ids_unique = np.unique(cold_user_ids)
                    warm_user_ids = np.array([u for u in b_users.cpu().numpy() if u not in set(cold_user_ids_unique)])
                    
                    if len(warm_user_ids) > 0 and len(cold_user_ids_unique) > 0:
                        cold_embeds = H_U[torch.tensor(cold_user_ids_unique, device=device)]
                        # Use batch users as warm pool
                        warm_embeds = H_U[torch.tensor(warm_user_ids, device=device)]
                        
                        # Use cached labels computed before the training loop
                        cold_cluster_labels = torch.tensor(user_labels[cold_user_ids_unique], device=device)
                        warm_cluster_labels = torch.tensor(user_labels[warm_user_ids], device=device)
                        
                        loss_cold = cold_start_contrastive_loss(
                            cold_embeds, warm_embeds,
                            cold_cluster_labels, warm_cluster_labels,
                            temperature
                        )

            # Combined total loss
            batch_loss = loss_bpr + beta * loss_kl + lambda_1 * loss_inter + lambda_2 * loss_struct
            if include_cold_start:
                batch_loss = batch_loss + lambda_3 * loss_cold
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss_bpr += loss_bpr.item()
            epoch_loss_kl += loss_kl.item()
            epoch_loss_inter += loss_inter.item()
            epoch_loss_struct += loss_struct.item()
            epoch_loss_cold += loss_cold.item()
            n_batches += 1
        
        # Step scheduler
        scheduler.step()
        
        # Record losses
        avg_bpr = epoch_loss_bpr / max(n_batches, 1)
        avg_kl = epoch_loss_kl / max(n_batches, 1)
        avg_inter = epoch_loss_inter / max(n_batches, 1)
        avg_struct = epoch_loss_struct / max(n_batches, 1)
        avg_cold = epoch_loss_cold / max(n_batches, 1)
        beta = lambda_4 * min(1.0, epoch / warmup_epochs)
        
        avg_total = avg_bpr + beta * avg_kl + lambda_1 * avg_inter + lambda_2 * avg_struct
        if include_cold_start:
            avg_total += lambda_3 * avg_cold
        
        train_losses.append(avg_total)
        loss_components_history['L_BPR'].append(avg_bpr)
        loss_components_history['L_inter'].append(avg_inter)
        loss_components_history['L_struct'].append(avg_struct)
        loss_components_history['L_KL'].append(avg_kl)
        loss_components_history['L_cold'].append(avg_cold)
        loss_components_history['L_total'].append(avg_total)
        
        epoch_time = time.time() - epoch_start
        
        # Evaluation
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            
            with torch.no_grad():
                # Link prediction on validation set (adding negative edges!)
                val_pos_users = torch.tensor(user_idx[val_idx], dtype=torch.long, device=device)
                val_pos_items = torch.tensor(item_idx[val_idx], dtype=torch.long, device=device)
                
                # Sample negative edges for validation
                val_neg_users, _, val_neg_items = negative_sampling_bpr(
                    user_idx[val_idx], item_idx[val_idx], n_items, n_neg=1,
                    user_pos_items=user_pos_all
                )
                val_neg_users = torch.tensor(val_neg_users, dtype=torch.long, device=device)
                val_neg_items = torch.tensor(val_neg_items, dtype=torch.long, device=device)
                
                val_users = torch.cat([val_pos_users, val_neg_users])
                val_items = torch.cat([val_pos_items, val_neg_items])
                val_labels = torch.cat([
                    torch.ones(len(val_idx), dtype=torch.float32, device=device),
                    torch.zeros(len(val_idx), dtype=torch.float32, device=device)
                ])
                
                val_logits, _, _, _ = model(
                    user_features, item_features, edge_index_train,
                    val_users, val_items,
                    user_modality_features, item_modality_features
                )
                
                current_threshold = select_best_threshold(val_logits, val_labels)
                lp_metrics = evaluate_link_prediction(val_logits, val_labels, threshold=current_threshold)
                
                # Full recommendation evaluation
                # Using optimized get_all_scores to compute matrix efficiently
                score_matrix_tensor = model.get_all_scores(
                    user_features, item_features, edge_index_train,
                    user_modality_features, item_modality_features
                )
                score_matrix = score_matrix_tensor.cpu().numpy()
                
                # Validation metrics should only evaluate val interactions:
                # mask train + test so candidates/targets are val-only.
                eval_val_mask = train_mask | test_interaction_mask
                rec_metrics = evaluate_model(score_matrix, rating_matrix, eval_val_mask, k=k)
                
                # Cold-start evaluation (journal version)
                if include_cold_start:
                    cold_metrics = evaluate_cold_start(
                        score_matrix, rating_matrix,
                        user_idx, item_idx, n_users, n_items,
                        keep_k=5, k=k, cold_ratio=0.2, seed=42
                    )
                else:
                    cold_metrics = {}
            
            for metric_name in epoch_metrics_history:
                if metric_name in rec_metrics:
                    epoch_metrics_history[metric_name].append(rec_metrics[metric_name])
            
            for metric_name in cold_start_metrics_history:
                if metric_name in cold_metrics:
                    cold_start_metrics_history[metric_name].append(cold_metrics[metric_name])
            
            # Track best model
            if rec_metrics['F1-Score@K'] > best_f1:
                best_f1 = rec_metrics['F1-Score@K']
                best_lp_threshold = current_threshold
                best_model_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}
            
            loss_str = (f"L_BPR: {avg_bpr:.4f} | L1: {loss_inter.item():.4f} | "
                       f"L2: {loss_struct.item():.4f} | L_KL: {avg_kl:.4f}")
            if include_cold_start:
                loss_str += f" | L3: {loss_cold.item():.4f}"
            
            metric_str = (f"P@{k}: {rec_metrics['Precision@K']:.4f} | R@{k}: {rec_metrics['Recall@K']:.4f} | "
                         f"NDCG@{k}: {rec_metrics['NDCG@K']:.4f} | F1@{k}: {rec_metrics['F1-Score@K']:.4f}")
            
            cold_str = ""
            if include_cold_start and cold_metrics:
                cold_str = (f" | Cold-Hit@{k}: {cold_metrics.get('ColdStart-Hit@K', 0):.4f}"
                           f" | Cold-NDCG@{k}: {cold_metrics.get('ColdStart-NDCG@K', 0):.4f}"
                           f" | Cold-R@{k}: {cold_metrics.get('ColdStart-Recall@K', 0):.4f}")
            
            print(f"Epoch {epoch:3d}/{epochs} | Total: {avg_total:.4f} | {loss_str} | "
                  f"Time: {epoch_time:.1f}s")
            print(f"  Metrics: {metric_str}{cold_str}")
        else:
            loss_str = (f"L_BPR: {avg_bpr:.4f} | L1: {loss_inter.item():.4f} | "
                       f"L2: {loss_struct.item():.4f} | L_KL: {avg_kl:.4f}")
            if include_cold_start:
                loss_str += f" | L3: {loss_cold.item():.4f}"
            print(f"Epoch {epoch:3d}/{epochs} | Total: {avg_total:.4f} | {loss_str} | "
                  f"Time: {epoch_time:.1f}s")
    
    # Load best model
    if best_model_state:
        model.load_state_dict({k_: v.to(device) for k_, v in best_model_state.items()})
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        # Test link prediction (adding negative edges!)
        test_pos_users = torch.tensor(user_idx[test_idx], dtype=torch.long, device=device)
        test_pos_items = torch.tensor(item_idx[test_idx], dtype=torch.long, device=device)
        
        # Sample negative edges for testing
        test_neg_users, _, test_neg_items = negative_sampling_bpr(
            user_idx[test_idx], item_idx[test_idx], n_items, n_neg=1,
            user_pos_items=user_pos_all
        )
        test_neg_users = torch.tensor(test_neg_users, dtype=torch.long, device=device)
        test_neg_items = torch.tensor(test_neg_items, dtype=torch.long, device=device)
        
        test_users = torch.cat([test_pos_users, test_neg_users])
        test_items = torch.cat([test_pos_items, test_neg_items])
        test_labels = torch.cat([
            torch.ones(len(test_idx), dtype=torch.float32, device=device),
            torch.zeros(len(test_idx), dtype=torch.float32, device=device)
        ])
        
        test_logits, _, _, _ = model(
            user_features, item_features, edge_index_train_val,
            test_users, test_items,
            user_modality_features, item_modality_features
        )
        
        test_lp_metrics = evaluate_link_prediction(test_logits, test_labels, threshold=best_lp_threshold)
        
        # Full recommendation evaluation on test
        print("[INFO] Computing test scores for all users...")
        score_matrix_tensor = model.get_all_scores(
            user_features, item_features, edge_index_train_val,
            user_modality_features, item_modality_features
        )
        score_matrix = score_matrix_tensor.cpu().numpy()
        
        # Test mask
        test_mask = np.zeros((n_users, n_items), dtype=bool)
        for idx in np.concatenate([train_idx, val_idx]):
            u, i = user_idx[idx], item_idx[idx]
            test_mask[u, i] = True
        
        final_metrics = evaluate_model(score_matrix, rating_matrix, test_mask, k=k)
        
        # Cold-start evaluation on test
        if include_cold_start:
            final_cold_metrics = evaluate_cold_start(
                score_matrix, rating_matrix,
                user_idx, item_idx, n_users, n_items,
                keep_k=5, k=k, cold_ratio=0.2, seed=42
            )
        else:
            final_cold_metrics = {}
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    for metric, value in final_metrics.items():
        print(f"{metric:<25} {value:<15.4f}")
    
    if include_cold_start and final_cold_metrics:
        print(f"\nCold-Start Metrics (K=5 shots):")
        for metric, value in final_cold_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nLink Prediction Metrics:")
    for metric, value in test_lp_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    print("\n[INFO] Generating visualizations...")
    plot_training_loss(train_losses, save_path='results/training_loss_ml1m.png',
                       title='Training Loss - MM-CLightRec on MovieLens 1M')
    plot_metrics_bar(final_metrics, save_path='results/metrics_ml1m.png',
                     title='MM-CLightRec Performance on MovieLens 1M')
    
    if epoch_metrics_history['Precision@K']:
        plot_metrics_over_epochs(epoch_metrics_history, save_path='results/metrics_over_epochs_ml1m.png',
                                  title='Metrics Over Epochs - MM-CLightRec')
    
    plot_link_prediction_metrics(test_lp_metrics, save_path='results/link_prediction_ml1m.png')
    
    if loss_components_history['L_BPR']:
        plot_loss_components(loss_components_history, save_path='results/loss_components_ml1m.png',
                            include_cold_start=include_cold_start)
    
    results = {
        'train_losses': train_losses,
        'loss_components': loss_components_history,
        'final_metrics': final_metrics,
        'cold_start_metrics': final_cold_metrics,
        'lp_metrics': test_lp_metrics,
        'epoch_metrics': epoch_metrics_history,
        'cold_start_epoch_metrics': cold_start_metrics_history,
        'config': config,
        'best_lp_threshold': best_lp_threshold,
    }
    
    return model, results
