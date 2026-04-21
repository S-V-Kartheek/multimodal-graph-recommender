"""
Evaluation Metrics for MM-CLightRec.

Standard metrics: Precision@K, Recall@K, NDCG@K, F1-Score@K, Accuracy@K, RMSE
Cold-start metrics (journal version): Hit@10, NDCG@10, Recall@10 on K=5-shot users
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(predicted, actual, k=10):
    """Precision@K: proportion of recommended items that are relevant."""
    predicted_k = predicted[:k]
    relevant = len(set(predicted_k) & set(actual))
    return relevant / k if k > 0 else 0.0


def recall_at_k(predicted, actual, k=10):
    """Recall@K: proportion of relevant items that are recommended."""
    predicted_k = predicted[:k]
    relevant = len(set(predicted_k) & set(actual))
    return relevant / len(actual) if len(actual) > 0 else 0.0


def ndcg_at_k(predicted, actual, k=10):
    """Normalized Discounted Cumulative Gain@K."""
    predicted_k = predicted[:k]
    actual_set = set(actual)
    
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)
    
    ideal_count = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
    
    return dcg / idcg if idcg > 0 else 0.0


def f1_score_at_k(predicted, actual, k=10):
    """F1-Score@K: harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(predicted, actual, k)
    r = recall_at_k(predicted, actual, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def accuracy_at_k(predicted, actual, k=10, n_items=None):
    """Hit-Accuracy@K: 1 if any relevant item appears in top-k, else 0."""
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    return 1.0 if len(predicted_k & actual_set) > 0 else 0.0


def hit_at_k(predicted, actual, k=10):
    """
    Hit@K: binary — 1 if ANY relevant item appears in top-K, else 0.
    Used for cold-start evaluation.
    """
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    return 1.0 if len(predicted_k & actual_set) > 0 else 0.0


def rmse(predicted_ratings, actual_ratings):
    """Root Mean Squared Error."""
    if len(predicted_ratings) == 0:
        return 0.0
    predicted_ratings = np.array(predicted_ratings)
    actual_ratings = np.array(actual_ratings)
    return np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))


def evaluate_model(score_matrix, rating_matrix, train_mask, k=10):
    """
    Evaluate recommendation model across all users.
    
    Returns dict of average metrics: Precision@K, Recall@K, NDCG@K, F1@K, Accuracy@K, RMSE
    """
    if isinstance(score_matrix, torch.Tensor):
        score_matrix = score_matrix.cpu().numpy()
    if isinstance(rating_matrix, torch.Tensor):
        rating_matrix = rating_matrix.cpu().numpy()
    if isinstance(train_mask, torch.Tensor):
        train_mask = train_mask.cpu().numpy()
    
    n_users, n_items = score_matrix.shape
    
    precision_list, recall_list, ndcg_list = [], [], []
    f1_list, accuracy_list, rmse_list = [], [], []
    
    for u in range(n_users):
        actual_items = np.where((rating_matrix[u] >= 4) & (~train_mask[u]))[0]
        if len(actual_items) == 0:
            continue
        
        scores = score_matrix[u].copy()
        scores[train_mask[u]] = -np.inf
        predicted_items = np.argsort(-scores)[:k]
        
        precision_list.append(precision_at_k(predicted_items, actual_items, k))
        recall_list.append(recall_at_k(predicted_items, actual_items, k))
        ndcg_list.append(ndcg_at_k(predicted_items, actual_items, k))
        f1_list.append(f1_score_at_k(predicted_items, actual_items, k))
        accuracy_list.append(accuracy_at_k(predicted_items, actual_items, k))
        
        rated_items = np.where((rating_matrix[u] > 0) & (~train_mask[u]))[0]
        if len(rated_items) > 0:
            # Sigmoid bounded [0, 1] scaled to rating [1, 5] based on formula: rating = 1 + 4 * score
            pred_ratings = 1.0 + 4.0 * scores[rated_items]
            actual_ratings = rating_matrix[u][rated_items]
            rmse_list.append(rmse(pred_ratings, actual_ratings))
    
    metrics = {
        'Precision@K': np.mean(precision_list) if precision_list else 0.0,
        'Recall@K': np.mean(recall_list) if recall_list else 0.0,
        'NDCG@K': np.mean(ndcg_list) if ndcg_list else 0.0,
        'F1-Score@K': np.mean(f1_list) if f1_list else 0.0,
        'Accuracy@K': np.mean(accuracy_list) if accuracy_list else 0.0,
        'RMSE': np.mean(rmse_list) if rmse_list else 0.0,
    }
    
    return metrics


def evaluate_cold_start(score_matrix, rating_matrix, user_idx, item_idx,
                        n_users, n_items, keep_k=5, k=10, cold_ratio=0.2, seed=42):
    """
    Cold-start evaluation: simulate cold-start users and measure recommendation quality.
    
    Simulates cold-start by keeping only `keep_k` interactions for randomly selected users.
    Evaluates Hit@K, NDCG@K, Recall@K on these cold users only.
    
    Args:
        score_matrix: (n_users, n_items) predicted scores
        rating_matrix: (n_users, n_items) ground truth ratings
        user_idx: All user indices from interactions
        item_idx: All item indices from interactions
        n_users: Total number of users
        n_items: Total number of items
        keep_k: K-shot — number of interactions to keep (default 5)
        k: Top-K for evaluation metrics (default 10)
        cold_ratio: Fraction of users to simulate as cold-start (default 0.2)
        seed: Random seed
    
    Returns:
        dict with Hit@K, NDCG@K, Recall@K metrics for cold-start users
    """
    if isinstance(score_matrix, torch.Tensor):
        score_matrix = score_matrix.cpu().numpy()
    if isinstance(rating_matrix, torch.Tensor):
        rating_matrix = rating_matrix.cpu().numpy()
    
    rng = np.random.RandomState(seed)
    
    # Select users who have enough interactions to simulate cold start
    user_interaction_count = np.bincount(user_idx, minlength=n_users)
    eligible_users = np.where(user_interaction_count > keep_k + 5)[0]  # Need enough for eval
    
    if len(eligible_users) == 0:
        return {'ColdStart-Hit@K': 0.0, 'ColdStart-NDCG@K': 0.0, 'ColdStart-Recall@K': 0.0}
    
    n_cold = max(1, int(len(eligible_users) * cold_ratio))
    cold_users = rng.choice(eligible_users, size=min(n_cold, len(eligible_users)), replace=False)
    
    hit_list, ndcg_list, recall_list = [], [], []
    
    for uid in cold_users:
        # Get all interactions for this user
        user_interactions = np.where(user_idx == uid)[0]
        user_items = item_idx[user_interactions]
        user_ratings = rating_matrix[uid]
        
        if len(user_interactions) <= keep_k:
            continue
        
        # Simulate cold-start: keep only K interactions as "known"
        perm = rng.permutation(len(user_interactions))
        known_items = set(user_items[perm[:keep_k]])
        eval_items = np.where((user_ratings >= 4))[0]
        # Remove known items from eval set
        eval_items = np.array([i for i in eval_items if i not in known_items])
        
        if len(eval_items) == 0:
            continue
        
        # Get scores, mask known items
        scores = score_matrix[uid].copy()
        for ki in known_items:
            scores[ki] = -np.inf
        
        predicted_items = np.argsort(-scores)[:k]
        
        hit_list.append(hit_at_k(predicted_items, eval_items, k))
        ndcg_list.append(ndcg_at_k(predicted_items, eval_items, k))
        recall_list.append(recall_at_k(predicted_items, eval_items, k))
    
    cold_metrics = {
        'ColdStart-Hit@K': np.mean(hit_list) if hit_list else 0.0,
        'ColdStart-NDCG@K': np.mean(ndcg_list) if ndcg_list else 0.0,
        'ColdStart-Recall@K': np.mean(recall_list) if recall_list else 0.0,
    }
    
    return cold_metrics


def evaluate_link_prediction(logits, labels, threshold=0.5):
    """
    Evaluate link prediction performance.
    Returns dict with accuracy, precision, recall, f1.
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        probs = 1 / (1 + np.exp(-logits))
    
    predictions = (probs > threshold).astype(int)
    labels_int = labels.astype(int)
    
    tp = np.sum((predictions == 1) & (labels_int == 1))
    fp = np.sum((predictions == 1) & (labels_int == 0))
    fn = np.sum((predictions == 0) & (labels_int == 1))
    tn = np.sum((predictions == 0) & (labels_int == 0))
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    try:
        auc = roc_auc_score(labels_int, probs)
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(labels_int, probs)
    except Exception:
        ap = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': float(threshold),
        'roc_auc': float(auc),
        'pr_auc': float(ap),
    }
