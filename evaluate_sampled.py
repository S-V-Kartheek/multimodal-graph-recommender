import torch
import numpy as np
import time
from data_loader_ml100k import load_and_preprocess_ml100k
from models.mm_clightrec import MM_CLightRec
from evaluate import precision_at_k, recall_at_k, ndcg_at_k, f1_score_at_k

def evaluate_sampled(model, user_features_t, item_features_t, bipartite_edge_index, 
                     user_modality_tensors, item_modality_tensors, 
                     rating_matrix, train_mask, k=10, num_negatives=1000):
    """
    Evaluates the model by raking positive test items against exactly N randomly sampled negative items
    (items the user has not interacted with at all).
    This exactly mimics the MGRS-HFA/MGAT multiple-choice baseline protocol.
    """
    print(f"\n[INFO] Starting Sampled Evaluation ({num_negatives} negatives per user)....")
    model.eval()
    n_users = user_features_t.shape[0]
    n_items = item_features_t.shape[0]

    # 1. Get all scores globally (fast matrix multiplication)
    with torch.no_grad():
        score_matrix = model.get_all_scores(
            user_features_t, item_features_t, bipartite_edge_index,
            user_modality_tensors, item_modality_tensors
        )
    score_matrix = score_matrix.cpu().numpy()

    # Metrics accumulators
    precisions, recalls, ndcgs, hits, f1s = [], [], [], [], []
    valid_users = 0

    print(f"[INFO] Computing ranking for {n_users} users...")
    start_time = time.time()
    
    # 2. Iterate per user and perform 1-to-N ranking
    for u in range(n_users):
        # Find positive items in test set (Rating >= 4 and not in Training set)
        pos_items = np.where((rating_matrix[u] >= 4) & (~train_mask[u]))[0]
        
        # If user has no positive test items, skip
        if len(pos_items) == 0:
            continue
            
        # Find all items the user never interacted with (Rating == 0)
        unseen_items = np.where(rating_matrix[u] == 0)[0]
        
        # Randomly sample 'num_negatives' (e.g., 1000)
        if len(unseen_items) > num_negatives:
            neg_items = np.random.choice(unseen_items, size=num_negatives, replace=False)
        else:
            neg_items = unseen_items

        # The evaluation candidate pool (Positives + 1000 Negatives)
        candidate_items = np.concatenate([pos_items, neg_items])
        
        # 3. Retrieve model predicted scores for ONLY these candidates
        candidate_scores = score_matrix[u][candidate_items]

        # 4. Rank the candidates and extract the top-K items
        # argsort sorts ascending, so [-k:][::-1] gets the indices of the largest scores descending
        top_k_candidate_indices = np.argsort(candidate_scores)[-k:][::-1]
        
        # Map those indices back to the actual item IDs
        top_k_items = candidate_items[top_k_candidate_indices]
        
        # 5. Compute metrics against pos_items
        p = precision_at_k(top_k_items, pos_items, k=k)
        r = recall_at_k(top_k_items, pos_items, k=k)
        ndcg = ndcg_at_k(top_k_items, pos_items, k=k)
        f1 = f1_score_at_k(top_k_items, pos_items, k=k)
        
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg)
        f1s.append(f1)
        hits.append(1.0 if r > 0 else 0.0)
        valid_users += 1

    print(f"[INFO] Evaluated {valid_users} users in {time.time() - start_time:.2f}s")
    
    # Average the metrics globally
    return {
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'NDCG@10': np.mean(ndcgs),
        'F1-Score@10': np.mean(f1s),
        'HitRatio@10': np.mean(hits)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=====================================================")
    print(f"   MGAT / MGRS-HFA Sample-Based Evaluation Protocol  ")
    print(f"=====================================================")
    
    # 1. Load the exact ML-100K data and pre-trained weights
    data = load_and_preprocess_ml100k()
    
    # 2. Instantiate Model
    model = MM_CLightRec(
        n_users=data['n_users'],
        n_items=data['n_items'],
        user_feature_dim=data['user_features'].shape[1],
        item_feature_dim=data['item_features'].shape[1],
        modality_dims=data['modality_dims'],
        cf_embed_dim=64,
        cbf_out_dim=64,
        vgae_latent_dim=64,
        include_cold_start=True
    ).to(device)
    
    # Load state dict
    model_path = "results/mm_clightrec_ml100k.pth"
    print(f"[INFO] Loading trained model from {model_path}...")
    try:
         model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
         print(f"[ERROR] Could not load model. Has main.py finished running? {e}")
         return
         
    # Compute train_mask manually just like in train.py
    n_users = data['n_users']
    n_items = data['n_items']
    train_mask = np.zeros((n_users, n_items), dtype=bool)
    for idx in data['train_idx']:
        u, i = data['user_idx'][idx], data['item_idx'][idx]
        train_mask[u, i] = True
    
    # Push tensors to device using exact ml100k keys
    u_feat = data['user_features'].to(device)
    i_feat = data['item_features'].to(device)
    edge_idx = data['edge_index'].to(device)
    u_mod = {k: v.to(device) for k, v in data['user_modality_features'].items()} if data['user_modality_features'] else None
    i_mod = {k: v.to(device) for k, v in data['item_modality_features'].items()} if data['item_modality_features'] else None

    # Re-build the Content-Based K-Means Cluster Graph
    # (These labels are generated dynamically and not saved in the .pth weights)
    print("\n[INFO] Building K-Means cluster graph (runs once)...")
    cluster_edge_index, user_labels, item_labels = model.cbf_module.build_cluster_graph(
        u_feat.cpu(), i_feat.cpu()
    )

    # 3. Run the exact 1000-sample multiple-choice evaluation
    metrics = evaluate_sampled(
        model=model,
        user_features_t=u_feat,
        item_features_t=i_feat,
        bipartite_edge_index=edge_idx,
        user_modality_tensors=u_mod,
        item_modality_tensors=i_mod,
        rating_matrix=data['rating_matrix'],
        train_mask=train_mask,
        k=10,
        num_negatives=1000
    )
    
    print("\n=====================================================")
    print(f" REPORTED METRICS FOR YOUR CONFERENCE TABLE")
    print(f" (Comparable 1:1 against MGRS-HFA baseline printed numbers)")
    print("=====================================================")
    print(f"  Sampled HitRatio@10 : {metrics['HitRatio@10']:.4f}")
    print(f"  Sampled NDCG@10     : {metrics['NDCG@10']:.4f}")
    print(f"  Sampled Recall@10   : {metrics['Recall@10']:.4f}")
    print(f"  Sampled Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Sampled F1-Score@10 : {metrics['F1-Score@10']:.4f}")
    print("=====================================================")

if __name__ == "__main__":
    main()
