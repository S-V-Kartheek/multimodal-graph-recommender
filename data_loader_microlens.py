import os
import pandas as pd
import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'microlens')

def load_microlens_data(data_dir=None):
    """
    Loads and preprocesses the MicroLens dataset.
    Reads interactions from microlens.inter and multimodal features from .npy files.
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    print("[INFO] Loading MicroLens interactions...")
    inter_path = os.path.join(data_dir, 'microlens.inter')
    
    # Read the .inter file (tab-separated)
    # Columns: userID, itemID, rating, timestamp, x_label
    df = pd.read_csv(inter_path, sep='\t')
    
    # Ensure rating is float
    df['rating'] = df['rating'].astype(float)
    
    # Get user and item mapped indices (they are already 0-indexed in the dataset!)
    user_indices = df['userID'].values
    item_indices = df['itemID'].values
    ratings_vals = df['rating'].values
    
    n_users = int(user_indices.max() + 1)
    n_items = int(item_indices.max() + 1)
    
    print(f"[INFO] Dataset stats: {n_users} users, {n_items} items, {len(df)} interactions")
    
    # ---------------------------------------------------------
    # Load Multimodal Features
    # ---------------------------------------------------------
    print("[INFO] Loading multimodal features (.npy files)...")
    
    text_feat_path = os.path.join(data_dir, 'text_feat.npy')
    image_feat_path = os.path.join(data_dir, 'image_feat.npy')
    video_feat_path = os.path.join(data_dir, 'video_feat.npy')
    
    # Load numpy arrays and convert to tensors
    # Expected shape: (n_items, feature_dim)
    text_feat = torch.tensor(np.load(text_feat_path), dtype=torch.float32)
    image_feat = torch.tensor(np.load(image_feat_path), dtype=torch.float32)
    video_feat = torch.tensor(np.load(video_feat_path), dtype=torch.float32)
    
    # Verify bounds
    assert text_feat.shape[0] >= n_items, f"text_feat has {text_feat.shape[0]} rows but max itemID is {n_items-1}"
    
    # Truncate to exact n_items just in case
    text_feat = text_feat[:n_items]
    image_feat = image_feat[:n_items]
    video_feat = video_feat[:n_items]
    
    # User features are not provided, so we initialize them as zero or random vectors
    # or just use small random embeddings. We'll use random (they will be augmented by LightGCN ID embeddings anyway)
    # Let's average item features to create weak user features as a base
    print("[INFO] Generating user features from interacted items...")
    combined_item_feats = torch.cat([text_feat, image_feat, video_feat], dim=1)
    user_features_t = torch.zeros((n_users, combined_item_feats.shape[1]), dtype=torch.float32)
    
    # Scatter add to get sum of item features per user, then divide by count
    for u, i in zip(user_indices, item_indices):
        user_features_t[u] += combined_item_feats[i]
    
    user_counts = np.bincount(user_indices, minlength=n_users)
    user_counts_t = torch.tensor(user_counts, dtype=torch.float32).unsqueeze(1)
    user_counts_t = torch.clamp(user_counts_t, min=1.0)
    user_features_t = user_features_t / user_counts_t
    
    # Project down slightly if it's too large, but for now just use it directly
    item_features_t = combined_item_feats
    
    feature_dim = item_features_t.shape[1]
    
    item_modality_tensors = {
        'text': text_feat,
        'image': image_feat,
        'video': video_feat
    }
    
    # For CBF memory fusion, we also need user specific modality features
    # Similar to above, mean pool item modalities for each user
    user_modality_tensors = {}
    for mod_name, mod_tensor in item_modality_tensors.items():
        u_mod = torch.zeros((n_users, mod_tensor.shape[1]), dtype=torch.float32)
        for u, i in zip(user_indices, item_indices):
            u_mod[u] += mod_tensor[i]
        u_mod = u_mod / user_counts_t
        user_modality_tensors[mod_name] = u_mod
    
    # ---------------------------------------------------------
    # Build Bipartite Graph & Split Data
    # ---------------------------------------------------------
    print("[INFO] Building bipartite graph...")
    
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([user_indices, item_indices + n_users]),
            np.concatenate([item_indices + n_users, user_indices])
        ]),
        dtype=torch.long
    )
    edge_attr = torch.tensor(np.concatenate([ratings_vals, ratings_vals]), dtype=torch.float)
    
    n_interactions = len(user_indices)
    np.random.seed(42)
    indices = np.random.permutation(n_interactions)
    train_end = int(n_interactions * 0.8)
    val_end = int(n_interactions * 0.9)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print("[INFO] Creating rating matrix...")
    # rating_matrix can be large (98k x 17k), keep it sparse-friendly if needed, 
    # but we'll stick to a NumPy array for now.
    rating_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i, r in zip(user_indices, item_indices, ratings_vals):
        rating_matrix[u, i] = r
        
    modality_dims = {
        'text': text_feat.shape[1],
        'image': image_feat.shape[1],
        'video': video_feat.shape[1]
    }
    
    data = {
        'user_features': user_features_t,
        'item_features': item_features_t,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'n_users': n_users,
        'n_items': n_items,
        'feature_dim': feature_dim,
        'user_idx': user_indices,
        'item_idx': item_indices,
        'ratings': ratings_vals,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'rating_matrix': rating_matrix,
        'user_modality_features': user_modality_tensors,
        'item_modality_features': item_modality_tensors,
        'modality_dims': modality_dims,
    }
    
    print("[INFO] Data loading complete!")
    return data

if __name__ == "__main__":
    # Test script
    d = load_microlens_data()
    print("User features shape:", d['user_features'].shape)
    print("Item features shape:", d['item_features'].shape)
    print("Edge index shape:", d['edge_index'].shape)
    print("Modality dims:", d['modality_dims'])
