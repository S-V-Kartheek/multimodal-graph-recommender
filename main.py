"""
MM-CLightRec: Contrastive Multimodal LightGCN Recommendation System.

Main entry point for training and evaluating the model on MovieLens 1M.

Architecture changes from MGRS-HFA base:
1. Collaborative Filtering: GCN+GAT → unified 3-layer LightGCN
2. Content Filtering: MixHopConv → 4 modality-specific LightGCN + adaptive fusion
3. Inter-Modal Contrastive Loss L₁ (NEW)
4. Structural Graph Contrastive Loss L₂ (NEW)
5. Cold-Start Contrastive Loss L₃ (NEW — journal only)
6. Unified loss: L_BPR + λ₁·L₁ + λ₂·L₂ + λ₃·L₃ + λ₄·L_KL

Usage:
    python main.py
    python main.py --epochs 100 --k 10 --lr 0.001
    python main.py --include_cold_start --epochs 100   # Journal version with L₃
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

from data_loader import load_and_preprocess_ml1m
from data_loader_ml100k import load_and_preprocess_ml100k
from data_loader_microlens import load_microlens_data
from train import train_mm_clightrec


def parse_args():
    parser = argparse.ArgumentParser(
        description='MM-CLightRec: Contrastive Multimodal LightGCN Recommendation'
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ml100k', choices=['ml1m', 'ml100k', 'microlens'],
                        help='Which dataset to use for training')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory for dataset storage')
    parser.add_argument('--feature_dim', type=int, default=200,
                        help='Total feature dimension (4 × mod_dim)')
    parser.add_argument('--mod_dim', type=int, default=50,
                        help='Per-modality feature dimension')
    
    # Model architecture — LightGCN CF
    parser.add_argument('--cf_embed_dim', type=int, default=64,
                        help='LightGCN embedding dimension')
    parser.add_argument('--cf_n_layers', type=int, default=3,
                        help='Number of LightGCN propagation layers')
    
    # Model architecture — Content Filtering
    parser.add_argument('--cbf_out_dim', type=int, default=64,
                        help='Content filtering output dimension')
    parser.add_argument('--cbf_n_layers', type=int, default=2,
                        help='Per-modality LightGCN layers in content filtering')
    parser.add_argument('--n_user_clusters', type=int, default=20,
                        help='Number of user clusters for K-means')
    parser.add_argument('--n_item_clusters', type=int, default=15,
                        help='Number of item clusters for K-means')
    
    # Model architecture — VGAE
    parser.add_argument('--vgae_hidden_dim', type=int, default=100,
                        help='Hidden dimension for VGAE encoder')
    parser.add_argument('--vgae_latent_dim', type=int, default=64,
                        help='Latent dimension for VGAE')
    
    # Contrastive learning
    parser.add_argument('--contrastive_proj_dim', type=int, default=64,
                        help='Projection dimension for contrastive heads')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='InfoNCE temperature τ')
    parser.add_argument('--lambda1', type=float, default=0.1,
                        help='Weight for L₁ (inter-modal contrastive)')
    parser.add_argument('--lambda2', type=float, default=0.1,
                        help='Weight for L₂ (structural contrastive)')
    parser.add_argument('--lambda3', type=float, default=0.05,
                        help='Weight for L₃ (cold-start contrastive, journal only)')
    parser.add_argument('--lambda4', type=float, default=0.01,
                        help='Weight for L_KL (VGAE regularization)')
    parser.add_argument('--include_cold_start', action='store_true', default=True,
                        help='Include L₃ cold-start contrastive loss (journal version)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Training batch size')
    parser.add_argument('--n_neg', type=int, default=1,
                        help='Number of negative samples per positive')
    
    # Evaluation
    parser.add_argument('--k', type=int, default=10,
                        help='Top-K for evaluation metrics')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    if args.dataset == 'ml100k':
        if args.epochs == 200:
            args.epochs = 120
        if args.eval_every == 10:
            args.eval_every = 5
        if args.n_neg == 1:
            args.n_neg = 3
    
    print("=" * 80)
    print("  MM-CLightRec: Contrastive Multimodal LightGCN Recommendation")
    print(f"  Dataset: {args.dataset}")
    print(f"L3 Cold-Start Active  : {'Yes' if args.include_cold_start else 'No'}")
    version = "JOURNAL (with L3 cold-start)" if args.include_cold_start else "CONFERENCE (without L3)"
    print(f"  Version: {version}")
    print("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load and preprocess data
    print(f"\n[STEP 1] Loading and preprocessing {args.dataset} dataset...")
    start_time = time.time()
    
    if args.dataset == 'ml1m':
        data = load_and_preprocess_ml1m(
            data_dir=args.data_dir
        )
    elif args.dataset == 'ml100k':
        data = load_and_preprocess_ml100k(
            data_dir=args.data_dir
        )
    elif args.dataset == 'microlens':
        data = load_microlens_data(
            data_dir=args.data_dir
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    load_time = time.time() - start_time
    print(f"[INFO] Data loading completed in {load_time:.1f} seconds")
    print(f"[INFO] Users: {data['n_users']}, Items: {data['n_items']}")
    print(f"[INFO] User features: {data['user_features'].shape}")
    print(f"[INFO] Item features: {data['item_features'].shape}")
    print(f"[INFO] Per-modality dimensions: {data['modality_dims']}")
    print(f"[INFO] Total interactions: {len(data['user_idx'])}")
    print(f"[INFO] Train/Val/Test: {len(data['train_idx'])}/{len(data['val_idx'])}/{len(data['test_idx'])}")
    
    # Step 2: Train model
    print(f"\n[STEP 2] Training MM-CLightRec model for {args.epochs} epochs...")
    
    config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'cf_embed_dim': args.cf_embed_dim,
        'cf_n_layers': args.cf_n_layers,
        'cbf_out_dim': args.cbf_out_dim,
        'cbf_n_layers': args.cbf_n_layers,
        'n_user_clusters': args.n_user_clusters,
        'n_item_clusters': args.n_item_clusters,
        'vgae_hidden_dim': args.vgae_hidden_dim,
        'vgae_latent_dim': args.vgae_latent_dim,
        'contrastive_proj_dim': args.contrastive_proj_dim,
        'temperature': args.temperature,
        'k': args.k,
        'eval_every': args.eval_every,
        'n_neg': args.n_neg,
        'include_cold_start': args.include_cold_start,
        'lambda_1': args.lambda1,
        'lambda_2': args.lambda2,
        'lambda_3': args.lambda3,
        'lambda_4': args.lambda4,
    }
    
    train_start = time.time()
    model, results = train_mm_clightrec(data, config)
    train_time = time.time() - train_start
    
    print(f"\n[INFO] Total training time: {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    
    # Step 3: Summary
    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    
    print(f"\nFinal Recommendation Metrics (K={args.k}):")
    for metric, value in results['final_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    if args.include_cold_start and results['cold_start_metrics']:
        print(f"\nCold-Start Metrics (K=5 shots, K={args.k} eval):")
        for metric, value in results['cold_start_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nLink Prediction Metrics:")
    for metric, value in results['lp_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nResults saved to: results/")
    print(f"  - training_loss_{args.dataset}.png")
    print(f"  - metrics_{args.dataset}.png")
    print(f"  - metrics_over_epochs_{args.dataset}.png")
    print(f"  - link_prediction_{args.dataset}.png")
    print(f"  - loss_components_{args.dataset}.png")
    
    # Save model
    model_path = f'results/mm_clightrec_{args.dataset}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"  - Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("  Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
