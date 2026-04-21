"""
Visualization module for MM-CLightRec.
Generates training loss curves, metric bar charts, loss component plots, and graph visualizations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_training_loss(losses, save_path='results/training_loss.png',
                       title='Training Loss - MM-CLightRec on MovieLens 1M'):
    """Plot training loss curve over epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(losses) + 1)
    
    ax.plot(epochs, losses, alpha=0.3, color='brown', label='Training Loss')
    
    window = max(5, len(losses) // 20)
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        offset = window // 2
        ax.plot(range(offset + 1, offset + 1 + len(moving_avg)), moving_avg,
                color='darkred', linewidth=2, label='Mean Training Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training loss plot saved to {save_path}")


def plot_metrics_bar(metrics_dict, save_path='results/metrics_comparison.png',
                     title='MM-CLightRec Performance on MovieLens 1M'):
    """Plot bar chart of evaluation metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metric_names = []
    metric_values = []
    
    for name, value in metrics_dict.items():
        if name != 'RMSE':
            metric_names.append(name)
            metric_values.append(value)
    
    colors = ['#8B4513', '#D2691E', '#DAA520', '#2E8B57', '#4682B4', '#6A5ACD']
    
    bars = ax.bar(metric_names, metric_values, color=colors[:len(metric_names)],
                  edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Metrics bar chart saved to {save_path}")


def plot_metrics_over_epochs(epoch_metrics, save_path='results/metrics_over_epochs.png',
                              title='Metrics Over Epochs - MM-CLightRec'):
    """Plot multiple metrics over training epochs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metric_names = ['Precision@K', 'Recall@K', 'NDCG@K', 'F1-Score@K', 'Accuracy@K', 'RMSE']
    colors = ['#8B4513', '#D2691E', '#DAA520', '#2E8B57', '#4682B4', '#6A5ACD']
    
    for idx, (metric_name, color) in enumerate(zip(metric_names, colors)):
        ax = axes[idx // 3, idx % 3]
        
        if metric_name in epoch_metrics:
            values = epoch_metrics[metric_name]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, color=color, linewidth=1.5, marker='o', markersize=3)
            ax.set_title(metric_name, fontsize=12)
            ax.set_xlabel('Eval Step', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Metrics over epochs plot saved to {save_path}")


def plot_link_prediction_metrics(lp_metrics, save_path='results/link_prediction_metrics.png'):
    """Plot link prediction evaluation metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    names = list(lp_metrics.keys())
    values = list(lp_metrics.values())
    colors = ['#2E8B57', '#4682B4', '#DAA520', '#8B4513']
    
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Link Prediction Performance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Link prediction metrics plot saved to {save_path}")


def plot_loss_components(loss_components, save_path='results/loss_components.png',
                          title='Loss Components Over Training - MM-CLightRec',
                          include_cold_start=False):
    """
    Plot all loss components over training epochs.
    
    Visualizes: L_BPR, L₁ (inter-modal), L₂ (structural), L_KL, L₃ (cold-start, if enabled)
    
    Args:
        loss_components: dict with keys 'L_BPR', 'L_inter', 'L_struct', 'L_KL', 'L_cold', 'L_total'
        save_path: Where to save the plot
        title: Plot title
        include_cold_start: Whether to show L₃
    """
    n_components = 5 if include_cold_start else 4
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    components = [
        ('L_BPR', 'L_BPR (Ranking)', '#e74c3c'),
        ('L_inter', 'L₁ (Inter-Modal)', '#3498db'),
        ('L_struct', 'L₂ (Structural)', '#2ecc71'),
        ('L_KL', 'L_KL (VGAE Reg.)', '#9b59b6'),
    ]
    
    if include_cold_start:
        components.append(('L_cold', 'L₃ (Cold-Start)', '#f39c12'))
    
    # Total loss in the last subplot
    components.append(('L_total', 'L_total', '#2c3e50'))
    
    for idx, (key, label, color) in enumerate(components):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if key in loss_components and loss_components[key]:
            values = loss_components[key]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, alpha=0.3, color=color)
            
            # Moving average
            window = max(3, len(values) // 20)
            if len(values) >= window:
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                offset = window // 2
                ax.plot(range(offset + 1, offset + 1 + len(moving_avg)), moving_avg,
                        color=color, linewidth=2)
            
            ax.set_title(label, fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(label + ' (N/A)', fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'Not computed', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    # Hide empty subplot if only 5 components
    if len(components) < 6:
        axes[1, 2].set_visible(False)
    
    plt.suptitle(title, fontsize=22, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Loss components plot saved to {save_path}")
