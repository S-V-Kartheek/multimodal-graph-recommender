# MM-CLightRec: AI Assistant Context & Architecture Guide

> **Welcome, AI Assistant!** (Cursor, Copilot, Antigravity, or any other tool)
> This file is your **single source of truth** for understanding the project.
> Read this COMPLETELY before making any changes to the codebase.

---

## ⚠️ CRITICAL — READ BEFORE TOUCHING ANY CODE

These bugs were found during development. **Do NOT reintroduce them:**

### Bug 1 — KL Collapse (KL = 0.0000 throughout training)
```python
# WRONG — default initialization causes KL collapse:
self.vgae_logvar = nn.Linear(d, 32)

# CORRECT — must initialize bias to -2.0:
self.vgae_logvar = nn.Linear(d, 32)
nn.init.normal_(self.vgae_logvar.weight, std=0.01)
nn.init.constant_(self.vgae_logvar.bias, -2.0)

# Also clamp logvar in forward pass:
logvar = logvar.clamp(-4, 4)

# Also lambda4 must be 0.1 NOT 0.01
# lambda4=0.01 causes KL to be ignored
```

### Bug 2 — K-means Inside Training Loop (184-min training)
```python
# WRONG — causes 3+ hour training time:
for epoch in range(n_epochs):
    clusters = model.build_cluster_graph(X_U, X_I)  # NEVER here

# CORRECT — run ONCE before the loop:
model.build_cluster_graph(X_U, X_I)  # runs once ~2min
for epoch in range(n_epochs):
    outputs = model(edge_index, features)
```

### Bug 3 — Link Prediction Data Leakage (accuracy=0.9999)
```python
# WRONG — evaluating on training edges:
evaluate_link_pred(model, train_edges)

# CORRECT — evaluate on test edges + negative samples only:
neg_edges = sample_negative_edges(n_nodes, train_edges)
evaluate_link_pred(model, test_edges, neg_edges)
```

### Bug 4 — RMSE Scale Mismatch
```python
# WRONG — BPR scores are unbounded:
rmse = sqrt(MSE(bpr_raw_scores, ratings))

# CORRECT — scale to rating range [1, 5]:
pred_rating = 1 + 4 * sigmoid(bpr_score)
rmse = sqrt(MSE(pred_rating, ratings))
```

### Bug 5 — PCA Missing Before K-means
```python
# WRONG — K-means on raw 4480-dim features is too slow:
kmeans.fit(user_features_4480_dim)

# CORRECT — PCA to 64-dim first:
from sklearn.decomposition import PCA
pca = PCA(n_components=64, random_state=42)
user_features_64 = pca.fit_transform(user_features)
kmeans.fit(user_features_64)
```

### Bug 6 — Variable Name Conflict in evaluate_cold_start
```python
# WRONG — k shadows config['K']:
k = config['K']
for k, v in data['modality_features'].items():  # conflict!

# CORRECT — rename loop variable:
top_k_count = config['K']
for mod_key, mod_val in data['modality_features'].items():
```

---

## 🎯 Project Goal

We are building **MM-CLightRec** (Multimodal Contrastive LightGCN Recommender)
— an improvement over the baseline paper **MGRS-HFA**
(published in International Journal of Computing and Digital Systems, 2025).

### Target Publication
```
Conference: SCI-2026
            8th International Conference on Smart Computing and Informatics
            Organized by: Swinburne University of Technology, Hanoi, Vietnam
            Date: April 28-29, 2026
            Published in: Springer Lecture Notes in Networks and Systems (LNNS)
            Indexed: SCOPUS
            Submission: https://cmt3.research.microsoft.com/SCI2026

Journal Extension (after conference):
            IEEE TKDE or Information Fusion (Elsevier)
```

### Datasets Used
```
Conference (SCI-2026):
  Primary:   MovieLens 1M   ← same as base paper, direct comparison

Journal Extension:
  Add: MicroLens-100K       ← all 4 real modalities, video domain
  Add: Amazon Sports        ← additional e-commerce dataset

NOT used (rejected):
  Amazon Beauty  → too large (2M users, 2-3 hrs/epoch on T4)
  Amazon Fashion → same problem
  KuaiRand       → no image features
  TikTok         → full dataset unavailable (copyright)
  MicroVideo 1.7M → too large (1.7M items, K-means impractical)
```

---

## 📊 What We Changed from Base Paper

| Component | MGRS-HFA (Base) | MM-CLightRec (Ours) | Status |
|---|---|---|---|
| CF architecture | 2-layer GCN (users) + 2-layer GAT (items) | 3-layer LightGCN (unified bipartite) | ✅ CHANGED |
| CBF architecture | 3× MixHopConv on fused features | 4 modality-specific LightGCN channels | ✅ NOVEL |
| CBF fusion | Fixed concatenation | Learnable α,β,γ,δ softmax weights | ✅ NOVEL |
| Modality alignment | None | L1 Inter-modal contrastive (InfoNCE) | ✅ NEW |
| Sparsity handling | None | L2 Structural graph contrastive | ✅ NEW |
| Cold-start | Not addressed (listed as limitation) | L3 Cluster-aware contrastive | ✅ NEW |
| Training objective | L_VGAE only | L_BPR + λ1·L1 + λ2·L2 + λ3·L3 + λ4·L_KL | ✅ CHANGED |
| Parameters | ~32M | ~2.6M | ✅ 91.8% reduction |
| Training time | ~65 min (ML-10M) | ~40 min/epoch (MovieLens 1M) | ✅ Much faster |

### KEPT from Base Paper (Unchanged):
- Multimodal preprocessing: RoBERTa, EfficientNet-V2, VideoTransformer, Prompt Gen
- K-means clustering concept for content filtering
- Cross-Attention mechanism: Q=H_U, K=H_I, V=H_UI
- VGAE for link prediction
- Bipartite graph G=(U,I,E) construction
- Dataset splits: 80/10/10 train/val/test
- Evaluation K=10 for all metrics

---

## 🏗️ Complete Architecture

### Pipeline Overview
```
INPUT DATA
    │
    ▼
PHASE 1: Multimodal Feature Extraction (UNCHANGED)
    RoBERTa → f_text
    EfficientNet-V2 → f_image      → X_U, X_I → Bipartite Graph G=(U,I,E)
    VideoTransformer → f_video
    PromptGen → f_meta
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
PHASE 2: LightGCN CF (CHANGED)        PHASE 3: MM-LightGCN CBF (NOVEL)
    3-layer LightGCN                      K-means → C_U, C_I clusters
    on bipartite G                        → G_cluster (cosine similarity)
    No W, no σ                            4 modality-specific LightGCN channels
    → H_U, H_I                            → Adaptive fusion α,β,γ,δ
                                          + L1 inter-modal contrastive
                                          → H_UI
    │                                      │
    └──────────────────────────────────────┘
                        │
                        ▼
                PHASE 4: L2 Structural Contrastive (NEW)
                    Graph augmentation (edge dropout + feature masking)
                    InfoNCE between augmented views
                    Compensates for interaction sparsity
                        │
                        ▼
                PHASE 5: L3 Cold-Start Contrastive (NEW) ← ENABLED
                    Simulate K=5 interaction cold users
                    Pull cold embeddings → warm cluster centers
                    No inference overhead
                    Adds exclusive ColdStart-Hit@K metric
                        │
                        ▼
                PHASE 6: Cross-Attention + VGAE (UNCHANGED)
                    Q=H_U, K=H_I, V=H_UI
                    Z = softmax(QK^T / sqrt(d)) · V
                    VGAE: encode → sample → decode → link probs
                        │
                        ▼
                    Top-K Recommendations (K=10)
```

---

## 🧩 Module Details

### Module 1: LightGCN (Collaborative Filtering)
**File:** `models/collaborative_filtering.py`

```python
# Core principle: NO weight matrix W, NO activation σ
# Propagation rule:
e_u^(l+1) = Σ_{i∈N(u)} [1/√(|N(u)|·|N(i)|)] · e_i^(l)
e_i^(l+1) = Σ_{u∈N(i)} [1/√(|N(i)|·|N(u)|)] · e_u^(l)

# Layer combination (final embedding):
e_u* = (1/L+1) · Σ_{l=0}^{L} e_u^(l)

# Implementation: pure PyTorch scatter_add_
# NO PyTorch Geometric dependency
# 3 layers (vs base paper's 2 layers)
```

**Outputs:** `H_U ∈ ℝ^(n_users × embed_dim)`, `H_I ∈ ℝ^(n_items × embed_dim)`

---

### Module 2: MM-LightGCN (Content-Based Filtering)
**File:** `models/content_filtering.py`

```python
# Step 1: K-means clustering (run ONCE before training)
# MUST apply PCA to 64-dim before K-means
pca = PCA(n_components=64)
user_feat_64 = pca.fit_transform(user_features)
item_feat_64 = pca.fit_transform(item_features)
kmeans_u.fit(user_feat_64)  # → user clusters C_U
kmeans_i.fit(item_feat_64)  # → item clusters C_I

# Step 2: Build cluster similarity graph
sim = cosine_similarity(user_centroids, item_centroids)
G_cluster = build_graph(sim, threshold=0.3)

# Step 3: 4 SEPARATE LightGCN channels (NOVEL)
e_text  = LightGCN(G_cluster, f_text)
e_image = LightGCN(G_cluster, f_image)
e_video = LightGCN(G_cluster, f_video)
e_meta  = LightGCN(G_cluster, f_meta)

# Step 4: Learnable adaptive fusion (NOVEL)
[α,β,γ,δ] = softmax(W_fuse · [e_text, e_image, e_video, e_meta])
H_UI = α·e_text + β·e_image + γ·e_video + δ·e_meta
# On MovieLens: α↑ (text/genres dominate)
# On TikTok:   γ↑ (video dominates)
# Model learns this automatically from data
```

**Outputs:** `H_UI ∈ ℝ^((n_users+n_items) × embed_dim)`

---

### Module 3: L1 Inter-Modal Contrastive
**File:** `models/contrastive_losses.py`

```python
# Problem solved: f_text and f_image live in completely
# different embedding spaces → LightGCN propagates noise

# Solution: MLP projection heads → shared contrastive space
class ModalityProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden=128, out=128):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # L2 norm!

# All heads project to SAME dim=128 (critical for InfoNCE)
proj_text  = ModalityProjectionHead(text_dim,  128, 128)
proj_image = ModalityProjectionHead(image_dim, 128, 128)
proj_video = ModalityProjectionHead(video_dim, 128, 128)

# InfoNCE loss (batch-level only — NOT all items)
# batch size for L1: min(256, batch_size)
L1 = InfoNCE(z_text, z_image) + InfoNCE(z_text, z_video)
     + InfoNCE(z_image, z_video)) / 3.0

# Applied BEFORE LightGCN propagation
```

---

### Module 4: L2 Structural Graph Contrastive
**File:** `models/contrastive_losses.py`

```python
# Problem solved: Sparse users have few neighbors
# → LightGCN gives poor embeddings for sparse users

# Solution: Graph augmentation contrastive
# View 1: original embeddings
# View 2: feature-masked embeddings (20% masking)
view1 = F.normalize(all_embeddings, dim=-1)
view2 = F.normalize(all_embeddings * mask, dim=-1)

# InfoNCE between views — same node should be consistent
L2 = InfoNCE(view1[sample_256], view2[sample_256])

# Theoretical grounding: SGL (SIGIR'21), SimGCL (WWW'22),
# LightGCL (WWW'23) — well-established citation chain
```

---

### Module 5: L3 Cold-Start Contrastive
**File:** `models/contrastive_losses.py`

```python
# STATUS: ENABLED for BOTH conference and journal
# lambda3 = 0.05

# Problem solved: Base paper ignores cold-start users
# (explicitly listed as their limitation)

# Solution: Cluster-aware contrastive during training
for cold_user in simulated_cold_users:  # <=5 interactions
    cold_cluster = user_labels[cold_user]

    # Positive: warm users in SAME cluster
    pos = warm_users_in(cold_cluster)
    # Negative: warm users in DIFFERENT clusters
    neg = warm_users_not_in(cold_cluster)

    L3 += InfoNCE(e_cold, e_pos, e_neg)

# At inference: cold user → forward pass → good embedding
# NO inner-loop gradient computation (unlike MAML)
# Adds EXCLUSIVE metrics base paper cannot report:
#   ColdStart-Hit@K
#   ColdStart-NDCG@K
#   ColdStart-Recall@K
```

---

### Module 6: Cross-Attention (UNCHANGED)
**File:** `models/cross_attention.py`

```python
Q = W_Q(H_U)    # queries from CF user embeddings
K = W_K(H_I)    # keys from CF item embeddings
V = W_V(H_UI)   # values from CBF content embeddings

A = softmax(Q @ K.T / sqrt(d_k))
Z = A @ V       # (n_users, embed_dim)
```

---

### Module 7: VGAE (UNCHANGED — but init fix required)
**File:** `models/vgae.py`

```python
# Encoder
mu     = GCN_mu(Z, edge_index)
logvar = GCN_logvar(Z, edge_index)

# CRITICAL initialization fix:
nn.init.constant_(vgae_logvar.bias, -2.0)
logvar = logvar.clamp(-4, 4)  # prevent explosion

# Reparameterization
z = mu + eps * exp(0.5 * logvar)

# Decoder
p(A_uv=1 | z) = sigmoid(z_u^T · z_v)

# Loss
L_KL = -0.5 * mean(1 + logvar - mu² - exp(logvar))
```

---

## 🧠 Loss Function

### Unified Training Objective
```
L_total = L_BPR + λ1·L1 + λ2·L2 + λ3·L3 + λ4·L_KL
```

### Both Conference AND Journal Version
```python
CONFIG = {
    'lambda1': 0.1,    # L1 inter-modal contrastive
    'lambda2': 0.1,    # L2 structural contrastive
    'lambda3': 0.05,   # L3 cold-start ← ENABLED for both
    'lambda4': 0.1,    # KL divergence (NOT 0.01)
    'include_cold_start': True,   # ENABLED for both
}
```

### Loss Term Summary
| Loss | Purpose | When Applied | λ |
|---|---|---|---|
| L_BPR | Ranking supervision | Every batch | 1.0 |
| L1 | Modality alignment | Before graph prop | 0.1 |
| L2 | Sparsity compensation | During training | 0.1 |
| L3 | Cold-start resolution | During training | 0.05 |
| L_KL | VGAE regularization | During training | 0.1 |

---

## 📁 File Structure

```
MM_CLightRec/
├── ARCHITECTURE.md              ← THIS FILE (read first)
│
├── models/
│   ├── mm_clightrec.py          ← main model orchestrator
│   ├── collaborative_filtering.py ← LightGCN CF
│   ├── content_filtering.py     ← MM-LightGCN CBF channels
│   ├── contrastive_losses.py    ← L1, L2, L3 losses
│   ├── cross_attention.py       ← retained from base paper
│   ├── vgae.py                  ← retained from base paper
│   └── mgrs_hfa.py              ← legacy base model (reference only)
│
├── data/
│   ├── data_loader.py           ← MovieLens 1M
│   └── data_loader_microlens.py ← MicroLens-100K (journal)
│
├── train.py                     ← training loop
├── evaluate.py                  ← evaluation metrics
├── main.py                      ← entry point + CLI args
└── config.py                    ← hyperparameter defaults
```

---

## 📊 Dataset Specifications

### MovieLens 1M (Conference Secondary)
```
Source:       grouplens.org/datasets/movielens/1m/
Users:        55,485
Items:         5,986
Interactions: 1,239,508
Sparsity:     99.63%

Feature files (build from raw data):
  text_feat  → TF-IDF/SVD   dim=100  (REAL)
  image_feat → PCA co-occur dim=64   (synthetic proxy)
  video_feat → temporal     dim=20   (synthetic)
  meta_feat  → genre multi-hot dim=18 (REAL — 18 exact genres)

T4 training: ~40 minutes/epoch ⚠️ use checkpoints
```

### MicroLens-100K (Journal Extension)
```
Source:  recsys.westlake.edu.cn/MicroLens-100k-Dataset/
         github.com/westlake-repl/MicroLens
Users:   100,000
Items:    19,738
Inter:   719,405
Sparsity: 99.96%

Feature files (pre-extracted, all REAL):
  image_feat.npy  ← cover images
  text_feat.npy   ← video titles
  video_feat.npy  ← audio features (UNIQUE advantage)
  microlens.inter ← interactions
  u_id_mapping.tsv ← REQUIRED for MicroLens
  i_id_mapping.tsv ← REQUIRED for MicroLens

T4 training: ~20-25 min/epoch
```

---

## ⚙️ Hyperparameters

```python
CONFIG = {
    # Dataset
    'dataset':         'ml1m',  # 'ml1m' | 'microlens'
    'data_dir':        '/content/data/ml1m',

    # Modality dims (auto-detected from data)
    # MovieLens 1M:  image=64,   text=100, video=20, meta=18
    # MicroLens-100K: image=auto, text=auto, video=auto, meta=5

    # Model architecture
    'embed_dim':       64,
    'contrastive_dim': 128,    # projection head output dim
    'n_layers':        3,      # LightGCN layers

    # Clustering (run ONCE before training)
    'n_clusters_u':    20,
    'n_clusters_i':    20,

    # Training
    'n_epochs':        300,
    'lr':              0.001,
    'weight_decay':    1e-5,
    'batch_size':      1024,

    # Loss weights
    'lambda1':         0.1,    # L1 inter-modal
    'lambda2':         0.1,    # L2 structural
    'lambda3':         0.05,   # L3 cold-start ENABLED
    'lambda4':         0.1,    # KL — must be 0.1 not 0.01

    # Contrastive settings
    'temperature':     0.1,    # τ for InfoNCE
    'edge_dropout':    0.1,    # L2 augmentation
    'feat_mask':       0.2,    # L2 augmentation

    # Cold-start
    'include_cold_start': True,
    'cold_fraction':      0.2,  # 20% of users simulated cold
    'keep_k':             5,    # K=5 interactions visible

    # Early stopping
    'patience':        20,      # stop after 20 no-improve evals

    # Evaluation
    'K':               10,      # Top-K recommendations
    'eval_every':      10,      # evaluate every N epochs
    'save_every':      10,      # checkpoint every N epochs

    # Dataset splits (same as base paper)
    'train_ratio':     0.8,
    'val_ratio':       0.1,
    'test_ratio':      0.1,
}
```

---

## 📈 Expected Results

### Loss Convergence
```
Loss   Epoch 1     Epoch 100   Target 300
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_BPR  0.69        0.57        0.40-0.50
L1     4.5-6.9     0.62        0.3-0.5
L2     0.08-0.13   0.22        0.2-0.3
L3     0.7         0.22        0.1-0.3
L_KL   0.0000*     0.0000*     0.1-0.5**
Total  1.15        0.66        0.45-0.55

* KL still 0.0000 — VGAE fix needs fresh model init
** Will improve with fixed model from Cell 10
```

---

## 🛠️ Technology Stack

```
Framework:     PyTorch (pure — NO PyTorch Geometric)
               LightGCN implemented with scatter_add_
               No external GNN library needed

Clustering:    scikit-learn KMeans
               + PCA preprocessing (always 64-dim)
               NO FAISS dependency

Feature ext:   Pre-extracted features used directly
               RoBERTa / EfficientNet-V2 / VideoTransformer
               only for MovieLens (if raw data available)

Hardware:      Google Colab T4 GPU (15GB VRAM)
               No local GPU required
               Use Google Drive for checkpoint storage
```

---

## 📋 Rules for AI Assistants

Read every rule before touching code:

```
1.  LightGCN = NO weight matrix W, NO activation σ
    Any code adding W or σ to LightGCN is WRONG

2.  K-means runs ONCE before training loop
    Never inside the training loop

3.  Always PCA to 64-dim BEFORE K-means
    Raw feature dims (4480+) are too slow

4.  vgae_logvar.bias initialized to -2.0
    logvar must be clamped(-4, 4)

5.  lambda4 = 0.1 (NOT 0.01)
    0.01 causes KL collapse throughout training

6.  L3 ENABLED for both conference and journal
    lambda3 = 0.05, include_cold_start = True

7.  No PyTorch Geometric — pure PyTorch only
    Use scatter_add_ for message passing

8.  All projection heads output dim=128
    L2 normalize: F.normalize(output, dim=-1)

9.  RMSE: scale scores to [1,5]
    pred = 1 + 4 * sigmoid(bpr_score)

10. Evaluate on test edges ONLY
    Never use training edges for link pred eval

11. MicroLens REQUIRES u_id_mapping + i_id_mapping

12. Early stopping patience=20
    Save best model by val NDCG not train loss

13. Move modality features to GPU ONCE before loop
    Not inside each batch iteration

14. L1 computed on batch items only (max 256)
    NOT on all 7050+ items at once
    This was causing L1=8.4 bug

15. clip_grad_norm_(params, max_norm=1.0)
    Always use gradient clipping
```

---

## 🔬 Paper Contributions (5 Total)

```
For SCI-2026 Conference submission:

① LightGCN replaces GCN+GAT in collaborative filtering
  → 91.8% parameter reduction (32M → 2.6M)
  → Faster training, better recommendation accuracy

② Modality-specific LightGCN channels in content filtering
  → 4 independent channels replace single MixHopConv
  → Learnable α,β,γ,δ weights discover dataset-specific
    modality importance automatically

③ L1 Inter-Modal Contrastive Loss
  → MLP projection heads align heterogeneous modality spaces
  → Applied before graph propagation — fixes root cause

④ L2 Structural Graph Contrastive Loss
  → Graph augmentation compensates for 99.88% sparsity
  → Self-supervised signal from augmented views
  → Theoretically grounded (SGL, SimGCL, LightGCL)

⑤ L3 Cluster-Aware Cold-Start Contrastive Loss
  → First paper to combine cluster contrastive with LightGCN
  → Exclusive metrics: ColdStart-Hit@K, ColdStart-NDCG@K
  → Zero inference overhead (unlike MAML)
  → Directly addresses base paper's stated limitation
```

---

## 📅 Timeline

```
Conference deadline: April 28, 2026 (SCI-2026)
Current date:        March 19, 2026
Time remaining:      ~6 weeks

Week 1-2:  Training on MovieLens 1M ← CURRENTLY RUNNING
Week 3:    Run MovieLens 1M experiments
Week 4:    Run baseline comparisons (MGAT, MGCF, LightGCN)
Week 5:    Write 10-12 page paper (LNNS Springer template)
Week 6:    Review + submit via Microsoft CMT

Journal (after conference):
  Add MicroLens-100K + Amazon Sports
  Extend to 15-20 pages
  Target: IEEE TKDE or Information Fusion
```

---

*Last updated: March 2026*
*Model status: Training with L3 enabled on MovieLens 1M*
*Current best NDCG@K: 0.0347 (epoch 130, continuing)*
