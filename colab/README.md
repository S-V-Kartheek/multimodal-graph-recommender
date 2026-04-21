# MGRS-HFA Colab Training Guide

## Quick Start (Google Colab)

### Step 1: Open Colab
Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2: Set GPU Runtime
`Runtime → Change runtime type → GPU (T4)`

### Step 3: Install Dependencies
Run this in the first cell:
```python
!pip install torch-geometric -q
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html -q
!pip install pandas scikit-learn matplotlib tqdm requests -q
```
> **Note:** Check your Colab's PyTorch version with `!python -c "import torch; print(torch.__version__)"` and adjust the URL above to match (e.g. `torch-2.5.0+cu124`).

### Step 4: Upload & Run the Training Script
```python
# Upload MGRS_HFA_Train.py to Colab, then run:
!python MGRS_HFA_Train.py
```
Or copy-paste the entire file into a Colab cell and run it.

### Step 5: Download Results
After training completes, the `results/` folder will contain:

| File | Description |
|------|-------------|
| `training_loss_ml1m.png` | Training loss curve over 100 epochs |
| `metrics_ml1m.png` | Bar chart of final evaluation metrics |
| `metrics_over_epochs_ml1m.png` | Metrics progression during training |
| `mgrs_hfa_ml1m.pth` | Trained model weights |
| `final_metrics.txt` | Text summary of all metrics vs paper targets |
| `training_losses.npy` | Raw loss values per epoch |

Download them:
```python
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

### Step 6: Place Results Locally
Unzip `results.zip` into:
```
c:\code playground\MGRS1\results\
```

## Expected Output
The paper targets for MovieLens 1M (Table II):

| Metric | Paper Target |
|--------|-------------|
| Precision@10 | 0.8269 |
| Recall@10 | 0.8718 |
| NDCG@10 | 0.6844 |
| F1-Score@10 | 0.8484 |
| Accuracy@10 | 0.5182 |
| RMSE | 0.8496 |

## Estimated Time
- **GPU (T4)**: ~15-20 minutes for 100 epochs
- **GPU (A100)**: ~5-10 minutes
