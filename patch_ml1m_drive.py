import json, os

notebook_path = "colab/MM_CLightRec_ML1M.ipynb"

NEW_CELL_SRC = r"""
# ==============================================================================
# 💾 STEP 2.5: Mount Google Drive & Protect Your Results
# ==============================================================================
# This connects your Google Drive and forces the model to save everything 
# directly to your Drive. If Colab disconnects, you won't lose your 8-hour run!

from google.colab import drive
import os

print("[1] Mounting Google Drive...")
drive.mount('/content/drive')

print("\n[2] Setting up permanent storage folder...")
DRIVE_DIR = '/content/drive/MyDrive/MM_CLightRec_ML1M_Results'
os.makedirs(DRIVE_DIR, exist_ok=True)

print("\n[3] Linking local results to Google Drive...")
!rm -rf results
!ln -s "{DRIVE_DIR}" results

print(f"\n✅ SUCCESS! All models, plots, and metrics will save directly to:\n   {DRIVE_DIR}")
"""

def patch_ml1m_notebook():
    if not os.path.exists(notebook_path):
        print(f"File not found: {notebook_path}")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Check if we already added it
    for cell in nb['cells']:
        if "Mount Google Drive & Protect" in "".join(cell.get('source', [])):
            print("Cell already exists!")
            return
            
    # Find the index of Step 3 (Run Training on MovieLens 1M)
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and "Step 3: Run Training on MovieLens 1M" in "".join(cell.get('source', [])):
            insert_idx = i
            break
            
    if insert_idx == -1:
        print("Couldn't find Step 3 cell, appending to end...")
        insert_idx = len(nb['cells'])
        
    # Setup cell format
    lines = NEW_CELL_SRC.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }
    
    # Insert right before Step 3
    nb['cells'].insert(insert_idx, new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Successfully injected Google Drive mount cell into {notebook_path}")

if __name__ == "__main__":
    patch_ml1m_notebook()
