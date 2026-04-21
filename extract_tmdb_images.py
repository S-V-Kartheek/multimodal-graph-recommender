import os
import time
import requests
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
from io import BytesIO
from sklearn.decomposition import TruncatedSVD
import warnings
import zipfile
import urllib.request

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. GET A FREE KEY: Go to https://www.themoviedb.org/signup
# 2. Once logged in, go to Settings -> API -> Request an API Key
# 3. Paste your 32-character key here:
TMDB_API_KEY = "45f47b7048a10dca38a260b5b6dbb4eb" 

DATA_DIR = "ml-1m"
IMAGE_DIM = 64  # The exact dimension our MM-CLightRec model expects for images
MOVIES_FILE = os.path.join(DATA_DIR, "movies.dat")
OUTPUT_FILE = os.path.join(DATA_DIR, "image_feat.npy")

# ==============================================================================
# MODEL SETUP (EfficientNet V2)
# ==============================================================================
print("[INFO] Loading EfficientNet-V2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)
# Remove the final classification layer to get raw embeddings (1280 dims)
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = weights.transforms()

# ==============================================================================
# SCRIPT
# ==============================================================================
def search_tmdb_poster(title, year):
    """Hits the TMDB Search API to find the official movie poster."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results and results[0].get('poster_path'):
                return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except Exception as e:
        pass
    return None

def download_and_embed_image(image_url):
    """Downloads the poster and pushes it through EfficientNet."""
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img_t).cpu().numpy().flatten()
            return embedding
    except Exception:
        pass
    return None

def download_dataset():
    """Downloads and extracts the ML-1M dataset if it missing locally."""
    if not os.path.exists(MOVIES_FILE):
        print("[INFO] MovieLens 1M dataset not found locally. Downloading from GroupLens...")
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = "ml-1m.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        print("[INFO] Dataset ready locally.")

def main():
    if TMDB_API_KEY == "PLACE_YOUR_TMDB_API_KEY_HERE":
        print("[ERROR] You must insert a TMDB API Key at the top of the script!")
        return

    # Automatically download ML-1M if running locally on Windows
    download_dataset()

    if not os.path.exists(MOVIES_FILE):
        print(f"[ERROR] Could not find {MOVIES_FILE}. Make sure the ML-1M dataset is extracted.")
        return

    print("[INFO] Parsing MovieLens 1M movies.dat...")
    # Read the movies mapping
    # movies.dat format: MovieID::Title (Year)::Genres
    movies = {}
    with open(MOVIES_FILE, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            movie_id = int(parts[0])
            title_year = parts[1]
            # Extract Year from string like "Toy Story (1995)"
            year = title_year[-5:-1] if title_year.endswith(')') else ""
            title = title_year[:-7].strip() if title_year.endswith(')') else title_year
            movies[movie_id] = {'title': title, 'year': year}

    n_movies = max(movies.keys()) + 1
    raw_embeddings = np.zeros((n_movies, 1280), dtype=np.float32)
    success_count = 0
    
    print(f"[INFO] Beginning extraction for {len(movies)} movies. This will take a while...")
    
    # Track missing to fill with mean later
    missing_ids = []

    for i, (movie_id, data) in enumerate(movies.items()):
        if i % 100 == 0:
            print(f"  -> Processed {i} / {len(movies)} movies... (Success: {success_count}, Missing: {len(missing_ids)})")
            
        poster_url = search_tmdb_poster(data['title'], data['year'])
        
        if poster_url:
            emb = download_and_embed_image(poster_url)
            if emb is not None:
                raw_embeddings[movie_id] = emb
                success_count += 1
                # TMDB API allows ~40 requests per 10 seconds, light sleep
                time.sleep(0.1)
                continue
                
        # If we failed to find/download
        missing_ids.append(movie_id)
        time.sleep(0.1)

    print(f"\n[INFO] Extraction complete! ({success_count} success, {len(missing_ids)} missing/failed)")

    # Fill missing embeddings with the mean of the successful ones (so it's neutral, not a zero vector)
    if success_count > 0:
        mean_emb = np.mean(raw_embeddings[np.linalg.norm(raw_embeddings, axis=1) > 0], axis=0)
        for mid in missing_ids:
            raw_embeddings[mid] = mean_emb

    # Our MM-CLightRec model config expects IMAGE_DIM = 64
    # EfficientNet outputs 1280 dimensions. We use SVD to compress it perfectly to 64 so it fits your graph!
    print(f"[INFO] Compressing 1280-dim EfficientNet vectors down to {IMAGE_DIM}-dim for MM-CLightRec...")
    svd = TruncatedSVD(n_components=IMAGE_DIM, random_state=42)
    final_embeddings = svd.fit_transform(raw_embeddings)
    
    # Save the numpy file precisely where data_loader.py expects it natively
    np.save(OUTPUT_FILE, final_embeddings)
    print(f"[SUCCESS] Saved final authentic image features to {OUTPUT_FILE}!")
    print(f"[NEXT STEP] You can now remove the 'synthetic proxy' lines in data_loader.py and load this .npy file!")

if __name__ == "__main__":
    main()
