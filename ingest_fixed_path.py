"""
ingest_fixed_path.py
────────────────────────────────────────────────────────
Hardcoded root directory (no CLI args needed)
────────────────────────────────────────────────────────
"""

import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
from pinecone import Pinecone

# ── CONFIG ─────────────────────────────────────────────
# Set these via environment variables before running:
#   export ROOT_DIR=/path/to/your/images
#   export PINECONE_API_KEY=your_key_here
#   export PINECONE_INDEX=stonex  (optional)
ROOT_DIR         = os.environ.get("ROOT_DIR", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "stonex")

if not ROOT_DIR:
    sys.exit("ERROR: ROOT_DIR environment variable is not set. Export it before running.")
if not PINECONE_API_KEY:
    sys.exit("ERROR: PINECONE_API_KEY environment variable is not set. Export it before running.")

BATCH_SIZE = 64
WORKERS    = 4

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

# ── Logging ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger()

# ── Device & Model ─────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

# ── Helpers ────────────────────────────────────────────

def make_vector_id(path: Path):
    return hashlib.md5(str(path).encode()).hexdigest()


def load_image(path: Path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        log.warning(f"Failed: {path}")
        return None


def embed_batch(images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :]
    return cls.cpu().numpy().tolist()


def collect_images(root: Path):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in SUPPORTED_EXTS:
                paths.append(Path(dirpath) / f)
    return paths


def build_metadata(path: Path, root: Path):
    rel = path.relative_to(root)

    return {
        "file_path"    : str(path),
        "file_name"    : path.name,
        "class"        : path.parent.name,
        "relative_path": str(rel),
    }

# ── MAIN ───────────────────────────────────────────────

def ingest():
    root = Path(ROOT_DIR)

    if not root.exists():
        raise Exception("Invalid path")

    log.info(f"Scanning: {root}")
    paths = collect_images(root)
    log.info(f"Total images: {len(paths)}")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    ingested = 0

    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i+BATCH_SIZE]

        # Load images (parallel)
        loaded = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(load_image, p): idx for idx, p in enumerate(batch)}
            for f in as_completed(futures):
                idx = futures[f]
                img = f.result()
                if img:
                    loaded[idx] = img

        if not loaded:
            continue

        valid_idx = sorted(loaded.keys())
        images = [loaded[i] for i in valid_idx]
        valid_paths = [batch[i] for i in valid_idx]

        # Embed
        vectors = embed_batch(images)

        # Prepare records
        records = []
        for p, v in zip(valid_paths, vectors):
            records.append({
                "id": make_vector_id(p),
                "values": v,
                "metadata": build_metadata(p, root)
            })

        # Upsert
        for j in range(0, len(records), 100):
            index.upsert(vectors=records[j:j+100])
            ingested += len(records[j:j+100])

        log.info(f"Done {i+len(batch)} / {len(paths)}")

    log.info(f"✅ Total ingested: {ingested}")


# ── RUN ────────────────────────────────────────────────
if __name__ == "__main__":
    ingest()