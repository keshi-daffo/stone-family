"""
ingest_data.py
─────────────────────────────────────────────────────────────────────────────
Ingest images into a ChromaDB collection using DINOv2-base embeddings.

This script is run once (or incrementally) to populate the vector database
that main.py's /search endpoint queries. It:

  1. Recursively scans a directory for image files.
  2. Skips images already present in ChromaDB (safe to re-run on the same
     folder — no duplicates will be created).
  3. Opens each image with PIL, embeds it through DINOv2, and stores the
     768-d vector alongside file metadata in ChromaDB.

The metadata stored per image is:
  - file_path     : absolute path on disk
  - filename      : bare filename (e.g. "cat.jpg")
  - folder_name   : immediate parent directory name — used by the API's
                    filter_folder query parameter
  - relative_path : path relative to the scanned root directory

Usage:
    python ingest_data.py --images_dir /path/to/images
    python ingest_data.py --images_dir /path/to/images --batch_size 32
    python ingest_data.py --images_dir /path/to/images --reset

Options:
    --images_dir   Directory to scan recursively for images (required)
    --batch_size   Number of images to embed and upsert per batch (default: 16)
    --reset        Drop and recreate the collection before ingesting
    --chroma_path  Path to ChromaDB storage (default: ./chroma_data)
    --collection   Collection name (default: stonex)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import hashlib
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
import chromadb
from tqdm import tqdm


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


# ── Supported image extensions ────────────────────────────────────────────────
# Only these extensions are considered during directory scanning.
# Pillow supports more formats, but these cover the vast majority of use cases.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest images into ChromaDB")
    parser.add_argument("--images_dir",  required=True,
                        help="Root directory to scan recursively for images")
    parser.add_argument("--batch_size",  type=int, default=16,
                        help="Number of images embedded and written per batch (default: 16). "
                             "Increase on GPU, decrease if running out of RAM.")
    parser.add_argument("--chroma_path", default="./chroma_data",
                        help="Directory where ChromaDB persists its data (default: ./chroma_data)")
    parser.add_argument("--collection",  default="stonex",
                        help="ChromaDB collection name to write into (default: stonex)")
    parser.add_argument("--reset",       action="store_true",
                        help="Drop the entire collection before ingesting — use to start fresh")
    return parser.parse_args()


def load_model(device: str) -> tuple:
    """
    Load the DINOv2-base processor and model onto the given device.

    The processor handles image pre-processing (resize, crop, normalise).
    The model is a Vision Transformer whose CLS token we use as the embedding.
    Both are downloaded from HuggingFace on first run and cached locally.

    Args:
        device: "cuda" or "cpu"

    Returns:
        Tuple of (processor, model) ready for inference.
    """
    log.info("Loading DINOv2-base …")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    # Switch to eval mode to disable dropout — required for deterministic
    # embeddings at inference time.
    model.eval()
    log.info("Model ready.")
    return processor, model


def embed_batch(
    images: list[Image.Image],
    processor,
    model,
    device: str,
) -> list[list[float]]:
    """
    Compute DINOv2 CLS-token embeddings for a batch of PIL images.

    Processing a batch together is significantly faster than embedding images
    one-by-one because GPU operations are parallelised across the batch.

    Args:
        images    : List of PIL Images (any size; processor will resize them).
        processor : HuggingFace image processor for DINOv2.
        model     : DINOv2 model in eval mode.
        device    : "cuda" or "cpu" — must match where the model lives.

    Returns:
        List of 768-d float lists, one per image, in the same order as input.
    """
    # Process all images together — the processor pads/resizes to a uniform
    # size and stacks them into a single tensor batch.
    inputs = processor(images=images, return_tensors="pt").to(device)

    # Disable gradient tracking — we only need the forward pass output,
    # and skipping gradient computation saves memory and time.
    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (batch_size, sequence_length, 768)
    # CLS token is always at position 0 — it aggregates global image context.
    cls_tokens = outputs.last_hidden_state[:, 0, :]  # shape: (B, 768)

    # Convert to nested Python lists for ChromaDB compatibility.
    return cls_tokens.cpu().numpy().tolist()


def collect_image_paths(root: Path) -> list[Path]:
    """
    Recursively find all image files under `root`.

    Only files whose extension matches IMAGE_EXTENSIONS are included.
    Results are sorted for deterministic ordering across runs.

    Args:
        root: Top-level directory to scan.

    Returns:
        Sorted list of absolute Path objects pointing to image files.
    """
    paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def make_id(path: Path, root: Path) -> str:
    """
    Generate a stable, unique ID for an image based on its relative path.

    Using a SHA1 hash of the relative path keeps IDs short, URL-safe, and
    consistent across machines — so the same image always maps to the same ID
    regardless of where the repo is checked out.

    Args:
        path: Absolute path to the image file.
        root: The root directory passed via --images_dir.

    Returns:
        40-character hex SHA1 string.
    """
    rel = str(path.relative_to(root))
    return hashlib.sha1(rel.encode()).hexdigest()


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Resolve to an absolute path so relative paths in metadata are consistent.
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"ERROR: '{images_dir}' is not a directory.")

    # ── Load model ────────────────────────────────────────────────────────────
    processor, model = load_model(device)

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    log.info(f"Connecting to ChromaDB at '{args.chroma_path}' …")
    client = chromadb.PersistentClient(path=args.chroma_path)

    if args.reset:
        # Wipe the collection completely — useful when re-ingesting from scratch
        # after changing the model or metadata schema.
        log.warning(f"--reset: dropping collection '{args.collection}'")
        client.delete_collection(args.collection)

    # get_or_create_collection is idempotent — safe on first run and reruns.
    # cosine distance is the right metric for normalised neural embeddings.
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Collection '{args.collection}' — {collection.count()} vectors before ingestion.")

    # ── Collect images ────────────────────────────────────────────────────────
    all_paths = collect_image_paths(images_dir)
    log.info(f"Found {len(all_paths)} image(s) under '{images_dir}'.")

    if not all_paths:
        log.warning("No images found. Nothing to ingest.")
        return

    # ── Skip already-ingested IDs ─────────────────────────────────────────────
    # Query ChromaDB for which of our computed IDs already exist.
    # This makes re-runs incremental — only new images are processed,
    # so you can safely point the script at a growing folder repeatedly.
    all_ids   = [make_id(p, images_dir) for p in all_paths]
    existing  = set(collection.get(ids=all_ids, include=[])["ids"])
    to_ingest = [(p, id_) for p, id_ in zip(all_paths, all_ids) if id_ not in existing]

    log.info(f"Skipping {len(existing)} already-ingested image(s).")
    log.info(f"Ingesting {len(to_ingest)} new image(s) in batches of {args.batch_size}.")

    if not to_ingest:
        log.info("Nothing new to ingest.")
        return

    # ── Ingest in batches ─────────────────────────────────────────────────────
    failed   = 0
    ingested = 0

    for batch_start in tqdm(range(0, len(to_ingest), args.batch_size), desc="Batches"):
        batch = to_ingest[batch_start : batch_start + args.batch_size]

        # Open each image in the batch, collecting only those that load cleanly.
        # Corrupt or truncated files are logged and counted but don't abort
        # the rest of the batch.
        images, ids, metadatas = [], [], []
        for path, id_ in batch:
            try:
                img = Image.open(path).convert("RGB")
                # image.load() forces full decoding now — catches corrupt files
                # before they reach the GPU and cause a harder-to-debug crash.
                img.load()
                images.append(img)
                ids.append(id_)
                metadatas.append({
                    "file_path"    : str(path),
                    "filename"     : path.name,
                    # folder_name is what the /search?filter_folder= parameter
                    # matches against — it's the immediate parent directory.
                    "folder_name"  : path.parent.name,
                    "relative_path": str(path.relative_to(images_dir)),
                })
            except (UnidentifiedImageError, Exception) as exc:
                log.warning(f"Skipping '{path.name}': {exc}")
                failed += 1

        # Skip the ChromaDB write if every image in this batch failed to open.
        if not images:
            continue

        try:
            embeddings = embed_batch(images, processor, model, device)
            # collection.add() writes IDs, embeddings, and metadata atomically.
            # It will raise if any ID already exists — the earlier deduplication
            # step above prevents that from happening.
            collection.add(
                ids        = ids,
                embeddings = embeddings,
                metadatas  = metadatas,
            )
            ingested += len(ids)
        except Exception as exc:
            log.error(f"Batch failed: {exc}")
            failed += len(ids)

    log.info(f"Done. Ingested: {ingested} | Skipped (errors): {failed}")
    log.info(f"Collection '{args.collection}' now has {collection.count()} vectors.")


if __name__ == "__main__":
    main()
