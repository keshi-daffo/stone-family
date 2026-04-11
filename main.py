"""
main.py
─────────────────────────────────────────────────────────────────────────────
DINOv2 × ChromaDB Image Search API

This service accepts image uploads, computes a 768-dimensional visual
embedding using Facebook's DINOv2-base vision transformer, and queries a
local ChromaDB vector store to return the most visually similar images.

The model runs entirely locally — no external AI API calls are made at
query time. ChromaDB is also local, so no cloud vector-DB account is needed.

Run directly:
    python main.py

Or with uvicorn (recommended):
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          → liveness check (always fast, no DB call)
    GET  /stats           → number of indexed vectors in ChromaDB
    POST /search          → upload an image, get top-k nearest neighbours
    POST /ingest          → start a background ingestion job from a server-side folder
    GET  /ingest/status   → poll progress of the running (or last) ingestion job
    POST /ingest/cancel   → request cancellation of a running ingestion
─────────────────────────────────────────────────────────────────────────────
"""

import io
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path as _FsPath

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
import chromadb

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = os.environ.get("CHROMA_PATH", "./chroma_data")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "stonex")

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


# ── Device & Model ────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device}")

log.info("Loading DINOv2-base …")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()
log.info("Model ready.")


# ── ChromaDB ──────────────────────────────────────────────────────────────────
log.info(f"Connecting to ChromaDB at '{CHROMA_PATH}' …")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
log.info(f"ChromaDB ready. Collection '{CHROMA_COLLECTION}' has {collection.count()} vectors.")

# Lock protecting the global `collection` reference.
# Held only long enough to read/swap the reference — never during DB I/O.
_collection_lock = threading.Lock()


# ── Ingestion state ───────────────────────────────────────────────────────────
_ingest_lock  = threading.Lock()
_cancel_event = threading.Event()

_ingest_state: dict = {
    "status"     : "idle",   # idle | running | done | error | cancelled
    "job_id"     : None,
    "folder_path": None,
    "progress"   : {"total": 0, "processed": 0, "ingested": 0, "failed": 0},
    "message"    : "No ingestion has been run yet.",
    "started_at" : None,
    "finished_at": None,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_embedding(image: Image.Image) -> list[float]:
    """Generate a 768-d DINOv2 CLS-token embedding for a single PIL image."""
    image  = image.convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vec = outputs.last_hidden_state[:, 0, :]   # (1, 768)
    return cls_vec.cpu().numpy()[0].tolist()


def _embed_batch(images: list[Image.Image]) -> list[list[float]]:
    """Batch-embed multiple PIL images — more efficient than one at a time."""
    rgb    = [img.convert("RGB") for img in images]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_tokens = outputs.last_hidden_state[:, 0, :]   # (B, 768)
    return cls_tokens.cpu().numpy().tolist()


def _collect_image_paths(root: _FsPath) -> list[_FsPath]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _make_id(path: _FsPath, root: _FsPath) -> str:
    return sha1(str(path.relative_to(root)).encode()).hexdigest()


def _patch_state(**kwargs):
    """Thread-safe partial update of _ingest_state."""
    with _ingest_lock:
        _ingest_state.update(kwargs)


def _patch_progress(**kwargs):
    with _ingest_lock:
        _ingest_state["progress"].update(kwargs)


# ── Background ingestion worker ───────────────────────────────────────────────

def _run_ingestion(job_id: str, folder_path: str, reset: bool, batch_size: int):
    global collection

    root = _FsPath(folder_path)
    log.info(f"[ingest:{job_id[:8]}] start  folder={folder_path}  reset={reset}  batch={batch_size}")

    try:
        # ── Optional reset ─────────────────────────────────────────────────────
        if reset:
            _patch_state(message="Resetting collection…")
            chroma_client.delete_collection(CHROMA_COLLECTION)
            new_col = chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            with _collection_lock:
                collection = new_col
            col = new_col
            log.info(f"[ingest:{job_id[:8]}] collection reset.")
        else:
            with _collection_lock:
                col = collection

        # ── Scan folder ────────────────────────────────────────────────────────
        _patch_state(message="Scanning folder for images…")
        all_paths = _collect_image_paths(root)
        all_ids   = [_make_id(p, root) for p in all_paths]
        log.info(f"[ingest:{job_id[:8]}] found {len(all_paths)} image(s).")

        # ── Skip already-indexed images ────────────────────────────────────────
        existing  = set(col.get(ids=all_ids, include=[])["ids"])
        to_ingest = [(p, id_) for p, id_ in zip(all_paths, all_ids) if id_ not in existing]
        skip_count = len(existing)
        log.info(f"[ingest:{job_id[:8]}] skipping {skip_count} already indexed, ingesting {len(to_ingest)} new.")

        _patch_state(message=f"Skipped {skip_count} already indexed. {len(to_ingest)} new image(s) to process.")
        _patch_progress(total=len(to_ingest), processed=0, ingested=0, failed=0)

        if not to_ingest:
            _patch_state(
                status="done",
                message="Nothing new to ingest — collection is already up to date.",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return

        # ── Batch ingest ───────────────────────────────────────────────────────
        ingested = 0
        failed   = 0

        for batch_start in range(0, len(to_ingest), batch_size):

            # Honour cancel requests between batches
            if _cancel_event.is_set():
                _patch_state(
                    status="cancelled",
                    message=f"Cancelled after {ingested} image(s) ingested.",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                log.info(f"[ingest:{job_id[:8]}] cancelled.")
                return

            batch  = to_ingest[batch_start : batch_start + batch_size]
            images, ids, metadatas = [], [], []

            for path, id_ in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    img.load()
                    images.append(img)
                    ids.append(id_)
                    metadatas.append({
                        "file_path"    : str(path),
                        "filename"     : path.name,
                        "folder_name"  : path.parent.name,
                        "relative_path": str(path.relative_to(root)),
                    })
                except (UnidentifiedImageError, Exception) as exc:
                    log.warning(f"[ingest:{job_id[:8]}] skipping '{path.name}': {exc}")
                    failed += 1

            if images:
                try:
                    embeddings = _embed_batch(images)
                    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    ingested += len(ids)
                except Exception as exc:
                    log.error(f"[ingest:{job_id[:8]}] batch error: {exc}")
                    failed += len(ids)

            processed = min(batch_start + batch_size, len(to_ingest))
            _patch_progress(processed=processed, ingested=ingested, failed=failed)
            _patch_state(message=f"Processing… {processed} / {len(to_ingest)} handled.")

        _patch_state(
            status="done",
            message=f"Done — {ingested} ingested, {failed} failed.",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        log.info(f"[ingest:{job_id[:8]}] complete  ingested={ingested}  failed={failed}")

    except Exception as exc:
        log.error(f"[ingest:{job_id[:8]}] unexpected error: {exc}")
        _patch_state(
            status="error",
            message=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


# ── Response models ───────────────────────────────────────────────────────────
# Pydantic models serve two purposes:
#   1. FastAPI uses them to generate the OpenAPI schema shown in Swagger UI.
#   2. They document exactly what callers should expect in each response.

class HealthResponse(BaseModel):
    status: str = Field(example="ok")
    device: str = Field(example="cpu", description="Compute device the model is running on.")


class StatsResponse(BaseModel):
    collection         : str = Field(example="stonex")
    total_vector_count : int = Field(example=1024, description="Number of image vectors currently indexed.")


class MatchItem(BaseModel):
    id      : str              = Field(description="SHA1 ID derived from the image's relative path.")
    score   : float            = Field(example=0.921, description="Cosine similarity score in [-1, 1]. Higher = more similar.")
    metadata: dict | None      = Field(default=None, description="Stored metadata: file_path, filename, folder_name, relative_path.")
    values  : list[float] | None = Field(default=None, description="Raw 768-d embedding vector. Only present when include_values=true.")


class SearchResponse(BaseModel):
    query_file   : str             = Field(example="query.jpg")
    total_results: int             = Field(example=5)
    filter_folder: str | None      = Field(default=None, example="products")
    matches      : list[MatchItem]


class IngestStartResponse(BaseModel):
    job_id: str = Field(example="3fa85f64-5717-4562-b3fc-2c963f66afa6")
    status: str = Field(example="started")


class IngestProgress(BaseModel):
    total    : int = Field(example=500,  description="Total new images to process.")
    processed: int = Field(example=128,  description="Images handled so far (success + failure).")
    ingested : int = Field(example=120,  description="Successfully embedded and stored.")
    failed   : int = Field(example=8,    description="Images that could not be opened or embedded.")


class IngestStatusResponse(BaseModel):
    status      : str            = Field(example="running",
                                         description="One of: idle | running | done | error | cancelled")
    job_id      : str | None     = Field(default=None, example="3fa85f64-5717-4562-b3fc-2c963f66afa6")
    folder_path : str | None     = Field(default=None, example="/data/images")
    progress    : IngestProgress
    message     : str            = Field(example="Processing… 128 / 500 handled.")
    started_at  : str | None     = Field(default=None, example="2024-01-15T10:30:00+00:00")
    finished_at : str | None     = Field(default=None, example="2024-01-15T10:35:22+00:00")


class CancelResponse(BaseModel):
    status: str = Field(example="cancel_requested")


# ── OpenAPI tag groups ────────────────────────────────────────────────────────
# Tags appear as collapsible sections in Swagger UI, grouping related endpoints.
_TAGS = [
    {
        "name"       : "Monitoring",
        "description": "Liveness and database statistics — lightweight probes with no side effects.",
    },
    {
        "name"       : "Search",
        "description": "Visual similarity search: upload an image and retrieve the nearest neighbours.",
    },
    {
        "name"       : "Ingestion",
        "description": (
            "Populate the ChromaDB collection from a server-side image folder. "
            "Jobs run in a background thread — use `/ingest/status` to poll progress."
        ),
    },
]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DINOv2 × ChromaDB Image Search",
    description = (
        "## Overview\n\n"
        "This API indexes images as 768-dimensional visual embeddings using "
        "[Facebook DINOv2-base](https://huggingface.co/facebook/dinov2-base), "
        "a self-supervised Vision Transformer, and stores them in a local "
        "[ChromaDB](https://www.trychroma.com/) vector database.\n\n"
        "**Query flow:** upload an image → DINOv2 embeds it → ChromaDB returns "
        "the top-k most visually similar indexed images ranked by cosine similarity.\n\n"
        "## Key facts\n\n"
        "- Runs entirely **locally** — no external AI API or cloud vector DB required.\n"
        "- Embeddings are **768-dimensional** cosine-normalised vectors.\n"
        "- Similarity scores are in **[-1, 1]**; 1.0 = identical, 0.0 = unrelated.\n"
        "- Use `POST /ingest` to populate the database before searching.\n"
        "- Use `GET /ingest/status` to monitor ingestion progress.\n"
    ),
    version     = "4.0.0",
    openapi_tags = _TAGS,
    docs_url    = "/docs",       # Swagger UI
    redoc_url   = "/redoc",      # ReDoc alternative
    contact     = {
        "name" : "StoneX Image Search",
        "email": "dev@stonex.com",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary      = "Liveness check",
    tags         = ["Monitoring"],
    response_model = HealthResponse,
)
def health():
    """
    Lightweight liveness probe — performs no database call, always responds fast.

    Returns the HTTP status and the compute device the model is running on
    (`cpu` or `cuda`), which confirms whether GPU acceleration is active.
    """
    return {"status": "ok", "device": device}


@app.get(
    "/stats",
    summary        = "ChromaDB collection statistics",
    tags           = ["Monitoring"],
    response_model = StatsResponse,
    responses      = {502: {"description": "ChromaDB unavailable"}},
)
def stats():
    """
    Return the name and total vector count of the active ChromaDB collection.

    Useful for confirming that ingestion has run and checking how many images
    are currently searchable.
    """
    try:
        with _collection_lock:
            col = collection
        return JSONResponse(content={
            "collection"        : CHROMA_COLLECTION,
            "total_vector_count": col.count(),
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB stats failed: {exc}")


@app.post(
    "/search",
    summary        = "Visual similarity search",
    tags           = ["Search"],
    response_model = SearchResponse,
    responses      = {
        400: {"description": "Uploaded file is not a valid image"},
        415: {"description": "Unsupported image format"},
        500: {"description": "Embedding failed (model error)"},
        502: {"description": "ChromaDB query failed"},
    },
)
async def search(
    file            : UploadFile = File(..., description="Image file (JPG / PNG / WEBP / BMP)"),
    top_k           : int        = Query(default=5, ge=1, le=100,
                                         description="Number of nearest neighbours to return."),
    include_values  : bool       = Query(default=False,
                                         description="Include raw embedding vectors in the response."),
    include_metadata: bool       = Query(default=True,
                                         description="Include stored metadata in the response."),
    filter_folder   : str | None = Query(default=None,
                                         description=(
                                             "Restrict results to a specific folder name "
                                             "(matches the 'folder_name' metadata field)."
                                         )),
):
    """
    Accept an uploaded image, embed it with DINOv2, and return the top-k
    most visually similar images from the ChromaDB collection.
    """
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp",
               "image/bmp", "image/tiff", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Accepted: {', '.join(sorted(allowed))}",
        )

    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    try:
        vector = generate_embedding(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    with _collection_lock:
        col = collection

    count = col.count()
    if count == 0:
        return JSONResponse(content={
            "query_file"   : file.filename,
            "total_results": 0,
            "filter_folder": filter_folder,
            "matches"      : [],
        })

    actual_top_k = min(top_k, count)

    include = ["distances"]
    if include_metadata:
        include.append("metadatas")
    if include_values:
        include.append("embeddings")

    try:
        query_kwargs = dict(
            query_embeddings=[vector],
            n_results=actual_top_k,
            include=include,
        )
        if filter_folder:
            query_kwargs["where"] = {"folder_name": filter_folder}

        results = col.query(**query_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ChromaDB query failed: {exc}")

    ids        = results["ids"][0]
    distances  = results["distances"][0]
    metadatas  = results.get("metadatas", [[]])[0] if include_metadata else [None] * len(ids)
    embeddings = results.get("embeddings", [[]])[0] if include_values  else [None] * len(ids)

    matches = [
        {
            "id"      : id_,
            "score"   : round(1 - dist, 6),
            "metadata": meta,
            "values"  : emb,
        }
        for id_, dist, meta, emb in zip(ids, distances, metadatas, embeddings)
    ]

    return JSONResponse(content={
        "query_file"   : file.filename,
        "total_results": len(matches),
        "filter_folder": filter_folder,
        "matches"      : matches,
    })


@app.post(
    "/ingest",
    summary        = "Start background image ingestion",
    tags           = ["Ingestion"],
    response_model = IngestStartResponse,
    responses      = {
        400: {"description": "folder_path does not exist or is not a directory"},
        409: {"description": "An ingestion job is already running"},
    },
)
def start_ingest(
    folder_path: str  = Query(...,
                              description="Absolute path to the image folder on the server."),
    reset      : bool = Query(default=False,
                              description="Drop and recreate the collection before ingesting."),
    batch_size : int  = Query(default=16, ge=1, le=128,
                              description="Number of images to embed per batch."),
):
    """
    Walk *folder_path* on the server, embed every image with DINOv2, and
    upsert the vectors into ChromaDB.

    - Already-indexed images are skipped automatically (idempotent).
    - Only one job can run at a time; returns 409 if one is already active.
    - Poll **GET /ingest/status** to track progress.
    - Send **POST /ingest/cancel** to abort between batches.
    """
    with _ingest_lock:
        if _ingest_state["status"] == "running":
            raise HTTPException(status_code=409, detail="An ingestion job is already running.")

    if not _FsPath(folder_path).is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist or is not a directory: {folder_path}",
        )

    job_id = str(uuid.uuid4())
    _cancel_event.clear()

    with _ingest_lock:
        _ingest_state.update({
            "status"     : "running",
            "job_id"     : job_id,
            "folder_path": folder_path,
            "progress"   : {"total": 0, "processed": 0, "ingested": 0, "failed": 0},
            "message"    : "Starting…",
            "started_at" : datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        })

    threading.Thread(
        target=_run_ingestion,
        args=(job_id, folder_path, reset, batch_size),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "started"}


@app.get(
    "/ingest/status",
    summary        = "Poll ingestion progress",
    tags           = ["Ingestion"],
    response_model = IngestStatusResponse,
)
def ingest_status():
    """
    Return the live state of the currently running ingestion job, or the
    final state of the last completed/failed/cancelled job.

    Poll this endpoint periodically after calling `POST /ingest` to track
    progress. The `progress.processed` field counts images handled so far.

    **Status values:**
    - `idle` — no job has been run yet
    - `running` — a job is currently in progress
    - `done` — last job completed successfully
    - `error` — last job failed with an unexpected error
    - `cancelled` — last job was stopped via `POST /ingest/cancel`
    """
    with _ingest_lock:
        return dict(_ingest_state)


@app.post(
    "/ingest/cancel",
    summary        = "Cancel the running ingestion",
    tags           = ["Ingestion"],
    response_model = CancelResponse,
    responses      = {409: {"description": "No ingestion is currently running"}},
)
def cancel_ingest():
    """
    Signal the running ingestion to stop after the current batch finishes.
    Returns immediately — poll /ingest/status to confirm cancellation.
    """
    with _ingest_lock:
        if _ingest_state["status"] != "running":
            raise HTTPException(status_code=409, detail="No ingestion is currently running.")
    _cancel_event.set()
    return {"status": "cancel_requested"}


# ── Direct run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8002,
        reload    = False,
        log_level = "info",
    )
