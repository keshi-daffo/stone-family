"""
app.py — VisualSearch Streamlit UI
DINOv2 × ChromaDB image similarity search & ingestion
"""

import io
import os
from hashlib import sha1
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
import chromadb

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH       = os.environ.get("CHROMA_PATH", "./chroma_data")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "stonex")
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

# ── Cached resources ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DINOv2 model…")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col    = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col


# ── Helpers ─────────────────────────────────────────────────────────────────────

def embed_image(image: Image.Image) -> list[float]:
    processor, model, device = load_model()
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].tolist()


def embed_batch(images: list[Image.Image]) -> list[list[float]]:
    processor, model, device = load_model()
    rgb    = [img.convert("RGB") for img in images]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def make_id(path: Path, root: Path) -> str:
    return sha1(str(path.relative_to(root)).encode()).hexdigest()


# ── Page setup ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VisualSearch",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("VisualSearch — DINOv2 × ChromaDB")

_, col = load_chroma()

# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Collection")
    count = col.count()
    st.metric("Indexed vectors", count)
    if count == 0:
        st.warning("No images indexed yet. Use the **Ingest** tab to add images.")

    st.divider()
    st.header("Search settings")
    top_k            = st.slider("Top K results", 1, 50, 5)
    include_metadata = st.toggle("Include metadata", value=True)
    filter_folder    = st.text_input("Filter by folder", placeholder="e.g. products")

    st.divider()
    st.header("Danger Zone")
    flush_confirmed = st.checkbox("Confirm — this deletes ALL vectors", key="flush_confirm")
    if st.button("Flush Vector DB", type="primary", disabled=not flush_confirmed,
                 use_container_width=True):
        chroma_client, _ = load_chroma()
        chroma_client.delete_collection(CHROMA_COLLECTION)
        chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        load_chroma.clear()
        st.success("Vector DB flushed.")
        st.rerun()


# ── Tabs ────────────────────────────────────────────────────────────────────────

tab_search, tab_ingest, tab_upload = st.tabs(["Search", "Ingest", "Upload & Embed"])


# ════════════════════════════════════════════════════════════════════════════════
# SEARCH TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.subheader("Upload a query image")
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        key="search_upload",
    )

    if uploaded:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            search_clicked = st.button("Search →", type="primary", use_container_width=True)

        with right:
            if search_clicked:
                if count == 0:
                    st.error("Collection is empty. Ingest images first.")
                else:
                    image = Image.open(io.BytesIO(uploaded.read()))

                    with st.spinner("Embedding image and querying…"):
                        vector = embed_image(image)

                        includes = ["distances"]
                        if include_metadata:
                            includes.append("metadatas")

                        query_kwargs = dict(
                            query_embeddings=[vector],
                            n_results=min(top_k, count),
                            include=includes,
                        )
                        if filter_folder.strip():
                            query_kwargs["where"] = {"class": filter_folder.strip()}

                        results   = col.query(**query_kwargs)

                    ids       = results["ids"][0]
                    distances = results["distances"][0]
                    metadatas = (results.get("metadatas") or [[]])[0] if include_metadata else [None] * len(ids)

                    st.success(f"{len(ids)} result{'s' if len(ids) != 1 else ''} returned")

                    for rank, (id_, dist, meta) in enumerate(zip(ids, distances, metadatas), 1):
                        score = round(1 - dist, 4)

                        with st.container(border=True):
                            img_col, info_col = st.columns([1, 3], gap="medium")

                            with img_col:
                                if meta and "file_path" in meta:
                                    try:
                                        st.image(meta["file_path"], use_container_width=True)
                                    except Exception:
                                        st.caption("(image not accessible)")
                                else:
                                    st.caption("(no preview)")

                            with info_col:
                                st.markdown(f"**#{rank}** &nbsp; `{id_[:16]}…`")
                                st.progress(
                                    max(0.0, min(1.0, float(score))),
                                    text=f"Similarity: **{score:.4f}**",
                                )
                                if meta:
                                    cols = st.columns(2)
                                    for i, (k, v) in enumerate(meta.items()):
                                        cols[i % 2].caption(f"**{k}:** {v}")


# ════════════════════════════════════════════════════════════════════════════════
# INGEST TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Index images into ChromaDB")

    folder_path = st.text_input(
        "Image folder path (absolute path on this machine)",
        placeholder="/Users/you/images",
    )
    col1, col2 = st.columns(2)
    batch_size  = col1.number_input("Batch size", min_value=1, max_value=128, value=16)
    reset_col   = col2.toggle("Reset collection before ingesting", value=False)

    if reset_col:
        st.warning("All existing vectors will be deleted before ingestion starts.")

    start = st.button("Start Ingestion →", type="primary", disabled=not folder_path.strip())

    if start:
        root = Path(folder_path.strip())

        if not root.is_dir():
            st.error(f"Path does not exist or is not a directory: `{folder_path}`")
        else:
            chroma_client, col = load_chroma()

            # Reset if requested
            if reset_col:
                with st.spinner("Resetting collection…"):
                    chroma_client.delete_collection(CHROMA_COLLECTION)
                    col = chroma_client.get_or_create_collection(
                        name=CHROMA_COLLECTION,
                        metadata={"hnsw:space": "cosine"},
                    )
                st.info("Collection reset.")

            # Scan
            with st.spinner("Scanning folder…"):
                all_paths = collect_images(root)
                all_ids   = [make_id(p, root) for p in all_paths]

            st.write(f"Found **{len(all_paths)}** image(s).")

            if not all_paths:
                st.warning("No supported images found in the folder.")
            else:
                # Skip already-indexed
                existing  = set(col.get(ids=all_ids, include=[])["ids"])
                to_ingest = [(p, id_) for p, id_ in zip(all_paths, all_ids) if id_ not in existing]

                st.write(f"Skipping **{len(existing)}** already indexed · "
                         f"Ingesting **{len(to_ingest)}** new")

                if not to_ingest:
                    st.success("Collection is already up to date — nothing to do.")
                else:
                    # Load model once before the loop
                    load_model()

                    progress_bar = st.progress(0, text="Starting…")
                    status_text  = st.empty()
                    metrics      = st.columns(4)
                    m_total      = metrics[0].empty()
                    m_processed  = metrics[1].empty()
                    m_ingested   = metrics[2].empty()
                    m_failed     = metrics[3].empty()

                    total    = len(to_ingest)
                    ingested = 0
                    failed   = 0

                    for batch_start in range(0, total, batch_size):
                        batch = to_ingest[batch_start : batch_start + batch_size]
                        images, ids, metadatas = [], [], []

                        for path, id_ in batch:
                            try:
                                img = Image.open(path).convert("RGB")
                                img.load()
                                images.append(img)
                                ids.append(id_)
                                metadatas.append({
                                    "file_path"    : str(path),
                                    "file_name"    : path.name,
                                    "class"        : path.parent.name,
                                    "relative_path": str(path.relative_to(root)),
                                })
                            except (UnidentifiedImageError, Exception):
                                failed += 1

                        if images:
                            try:
                                embeddings = embed_batch(images)
                                col.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                                ingested += len(ids)
                            except Exception as exc:
                                st.warning(f"Batch error: {exc}")
                                failed += len(ids)

                        processed = min(batch_start + batch_size, total)
                        pct       = processed / total

                        progress_bar.progress(pct, text=f"Processing… {processed} / {total}")
                        status_text.caption(f"Batch {batch_start // batch_size + 1} done")
                        m_total.metric("Total", total)
                        m_processed.metric("Processed", processed)
                        m_ingested.metric("Ingested", ingested)
                        m_failed.metric("Failed", failed)

                    progress_bar.progress(1.0, text="Done")
                    if failed:
                        st.warning(f"Completed — {ingested} ingested, {failed} failed.")
                    else:
                        st.success(f"Done — {ingested} image(s) ingested successfully.")

                    # Bust the cached collection count in sidebar
                    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# UPLOAD & EMBED TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.subheader("Upload images and embed directly into ChromaDB")

    upload_class = st.text_input(
        "Class label (folder / category name)",
        placeholder="e.g. granite, marble, sandstone",
        help="Stored as the 'class' metadata field for every uploaded image.",
    )

    uploaded_files = st.file_uploader(
        "Drop images here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="upload_embed_files",
    )

    if uploaded_files:
        # Thumbnail preview grid (4 columns)
        st.caption(f"{len(uploaded_files)} file(s) selected")
        preview_cols = st.columns(4)
        for i, f in enumerate(uploaded_files):
            preview_cols[i % 4].image(f, caption=f.name, use_container_width=True)

        st.divider()
        embed_clicked = st.button(
            "Embed & Save →", type="primary",
            disabled=not upload_class.strip(),
            help="Enter a class label above before embedding.",
        )

        if embed_clicked:
            _, upload_col = load_chroma()
            load_model()   # warm up model before the loop

            prog       = st.progress(0, text="Starting…")
            n          = len(uploaded_files)
            saved      = 0
            errs       = 0

            for i, f in enumerate(uploaded_files):
                try:
                    image  = Image.open(io.BytesIO(f.read())).convert("RGB")
                    vector = embed_image(image)
                    id_    = sha1(f"{upload_class.strip()}/{f.name}".encode()).hexdigest()
                    upload_col.upsert(
                        ids        = [id_],
                        embeddings = [vector],
                        metadatas  = [{
                            "file_path"    : f.name,
                            "file_name"    : f.name,
                            "class"        : upload_class.strip(),
                            "relative_path": f"{upload_class.strip()}/{f.name}",
                        }],
                    )
                    saved += 1
                except Exception as exc:
                    st.warning(f"Failed to embed `{f.name}`: {exc}")
                    errs += 1

                prog.progress((i + 1) / n, text=f"Embedding… {i + 1} / {n}")

            prog.progress(1.0, text="Done")
            if errs:
                st.warning(f"Completed — {saved} saved, {errs} failed.")
            else:
                st.success(f"{saved} image(s) embedded and saved under class **{upload_class.strip()}**.")

            st.rerun()
