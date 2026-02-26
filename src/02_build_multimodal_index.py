import fitz  # PyMuPDF
import os
import io
import uuid
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- CONFIGURATION ---
PDF_PATH = "data/MG4_EV_Owner_Manual.pdf"
IMAGE_OUTPUT_DIR = "data/images_extracted"
DB_PATH = "./qdrant_db"

# Heuristics for image filtering
MIN_IMAGE_SIZE_BYTES = 5000
MIN_IMAGE_DIM = 200

# Citation adjustment (0 = no offset)
PAGE_OFFSET = 0

# Single embedding dimension — CLIP ViT-B/32 shared text+image space
EMBED_DIM = 512  # clip-ViT-B-32


def extract_images_from_pdf(pdf_path, output_dir):
    """
    Smart Extraction: Rips images from PDF but ignores 'noise'.
    Returns a list of dicts (not LlamaIndex ImageDocument).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    image_dicts = []

    print(f"Processing {len(doc)} pages for images...")

    saved_count = 0
    for page_index in range(len(doc)):
        human_page_num = page_index + 1 + PAGE_OFFSET
        page = doc[page_index]
        image_list = page.get_images(full=True)

        if human_page_num % 50 == 0:
            print(f"  Scanning PDF Page {human_page_num}...")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
            except Exception as e:
                print(f"  Warning: Could not extract image on Page {human_page_num}: {e}")
                continue

            # Heuristic filtering
            if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                continue

            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size
                if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                    continue
                # Convert to PNG bytes for consistent storage
                png_buf = io.BytesIO()
                pil_img.convert("RGB").save(png_buf, format="PNG")
                png_bytes = png_buf.getvalue()
            except Exception:
                continue

            # Save to disk
            filename = f"page_{human_page_num}_img_{img_index}.png"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(png_bytes)

            image_dicts.append({
                "filepath": filepath,
                "page_label": str(human_page_num),
                "source": "mg4_manual",
            })
            saved_count += 1

    print(f"Extracted {saved_count} valid images to {output_dir}")
    return image_dicts


def extract_text_from_pdf(pdf_path, min_chars=50):
    """
    Extract page-level text chunks from the PDF using PyMuPDF.
    Skips pages with fewer than min_chars characters (covers, blanks, etc.).
    Returns a list of dicts with 'text' and 'page_label'.
    """
    doc = fitz.open(pdf_path)
    text_chunks = []

    print(f"Extracting text from {len(doc)} pages...")
    for page_index in range(len(doc)):
        human_page_num = page_index + 1 + PAGE_OFFSET
        page = doc[page_index]
        text = page.get_text("text").strip()

        if len(text) < min_chars:
            continue

        text_chunks.append({
            "text": text,
            "page_label": str(human_page_num),
            "source": "mg4_manual",
        })

    print(f"  Kept {len(text_chunks)} text pages (>= {min_chars} chars)")
    return text_chunks


def create_qdrant_collections(client):
    """Create mg4_text and mg4_image collections, dropping old ones if they exist."""
    for name in ["mg4_text", "mg4_image"]:
        existing = [c.name for c in client.get_collections().collections]
        if name in existing:
            print(f"  Dropping existing collection: {name}")
            client.delete_collection(name)

    client.create_collection(
        collection_name="mg4_text",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection: mg4_text (dim={EMBED_DIM})")

    client.create_collection(
        collection_name="mg4_image",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection: mg4_image (dim={EMBED_DIM})")


def embed_and_upsert_text(client, text_chunks, model, batch_size=32):
    """Embed text chunks with CLIP in batches and upsert to mg4_text collection."""
    print(f"\nEmbedding {len(text_chunks)} text chunks (batch={batch_size})...")
    points = []

    for i in tqdm(range(0, len(text_chunks), batch_size), desc="Text batches"):
        batch = text_chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        for chunk, vec in zip(batch, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "text": chunk["text"],
                        "page_label": chunk["page_label"],
                        "source": chunk["source"],
                    },
                )
            )

    client.upsert(collection_name="mg4_text", points=points)
    print(f"  Upserted {len(points)} text vectors")
    return len(points)


def embed_and_upsert_images(client, image_dicts, model):
    """Embed images with CLIP and upsert to mg4_image collection."""
    print(f"\nEmbedding {len(image_dicts)} images with CLIP...")
    points = []

    for img in tqdm(image_dicts, desc="Images"):
        try:
            pil_img = Image.open(img["filepath"]).convert("RGB")
            vec = model.encode(pil_img, normalize_embeddings=True)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "filepath": img["filepath"],
                        "page_label": img["page_label"],
                        "source": img["source"],
                    },
                )
            )
        except Exception as e:
            print(f"  Warning: skipping {img['filepath']}: {e}")

    client.upsert(collection_name="mg4_image", points=points)
    print(f"  Upserted {len(points)} image vectors")
    return len(points)


def build_index():
    # 1. Extract images from PDF
    image_dicts = extract_images_from_pdf(PDF_PATH, IMAGE_OUTPUT_DIR)

    # 2. Extract text from PDF
    text_chunks = extract_text_from_pdf(PDF_PATH)

    # 3. Load embedding model (downloads on first run, then cached)
    print("\nLoading embedding model...")
    print("  Loading CLIP ViT-B/32 (text + image)...")
    model = SentenceTransformer("clip-ViT-B-32")

    # 4. Initialize Qdrant and create collections
    print("\nInitializing Qdrant...")
    client = QdrantClient(path=DB_PATH)
    create_qdrant_collections(client)

    # 5. Embed and upsert text
    n_text = embed_and_upsert_text(client, text_chunks, model)

    # 6. Embed and upsert images
    n_images = embed_and_upsert_images(client, image_dicts, model)

    # 7. Verification
    print("\n--- Indexing Complete ---")
    text_count = client.count(collection_name="mg4_text").count
    image_count = client.count(collection_name="mg4_image").count
    print(f"  mg4_text  : {text_count} vectors (expected {n_text})")
    print(f"  mg4_image : {image_count} vectors (expected {n_images})")
    print(f"  Database saved to: {DB_PATH}")


if __name__ == "__main__":
    build_index()
