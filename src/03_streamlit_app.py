import os
import io
import base64
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load environment (must happen before reading env vars)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./qdrant_db"
TEXT_COLLECTION = "mg4_text"
IMAGE_COLLECTION = "mg4_image"

TOP_K_TEXT = 3
TOP_K_IMAGE = 2

# Configurable via .env — go to https://openrouter.ai/models and filter Free + Vision
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-flash-02-23")

SYSTEM_PROMPT = (
    "You are a helpful assistant for the MG4 EV Owner's Manual. "
    "Answer the user's question accurately using ONLY the provided context. "
    "If the context does not contain enough information, say so honestly. "
    "Always cite the page number(s) from the manual when you use information from them."
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Qdrant...")
def get_qdrant_client():
    return QdrantClient(path=DB_PATH)


@st.cache_resource(show_spinner="Loading embedding model (CLIP)...")
def get_embed_model():
    return SentenceTransformer("clip-ViT-B-32")


@st.cache_resource(show_spinner="Initializing OpenRouter client...")
def get_llm_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate_startup(qdrant_client: QdrantClient, llm_client) -> list[str]:
    errors = []

    if llm_client is None:
        errors.append(
            "OPENROUTER_API_KEY is not set. Add it to your `.env` file "
            "and restart the app."
        )

    existing = {c.name for c in qdrant_client.get_collections().collections}
    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        if name not in existing:
            errors.append(
                f"Qdrant collection `{name}` not found. "
                "Run `python src/02_build_multimodal_index.py` first."
            )

    return errors


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def retrieve_text(query: str, client: QdrantClient, model: SentenceTransformer):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    return client.query_points(
        collection_name=TEXT_COLLECTION,
        query=vec,
        limit=TOP_K_TEXT,
        with_payload=True,
    ).points


def retrieve_images(query: str, client: QdrantClient, model: SentenceTransformer):
    vec = model.encode(query, normalize_embeddings=True).tolist()
    return client.query_points(
        collection_name=IMAGE_COLLECTION,
        query=vec,
        limit=TOP_K_IMAGE,
        with_payload=True,
    ).points


def image_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(
    query: str,
    text_hits,
    image_hits,
    user_image: Image.Image | None = None,
) -> list:
    """
    Build OpenAI-compatible messages.
    - user_image: an optional PIL image uploaded by the user
    - image_hits: CLIP-retrieved manual images (also sent to the LLM for context)
    """
    user_content = []

    # User-uploaded image goes first so the model sees it alongside the question
    if user_image is not None:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_b64(user_image)}"},
        })

    # Retrieved text context
    if text_hits:
        ctx_lines = []
        for i, hit in enumerate(text_hits, 1):
            page = hit.payload.get("page_label", "?")
            text = hit.payload.get("text", "")
            ctx_lines.append(f"[Context {i} — Page {page}]\n{text}")
        user_content.append({
            "type": "text",
            "text": f"MANUAL CONTEXT:\n" + "\n\n".join(ctx_lines),
        })

    # CLIP-retrieved manual images
    for hit in image_hits:
        filepath = hit.payload.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            continue
        try:
            img = Image.open(filepath).convert("RGB")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_to_b64(img)}"},
            })
        except Exception:
            pass

    # The actual question
    user_content.append({"type": "text", "text": f"USER QUESTION: {query}"})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def run_rag(
    query: str,
    client,
    model,
    llm_client,
    user_image: Image.Image | None = None,
) -> dict:
    text_hits = retrieve_text(query, client, model)
    image_hits = retrieve_images(query, client, model)
    messages = build_messages(query, text_hits, image_hits, user_image)

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content,
        "text_hits": text_hits,
        "image_hits": image_hits,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="MG4 EV Manual Assistant",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 MG4 EV Manual Assistant")
    st.caption(f"Powered by OpenRouter · `{LLM_MODEL}` · CLIP · Qdrant")

    # Load resources
    qdrant_client = get_qdrant_client()
    embed_model = get_embed_model()
    llm_client = get_llm_client()

    # Startup validation
    errors = validate_startup(qdrant_client, llm_client)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Database Stats")
        try:
            n_text = qdrant_client.count(collection_name=TEXT_COLLECTION).count
            n_img = qdrant_client.count(collection_name=IMAGE_COLLECTION).count
            st.metric("Text chunks", n_text)
            st.metric("Images", n_img)
        except Exception:
            st.warning("Could not load collection stats.")

        st.divider()
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "**Embedding model:** `clip-ViT-B-32` (text + image)\n\n"
            f"**LLM:** `{LLM_MODEL}`\n\n"
            "To change the LLM, set `OPENROUTER_MODEL` in `.env` and restart."
        )

    # Chat history initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("user_image_bytes"):
                st.image(msg["user_image_bytes"], width=300)
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                st.caption(f"Sources: pages {', '.join(msg['pages'])}")
            if msg["role"] == "assistant" and msg.get("image_paths"):
                with st.expander("Relevant images from manual"):
                    cols = st.columns(len(msg["image_paths"]))
                    for col, (path, page) in zip(cols, msg["image_paths"]):
                        if os.path.exists(path):
                            col.image(path, caption=f"Page {page}", use_container_width=True)

    # Image upload + chat input
    uploaded_file = st.file_uploader(
        "Attach an image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Upload a photo (e.g. dashboard warning light) to include in your question.",
    )

    user_image: Image.Image | None = None
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")
        st.image(user_image, caption="Attached image", width=300)

    if prompt := st.chat_input("Ask anything about the MG4 EV…"):
        # Capture uploaded image bytes for history rendering
        user_image_bytes = None
        if user_image is not None:
            buf = io.BytesIO()
            user_image.save(buf, format="PNG")
            user_image_bytes = buf.getvalue()

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "user_image_bytes": user_image_bytes,
        })
        with st.chat_message("user"):
            if user_image_bytes:
                st.image(user_image_bytes, width=300)
            st.markdown(prompt)

        # Run RAG and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating answer…"):
                try:
                    result = run_rag(
                        prompt,
                        qdrant_client,
                        embed_model,
                        llm_client,
                        user_image=user_image,
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.stop()

            answer = result["answer"]
            text_hits = result["text_hits"]
            image_hits = result["image_hits"]

            pages = list(dict.fromkeys(
                hit.payload.get("page_label", "?") for hit in text_hits
            ))

            image_paths = [
                (hit.payload["filepath"], hit.payload.get("page_label", "?"))
                for hit in image_hits
                if hit.payload.get("filepath") and os.path.exists(hit.payload["filepath"])
            ]

            st.markdown(answer)

            if pages:
                st.caption(f"Sources: pages {', '.join(pages)}")

            if image_paths:
                with st.expander("Relevant images from manual"):
                    cols = st.columns(len(image_paths))
                    for col, (path, page) in zip(cols, image_paths):
                        col.image(path, caption=f"Page {page}", use_container_width=True)

        # Persist to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pages": pages,
            "image_paths": image_paths,
        })


if __name__ == "__main__":
    main()
