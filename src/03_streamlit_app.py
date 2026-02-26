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
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./qdrant_db"
TEXT_COLLECTION = "mg4_text"
IMAGE_COLLECTION = "mg4_image"

BGE_QUERY_PREFIX = "Represent this question for searching relevant passages: "

TOP_K_TEXT = 3
TOP_K_IMAGE = 2

# Free vision-capable model on OpenRouter
LLM_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"

SYSTEM_PROMPT = (
    "You are a helpful assistant for the MG4 EV Owner's Manual. "
    "Answer the user's question accurately using ONLY the provided context. "
    "If the context does not contain enough information, say so honestly. "
    "Always cite the page number(s) from the manual when you use information from them."
)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Qdrant...")
def get_qdrant_client():
    return QdrantClient(path=DB_PATH)


@st.cache_resource(show_spinner="Loading text embedding model (BGE)...")
def get_text_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Loading image embedding model (CLIP)...")
def get_image_model():
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
def validate_startup(client: QdrantClient, llm_client) -> list[str]:
    """Return a list of error strings. Empty list = all good."""
    errors = []

    if llm_client is None:
        errors.append(
            "OPENROUTER_API_KEY is not set. Create a `.env` file with your OpenRouter API key "
            "(copy `.env.example`) and restart the app."
        )

    existing = {c.name for c in client.get_collections().collections}
    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        if name not in existing:
            errors.append(
                f"Qdrant collection `{name}` not found. "
                "Run `python src/02_build_multimodal_index.py` first to build the index."
            )

    return errors


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def retrieve_text(query: str, client: QdrantClient, text_model: SentenceTransformer):
    """Encode query with BGE and search mg4_text."""
    vec = text_model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()
    results = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=vec,
        limit=TOP_K_TEXT,
        with_payload=True,
    ).points
    return results


def retrieve_images(query: str, client: QdrantClient, image_model: SentenceTransformer):
    """Encode query with CLIP (cross-modal: text → image) and search mg4_image."""
    vec = image_model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=IMAGE_COLLECTION,
        query=vec,
        limit=TOP_K_IMAGE,
        with_payload=True,
    ).points
    return results


def build_messages(query: str, text_hits, image_hits) -> list:
    """
    Build the OpenAI-compatible messages list.
    Images are sent as base64 data URLs (supported by vision models on OpenRouter).
    """
    user_content = []

    # Context text
    if text_hits:
        ctx_lines = []
        for i, hit in enumerate(text_hits, 1):
            page = hit.payload.get("page_label", "?")
            text = hit.payload.get("text", "")
            ctx_lines.append(f"[Context {i} — Page {page}]\n{text}")
        context_block = "\n\n".join(ctx_lines)
        user_content.append({"type": "text", "text": f"MANUAL CONTEXT:\n{context_block}"})

    # Images as base64 data URLs
    for hit in image_hits:
        filepath = hit.payload.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            continue
        try:
            img = Image.open(filepath).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception:
            pass  # Skip unreadable images silently

    # Question
    user_content.append({"type": "text", "text": f"USER QUESTION: {query}"})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def run_rag(query: str, client, text_model, image_model, llm_client) -> dict:
    """
    Full RAG pipeline. Returns:
        {
            "answer": str,
            "text_hits": [...],
            "image_hits": [...],
        }
    """
    text_hits = retrieve_text(query, client, text_model)
    image_hits = retrieve_images(query, client, image_model)

    messages = build_messages(query, text_hits, image_hits)

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
    st.caption("Powered by OpenRouter (Llama 3.2 Vision) · BGE + CLIP embeddings · Qdrant")

    # Load resources
    qdrant_client = get_qdrant_client()
    text_model = get_text_model()
    image_model = get_image_model()
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
            "**Models used:**\n"
            f"- Text: `BAAI/bge-small-en-v1.5`\n"
            f"- Image: `clip-ViT-B-32`\n"
            f"- LLM: `{LLM_MODEL}`"
        )

    # Chat history initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                st.caption(f"Sources: pages {', '.join(msg['pages'])}")
            if msg["role"] == "assistant" and msg.get("image_paths"):
                with st.expander("Relevant images from manual"):
                    cols = st.columns(len(msg["image_paths"]))
                    for col, (path, page) in zip(cols, msg["image_paths"]):
                        if os.path.exists(path):
                            col.image(path, caption=f"Page {page}", use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Ask anything about the MG4 EV…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run RAG and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating answer…"):
                try:
                    result = run_rag(
                        prompt,
                        qdrant_client,
                        text_model,
                        image_model,
                        llm_client,
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.stop()

            answer = result["answer"]
            text_hits = result["text_hits"]
            image_hits = result["image_hits"]

            # Collect unique page labels from text hits
            pages = list(
                dict.fromkeys(
                    hit.payload.get("page_label", "?") for hit in text_hits
                )
            )

            # Collect image paths + page labels
            image_paths = []
            for hit in image_hits:
                path = hit.payload.get("filepath", "")
                page = hit.payload.get("page_label", "?")
                if path and os.path.exists(path):
                    image_paths.append((path, page))

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
