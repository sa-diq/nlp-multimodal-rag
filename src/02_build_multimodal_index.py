import fitz  # PyMuPDF
import os
from PIL import Image
import io

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageDocument
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
PDF_PATH = "data/MG4_EV_Owner_Manual.pdf"
IMAGE_OUTPUT_DIR = "data/images_extracted"
DB_PATH = "./qdrant_db"

# HEURISTICS
MIN_IMAGE_SIZE_BYTES = 5000  
MIN_IMAGE_DIM = 200          

# CITATION ADJUSTMENT
PAGE_OFFSET = 0 

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Smart Extraction: Rips images from PDF but ignores 'noise'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    doc = fitz.open(pdf_path)
    image_docs = []
    
    print(f"Processing {len(doc)} pages for images...")
    
    saved_count = 0
    for page_index in range(len(doc)):
        # HUMAN READABLE PAGE NUMBER (What the PDF Viewer says)
        human_page_num = page_index + 1 + PAGE_OFFSET
        
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        # Log progress every 50 pages so you know it's working
        if human_page_num % 50 == 0:
            print(f"Scanning PDF Page {human_page_num}...")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
            except Exception as e:
                print(f"Warning: Could not extract image on Page {human_page_num}: {e}")
                continue
            
            # --- HEURISTIC FILTERING ---
            if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                continue
                
            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size
                if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
                    continue
            except:
                continue 
                
            # --- SAVE VALID IMAGE ---
            # Filename includes the Human Page Number for easy debugging
            filename = f"page_{human_page_num}_img_{img_index}.png"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Create LlamaIndex Document
            # We cite 'human_page_num' so the bot answer matches the PDF Viewer
            image_doc = ImageDocument(
                image_path=filepath,
                metadata={
                    "page_label": str(human_page_num), 
                    "source": "mg4_manual"
                }
            )
            image_docs.append(image_doc)
            saved_count += 1
            
    print(f"Extracted {saved_count} valid images to {output_dir}")
    return image_docs

def build_index():
    # 1. Extract Images
    image_documents = extract_images_from_pdf(PDF_PATH, IMAGE_OUTPUT_DIR)
    
    # 2. Extract Text
    print("Loading Text...")
    # SimpleDirectoryReader automatically handles PDF text extraction
    text_documents = SimpleDirectoryReader(
        input_files=[PDF_PATH]
    ).load_data()
    print(f"Loaded {len(text_documents)} text pages.")

    # 3. Setup Qdrant
    print("Initializing Qdrant Vector DB...")
    client = QdrantClient(path=DB_PATH)
    
    text_store = QdrantVectorStore(client=client, collection_name="mg4_text")
    image_store = QdrantVectorStore(client=client, collection_name="mg4_image")
    
    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store
    )

    # 4. Create Multi-Modal Index
    print("Embedding Data into Shared Vector Space...")
    index = MultiModalVectorStoreIndex.from_documents(
        text_documents + image_documents,
        storage_context=storage_context,
    )
    
    print("Indexing Complete! Database saved to ./qdrant_db")

if __name__ == "__main__":
    build_index()