import fitz  
import os

PDF_PATH = "data/MG4_EV_Owner_Manual.pdf"

def analyze_pdf(path):
    if not os.path.exists(path):
        print(f" Error: File not found at {path}")
        return

    print(f" Analyzing: {path}...")
    try:
        doc = fitz.open(path)
        print(f" PDF Load Success. Total Pages: {len(doc)}")
        
        # Check Page 10 (usually has real content, not just TOC)
        page = doc[62] 
        text = page.get_text()
        images = page.get_images(full=True)
        
        print("\n--- SAMPLE TEXT (Page 10) ---")
        print(text[:500].strip()) # First 500 chars
        print("\n-----------------------------")
        
        print(f"  Images found on Page 62: {len(images)}")
        if len(images) > 0:
            print("Image extraction is possible!")
        else:
            print(" No images on Page 62 (This is normal if it's text-only, but check later pages).")
    except Exception as e:
        print(f" CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    analyze_pdf(PDF_PATH)