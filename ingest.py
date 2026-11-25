"""Extract page-aware text from PDFs and split it into overlapping chunks."""
from pathlib import Path
import pymupdf
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + chunk_size]).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(words):
            break
    return chunks

def extract_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    records = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            text = " ".join(text.split())
            if not text:
                continue
            for chunk_number, chunk in enumerate(split_into_chunks(text), start=1):
                records.append({
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                })
    return records

def ingest_folder(folder=DATA_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {folder}. Add lecture-note PDFs and run ingestion again."
        )
    records = []
    for pdf in pdfs:
        print(f"Reading: {pdf.name}")
        records.extend(extract_pdf(pdf))
    if not records:
        raise ValueError("No selectable text was extracted. Scanned PDFs may need OCR first.")
    return records

if __name__ == "__main__":
    from index import build_index
    chunks = ingest_folder()
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks from PDF files.")
