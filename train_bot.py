import os
from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from vector_store import create_faiss_index

def train_bot(pdf_folder):
    """Extracts text from PDFs, chunks it, and stores embeddings."""
    all_text = ""

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            all_text += text + "\n"

    # Split text into chunks
    text_chunks = chunk_text(all_text)

    # Generate embeddings and store in FAISS
    create_faiss_index(text_chunks)

if __name__ == "__main__":
    train_bot("syllabus_pdfs")
