import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def create_faiss_index(text_chunks):
    print(embeddings)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("completed document embeddings in FAISS.")

def load_faiss_index():
    print("Load FAISS index from storage.")
    return FAISS.load_local("faiss_index", embeddings)
