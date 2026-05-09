import os

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
