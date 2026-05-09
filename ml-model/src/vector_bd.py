from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHROMA_PERSIST_DIR

def init_or_load_vectorstore(embeddings):
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

def add_texts_to_db(vectorstore, texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(texts)
    vectorstore.add_documents(docs)
    vectorstore.persist()  # Сохраняем на диск
