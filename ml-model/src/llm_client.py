from langchain_community.chat_models import GigaChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import GIGACHAT_CREDENTIALS, EMBEDDING_MODEL

def get_llm():
    return GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
