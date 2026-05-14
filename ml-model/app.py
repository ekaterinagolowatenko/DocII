from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.router import route_and_execute_async
from src.llm_client import get_llm, get_embeddings
from src.vector_db import init_or_load_vectorstore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Инициализация ML-сервиса...")
    app.state.llm = get_llm()
    app.state.embeddings = get_embeddings()
    app.state.vector_db = await init_or_load_vectorstore(app.state.embeddings)
    logger.info("✅ ML-сервис готов")
    yield

app = FastAPI(title="Document AI Service", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/process")
async def process_query(req: QueryRequest):
    try:
        result = await route_and_execute_async(
            query=req.query,
            llm=app.state.llm,
            vector_db=app.state.vector_db
        )
        return result
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка ML-сервиса")

@app.post("/api/upload")
async def upload_documents(folder_path: str):
    # Фоновая индексация (вызывается из 1С/бэкенда)
    from src.doc_loader import scan_files, extract_text_async_batch
    from src.vector_db import add_texts_to_vectorstore
    
    files = scan_files(folder_path)
    # Асинхронно читаем файлы в пуле потоков
    texts = await extract_text_async_batch(files)
    await add_texts_to_vectorstore(app.state.vector_db, texts)
    return {"status": "ok", "indexed_files": len(files)}
