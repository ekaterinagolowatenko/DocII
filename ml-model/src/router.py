# ml-model/src/router.py
from langchain_core.prompts import PromptTemplate
from .utils import parse_llm_json

# 1. Промпт для определения намерения (интента)
INTENT_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
    Ты - диспетчер задач AI-системы документооборота. 
    Проанализируй запрос пользователя и верни ТОЛЬКО JSON в формате:
    {"task": "qa | extract | classify | analytics | unknown", "original_query": "текст запроса"}
    Запрос: {query}
    """
)

# 2. Промпты для конкретных задач
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="Используй только контекст. Если ответа нет, скажи 'Нет данных'. Контекст: {context}\nВопрос: {question}\nОтвет:"
)

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="Извлеки из текста: дата, контрагент, сумма, тип_расхода. Ответь ТОЛЬКО валидным JSON: {\"дата\": \"...\", \"контрагент\": \"...\", \"сумма\": \"...\", \"тип_расхода\": \"...\"}\nКонтекст: {context}"
)

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["context", "categories"],
    template="Выбери ОДНУ категорию из списка {categories} или 'Другое'. Контекст: {context}\nОтвет (только название):"
)

def route_and_execute(query: str, llm, vector_db) -> dict:
    """Единая точка входа. Читает запрос, понимает задачу, возвращает результат."""
    
    # 1. Определяем интент
    intent_chain = INTENT_PROMPT | llm
    intent_raw = intent_chain.invoke({"query": query})
    intent = parse_llm_json(intent_raw.content)
    task = intent.get("task", "unknown")
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Выполняем задачу
    if task == "qa":
        docs = retriever.invoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        result = (QA_PROMPT | llm).invoke({"context": context, "question": query})
        return {"task": "qa", "answer": result.content}
        
    elif task == "extract":
        docs = retriever.invoke("дата сумма контрагент договор счет акт")
        context = "\n---\n".join([d.page_content for d in docs])
        result = (EXTRACT_PROMPT | llm).invoke({"context": context})
        return {"task": "extract", "data": parse_llm_json(result.content)}
        
    elif task == "classify":
        docs = retriever.invoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        # Можно захардкодить категории или брать из запроса
        categories = "Договор, Счет, Акт, Накладная, Другое"
        result = (CLASSIFY_PROMPT | llm).invoke({"context": context, "categories": categories})
        return {"task": "classify", "category": result.content.strip()}
        
    elif task == "analytics":
        # Для аналитики возвращаем сырой текст + просим LLM сгенерировать pandas-код
        # (в проде лучше использовать безопасный eval или pre-defined templates)
        return {"task": "analytics", "note": "Аналитика требует передачи таблицы. Передайте данные явно."}
        
    else:
        # Fallback: обычный QA по всем документам
        docs = retriever.invoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        result = (QA_PROMPT | llm).invoke({"context": context, "question": query})
        return {"task": "fallback_qa", "answer": result.content}
