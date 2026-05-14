import json, re, asyncio
from langchain_core.prompts import PromptTemplate

INTENT_PROMPT = PromptTemplate.from_template(
    "Определи задачу: extract, qa, classify. Ответ только JSON: {{\"task\": \"...\"}}\nЗапрос: {query}"
)

QA_PROMPT = PromptTemplate.from_template(
    "Используй только контекст. Если ответа нет, скажи 'Нет данных'.\nКонтекст: {context}\nВопрос: {question}"
)

EXTRACT_PROMPT = PromptTemplate.from_template(
    "Извлеки: дата, контрагент, сумма, тип_расхода. Только JSON.\nКонтекст: {context}"
)

def _safe_parse_json(text: str) -> dict:
    try:
        cleaned = re.sub(r'`(?:json)?\s*|\s*```', '', text).strip()
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        return {}

async def route_and_execute_async(query: str, llm, vector_db) -> dict:
    # 1. Определяем интент (async)
    intent_text = await (INTENT_PROMPT | llm).ainvoke({"query": query})
    intent = _safe_parse_json(intent_text.content)
    task = intent.get("task", "unknown")

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Выполняем задачу
    if task == "extract":
        docs = await retriever.ainvoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        result = await (EXTRACT_PROMPT | llm).ainvoke({"context": context})
        return {"task": "extract", "status": "success", "data": _safe_parse_json(result.content), "message": None}

    elif task == "qa":
        docs = await retriever.ainvoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        result = await (QA_PROMPT | llm).ainvoke({"context": context, "question": query})
        return {"task": "qa", "status": "success", "data": {"answer": result.content}, "message": None}

    elif task == "classify":
        docs = await retriever.ainvoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        cat_prompt = PromptTemplate.from_template("Выбери 1 категорию из: Договор, Счет, Акт, Другое.\nКонтекст: {context}")
        result = await (cat_prompt | llm).ainvoke({"context": context})
        return {"task": "classify", "status": "success", "data": {"category": result.content.strip()}, "message": None}

    else:
        # Fallback: обычный QA
        docs = await retriever.ainvoke(query)
        context = "\n---\n".join([d.page_content for d in docs])
        result = await (QA_PROMPT | llm).ainvoke({"context": context, "question": query})
        return {"task": "fallback", "status": "success", "data": {"answer": result.content}, "message": None}