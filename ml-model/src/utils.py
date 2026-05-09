import json, re

def parse_llm_json(text: str) -> dict:
    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    return json.loads(match.group()) if match else {}
