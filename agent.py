import requests
def ask_ollama(question, context):
    prompt = (
        "You are a helpful expert assistant. "
        "Answer ONLY based on the provided CONTEXT below. "
        "If the answer is not present in the context, reply: "
        "'Sorry, I cannot answer this question from the available context.'\n\n"
        "Guidelines:\n"
        "- Provide a **brief but informative explanation**.\n"
        "- Write at least **two paragraphs** in your answer. Expand with details, examples, and facts found in the context.\n"
        "- Do NOT list just bullet points; use paragraph form.\n"
        "- Be clear and educational; do not invent or assume information not present in the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    data = {"model": "gemma3:latest", "prompt": prompt, "stream": False}
    try:
        r = requests.post("http://localhost:11434/api/generate", json=data)
        res = r.json()
        return res.get("response", "").strip() or "No response."
    except Exception as ex:
        return str(ex)